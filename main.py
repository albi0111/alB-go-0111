"""
main.py
───────
Entry point for the Nifty Options Algo Trading System.

This module is the `algo` CLI command (installed via setup.py).
All mode routing, capital prompts, and the real-time engine loop live here.

Commands:
    algo --uptrade               Live trading with real Upstox orders
    algo --papertrade            Live analysis, virtual orders
    algo --simulate              Historical replay (prompts for date/days)
    algo --simulate --days 10    Last 10 trading days
    algo --simulate --date YYYY-MM-DD  Specific date
    algo --simulate --from DATE --to DATE  Date range
    algo --status                Show running mode and open position
    algo --logs                  Tail live logs
    algo --stop                  Stop any running service
    algo --train-ai              Train AI model on collected dataset
    algo --backtest --days N     Full backtest with rich report
"""

from __future__ import annotations

import os
import sys
import time
import signal
import getpass
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ── Bootstrap ──────────────────────────────────────────────────────────────────
# Must happen before any other local imports so logger is ready
from core.logger import setup_logging
setup_logging()

from core.logger import log
import core.config as cfg
from data.data_storage import init_db
from data.data_fetcher import (
    fetch_1min_candles, aggregate_to_3min,
    fetch_spot_price, fetch_option_chain,
    get_futures_instrument_key, fetch_account_balance,
)
from strategy.breakout_strategy import run_strategy
from options.option_selector import select_option
from execution.order_manager import OrderManager
from ai.dataset_builder import record_signal_features, label_trade_outcome
from ai.inference import should_trade

console = Console()

# ── Shared state file for --status / --stop ────────────────────────────────────
_STATE_FILE = cfg.PROJECT_ROOT / ".algo_state"


def _write_state(mode: str, pid: int) -> None:
    _STATE_FILE.write_text(f"{mode}:{pid}")


def _clear_state() -> None:
    _STATE_FILE.unlink(missing_ok=True)


def _read_state() -> Optional[tuple[str, int]]:
    if not _STATE_FILE.exists():
        return None
    parts = _STATE_FILE.read_text().strip().split(":")
    return (parts[0], int(parts[1])) if len(parts) == 2 else None


# ── Token prompt (live mode only) ─────────────────────────────────────────────

def _prompt_trading_token() -> str:
    """
    Prompt the user for the daily Upstox trading access token.
    Saves it to .env and updates the runtime config.
    """
    console.print(Panel(
        "[bold yellow]Upstox Daily Trading Token Required[/bold yellow]\n\n"
        "Generate your token at: [link=https://upstox.com]upstox.com[/link]\n"
        "Login → My Account → API → Generate Token\n\n"
        "[dim]This token expires daily. It is stored in .env for this session.[/dim]",
        title="🔐 Token Setup",
        border_style="yellow",
    ))

    token = getpass.getpass("Paste your Upstox trading token: ").strip()
    if not token:
        console.print("[red]No token entered. Exiting.[/red]")
        sys.exit(1)

    # Persist to .env file
    env_path = cfg.PROJECT_ROOT / ".env"
    if env_path.exists():
        lines = env_path.read_text().splitlines()
        new_lines = []
        replaced = False
        for line in lines:
            if line.startswith("UPSTOX_TRADING_TOKEN="):
                new_lines.append(f"UPSTOX_TRADING_TOKEN={token}")
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            new_lines.append(f"UPSTOX_TRADING_TOKEN={token}")
        env_path.write_text("\n".join(new_lines) + "\n")
    else:
        env_path.write_text(f"UPSTOX_TRADING_TOKEN={token}\n")

    # Update runtime value
    os.environ["UPSTOX_TRADING_TOKEN"] = token
    cfg.UPSTOX_TRADING_TOKEN = token   # type: ignore[attr-defined]
    console.print("[green]✓ Token saved.[/green]\n")
    return token


def _prompt_capital() -> float:
    """Ask the user for virtual capital amount (paper/simulate modes)."""
    while True:
        try:
            val = click.prompt(
                "Enter virtual capital (₹)",
                default=10_000,
                type=float,
            )
            if val <= 0:
                console.print("[red]Capital must be positive.[/red]")
            else:
                return val
        except (ValueError, click.Abort):
            console.print("[red]Invalid input.[/red]")


# ── Real-Time Engine Loop ──────────────────────────────────────────────────────

def _run_live_loop(mode: str, capital: float) -> None:
    """
    Main real-time trading loop for live and papertrade modes.
    Runs every FETCH_INTERVAL_MIN minutes, aligned to market hours.
    """
    from core.constants import MARKET_OPEN, MARKET_CLOSE, FETCH_INTERVAL_MIN
    import schedule

    fut_key    = get_futures_instrument_key()
    order_mgr  = OrderManager(mode=mode)
    df_1min_rolling: list[dict] = []
    prev_trade_id: Optional[int] = None
    prev_pnl: Optional[float] = None

    console.print(Panel(
        f"[bold green]Mode: {mode.upper()}[/bold green]  |  "
        f"Capital: [cyan]₹{capital:,.0f}[/cyan]  |  "
        f"Futures: [cyan]{fut_key}[/cyan]\n"
        f"[dim]Press Ctrl+C to stop cleanly.[/dim]",
        title="🚀 Algo Engine Started",
        border_style="green",
    ))

    def _tick():
        nonlocal prev_trade_id, prev_pnl, df_1min_rolling

        now = datetime.now().time()
        if now < MARKET_OPEN or now > MARKET_CLOSE:
            return

        today = date.today()

        # ── Fetch latest 1-min candles and append to rolling window ────────
        df_new = fetch_1min_candles(fut_key, today, today)
        if df_new.empty:
            log.warning("No candle data returned. Retrying next tick.")
            return

        # Keep only candles we haven't seen yet (last N minutes)
        if df_1min_rolling:
            last_ts = df_1min_rolling[-1]["ts"]
            new_rows = df_new[df_new["ts"] > last_ts]
            df_1min_rolling.extend(new_rows.to_dict("records"))
        else:
            df_1min_rolling = df_new.to_dict("records")

        import pandas as pd
        df_1min = pd.DataFrame(df_1min_rolling)
        df_3min = aggregate_to_3min(df_1min)

        if df_3min.empty or len(df_3min) < 2:
            return

        spot = fetch_spot_price()
        if spot is None:
            return

        atr_val = current_atr(df_3min) if not df_3min.empty else 0.0
        
        # ── 1. Run Strategy & AI Filter (Ensures we have fresh scores for exits) ──
        signal = run_strategy(df_1min, df_3min, spot)
        record_signal_features(signal) 
        
        # ── 2. Manage open position(s) ───────────────────────────────────────────
        if order_mgr.has_open_position:
            from data.data_fetcher import fetch_market_quotes
            instr_keys = [p.instrument_key for p in order_mgr.positions]
            quotes = fetch_market_quotes(instr_keys) if instr_keys else {}
            
            exit_reasons = order_mgr.update(quotes, signal)
            if exit_reasons and prev_trade_id is not None:
                final_pnl = order_mgr.daily_pnl - (prev_pnl or 0)
                # prev_trade_id is already used for labelling in OrderManager._close now
                prev_trade_id = None
            return

        # ── 3. Check if past cutoff (Only if no position open) ───────────────
        from datetime import time as dtime
        if now >= dtime(15, 15):   # No new entries after 15:15
            order_mgr.close_eod(spot)
            return

        # ── 4. AI filter for new entries ────────────────────────────────────────
        ai_allow, ai_prob = should_trade(signal)
        if not ai_allow:
            log.info("Trade skipped by AI filter (P={prob:.3f}).", prob=ai_prob)
            return

        # ── Option selection ───────────────────────────────────────────────
        opt_chain = fetch_option_chain(spot)
        selection = select_option(opt_chain, spot, signal.option_type, capital)
        if selection is None:
            log.warning("No suitable option found. Skipping trade.")
            return

        # ── Build instrument key for the selected option ───────────────────
        from data.data_fetcher import get_options_expiry
        expiry = get_options_expiry()
        from core.constants import NIFTY_FUT_EXCHANGE
        # Option key format: NSE_FO|NIFTY<YY><MMM><STRIKE><TYPE>
        year_2   = str(expiry.year)[-2:]
        from core.constants import MONTH_ABBR
        month_str = MONTH_ABBR[expiry.month]
        strike_str = str(int(selection.strike))
        opt_key = f"NSE_FO|NIFTY{year_2}{month_str}{strike_str}{selection.option_type}"

        # ── Open trade ────────────────────────────────────────────────────
        prev_pnl = order_mgr.daily_pnl
        opened = order_mgr.try_open(signal, selection, opt_key)
        if opened:
            prev_trade_id = order_mgr.position.trade_id if order_mgr.position else None
            record_signal_features(signal, prev_trade_id)   # Record with trade_id

    # ── Schedule the tick and run ──────────────────────────────────────────
    schedule.every(FETCH_INTERVAL_MIN).minutes.do(_tick)

    def _handle_exit(sig, frame):
        console.print("\n[yellow]Stopping engine...[/yellow]")
        if order_mgr.has_open_position:
            spot = fetch_spot_price() or 0.0
            order_mgr.close_eod(spot)
        _clear_state()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    _write_state(mode, os.getpid())

    # Run once immediately, then on schedule
    _tick()
    while True:
        schedule.run_pending()
        time.sleep(1)


# ── ATR helper import at module scope ──────────────────────────────────────────
from indicators.atr import current_atr


# ── Click CLI ─────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--uptrade",    is_flag=True, help="Live trading with real Upstox orders.")
@click.option("--papertrade", is_flag=True, help="Live analysis, virtual orders only.")
@click.option("--simulate",   is_flag=True, help="Historical candle replay.")
@click.option("--backtest",   is_flag=True, help="Run backtest and show report.")
@click.option("--train-ai",   is_flag=True, help="Train AI model on collected dataset.")
@click.option("--status",     is_flag=True, help="Show current running service status.")
@click.option("--logs",       is_flag=True, help="Tail live log file.")
@click.option("--stop",       is_flag=True, help="Stop any running service (sends SIGTERM).")
@click.option("--days",    type=int,  default=None, help="Number of trading days for simulate/backtest.")
@click.option("--date",    type=str,  default=None, help="Specific date: YYYY-MM-DD.")
@click.option("--from",    "from_date", type=str, default=None, help="Start date: YYYY-MM-DD.")
@click.option("--to",      "to_date",   type=str, default=None, help="End date: YYYY-MM-DD.")
def cli(
    uptrade, papertrade, simulate, backtest, train_ai,
    status, logs, stop,
    days, date, from_date, to_date,
):
    """
    \b
    ╔═══════════════════════════════════════╗
    ║   Nifty Options Algo Trading System   ║
    ╚═══════════════════════════════════════╝

    \b
    Examples:
      algo --uptrade                        # Live real-money trading
      algo --papertrade                     # Paper trading (no real orders)
      algo --simulate --days 5             # Replay last 5 trading days
      algo --simulate --date 2026-03-24    # Replay a specific date
      algo --backtest --days 30            # Backtest report
      algo --train-ai                      # Train AI model
      algo --status                        # Check running service
      algo --stop                          # Stop running service
      algo --logs                          # Tail live logs
    """
    # ── Initialise DB on every invocation ────────────────────────────────
    init_db()

    # ╔═══════════════╗
    # ║  --status     ║
    # ╚═══════════════╝
    if status:
        state = _read_state()
        if state:
            console.print(f"[green]● Running:[/green] mode={state[0]}  PID={state[1]}")
        else:
            console.print("[dim]● No service currently running.[/dim]")
        return

    # ╔═══════════╗
    # ║  --stop   ║
    # ╚═══════════╝
    if stop:
        state = _read_state()
        if state:
            import signal as _sig
            try:
                os.kill(state[1], _sig.SIGTERM)
                console.print(f"[yellow]Sent SIGTERM to PID {state[1]} ({state[0]} mode).[/yellow]")
                _clear_state()
            except ProcessLookupError:
                console.print("[dim]Process not found. Clearing state.[/dim]")
                _clear_state()
        else:
            console.print("[dim]No running service found.[/dim]")
        return

    # ╔═══════════╗
    # ║  --logs   ║
    # ╚═══════════╝
    if logs:
        log_dir = cfg.LOGS_DIR
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file  = log_dir / f"algo_{today_str}.log"
        if not log_file.exists():
            console.print(f"[dim]No log file for today: {log_file}[/dim]")
            return
        subprocess.run(["tail", "-f", str(log_file)])
        return

    # ╔═════════════╗
    # ║  --train-ai ║
    # ╚═════════════╝
    if train_ai:
        from ai.model import train
        train()
        return

    # ╔════════════════╗
    # ║  --backtest    ║
    # ╚════════════════╝
    if backtest:
        from backtest.simulator import run_backtest
        capital = _prompt_capital()
        _pd = _parse_date(date)
        _fd = _parse_date(from_date)
        _td = _parse_date(to_date)
        fut_key = get_futures_instrument_key()
        # Extract instrument name from key (e.g. "NSE_FO|NIFTY26MARFUT" → "NIFTY26MARFUT")
        fut_inst = fut_key.split("|")[-1]
        run_backtest(
            days=days, from_date=_fd, to_date=_td,
            specific_date=_pd, capital=capital, fut_instrument=fut_inst,
        )
        return

    # ╔══════════════╗
    # ║  --simulate  ║
    # ╚══════════════╝
    if simulate:
        from backtest.simulator import run_simulation

        # Check token availability for data fetch
        if not cfg.UPSTOX_ANALYSIS_TOKEN:
            console.print("[red]UPSTOX_ACCESS_TOKEN not set in .env. Cannot fetch market data.[/red]")
            console.print("[dim]Set UPSTOX_ACCESS_TOKEN in .env then retry.[/dim]")
            sys.exit(1)

        capital = _prompt_capital()
        # If no date flags, prompt interactively
        if not days and not date and not from_date and not to_date:
            console.print("[cyan]No date specified. Choose:[/cyan]")
            console.print("  1. Last N days    2. Specific date    3. Date range")
            choice = click.prompt("Enter choice", type=click.Choice(["1", "2", "3"]), default="1")
            if choice == "1":
                days = click.prompt("Number of days", type=int, default=5)
            elif choice == "2":
                date = click.prompt("Date (YYYY-MM-DD)")
            else:
                from_date = click.prompt("From date (YYYY-MM-DD)")
                to_date   = click.prompt("To date (YYYY-MM-DD)")

        _pd = _parse_date(date)
        _fd = _parse_date(from_date)
        _td = _parse_date(to_date)
        fut_key  = get_futures_instrument_key()
        fut_inst = fut_key.split("|")[-1]
        run_simulation(
            days          = days,
            from_date     = _fd,
            to_date       = _td,
            specific_date = _pd,
            capital       = capital,
            fut_instrument = fut_inst,
            auto_fetch    = True,   # Fetch Upstox data before simulating
            source        = "simulate",
        )
        return

    # ╔════════════════╗
    # ║  --papertrade  ║
    # ╚════════════════╝
    if papertrade:
        if not cfg.UPSTOX_ANALYSIS_TOKEN:
            console.print("[red]UPSTOX_ANALYSIS_TOKEN not set in .env. Required for market data.[/red]")
            sys.exit(1)
        capital = _prompt_capital()
        _run_live_loop(mode="papertrade", capital=capital)
        return

    # ╔═════════════╗
    # ║  --uptrade  ║
    # ╚═════════════╝
    if uptrade:
        if not cfg.UPSTOX_ANALYSIS_TOKEN:
            console.print("[red]UPSTOX_ANALYSIS_TOKEN not set in .env. Required for market data.[/red]")
            sys.exit(1)
        _prompt_trading_token()
        capital = fetch_account_balance()
        if capital <= 0:
            console.print("[red]Could not fetch Upstox balance or balance is zero. Exiting.[/red]")
            sys.exit(1)
        console.print(f"[green]Available capital: ₹{capital:,.2f}[/green]")
        _run_live_loop(mode="live", capital=capital)
        return

    # ── No flag provided ───────────────────────────────────────────────────
    console.print("[yellow]No command specified. Use --help for usage.[/yellow]")


def _parse_date(s: Optional[str]) -> Optional[date]:
    if s is None:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        console.print(f"[red]Invalid date format: {s!r}. Use YYYY-MM-DD.[/red]")
        sys.exit(1)


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli()
