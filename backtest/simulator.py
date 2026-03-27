"""
backtest/simulator.py
──────────────────────
Historical candle replay simulation engine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIMULATION PIPELINE (per day)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each 1-min candle (chronologically):
  1. Compute VWAP, EMA, RSI, ATR, ADX on growing window
  2. Detect market regime (TRENDING / RANGING) using ADX + ATR
  3. Evaluate signal (bullish vs bearish composite scoring)
  4. If score ≥ threshold and no open position:
       a. Build mock OptionSelection (virtual premium = 2×ATR)
       b. record_signal_features() ← Stage 1 dataset capture
       c. try_open() ← opens position
  5. Each candle while position is open:
       ExitEngine (6-layer) + PhaseManager evaluate exit
  6. On exit:
       label_trade_outcome() ← Stage 2 dataset labelling

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After all days complete, validate_dataset() is called automatically:
  - Counts total labelled records
  - Checks for NULL feature columns
  - Logs win rate and average return
  - Raises SimulationDataError if records = 0

Usage:
    algo --simulate --days 10
    algo --simulate --date 2026-03-24
    algo --simulate --from 2026-03-01 --to 2026-03-24
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

import core.config as cfg
from data.data_storage import (
    get_session, Candle, get_training_dataframe, AIDataset,
)
from data.data_fetcher import aggregate_to_3min
from indicators.vwap   import compute_vwap
from indicators.rsi    import compute_rsi
from indicators.ema    import compute_ema_pair
from indicators.volume import volume_spike_ratio, volume_trend_series
from indicators.atr    import compute_atr
from indicators.adx    import compute_adx
from strategy.conditions import evaluate_signal
from strategy.regime    import detect_regime, MarketRegime, detect_regime_series
from options.option_selector import OptionSelection
from execution.order_manager import OrderManager
from core.logger import log
from core.constants import MARKET_OPEN, MARKET_CLOSE, BREAKOUT_WINDOW_END

console = Console()


# ── Custom Exceptions ──────────────────────────────────────────────────────────

class SimulationDataError(RuntimeError):
    """Raised when simulation produces no dataset records — silent failure guard."""


# ── Date Range Helpers ────────────────────────────────────────────────────────

def _trading_days_back(n: int) -> list[date]:
    """Return the last N trading days (Mon–Fri), most recent last."""
    days: list[date] = []
    d = date.today() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def _date_range(from_date: date, to_date: date) -> list[date]:
    days = []
    d = from_date
    while d <= to_date:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


# ── Candle Loading ─────────────────────────────────────────────────────────────

def _load_candles_from_db(instrument: str, trade_date: date) -> pd.DataFrame:
    """Load 1-min candles for a single trading day from the database."""
    with get_session() as session:
        rows = (
            session.query(Candle)
            .filter(
                Candle.instrument == instrument,
                Candle.timeframe == 1,
                Candle.ts >= datetime.combine(trade_date, datetime.min.time()),
                Candle.ts <  datetime.combine(trade_date + timedelta(days=1), datetime.min.time()),
            )
            .order_by(Candle.ts)
            .all()
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {"ts": r.ts, "open": r.open, "high": r.high,
         "low": r.low, "close": r.close, "volume": r.volume}
        for r in rows
    ])


# ── Single-Day Simulation ──────────────────────────────────────────────────────

def simulate_day(
    trade_date: date,
    fut_instrument: str,
    capital: float,
    mode: str = "simulate",
    om: Optional[OrderManager] = None
) -> tuple[dict, list, list, list]:
    """
    Replay one trading day's 1-min candles through the full strategy pipeline.
    O(n) performance: ALL indicators pre-computed and aligned before loop.
    """
    # 1. Load Data
    df_1min = _load_candles_from_db(fut_instrument, trade_date)
    if df_1min.empty:
        df_1min = _load_candles_from_db("NIFTY50", trade_date)

    if df_1min.empty:
        log.warning(f"No data for {trade_date} ({fut_instrument})")
        return ({"date": trade_date, "trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, 
                 "max_drawdown": 0.0, "skipped": True, "dataset_records": 0}, [], [], [])

    # 2. Vectorized Pre-computation (1-min)
    vwap_series  = compute_vwap(df_1min)
    spike_series = volume_spike_ratio(df_1min)
    vol_trend_s  = volume_trend_series(df_1min, lookback=5)
    
    # 3. Vectorized Pre-computation (3-min)
    df_3min = aggregate_to_3min(df_1min)
    if df_3min.empty:
        return ({"date": trade_date, "trades": 0, "pnl": 0.0, "wins": 0, "losses": 0, "skipped": True, "dataset_records": 0}, [], [], [])

    atr_series   = compute_atr(df_3min)
    rsi_series   = compute_rsi(df_3min)
    ema_df       = pd.concat(compute_ema_pair(df_3min), axis=1)
    ema_df.columns = ["ema20", "ema50"]
    regime_df    = detect_regime_series(df_3min)
    
    # 4. Multi-Timeframe Alignment (STRICT NO-LEAKAGE)
    # Shift 3-min indicators by 1 to ensure only CLOSED candles are used.
    # A 3-min bar for [09:15, 09:18) is timestamped 09:15. 
    # Shifting it makes it available to 1-min candles from 09:18 onwards.
    df_3min["ts_obj"] = pd.to_datetime(df_3min.index)
    morning_mask = (df_3min["ts_obj"].dt.time >= MARKET_OPEN) & (df_3min["ts_obj"].dt.time <= BREAKOUT_WINDOW_END)
    
    df_3min["brk_high"] = 0.0
    df_3min["brk_low"]  = 0.0
    if not df_3min.loc[morning_mask].empty:
        df_3min.loc[morning_mask, "brk_high"] = df_3min.loc[morning_mask, "high"].expanding().max()
        df_3min.loc[morning_mask, "brk_low"]  = df_3min.loc[morning_mask, "low"].expanding().min()
        
        final_h = df_3min.loc[morning_mask, "brk_high"].iloc[-1]
        final_l = df_3min.loc[morning_mask, "brk_low"].iloc[-1]
        after_mask = df_3min["ts_obj"].dt.time > BREAKOUT_WINDOW_END
        df_3min.loc[after_mask, "brk_high"] = final_h
        df_3min.loc[after_mask, "brk_low"]  = final_l

    # Group all 3-min features for shifting
    features_3min = pd.concat([
        atr_series.rename("atr"), 
        rsi_series.rename("rsi"), 
        ema_df, 
        regime_df, 
        df_3min[["brk_high", "brk_low"]]
    ], axis=1)
    
    # ── LEAKAGE GUARD: SHIFT BY 1 ──
    # This ensures the 3-min indicator for period T is only used at T + 3mins.
    features_3min_shifted = features_3min.shift(1)
    features_3min_shifted.index.name = "ts"
    features_3min_shifted = features_3min_shifted.reset_index()
    features_3min_shifted["ts"] = pd.to_datetime(features_3min_shifted["ts"])
    
    # Merge with 1-min
    df_1min["ts"] = pd.to_datetime(df_1min["ts"])
    df_merged = pd.merge_asof(
        df_1min.sort_values("ts"),
        features_3min_shifted.sort_values("ts"),
        on="ts",
        direction="backward"
    )
    
    # Join 1-min indicators
    df_merged["vwap"] = vwap_series.values
    df_merged["spike_ratio"] = spike_series.values
    df_merged["vol_trend"] = vol_trend_s.values
    
    # 5. O(1) Candle Loop (STRICTLY NO RECOMPUTATION)
    peak_pnl = 0.0
    # Track day-level metrics
    prev_pnl = 0.0
    day_stats = {
        "date": trade_date, "trades": 0, "wins": 0, "losses": 0, 
        "pnl": 0.0, "max_drawdown": 0.0, "skipped": False, "dataset_records": 0
    }
    
    # NEW: Diagnostic Collection (Phase 1 & 2)
    day_scores: list[float] = []
    rejected_signals = []
    trade_results: list[dict] = []   # To collect individual trade outcomes
    # Initialize or reset OrderManager
    if om is None:
        om = OrderManager(mode="simulate", capital=capital)
    else:
        om.reset_daily_pnl()
    
    # Frequency Guards: 15-min per side (15 candles)
    last_buy_idx = -100
    last_sell_idx = -100
    COOLDOWN_SIZE = 15

    for i, row in df_merged.iterrows():
        # Warmup: need indicators to be populated
        if pd.isna(row["atr"]) or pd.isna(row["rsi"]): continue
        
        spot = float(row["close"])
        atr  = float(row["atr"])

        om.tick_cooldown()
        
        # Signal Evaluation (using pre-computed features)
        # We compute this every candle because both entry and exit logic depend on current scores.
        signal = evaluate_signal(
            price         = spot,
            vwap          = row["vwap"],
            rsi_val       = row["rsi"],
            ema20         = row["ema20"],
            ema50         = row["ema50"],
            atr_val       = atr,
            spike_ratio   = row["spike_ratio"],
            df_3min       = df_3min,
            adx_val       = row["adx"],
            vol_trend     = row["vol_trend"],
            breakout_high = row["brk_high"],
            breakout_low  = row["brk_low"]
        )
        
        # Adaptive Threshold: rolling mean + 0.5 * std (window=200)
        current_threshold = cfg.SIGNAL_STRENGTH_THRESHOLD
        if len(day_scores) > 100:
            window = day_scores[-200:]
            std_val = np.std(window)
            current_threshold = np.mean(window) + 0.5 * std_val
            # Leveling: Threshold = max(0.50, current_threshold)
            current_threshold = max(0.50, current_threshold)
        
        # Acceptance check
        is_accepted = signal.signal_strength >= current_threshold
        
        # Log entry for rejected_signals logic
        rejection_override = None

        # Position Guard
        if is_accepted and om.has_open_position:
            rejection_override = "EXISTING_POSITION"
        
        # Directional Guard: Max 1 per side per 15 mins
        if is_accepted and not rejection_override:
            if signal.direction == "BUY" and (i - last_buy_idx) < COOLDOWN_SIZE:
                rejection_override = "DIRECTIONAL_COOLDOWN_BUY"
            elif signal.direction == "SELL" and (i - last_sell_idx) < COOLDOWN_SIZE:
                rejection_override = "DIRECTIONAL_COOLDOWN_SELL"
        
        # Final Decision
        final_decision = is_accepted and not rejection_override
        
        # Track all scores for distribution analysis
        day_scores.append(signal.signal_strength)
        
        # ── Execution Filter: Minimum RR Constraint ───────────────────────────
        # Target / Risk must be >= 1.2 to consider entry
        if final_decision:
            rr = 0.0
            risk = abs(signal.spot_price - signal.stop_loss)
            reward = abs(signal.target - signal.spot_price)
            if risk > 0:
                rr = reward / risk
            
            if rr < 1.2:
                final_decision = False
                rejection_override = f"LOW_RR ({rr:.2f})"
        
        # Track rejected signals (if not entered)
        if not final_decision and signal.raw_score >= 0.45:
            rejected_signals.append({
                "ts": row["ts"],
                "dir": "BUY" if signal.bullish_score >= signal.bearish_score else "SELL",
                "score": signal.signal_strength,
                "raw": signal.raw_score,
                "thresh": round(current_threshold, 3),
                "reason": rejection_override or signal.rejection_reason or "BELOW_THRESHOLD",
                "penalties": signal.penalty_log
            })
        
        # Position Management
        # ── Update & Exit Management ──────────────────────────────────────────
        prev_pnl_before_update = om.daily_pnl
        exit_reasons = om.update(
            current_spot   = spot,
            atr_value      = atr,
            vwap           = row["vwap"],
            vol_trend      = row["vol_trend"],
            adx_val        = row["adx"],
            breakout_high  = row["brk_high"],
            breakout_low   = row["brk_low"],
            bullish_score  = signal.bullish_score,
            bearish_score  = signal.bearish_score,
        )
        
        if exit_reasons:
            for reason in exit_reasons:
                day_stats["trades"] += 1
                day_stats["dataset_records"] += 1
                # Note: Exact per-trade PnL is tracked in trade_results
                # We can't easily isolate individual PnL here since multiple could close,
                # but we can record the total PnL jump.
                pass
            
            pnl_jump = om.daily_pnl - prev_pnl_before_update
            trade_results.append({
                "pnl": round(pnl_jump, 2),
                "reason": "|".join(exit_reasons),
                "ts": row["ts"]
            })
            if om.daily_pnl > peak_pnl: peak_pnl = om.daily_pnl
            prev_pnl = om.daily_pnl
            # We don't continue anymore, as multiple positions might still be open
            # or a new entry might be allowed in the same candle (though usually blocked by cooldown)

        # Market Close Check (no new entries after 15:15)
        if pd.to_datetime(row["ts"]).time() >= MARKET_CLOSE: break

        # Use our final decision flag
        if not final_decision: continue

        # Regime Context from Merged Data
        regime = MarketRegime(
            type             = row["type"],
            adx              = row["adx"],
            volatility       = row["volatility"],
            use_trailing_sl  = row["use_trailing_sl"],
            exit_tolerance   = row["exit_tolerance"],
            signal_threshold = row["signal_threshold"]
        )

        # Build Mock Option
        premium = max(atr * 2, 50.0)
        sl_points = abs(signal.spot_price - signal.stop_loss)
        lots = om.calculate_institutional_lots(
            premium    = premium,
            atr        = atr,
            sl_points  = sl_points,
            ml_prob    = signal.ml_quality_prob
        )
        
        mock_selection = OptionSelection(
            strike      = round(spot / 50) * 50,
            option_type = signal.option_type,
            expiry      = "simulated",
            ltp         = premium,
            bid         = premium - 0.5,
            ask         = premium + 0.5,
            oi          = 10000.0,
            volume      = 100000.0,
            lots        = lots,
            total_cost  = premium * lots * cfg.LOT_SIZE
        )

        # Open Position
        if om.try_open(
            signal          = signal,
            selection       = mock_selection,
            instrument_key  = "SIMULATED",
            regime          = regime,
            entry_ts        = pd.to_datetime(row["ts"]).to_pydatetime()
        ):
            day_stats["dataset_records"] += 1
            if signal.direction == "BUY": last_buy_idx = i
            else:                         last_sell_idx = i

        # Track Drawdown
        if om.daily_pnl < peak_pnl:
            day_stats["max_drawdown"] = max(day_stats["max_drawdown"], peak_pnl - om.daily_pnl)
        else:
            peak_pnl = om.daily_pnl

    # EOD Close
    if om.has_open_position:
        prev_pnl_before_eod = om.daily_pnl
        spot_at_close = float(df_1min["close"].iloc[-1])
        reasons = om.close_eod(current_spot=spot_at_close)
        
        pnl_diff = round(om.daily_pnl - prev_pnl_before_eod, 2)
        trade_results.append({
            "pnl": pnl_diff,
            "reason": "EOD_FLUSH",
            "ts": df_1min["ts"].iloc[-1]
        })

        day_stats["dataset_records"] += len(reasons)
        day_stats["trades"] = om.trade_count

    # Final Stats
    day_stats["pnl"] = round(om.daily_pnl, 2)
    day_stats["trades"] = om.trade_count
    if om.trade_count > 0:
        day_stats["wins"] = sum(1 for t in trade_results if t["pnl"] > 0)
        day_stats["losses"] = sum(1 for t in trade_results if t["pnl"] <= 0)
        
    log.info(f"Day {trade_date} complete: {day_stats['trades']} trades, PnL {day_stats['pnl']}")
    return day_stats, day_scores, rejected_signals, trade_results


# ── Dataset Validation ─────────────────────────────────────────────────────────

def validate_dataset() -> dict:
    """
    Query the ai_dataset table and produce a validation report.

    Returns dict:
        total_records, labelled_records, null_feature_count,
        win_rate, avg_return_pct

    Raises:
        SimulationDataError if no labelled records exist (silent failure guard).
    """
    with get_session() as session:
        total     = session.query(AIDataset).count()
        labelled  = session.query(AIDataset).filter(AIDataset.label.isnot(None)).count()
        null_feat = (
            session.query(AIDataset)
            .filter(
                AIDataset.label.isnot(None),
                AIDataset.bullish_score.is_(None),
            )
            .count()
        )
        wins = session.query(AIDataset).filter(AIDataset.label == 1).count()

    if labelled == 0:
        raise SimulationDataError(
            "Dataset is empty after simulation. "
            "No trades were executed or labelling failed. "
            "Check logs for errors."
        )

    # Average return from DataFrame (supports clipping check)
    try:
        df = get_training_dataframe()
        avg_return = float(df["return_pct"].mean()) if not df.empty else 0.0
    except Exception:
        avg_return = 0.0

    win_rate = (wins / labelled * 100) if labelled > 0 else 0.0

    report = {
        "total_records":    total,
        "labelled_records": labelled,
        "null_feat_count":  null_feat,
        "win_rate_pct":     round(win_rate, 1),
        "avg_return_pct":   round(avg_return, 2),
    }

    console.print(Panel(
        f"[bold]Dataset Validation[/bold]\n\n"
        f"  Total records:     [cyan]{total}[/cyan]\n"
        f"  Labelled records:  [green]{labelled}[/green]\n"
        f"  NULL feature rows: [{'red' if null_feat else 'green'}]{null_feat}[/{'red' if null_feat else 'green'}]\n\n"
        f"  Win rate:          [bold cyan]{win_rate:.1f}%[/bold cyan]\n"
        f"  Avg return:        [{'green' if avg_return >= 0 else 'red'}]{avg_return:+.2f}%[/{'green' if avg_return >= 0 else 'red'}]",
        title="📊 AI Dataset Validation",
        border_style="cyan",
    ))

    if null_feat > 0:
        log.warning(
            "{n} labelled rows have NULL feature columns. "
            "Check that record_signal_features() receives the full score_breakdown.",
            n=null_feat,
        )

    return report


# ── Full Simulation Runner ─────────────────────────────────────────────────────

def run_simulation(
    days: Optional[int] = None,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    specific_date: Optional[date] = None,
    capital: float = 50_000.0,
    fut_instrument: str = "NIFTY26MARFUT",
    auto_fetch: bool = True,
    source: str = "simulate",
) -> None:
    """
    Orchestrate the full simulation pipeline:
      1. Fetch missing historical candles from Upstox (if auto_fetch=True)
      2. Simulate each day
      3. Print rich results table
      4. Run dataset validation

    Args:
        days:          Last N trading days.
        from_date:     Start of custom date range.
        to_date:       End of custom date range.
        specific_date: Single day to simulate.
        capital:       Virtual capital per day.
        fut_instrument: Nifty Futures instrument name (no exchange prefix).
        auto_fetch:    Automatically fetch Upstox candles before simulating.
        source:        Tag for dataset records ("simulation" / "papertrade").
    """
    # ── Determine date list ────────────────────────────────────────────────────
    if specific_date:
        trade_dates = [specific_date]
    elif days:
        trade_dates = _trading_days_back(days)
    elif from_date and to_date:
        trade_dates = _date_range(from_date, to_date)
    else:
        raise ValueError("Specify --days, --date, or --from/--to.")

    from data.data_storage import init_db
    init_db()

    console.rule(f"[bold cyan]🚀 Simulation: {len(trade_dates)} trading days | "
                 f"Capital ₹{capital:,.0f}")

    # ── Step 1: Fetch data ─────────────────────────────────────────────────────
    if auto_fetch:
        from data.historical_fetcher import fetch_and_store_history
        from data.data_fetcher import get_futures_instrument_key
        # Derive full instrument key for the current active contract
        fut_key = get_futures_instrument_key()
        try:
            fetch_summary = fetch_and_store_history(
                instrument_key = fut_key,
                trade_dates    = trade_dates,
            )
            if fetch_summary["missing_days"]:
                console.print(
                    f"[yellow]⚠ Missing data for: "
                    f"{', '.join(fetch_summary['missing_days'])}[/yellow]"
                )
        except RuntimeError as exc:
            console.print(f"[red]Data fetch failed: {exc}[/red]")
            console.print("[yellow]Continuing with cached DB data (if any).[/yellow]")

    # ── Step 2: Simulate each day ──────────────────────────────────────────────
    all_day_stats = []
    total_scores = []
    all_rejected = []
    all_individual_trades = []

    # Initialize persistent state
    om = OrderManager(mode=source, capital=capital)

    for day in trade_dates:
        try:
            # result = (stats, scores, rejected, trade_results)
            stats, scores, rejected, individual_trades = simulate_day(
                day, fut_instrument, capital, mode=source, om=om
            )
            all_day_stats.append(stats)
            total_scores.extend(scores)
            all_rejected.extend(rejected)
            all_individual_trades.extend(individual_trades)
        except Exception as e:
            log.error("Failed to simulate {d}: {e}", d=day, e=str(e))
            all_day_stats.append({"date": day, "skipped": True, "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "max_drawdown": 0.0, "dataset_records": 0})

    # ── Phase 1: Distribution Analysis ─────────────────────────────────────────
    console.print(f"\n[bold cyan][Simulator][/bold cyan] Computing distribution for {len(total_scores)} signals...")
    log.info("Computing distribution for {n} signals.", n=len(total_scores))
    if total_scores:
        import numpy as np
        p50 = np.percentile(total_scores, 50)
        p75 = np.percentile(total_scores, 75)
        p90 = np.percentile(total_scores, 90)
        p95 = np.percentile(total_scores, 95)
        dist_str = (f"min: {min(total_scores):.3f} | "
                    f"p50: {p50:.3f} | p75: {p75:.3f} | p90: {p90:.3f} | p95: {p95:.3f} | "
                    f"max: {max(total_scores):.3f}")
        
        console.print(Panel(
            f"[bold yellow]Signal Strength Distribution[/bold yellow]\n\n{dist_str}",
            title="📊 Diagnostics", border_style="yellow"
        ))

    # ── Phase 2: Filter Impact Analysis ─────────────────────────────────────────
    if all_rejected:
        from rich.table import Table
        from rich import box
        impact_table = Table(title="Filter Rejection Breakdown", box=box.ROUNDED)
        # Sort by score (strongest signals that didn't enter)
        all_rejected.sort(key=lambda x: x["score"], reverse=True)
        top_20 = all_rejected[:20]
        
        rej_table = Table(title="Top 20 Rejected Signals (Potential missed entries)", box=box.SIMPLE)
        rej_table.add_column("Timestamp", style="dim")
        rej_table.add_column("Dir")
        rej_table.add_column("Final", justify="right")
        rej_table.add_column("Raw", justify="right")
        rej_table.add_column("Reason", style="red")
        rej_table.add_column("Penalties", style="yellow")
        
        for r in top_20:
            rej_table.add_row(
                str(r["ts"]), r["dir"], f"{r['score']:.3f}", f"{r['raw']:.3f}", 
                r["reason"], str(r["penalties"])
            )
        console.print(rej_table)

    # ── Step 3: Results table ──────────────────────────────────────────────────
    import rich.box
    table = Table(
        title=f"Simulation Results ({fut_instrument})",
        box=rich.box.DOUBLE_EDGE,
        header_style="bold magenta",
        show_footer=True
    )
    
    table.add_column("Date",       style="dim", footer="TOTAL")
    table.add_column("Trades",     justify="right")
    table.add_column("W",          justify="center")
    table.add_column("L",          justify="center")
    table.add_column("PnL (₹)",    justify="right", style="bold")
    table.add_column("Max DD (₹)", justify="right", style="red")
    table.add_column("Dataset",    justify="right")
    table.add_column("Status",     justify="center")

    total_pnl    = 0.0
    total_trades = 0
    total_wins   = 0
    total_losses = 0
    total_ds     = 0

    for s in all_day_stats:
        pnl_str = (f"[green]+{s['pnl']:,.0f}[/green]"
                   if s["pnl"] >= 0 else f"[red]{s['pnl']:,.0f}[/red]")
        status  = "[dim]No data[/dim]" if s.get("skipped") else "✅"
        table.add_row(
            str(s["date"]),
            str(s["trades"]),
            str(s.get("wins", 0)),
            str(s.get("losses", 0)),
            pnl_str,
            f"₹{s.get('max_drawdown', 0):,.0f}",
            str(s.get("dataset_records", 0)),
            status,
        )
        if not s.get("skipped"):
            total_pnl    += s["pnl"]
            total_trades += s["trades"]
            total_wins   += s.get("wins", 0)
            total_losses += s.get("losses", 0)
            total_ds     += s.get("dataset_records", 0)

    # Update footers after aggregation
    table.columns[1].footer = str(total_trades)
    table.columns[4].footer = f"₹{total_pnl:+,.2f}"
    table.columns[6].footer = str(total_ds)

    console.print(table)

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    # Detailed Expectancy Breakdown
    pnl_records = [t["pnl"] for t in all_individual_trades]
    win_records = [p for p in pnl_records if p > 0]
    loss_records = [p for p in pnl_records if p <= 0]
    
    avg_win  = np.mean(win_records) if win_records else 0.0
    avg_loss = np.mean(loss_records) if loss_records else 0.0
    # win_rate defined by individual trades now
    win_rate_actual = len(win_records) / len(pnl_records) if pnl_records else 0.0
    expectancy = (win_rate_actual * avg_win) + ((1 - win_rate_actual) * avg_loss)
    
    # Simplified R multiple: We'll log the average win/loss ratio
    r_mult = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    console.print(Panel(
        f"[bold cyan]Expectancy Breakdown[/bold cyan]\n\n"
        f"Win Rate: [green]{win_rate_actual*100:.1f}%[/green] | "
        f"Avg Win: [green]₹{avg_win:,.2f}[/green] | "
        f"Avg Loss: [red]₹{avg_loss:,.2f}[/red]\n"
        f"Reward/Risk Ratio: [yellow]{r_mult:.2f}[/yellow] | "
        f"Expectancy: [bold]{'positive' if expectancy > 0 else 'negative'} (₹{expectancy:,.2f} per trade)[/bold]",
        title="Profitability Report", expand=False
    ))
    pnl_color = "green" if total_pnl >= 0 else "red"

    console.print(f"\n[bold]Total Trades:[/bold]   {total_trades}")
    console.print(f"[bold]Win Rate:[/bold]       {win_rate_actual*100:.1f}%  ({total_wins}W / {total_losses}L)")
    console.print(f"[bold]Total PnL:[/bold]      [{pnl_color}]₹{total_pnl:,.2f}[/{pnl_color}]")
    console.print(f"[bold]Dataset Records:[/bold] {total_ds} new entries added")

    # ── Step 4: Dataset validation ─────────────────────────────────────────────
    console.print()
    try:
        validate_dataset()
    except Exception as exc:
        console.print(f"[bold red]❌ Dataset validation failed:[/bold red] {exc}")
        if total_trades == 0:
            console.print(
                "[yellow]ℹ No trades were triggered. "
                "Check signal threshold (SIGNAL_STRENGTH_THRESHOLD) "
                "or verify that candle data matches market conditions.[/yellow]"
            )

    console.rule()


# ── Backward-Compatible Alias ─────────────────────────────────────────────────

def run_backtest(
    days: Optional[int] = None,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    specific_date: Optional[date] = None,
    capital: float = 50_000.0,
    fut_instrument: str = "NIFTY26MARFUT",
) -> None:
    """Alias for run_simulation() (used by --backtest CLI flag)."""
    run_simulation(
        days=days, from_date=from_date, to_date=to_date,
        specific_date=specific_date, capital=capital,
        fut_instrument=fut_instrument,
        auto_fetch=True, source="simulation",
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int)
    parser.add_argument("--date")
    parser.add_argument("--from", dest="from_date")
    parser.add_argument("--to", dest="to_date")
    parser.add_argument("--capital", type=float, default=50000.0)
    parser.add_argument("--instrument", default="NIFTY26MARFUT")
    args = parser.parse_args()

    run_simulation(
        days      = args.days,
        from_date = args.from_date,
        to_date   = args.to_date,
        specific_date = args.date,
        capital   = args.capital,
        fut_instrument = args.instrument,
        auto_fetch = True
    )
