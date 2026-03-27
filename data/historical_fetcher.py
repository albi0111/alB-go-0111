"""
data/historical_fetcher.py
──────────────────────────
Fetches historical 1-min candles from Upstox and stores them in the database.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA SOURCE STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Upstox historical candle endpoint (v2) accepts:

  /v2/historical-candle/{instrument_key}/{unit}/{to_date}
  /v2/historical-candle/{instrument_key}/{unit}/{to_date}/{from_date}

Tested working keys:
  NSE_INDEX|Nifty 50   → 375 candles/day (primary price source)
  NSE_INDEX|Nifty Bank → works

Nifty Futures keys (e.g. NSE_FO|NIFTY26MARFUT) return 400 with the
read-only analysis token. The exact key requires the full Upstox
instrument master CSV which requires authentication.

Decision: Use Nifty 50 SPOT as the price source for simulation.
Volume analysis uses the same series (reasonable approximation for
the signal engine which uses volume spike ratio against rolling avg).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  fetch_and_store_history() is the entry point:
    1. Fetch 1-min Nifty 50 spot candles (up to 20 days per call)
    2. Validate completeness (≥ MIN_CANDLES_PER_DAY expected)
    3. Split by day and bulk-insert into DB under instrument='NIFTY50'
    4. Skip days already populated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import time
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

import core.config as cfg
from core.logger import log
from data.data_storage import get_session, Candle, save_candles


# ── Constants ─────────────────────────────────────────────────────────────────

# Upstox historical candle endpoint (v2) — works for index instruments
_BASE = "https://api.upstox.com/v2"

# Primary price source (Nifty 50 Spot index — works with read-only token)
NIFTY_SPOT_UPSTOX_KEY = "NSE_INDEX|Nifty 50"

# DB instrument name for simulation lookup
NIFTY_SPOT_DB_NAME = "NIFTY50"

# Minimum 1-min candles expected per full trading day (09:15–15:30 = 375 min).
# Lenient value to handle early close or holiday half-days.
MIN_CANDLES_PER_DAY = 60


# ── Retry-Backed HTTP Fetch ────────────────────────────────────────────────────

def _auth_headers() -> dict:
    token = cfg.UPSTOX_ANALYSIS_TOKEN
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _get_with_retry(
    url: str,
    params: dict = None,
    max_retries: int = 4,
    base_delay: float = 2.0,
) -> dict:
    """
    GET the URL with exponential backoff retry.

    Retries on: 429 (rate limit), 5xx, network errors, timeouts.
    Raises on: unrecoverable 4xx or exhausted retries.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url, headers=_auth_headers(), params=params or {}, timeout=15
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", base_delay * attempt))
                log.warning("Rate limited. Waiting {s:.1f}s (attempt {a}/{m}).",
                            s=wait, a=attempt, m=max_retries)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                log.warning("Server error {c}. Retry {a}/{m}.",
                            c=resp.status_code, a=attempt, m=max_retries)
                time.sleep(base_delay * attempt)
                continue
            # 4xx — not retryable
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            log.warning("Timeout. Retry {a}/{m}.", a=attempt, m=max_retries)
            time.sleep(base_delay * attempt)
            last_exc = TimeoutError("Upstox request timed out")
        except requests.exceptions.ConnectionError as exc:
            log.warning("Connection error: {e}. Retry {a}/{m}.", e=exc, a=attempt, m=max_retries)
            time.sleep(base_delay * attempt)
            last_exc = exc

    raise RuntimeError(f"Upstox API failed after {max_retries} retries. Last: {last_exc}")


# ── Core Fetch Function ────────────────────────────────────────────────────────

def fetch_candles_range(
    upstox_instrument_key: str,
    from_date: date,
    to_date: date,
    unit: str = "1minute",
) -> pd.DataFrame:
    """
    Fetch all 1-min OHLCV candles between from_date and to_date (inclusive).

    Upstox historical endpoint supports multi-day ranges in a single call;
    returns up to ~20 trading days per request.

    Returns DataFrame with [ts, open, high, low, close, volume].
    """
    inst_enc = urllib.parse.quote(upstox_instrument_key, safe="")
    url = (
        f"{_BASE}/historical-candle/{inst_enc}/{unit}"
        f"/{to_date.isoformat()}/{from_date.isoformat()}"
    )
    log.info("Fetching {key} | {f} → {t}", key=upstox_instrument_key, f=from_date, t=to_date)

    try:
        data = _get_with_retry(url)
    except Exception as exc:
        log.error("Fetch failed: {e}", e=exc)
        return pd.DataFrame()

    candles_raw = data.get("data", {}).get("candles", [])
    if not candles_raw:
        log.warning("No candles returned for {key} {f}→{t}.",
                    key=upstox_instrument_key, f=from_date, t=to_date)
        return pd.DataFrame()

    # Format: [timestamp, open, high, low, close, volume, oi]
    df = pd.DataFrame(
        candles_raw,
        columns=["ts", "open", "high", "low", "close", "volume", "oi"],
    )
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "volume"]]


# ── DB Helpers ────────────────────────────────────────────────────────────────

def _candles_exist_in_db(instrument: str, trade_date: date) -> bool:
    """True if the DB already has enough candles for this instrument + date."""
    day_start = datetime.combine(trade_date, datetime.min.time())
    day_end   = datetime.combine(trade_date + timedelta(days=1), datetime.min.time())
    with get_session() as session:
        count = (
            session.query(Candle)
            .filter(
                Candle.instrument == instrument,
                Candle.timeframe == 1,
                Candle.ts >= day_start,
                Candle.ts <  day_end,
            )
            .count()
        )
    return count >= MIN_CANDLES_PER_DAY


def _store_day_candles(df: pd.DataFrame, instrument_name: str, trade_date: date) -> int:
    """Filter DataFrame to a single date and bulk-insert into DB. Returns count."""
    # Filter to this date only (9:00–16:00 window to catch all sessions)
    day_start = datetime.combine(trade_date, datetime.min.time())
    day_end   = datetime.combine(trade_date + timedelta(days=1), datetime.min.time())
    day_df = df[(df["ts"] >= day_start) & (df["ts"] < day_end)]

    if day_df.empty:
        return 0

    records = [
        {
            "instrument": instrument_name,
            "timeframe":  1,
            "ts":         row.ts,
            "open":       float(row.open),
            "high":       float(row.high),
            "low":        float(row.low),
            "close":      float(row.close),
            "volume":     float(row.volume),
        }
        for row in day_df.itertuples(index=False)
    ]
    save_candles(records)
    return len(records)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def _trading_days_back(n: int) -> list[date]:
    """Return the last N trading days (Mon–Fri), oldest first."""
    days: list[date] = []
    d = date.today() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def fetch_and_store_history(
    instrument_key: Optional[str] = None,
    days: int = 5,
    trade_dates: Optional[list[date]] = None,
    also_fetch_spot: bool = True,   # Kept for backward compat — always True now
) -> dict:
    """
    Fetch Nifty 50 spot 1-min candles for the last N trading days.

    Single API call fetches the whole range, then data is split per day for storage.
    Days already in the DB are skipped automatically.

    Returns:
        {"dates_fetched": int, "total_candles": int, "missing_days": list[str]}
    """
    if not cfg.UPSTOX_ANALYSIS_TOKEN:
        raise RuntimeError(
            "UPSTOX_ACCESS_TOKEN not set in .env. Cannot fetch historical data."
        )

    dates = trade_dates or _trading_days_back(days)
    if not dates:
        return {"dates_fetched": 0, "total_candles": 0, "missing_days": []}

    # Check which days already exist in DB
    missing_dates = [d for d in dates if not _candles_exist_in_db(NIFTY_SPOT_DB_NAME, d)]
    already_have  = len(dates) - len(missing_dates)

    if already_have:
        log.info("✓ {n} day(s) already in DB — skipping.", n=already_have)

    summary = {"dates_fetched": already_have, "total_candles": 0, "missing_days": []}

    if not missing_dates:
        log.info("All requested days already cached in DB.")
        return summary

    # Fetch the whole date range in one API call (Upstox supports up to ~20 days)
    from_date = min(missing_dates)
    to_date   = max(missing_dates)

    log.info("=" * 60)
    log.info("Fetching Nifty 50 candles: {f} → {t} ({n} missing days)",
             f=from_date, t=to_date, n=len(missing_dates))
    log.info("=" * 60)

    df_all = fetch_candles_range(
        upstox_instrument_key = NIFTY_SPOT_UPSTOX_KEY,
        from_date             = from_date,
        to_date               = to_date,
    )

    if df_all.empty:
        summary["missing_days"] = [str(d) for d in missing_dates]
        log.error("No data returned from Upstox. Simulation will use DB cache only.")
        return summary

    log.info("Total candles received: {n}", n=len(df_all))

    # Split by day and store
    for trade_date in missing_dates:
        if _candles_exist_in_db(NIFTY_SPOT_DB_NAME, trade_date):
            continue   # Another process may have stored it between checks

        stored = _store_day_candles(df_all, NIFTY_SPOT_DB_NAME, trade_date)
        if stored >= MIN_CANDLES_PER_DAY:
            log.info("  → Stored {n} candles for {d}.", n=stored, d=trade_date)
            summary["dates_fetched"] += 1
            summary["total_candles"] += stored
        elif stored > 0:
            log.warning("  → Only {n} candles for {d} (partial day).", n=stored, d=trade_date)
            summary["dates_fetched"] += 1
            summary["total_candles"] += stored
        else:
            log.warning("  → No candles found for {d} (market holiday?).", d=trade_date)
            summary["missing_days"].append(str(trade_date))

    if summary["missing_days"]:
        log.warning("Missing data for: {days}", days=", ".join(summary["missing_days"]))

    log.info("Fetch complete: {fd} days | {tc} candles | {md} missing",
             fd=summary["dates_fetched"],
             tc=summary["total_candles"],
             md=len(summary["missing_days"]))

    return summary
