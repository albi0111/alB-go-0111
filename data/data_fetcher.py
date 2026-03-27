"""
data/data_fetcher.py
────────────────────
Fetches market data from Upstox API.

Design:
  - Fetches 1-min OHLCV candles (Nifty Spot for price, Nifty Fut for volume)
  - Aggregates 1-min → 3-min in-memory so no intrabar breakouts are missed
  - Fetches option chain snapshot for the relevant weekly expiry
  - Handles expiry rollover:
      Futures  → switches to next-month contract in the last week of month
      Options  → switches to next-week expiry on Monday and Tuesday
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Optional
import requests
import pandas as pd
import urllib.parse

import core.config as cfg
import core.constants as C
from core.logger import log


# ── Expiry Helpers ──────────────────────────────────────────────────────────────

def _last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of the given month (futures expiry)."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:   # 3 = Thursday
        d -= timedelta(days=1)
    return d


def _next_tuesday(from_date: date) -> date:
    """Return the next Tuesday on or after from_date (options weekly expiry)."""
    days_ahead = (1 - from_date.weekday()) % 7   # 1 = Tuesday
    if days_ahead == 0:
        days_ahead = 7
    return from_date + timedelta(days=days_ahead)


def get_futures_expiry(today: Optional[date] = None) -> date:
    """
    Return the active Nifty Futures expiry date.
    Rolls over to next month in the last FUTURES_ROLLOVER_DAYS_BEFORE_EXPIRY days.
    """
    today = today or date.today()
    expiry = _last_thursday_of_month(today.year, today.month)
    if (expiry - today).days < C.FUTURES_ROLLOVER_DAYS_BEFORE_EXPIRY:
        # Roll to next month
        next_month = today.month % 12 + 1
        next_year  = today.year + (1 if today.month == 12 else 0)
        expiry = _last_thursday_of_month(next_year, next_month)
    return expiry


def get_options_expiry(today: Optional[date] = None) -> date:
    """
    Return the active Nifty Options weekly expiry (Tuesday).
    On Monday / Tuesday → use NEXT week's expiry to avoid last-day illiquidity.
    """
    today = today or date.today()
    upcoming = _next_tuesday(today)
    if today.weekday() in C.OPTIONS_ROLLOVER_WEEKDAYS:
        # Already pointing to this week's Tuesday; push to next week
        upcoming += timedelta(days=7)
    return upcoming


def get_futures_instrument_key(today: Optional[date] = None) -> str:
    """Build the Upstox instrument key for the active Nifty Futures contract."""
    today = today or date.today()
    expiry = get_futures_expiry(today)
    year_2 = str(expiry.year)[-2:]   # e.g. 2026 → "26"
    month_str = C.MONTH_ABBR[expiry.month]
    symbol = f"NIFTY{year_2}{month_str}FUT"
    return f"{C.NIFTY_FUT_EXCHANGE}|{symbol}"


# ── Upstox REST Helpers ─────────────────────────────────────────────────────────

def _auth_headers(trading: bool = False) -> dict:
    """Return auth headers. Use analysis token by default (read-only)."""
    token = cfg.UPSTOX_TRADING_TOKEN if trading else cfg.UPSTOX_ANALYSIS_TOKEN
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _get(endpoint: str, params: dict = None, trading: bool = False) -> dict:
    """Make an authenticated GET to the Upstox v2 API."""
    url = f"{C.UPSTOX_BASE_URL}{endpoint}"
    response = requests.get(url, headers=_auth_headers(trading), params=params, timeout=10)
    response.raise_for_status()
    return response.json()


# ── 1-Min Candle Fetch ──────────────────────────────────────────────────────────

def fetch_1min_candles(
    instrument_key: str,
    from_date: date,
    to_date: date,
) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV candles from Upstox historical API.
    Returns a DataFrame with columns: [ts, open, high, low, close, volume]
    sorted chronologically.
    """
    inst_enc = urllib.parse.quote(instrument_key, safe="")
    endpoint = f"/historical-candle/intraday/{inst_enc}/1minute"
    
    try:
        is_fo = instrument_key.startswith("NSE_FO")
        data = _get(endpoint, trading=is_fo)
        candles = data.get("data", {}).get("candles", [])
    except Exception as exc:
        if instrument_key != C.NIFTY_SPOT_KEY:
            log.warning("Failed to fetch {key}. Falling back to Index Spot.", key=instrument_key)
            return fetch_1min_candles(C.NIFTY_SPOT_KEY, from_date, to_date)
            
        log.error("Failed to fetch Index candles: {err}", err=exc)
        return pd.DataFrame()

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume", "oi"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "volume"]]


# ── 1-Min → 3-Min Aggregation ──────────────────────────────────────────────────

def aggregate_to_3min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a 1-min OHLCV DataFrame into 3-min candles.
    Groups starting from market open (09:15) to avoid misaligned bars.
    """
    if df_1min.empty:
        return df_1min

    df = df_1min.copy()
    df = df.set_index("ts")

    # Anchor to 09:15 so bars are 09:15, 09:18, 09:21, ...
    df_3min = df.resample("3min", origin="start_day", offset="9h15min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    return df_3min.reset_index().rename(columns={"ts": "ts"})


# ── Spot Price ──────────────────────────────────────────────────────────────────

def fetch_spot_price() -> Optional[float]:
    """Fetch the current Nifty 50 index spot price (LTP)."""
    try:
        data = _get(
            "/market-quote/ltp",
            params={"instrument_key": C.NIFTY_SPOT_KEY},
        )
        ltp = data["data"][C.NIFTY_SPOT_KEY]["last_price"]
        return float(ltp)
    except Exception as exc:
        log.error("Failed to fetch Nifty spot price: {err}", err=exc)
        return None


def fetch_market_quotes(instrument_keys: list[str]) -> dict[str, dict]:
    """
    Fetch full market quotes (LTP, Bid, Ask) for multiple instruments.
    Returns: {"instrument_key": {"ltp": X, "bid": Y, "ask": Z}}
    """
    if not instrument_keys:
        return {}

    try:
        keys_str = ",".join(instrument_keys)
        data = _get("/market-quote/quotes", params={"instrument_key": keys_str})
        quotes = data.get("data", {})
        
        result = {}
        for key, qdata in quotes.items():
            result[key] = {
                "ltp": float(qdata.get("last_price", 0)),
                "bid": float(qdata.get("depth", {}).get("buy", [{}])[0].get("price", 0)),
                "ask": float(qdata.get("depth", {}).get("sell", [{}])[0].get("price", 0)),
            }
        return result
    except Exception as exc:
        log.error("Failed to fetch market quotes for {keys}: {err}", keys=instrument_keys, err=exc)
        return {}


# ── Option Chain ────────────────────────────────────────────────────────────────

def fetch_option_chain(spot_price: float, today: Optional[date] = None) -> pd.DataFrame:
    """
    Fetch the Nifty option chain for the active weekly expiry.
    Filters to ATM ± (ATM_RANGE_STRIKES × STRIKE_STEP).
    Returns DataFrame: [expiry, strike, option_type, ltp, bid, ask, oi, volume]
    """
    today = today or date.today()
    expiry = get_options_expiry(today)

    try:
        data = _get(
            "/option/chain",
            params={
                "instrument_key": C.NIFTY_SPOT_KEY,
                "expiry_date": expiry.isoformat(),
            },
        )
        chain = data.get("data", [])
    except Exception as exc:
        log.error("Failed to fetch option chain: {err}", err=exc)
        return pd.DataFrame()

    records = []
    atm = round(spot_price / C.STRIKE_STEP) * C.STRIKE_STEP
    for item in chain:
        strike = item.get("strike_price", 0)
        if abs(strike - atm) > C.ATM_RANGE_STRIKES * C.STRIKE_STEP:
            continue
        for otype in ("call_options", "put_options"):
            opt = item.get(otype, {}).get("market_data", {})
            records.append({
                "expiry":      expiry.isoformat(),
                "strike":      strike,
                "option_type": "CE" if otype == "call_options" else "PE",
                "ltp":         opt.get("ltp", 0),
                "bid":         opt.get("bid_price", 0),
                "ask":         opt.get("ask_price", 0),
                "oi":          opt.get("oi", 0),
                "volume":      opt.get("volume", 0),
            })

    return pd.DataFrame(records)


# ── Account Balance (live mode only) ───────────────────────────────────────────

def fetch_account_balance() -> float:
    """
    Fetch available margin / funds from Upstox (used in live mode).
    Returns usable capital in INR.
    """
    try:
        data = _get("/user/get-funds-and-margin", trading=True)
        equity = data.get("data", {}).get("equity", {})
        available = float(equity.get("available_margin", 0))
        log.info("Upstox available margin: ₹{amount:,.2f}", amount=available)
        return available
    except Exception as exc:
        log.error("Failed to fetch Upstox balance: {err}", err=exc)
        return 0.0
