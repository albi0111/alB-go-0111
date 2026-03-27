"""
indicators/vwap.py
──────────────────
Intraday Volume Weighted Average Price (VWAP).

VWAP resets at market open (09:15 IST) each day.
Uses Nifty Futures candles since the index has no volume data.
"""

from __future__ import annotations

import pandas as pd

import core.constants as C
from core.logger import log


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute intraday VWAP for each row in df.

    Args:
        df: DataFrame with columns [ts, open, high, low, close, volume].
            ts must be timezone-naive or IST-localised datetime.

    Returns:
        pd.Series of VWAP values aligned to df's index.
        Values for bars before 09:15 are NaN.

    Formula:
        typical_price = (high + low + close) / 3
        VWAP_t = cumsum(typical_price × volume) / cumsum(volume)
        Accumulation resets at each new trading day.
    """
    if df.empty:
        return pd.Series(dtype=float)

    df = df.copy()
    df["date"] = pd.to_datetime(df["ts"]).dt.date
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = df["typical_price"] * df["volume"]

    # Cumulative sums reset each trading day
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"]    = df.groupby("date")["volume"].cumsum()

    vwap = df["cum_tp_vol"] / df["cum_vol"]
    vwap.name = "vwap"
    return vwap


def current_vwap(df: pd.DataFrame) -> float:
    """
    Return the latest VWAP value from the DataFrame.
    Convenience wrapper over compute_vwap().
    """
    series = compute_vwap(df)
    if series.empty or series.isna().all():
        log.warning("VWAP could not be computed — insufficient data.")
        return float("nan")
    return float(series.iloc[-1])
