"""
indicators/rsi.py
─────────────────
Relative Strength Index (RSI) using Wilder's smoothing method.
Period: configurable (default 14 from constants).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

import core.constants as C


def compute_rsi(df: pd.DataFrame, period: int = C.RSI_PERIOD) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EMA with alpha = 1/period).

    Args:
        df:     DataFrame with a 'close' column.
        period: Lookback period (default 14).

    Returns:
        pd.Series of RSI values (0–100), aligned to df's index.
        First `period` values will be NaN until enough data accumulates.
    """
    close = df["close"].astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing = EMA with alpha = 1/period (equivalent to ewm min_periods)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # When avg_loss = 0 (perfect uptrend), RS = infinity → RSI = 100
    # Use numpy where to handle this gracefully without NaN
    rsi = pd.Series(index=avg_gain.index, dtype=float)
    zero_loss = avg_loss == 0
    rsi[zero_loss]  = 100.0
    rsi[~zero_loss] = 100 - (100 / (1 + avg_gain[~zero_loss] / avg_loss[~zero_loss]))
    rsi.name = "rsi"
    return rsi


def current_rsi(df: pd.DataFrame, period: int = C.RSI_PERIOD) -> float:
    """Return the latest RSI value as a float."""
    series = compute_rsi(df, period)
    if series.empty or series.isna().all():
        return float("nan")
    return float(series.iloc[-1])
