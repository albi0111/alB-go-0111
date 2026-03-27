"""
indicators/atr.py
─────────────────
Average True Range (ATR) via Wilder's smoothing.
Used for adaptive stop-loss placement and trailing stop calculation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

import core.constants as C


def compute_atr(df: pd.DataFrame, period: int = C.ATR_PERIOD) -> pd.Series:
    """
    Compute ATR using Wilder's smoothing (alpha = 1/period).

    True Range = max(
        high - low,
        |high - prev_close|,
        |low  - prev_close|
    )

    Args:
        df:     DataFrame with columns [high, low, close].
        period: ATR lookback period (default 14).

    Returns:
        pd.Series named 'atr'.
    """
    df = df.copy().astype({"high": float, "low": float, "close": float})
    prev_close = df["close"].shift(1)

    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    atr.name = "atr"
    return atr


def current_atr(df: pd.DataFrame, period: int = C.ATR_PERIOD) -> float:
    """Return the latest ATR value as a float."""
    series = compute_atr(df, period)
    if series.empty or series.isna().all():
        return float("nan")
    return float(series.iloc[-1])


def stop_loss_distance(atr_value: float, multiplier: float = C.ATR_SL_MULTIPLIER) -> float:
    """
    Compute the stop-loss distance from the entry price.
    Returns: atr_value × multiplier
    """
    return atr_value * multiplier


def trailing_stop_distance(atr_value: float, multiplier: float = C.ATR_TRAIL_MULTIPLIER) -> float:
    """
    Compute the trailing stop distance (tighter than initial SL).
    Returns: atr_value × multiplier
    """
    return atr_value * multiplier


def classify_volatility(
    df: pd.DataFrame,
    period: int = C.ATR_PERIOD,
    lookback: int = C.REGIME_LOOKBACK,
    high_multiplier: float = C.ATR_HIGH_MULTIPLIER,
) -> str:
    """Legacy scalar version of classify_volatility."""
    res = classify_volatility_series(df, period, lookback, high_multiplier)
    return str(res.iloc[-1]) if not res.empty else "normal"


def classify_volatility_series(
    df: pd.DataFrame,
    period: int = C.ATR_PERIOD,
    lookback: int = C.REGIME_LOOKBACK,
    high_multiplier: float = C.ATR_HIGH_MULTIPLIER,
) -> pd.Series:
    """
    Vectorized volatility classification.
    Returns Series of 'high', 'normal', 'low'.
    """
    atr_series = compute_atr(df, period)
    rolling_median = atr_series.rolling(lookback, min_periods=lookback // 2).median()
    
    # Avoid division by zero
    valid_median = rolling_median.replace(0, np.nan)
    ratio = atr_series / valid_median

    res = pd.Series("normal", index=df.index)
    res[ratio >= high_multiplier] = "high"
    res[ratio <= 1.0 / high_multiplier] = "low"
    
    # Fill NaN (warmup) with 'normal'
    return res.fillna("normal")
