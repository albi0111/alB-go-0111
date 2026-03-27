"""
indicators/ema.py
─────────────────
Exponential Moving Average (EMA) for configurable periods.
Provides EMA-20 (fast) and EMA-50 (slow) by default.
"""

from __future__ import annotations

import pandas as pd

import core.constants as C


def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Compute EMA for the given period on the 'close' column.

    Uses pandas ewm with span=period (standard EMA formula).
    The first `period - 1` values are partial estimates; they converge
    as more data becomes available.

    Args:
        df:     DataFrame with a 'close' column.
        period: EMA span (e.g. 20 or 50).

    Returns:
        pd.Series named 'ema_{period}'.
    """
    ema = df["close"].astype(float).ewm(span=period, adjust=False).mean()
    ema.name = f"ema_{period}"
    return ema


def compute_ema_pair(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute both EMA-fast (20) and EMA-slow (50) in one call.
    Returns: (ema_fast, ema_slow)
    """
    return (
        compute_ema(df, C.EMA_FAST_PERIOD),
        compute_ema(df, C.EMA_SLOW_PERIOD),
    )


def current_emas(df: pd.DataFrame) -> tuple[float, float]:
    """
    Return (ema_fast_now, ema_slow_now) as floats.
    """
    fast, slow = compute_ema_pair(df)
    return (
        float(fast.iloc[-1]) if not fast.empty else float("nan"),
        float(slow.iloc[-1]) if not slow.empty else float("nan"),
    )
