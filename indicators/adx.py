"""
indicators/adx.py
─────────────────
Average Directional Index (ADX) — measures TREND STRENGTH, not direction.

ADX tells us *how strongly* the market is trending, not which way.
Combined with DI+ (bullish pressure) and DI- (bearish pressure), it gives
a complete picture of trend quality.

Why ADX matters:
  VWAP and EMAs tell you direction, but not strength. In a ranging/choppy
  market, both can give false signals — Nifty oscillates around VWAP without
  committing to a direction. ADX filters these out by reducing the score
  contribution when the trend is weak (ADX < 20).

  Crucially: ADX does NOT hard-block entries. It's a MODIFIER.
  A high-quality signal in a ranging market can still fire if all other
  conditions are strong — ADX just makes it harder to cross the threshold.

ADX Interpretation:
  ADX < 20:  Ranging / weak trend (score penalty)
  ADX 20–25: Trend beginning to form
  ADX 25–40: Strong trend (ideal entry zone)
  ADX > 40:  Very strong trend (bonus score)

Computation (Wilder's smoothing — same as ATR/RSI):
  1. True Range (TR) per candle
  2. +DM = high - prev_high if positive, else 0
  3. -DM = prev_low - low if positive, else 0
  4. Smooth TR, +DM, -DM with Wilder's EWM (alpha = 1/period)
  5. DI+ = 100 × smooth_+DM / smooth_TR
  6. DI- = 100 × smooth_-DM / smooth_TR
  7. DX  = 100 × |DI+ - DI-| / (DI+ + DI-)
  8. ADX = Wilder-smoothed DX
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import core.constants as C


def compute_adx(df: pd.DataFrame, period: int = C.ADX_PERIOD) -> pd.DataFrame:
    """
    Compute ADX, DI+, and DI- for a candle DataFrame.

    Args:
        df:     OHLCV DataFrame with at least [high, low, close] columns.
        period: Smoothing period (default: ADX_PERIOD = 14).

    Returns:
        DataFrame with columns [adx, di_plus, di_minus], same index as input.
        First (period × 2) rows will be NaN during warmup.
    """
    if df.empty or len(df) < period + 1:
        empty = pd.DataFrame(
            {"adx": np.nan, "di_plus": np.nan, "di_minus": np.nan},
            index=df.index,
        )
        return empty

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # ── True Range ────────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # ── Directional Movement ──────────────────────────────────────────────────
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # ── Wilder Smoothing (alpha = 1 / period) ─────────────────────────────────
    alpha = 1.0 / period
    smooth_tr   = tr.ewm(alpha=alpha,       adjust=False, min_periods=period).mean()
    smooth_pdm  = plus_dm.ewm(alpha=alpha,  adjust=False, min_periods=period).mean()
    smooth_mdm  = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # ── DI+ / DI- ─────────────────────────────────────────────────────────────
    di_plus  = 100 * smooth_pdm / smooth_tr.replace(0, np.nan)
    di_minus = 100 * smooth_mdm / smooth_tr.replace(0, np.nan)

    # ── DX and ADX ────────────────────────────────────────────────────────────
    di_sum  = di_plus + di_minus
    dx = (100 * (di_plus - di_minus).abs() / di_sum.replace(0, np.nan))
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    result = pd.DataFrame({
        "adx":      adx,
        "di_plus":  di_plus,
        "di_minus": di_minus,
    }, index=df.index)

    return result


def current_adx(df: pd.DataFrame, period: int = C.ADX_PERIOD) -> float:
    """Return the most recent ADX value. Returns 0.0 if insufficient data."""
    result = compute_adx(df, period)
    val = result["adx"].dropna()
    return float(val.iloc[-1]) if not val.empty else 0.0


def current_di(df: pd.DataFrame, period: int = C.ADX_PERIOD) -> tuple[float, float]:
    """Return (DI+, DI-) current values. Useful for directional bias."""
    result = compute_adx(df, period)
    di_plus  = result["di_plus"].dropna()
    di_minus = result["di_minus"].dropna()
    dp = float(di_plus.iloc[-1])  if not di_plus.empty  else 0.0
    dm = float(di_minus.iloc[-1]) if not di_minus.empty else 0.0
    return dp, dm


def is_trending(df: pd.DataFrame, threshold: float = C.ADX_TREND_THRESHOLD) -> bool:
    """Returns True if ADX exceeds the trend threshold."""
    return current_adx(df) >= threshold
