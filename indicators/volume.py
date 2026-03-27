"""
indicators/volume.py
--------------------
Volume analysis: rolling average and spike detection.
"""

from __future__ import annotations
import pandas as pd
import core.constants as C
import core.config as cfg
from core.logger import log

def rolling_avg_volume(df: pd.DataFrame, period: int = C.VOLUME_ROLLING_PERIOD) -> pd.Series:
    """Compute the rolling mean of volume."""
    avg = df["volume"].rolling(window=period, min_periods=1).mean()
    avg.name = "avg_volume"
    return avg

def current_spike_ratio(df: pd.DataFrame, period: int = C.VOLUME_ROLLING_PERIOD) -> float:
    """Return the latest spike ratio."""
    series = volume_spike_ratio(df, period)
    return float(series.iloc[-1]) if not series.empty else 0.0

def volume_spike_ratio(df: pd.DataFrame, period: int = C.VOLUME_ROLLING_PERIOD) -> pd.Series:
    """Vectorized volume spike ratio calculation."""
    avg = rolling_avg_volume(df, period)
    valid_avg = avg.replace(0, float("nan"))
    return df["volume"] / valid_avg

def detect_volume_spike(
    df: pd.DataFrame,
    threshold: float | None = None,
    period: int = C.VOLUME_ROLLING_PERIOD,
) -> bool:
    """Legacy scalar version of detect_volume_spike."""
    eff_threshold = threshold or cfg.VOLUME_SPIKE_RATIO
    ratio = current_spike_ratio(df, period)
    return ratio >= eff_threshold

def volume_trend(df: pd.DataFrame, lookback: int = 5) -> str:
    """Legacy scalar version of volume_trend."""
    res = volume_trend_series(df, lookback)
    return str(res.iloc[-1]) if not res.empty else "flat"

def volume_trend_series(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Vectorized volume trend."""
    if len(df) < lookback:
        return pd.Series("flat", index=df.index)

    half = lookback // 2
    early_avg = df["volume"].rolling(window=half).mean().shift(lookback - half)
    late_avg  = df["volume"].rolling(window=lookback - half).mean()

    valid_early = early_avg.replace(0, float("nan"))
    change_pct = (late_avg - valid_early) / valid_early

    res = pd.Series("flat", index=df.index)
    res[change_pct > 0.15] = "rising"
    res[change_pct < -0.15] = "declining"
    return res

def is_volume_trap(
    df: pd.DataFrame,
    spike_threshold: float | None = None,
    collapse_pct: float = 0.40,
    lookback: int = 3,
) -> bool:
    """Legacy scalar version of is_volume_trap."""
    res = volume_trap_series(df, spike_threshold, collapse_pct, lookback)
    return bool(res.iloc[-1]) if not res.empty else False

def volume_trap_series(
    df: pd.DataFrame,
    spike_threshold: float | None = None,
    collapse_pct: float = 0.40,
    lookback: int = 3,
) -> pd.Series:
    """Vectorized volume trap detection."""
    threshold = spike_threshold or cfg.VOLUME_SPIKE_RATIO
    spike_ratio = volume_spike_ratio(df)
    is_spike = spike_ratio >= threshold
    
    spike_vol_past = df["volume"].shift(lookback)
    avg_post = df["volume"].rolling(lookback).mean()
    is_spike_past = is_spike.shift(lookback)
    
    collapse = (spike_vol_past - avg_post) / spike_vol_past.replace(0, float("nan"))
    res = (is_spike_past) & (collapse >= collapse_pct)
    return res.fillna(False)
