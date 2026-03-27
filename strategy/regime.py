"""
strategy/regime.py
------------------
Market Regime Detection - classifies the current market as TRENDING or RANGING.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import core.config as cfg
from core.logger import log
from indicators.adx import compute_adx
from indicators.atr import classify_volatility_series

@dataclass
class MarketRegime:
    """
    Snapshot of current market regime for one evaluation tick.
    """
    type: str
    adx: float
    volatility: str
    use_trailing_sl: bool
    exit_tolerance: float
    signal_threshold: float

def detect_regime(df_3min: pd.DataFrame) -> MarketRegime:
    """Legacy scalar version of detect_regime."""
    df_regime = detect_regime_series(df_3min)
    if df_regime.empty:
        return MarketRegime("RANGING", 0.0, "normal", False, 0.45, 0.70)
    
    row = df_regime.iloc[-1]
    return MarketRegime(
        type             = str(row["type"]),
        adx              = float(row["adx"]),
        volatility       = str(row["volatility"]),
        use_trailing_sl  = bool(row["use_trailing_sl"]),
        exit_tolerance   = float(row["exit_tolerance"]),
        signal_threshold = float(row["signal_threshold"]),
    )

def detect_regime_series(df_3min: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized market regime detection.
    Returns DataFrame with columns:
    ['type', 'adx', 'volatility', 'use_trailing_sl', 'exit_tolerance', 'signal_threshold']
    """
    if df_3min.empty:
        return pd.DataFrame()

    adx_df = compute_adx(df_3min)
    adx_series = adx_df["adx"]
    vol_series = classify_volatility_series(df_3min)
    
    is_trending = adx_series >= cfg.ADX_TREND_MIN
    
    res = pd.DataFrame(index=df_3min.index)
    res["type"] = "RANGING"
    res.loc[is_trending, "type"] = "TRENDING"
    res["adx"] = adx_series.fillna(0.0)
    res["volatility"] = vol_series
    
    res["use_trailing_sl"] = False
    res.loc[is_trending, "use_trailing_sl"] = True
    
    res["exit_tolerance"] = 0.45
    res.loc[is_trending, "exit_tolerance"] = 0.35
    
    # Ranging threshold is slightly higher
    base_threshold = cfg.SIGNAL_STRENGTH_THRESHOLD
    res["signal_threshold"] = base_threshold
    res.loc[~is_trending, "signal_threshold"] = min(base_threshold + 0.05, 0.72)
    
    return res
