"""
strategy/breakout_strategy.py
──────────────────────────────
Strategy orchestrator: pulls in data → computes all indicators →
calls conditions.py → returns a SignalResult.

This module is the single point of contact for the main engine.
It is stateless — it operates on the candle DataFrame passed to it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from indicators.vwap   import current_vwap
from indicators.rsi    import current_rsi
from indicators.ema    import current_emas
from indicators.volume import current_spike_ratio
from indicators.atr    import current_atr
from strategy.conditions import evaluate_signal, SignalResult
from core.logger import log


def run_strategy(
    df_1min_fut: pd.DataFrame,    # 1-min Nifty Futures candles (volume source)
    df_3min_fut: pd.DataFrame,    # 3-min aggregated Nifty Futures candles
    spot_price: float,            # Nifty 50 spot price (index, for price reference)
) -> SignalResult:
    """
    Run the full breakout strategy on the current candle snapshot.

    Args:
        df_1min_fut: Rolling 1-min candles for Nifty Futures (used for VWAP, volume).
        df_3min_fut: Aggregated 3-min candles (used for indicators + breakout levels).
        spot_price:  Current Nifty 50 index price (for strategy conditions).

    Returns:
        SignalResult — direction is None if no valid signal.
    """
    if df_3min_fut.empty or len(df_3min_fut) < 2:
        log.warning("Insufficient candle data for strategy evaluation.")
        return SignalResult()

    # ── Compute all indicators ─────────────────────────────────────────────
    vwap_val    = current_vwap(df_1min_fut)     # VWAP uses raw 1-min for precision
    rsi_val     = current_rsi(df_3min_fut)
    ema20, ema50 = current_emas(df_3min_fut)
    spike_ratio = current_spike_ratio(df_1min_fut)   # Volume from futures
    atr_val     = current_atr(df_3min_fut)

    log.debug(
        "Indicators | price={p:.2f} VWAP={v:.2f} RSI={r:.2f} "
        "EMA20={e20:.2f} EMA50={e50:.2f} ATR={a:.2f} SpikeRatio={sr:.2f}",
        p=spot_price, v=vwap_val, r=rsi_val,
        e20=ema20, e50=ema50, a=atr_val, sr=spike_ratio,
    )

    # ── Detect Regime ──────────────────────────────────────────────────────
    from strategy.regime import detect_regime
    regime = detect_regime(df_3min_fut)
    adx_val = regime.adx

    # ── Evaluate signal ────────────────────────────────────────────────────
    signal = evaluate_signal(
        price        = spot_price,
        vwap         = vwap_val,
        rsi_val      = rsi_val,
        ema20        = ema20,
        ema50        = ema50,
        atr_val      = atr_val,
        spike_ratio  = spike_ratio,
        df_3min      = df_3min_fut,
        adx_val      = adx_val,
    )

    return signal
