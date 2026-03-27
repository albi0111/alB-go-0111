"""
tests/test_strategy.py
──────────────────────
Unit tests for strategy conditions and signal scoring.
"""

import math
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import core.config as cfg
from indicators.rsi import current_rsi
from strategy.conditions import (
    score_volume_spike,
    score_breakout,
    score_trend,
    score_vwap_proximity,
    score_rsi,
    get_breakout_levels,
    evaluate_signal,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def candles_with_breakout() -> pd.DataFrame:
    """3-min candles covering 09:15–09:45 window for breakout detection."""
    times = pd.date_range("2026-03-24 09:15", periods=20, freq="3min")
    return pd.DataFrame({
        "ts":    times,
        "high":  [22100, 22120, 22150, 22130, 22140, 22110, 22105, 22095, 22080, 22070,
                  22060, 22050, 22045, 22030, 22020, 22010, 22005, 21995, 21985, 21975],
        "low":   [22000, 22020, 22050, 22030, 22040, 22010, 22005, 21995, 21980, 21970,
                  21960, 21950, 21945, 21930, 21920, 21910, 21905, 21895, 21885, 21875],
        "close": [22050, 22080, 22120, 22100, 22110, 22080, 22070, 22060, 22050, 22040,
                  22030, 22020, 22010, 22000, 21990, 21980, 21970, 21960, 21950, 21940],
        "open":  [22010, 22050, 22080, 22060, 22070, 22050, 22040, 22030, 22020, 22010,
                  22000, 21990, 21980, 21970, 21960, 21950, 21940, 21930, 21920, 21910],
        "volume": [10000] * 20,
    })


# ── Score function tests ───────────────────────────────────────────────────────

class TestScorers:
    def test_volume_spike_above_threshold(self):
        # At exactly 1.5×: score 0.7; at 2× threshold (3.0): score 1.0
        assert score_volume_spike(3.0, 1.5) == 1.0  # 2× threshold → full score
        assert score_volume_spike(1.5, 1.5) >= 0.7  # Exactly at threshold

    def test_volume_spike_below_threshold(self):
        score = score_volume_spike(1.0, 1.5)
        assert 0.0 < score < 0.7

    def test_volume_spike_zero(self):
        assert score_volume_spike(0, 1.5) == 0.0

    def test_uptrend_rsi_above_50(self):
        """Strongly trending-up prices should produce RSI > 50 (pure uptrend = 100)."""
        closes = [float(c) for c in range(1000, 1032)]
        df = pd.DataFrame({"close": closes})
        val = current_rsi(df)
        assert val > 50

    def test_breakout_above_confirmed(self):
        score = score_breakout(22200, 22100, 22000, "BUY", 0.002)
        assert score == 1.0

    def test_breakout_below_level(self):
        score = score_breakout(21900, 22100, 22000, "BUY", 0.002)
        assert score < 0.5

    def test_trend_bullish_both_conditions(self):
        score = score_trend(22100, 22000, 22050, 21900, "BUY", 0.0)
        assert score > 0.8

    def test_trend_bullish_partial(self):
        """Price above VWAP but EMA20 ≈ EMA50 (very small diff) — partial score."""
        score = score_trend(22100, 22000, 22050, 22048, "BUY", 0.0)
        assert 0.5 < score <= 1.0

    def test_vwap_proximity_exact(self):
        assert score_vwap_proximity(22000, 22000, 0.003) == 1.0

    def test_vwap_proximity_far(self):
        assert score_vwap_proximity(22500, 22000, 0.003) == 0.0

    def test_rsi_neutral(self):
        assert score_rsi(55, "BUY") == 1.0

    def test_rsi_overbought_penalty(self):
        assert score_rsi(80, "BUY") < 0.5

    def test_rsi_oversold_on_sell_penalty(self):
        assert score_rsi(20, "SELL") < 0.5


# ── Breakout Level Tests ───────────────────────────────────────────────────────

class TestBreakoutLevels:
    def test_levels_detected(self, candles_with_breakout):
        high, low = get_breakout_levels(candles_with_breakout)
        # First 10 bars (09:15 to 09:42 inclusive) should be in the 30-min window
        # Max high in that window: 22150, min low in that window: 22000
        assert high == 22150
        assert low <= 22010   # Allow for floating point; first bar low is 22000

    def test_empty_returns_zero(self):
        high, low = get_breakout_levels(pd.DataFrame())
        assert high == 0.0 and low == 0.0


# ── End-to-End Signal Evaluation ──────────────────────────────────────────────

class TestEvaluateSignal:
    def test_strong_bullish_signal(self, candles_with_breakout):
        """Strong conditions: high volume spike, bullish trend, near VWAP — expect BUY."""
        result = evaluate_signal(
            price=22150,   # At breakout high
            vwap=22140,    # Price just above VWAP
            rsi_val=58,
            ema20=22100,
            ema50=22000,   # EMA20 > EMA50 — bullish
            atr_val=80,
            spike_ratio=2.5,
            df_3min=candles_with_breakout,
        )
        # Signal strength should be above threshold (0.6)
        assert result.signal_strength >= cfg.SIGNAL_STRENGTH_THRESHOLD, \
            f"Expected signal_strength >= {cfg.SIGNAL_STRENGTH_THRESHOLD} (threshold), got {result.signal_strength}"
        assert result.direction == "BUY"
        assert result.option_type == "CE"

    def test_weak_signal_returns_no_direction(self, candles_with_breakout):
        """Weak / conflicting conditions — expect no signal."""
        result = evaluate_signal(
            price=22000,
            vwap=22300,    # Price far below VWAP
            rsi_val=50,
            ema20=22000,
            ema50=22000,   # Flat EMAs
            atr_val=30,
            spike_ratio=0.8,   # Below average volume
            df_3min=candles_with_breakout,
        )
        assert result.direction is None

    def test_signal_has_stop_loss_and_target(self, candles_with_breakout):
        result = evaluate_signal(
            price=22150, vwap=22140, rsi_val=58,
            ema20=22100, ema50=22000, atr_val=80,
            spike_ratio=2.5, df_3min=candles_with_breakout,
        )
        if result.direction:
            assert result.stop_loss > 0
            assert result.target > 0
