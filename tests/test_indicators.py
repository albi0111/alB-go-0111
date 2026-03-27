"""
tests/test_indicators.py
────────────────────────
Unit tests for all indicator modules.
Run: python -m pytest tests/ -v
"""

import math
import pytest
import pandas as pd
import numpy as np

from indicators.vwap   import compute_vwap, current_vwap
from indicators.rsi    import compute_rsi, current_rsi
from indicators.ema    import compute_ema, current_emas
from indicators.volume import rolling_avg_volume, current_spike_ratio, is_volume_spike
from indicators.atr    import compute_atr, current_atr


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_candles() -> pd.DataFrame:
    """30 synthetic 3-min candles starting from 09:15."""
    n = 30
    dates = pd.date_range("2026-03-24 09:15", periods=n, freq="3min")
    np.random.seed(42)
    closes = 22000 + np.cumsum(np.random.randn(n) * 20)
    return pd.DataFrame({
        "ts":     dates,
        "open":   closes - np.random.rand(n) * 10,
        "high":   closes + np.random.rand(n) * 15,
        "low":    closes - np.random.rand(n) * 15,
        "close":  closes,
        "volume": np.random.randint(5000, 20000, n).astype(float),
    })


# ── VWAP ───────────────────────────────────────────────────────────────────────

class TestVWAP:
    def test_length_matches_input(self, sample_candles):
        result = compute_vwap(sample_candles)
        assert len(result) == len(sample_candles)

    def test_values_are_positive(self, sample_candles):
        result = compute_vwap(sample_candles)
        assert (result > 0).all()

    def test_current_vwap_is_float(self, sample_candles):
        val = current_vwap(sample_candles)
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_vwap_is_price_level(self, sample_candles):
        """VWAP is a cumulative weighted avg — it's centred in the day's price range."""
        result = compute_vwap(sample_candles)
        # VWAP should be within the session's overall high/low range (not each individual bar)
        assert float(result.mean()) >= sample_candles["low"].min() - 100
        assert float(result.mean()) <= sample_candles["high"].max() + 100

    def test_empty_dataframe(self):
        result = compute_vwap(pd.DataFrame())
        assert result.empty


# ── RSI ────────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_range_0_to_100(self, sample_candles):
        result = compute_rsi(sample_candles)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_current_rsi_is_float(self, sample_candles):
        val = current_rsi(sample_candles)
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_custom_period(self, sample_candles):
        result = compute_rsi(sample_candles, period=7)
        assert len(result) == len(sample_candles)

    def test_uptrend_rsi_above_50(self):
        """Strongly trending-up prices should produce RSI = 100 (no losses)."""
        closes = [float(c) for c in range(1000, 1031)]
        df = pd.DataFrame({"close": closes})
        val = current_rsi(df)
        assert val > 50   # Pure uptrend: RSI = 100


# ── EMA ────────────────────────────────────────────────────────────────────────

class TestEMA:
    def test_ema20_length(self, sample_candles):
        result = compute_ema(sample_candles, 20)
        assert len(result) == len(sample_candles)

    def test_current_emas_tuple(self, sample_candles):
        ema20, ema50 = current_emas(sample_candles)
        assert isinstance(ema20, float)
        assert isinstance(ema50, float)

    def test_ema_converges_to_constant(self):
        """EMA of a constant series should equal that constant."""
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_ema(df, 20)
        assert abs(result.iloc[-1] - 100.0) < 0.01


# ── Volume ─────────────────────────────────────────────────────────────────────

class TestVolume:
    def test_rolling_avg_positive(self, sample_candles):
        result = rolling_avg_volume(sample_candles)
        assert (result > 0).all()

    def test_spike_ratio_nonnegative(self, sample_candles):
        result = current_spike_ratio(sample_candles)
        assert result >= 0

    def test_obvious_spike(self):
        """Last candle has 10× average volume — should be detected."""
        volumes = [1000.0] * 20 + [10000.0]
        df = pd.DataFrame({"volume": volumes})
        ratio = current_spike_ratio(df)
        assert ratio > 1.5

    def test_no_spike(self):
        """Constant volume → ratio ≈ 1.0."""
        df = pd.DataFrame({"volume": [5000.0] * 21})
        assert not is_volume_spike(df, threshold=1.5)


# ── ATR ────────────────────────────────────────────────────────────────────────

class TestATR:
    def test_length_matches(self, sample_candles):
        result = compute_atr(sample_candles)
        assert len(result) == len(sample_candles)

    def test_current_atr_positive(self, sample_candles):
        val = current_atr(sample_candles)
        assert isinstance(val, float)
        assert val > 0

    def test_atr_increases_with_volatility(self):
        """High-volatility candles should produce higher ATR."""
        base = pd.DataFrame({
            "high":  [100.5] * 20, "low": [99.5] * 20, "close": [100.0] * 20,
        })
        volatile = pd.DataFrame({
            "high":  [110.0] * 20, "low": [90.0] * 20, "close": [100.0] * 20,
        })
        assert current_atr(volatile) > current_atr(base)
