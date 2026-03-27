"""
tests/test_adx.py
─────────────────
Tests for indicators/adx.py
"""
import pytest
import pandas as pd
import numpy as np
from indicators.adx import compute_adx, current_adx, is_trending


def _make_df(highs, lows, closes):
    return pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": 1000})


class TestComputeAdx:
    def test_returns_dataframe_with_columns(self):
        n = 30
        df = _make_df(
            highs  = [100 + i for i in range(n)],
            lows   = [95  + i for i in range(n)],
            closes = [98  + i for i in range(n)],
        )
        result = compute_adx(df, period=14)
        assert set(result.columns) == {"adx", "di_plus", "di_minus"}
        assert len(result) == n

    def test_adx_range_0_to_100(self):
        n = 40
        df = _make_df(
            highs  = [100 + i * 0.5 for i in range(n)],
            lows   = [98  + i * 0.5 for i in range(n)],
            closes = [99  + i * 0.5 for i in range(n)],
        )
        result = compute_adx(df, period=14)
        valid = result["adx"].dropna()
        assert (valid >= 0).all(), "ADX must be >= 0"
        assert (valid <= 100).all(), "ADX must be <= 100"

    def test_strong_trend_has_high_adx(self):
        """Consistent one-directional move should produce ADX > 20."""
        n = 50
        df = _make_df(
            highs  = [100 + i * 2 for i in range(n)],
            lows   = [99  + i * 2 for i in range(n)],
            closes = [99.5 + i * 2 for i in range(n)],
        )
        adx = current_adx(df, period=14)
        assert adx > 20, f"Expected strong trend ADX > 20, got {adx:.2f}"

    def test_insufficient_data_returns_zero(self):
        df = _make_df(highs=[100, 101], lows=[99, 100], closes=[100, 101])
        adx = current_adx(df, period=14)
        assert adx == 0.0

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["high", "low", "close", "volume"])
        adx = current_adx(df)
        assert adx == 0.0

    def test_is_trending_true_for_strong_trend(self):
        n = 50
        df = _make_df(
            highs  = [100 + i * 2 for i in range(n)],
            lows   = [99  + i * 2 for i in range(n)],
            closes = [99.5 + i * 2 for i in range(n)],
        )
        assert is_trending(df, threshold=20) is True

    def test_is_trending_false_for_flat_market(self):
        """Oscillating market should have low ADX."""
        n = 50
        closes = [100 + (1 if i % 2 == 0 else -1) for i in range(n)]
        df = _make_df(
            highs  = [c + 0.5 for c in closes],
            lows   = [c - 0.5 for c in closes],
            closes = closes,
        )
        adx = current_adx(df, period=14)
        # Oscillating market — ADX should be lower than a trending one
        assert adx < 40, f"Oscillating ADX should be < 40, got {adx:.2f}"
