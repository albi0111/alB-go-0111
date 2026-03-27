"""
tests/test_dataset.py
─────────────────────
Tests for ai/dataset_builder.py — feature capture, outcome labelling,
leakage guard, quality filter, and return clipping.
"""
import math
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from ai.dataset_builder import (
    record_signal_features,
    label_trade_outcome,
    DataLeakageError,
    _encode_regime,
    _encode_volatility,
    _encode_direction,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_signal(bullish=0.72, bearish=0.30, strength=0.72, rsi=55.0, atr=80.0):
    sig = MagicMock()
    sig.bullish_score      = bullish
    sig.bearish_score      = bearish
    sig.signal_strength    = strength
    sig.volume_spike_ratio = 1.5
    sig.rsi                = rsi
    sig.vwap_distance_pct  = 0.003
    sig.ema_diff_pct       = 0.002
    sig.atr                = atr
    sig.direction          = "BUY"
    # New fields (may be absent on older signal objects — defaulting gracefully)
    sig.adx                = 28.0
    return sig


def _make_regime(type_="TRENDING", volatility="normal"):
    regime = MagicMock()
    regime.type       = type_
    regime.volatility = volatility
    return regime


def _make_selection(strike=22000, oi=50000, volume=1200):
    sel = MagicMock()
    sel.strike   = strike
    sel.oi       = oi
    sel.volume   = volume
    sel.oi_change = 200.0
    sel.ltp      = 145.0
    return sel


def _make_score_breakdown():
    return {
        "volume_spike":   0.85,
        "trend":          0.70,
        "breakout":       0.75,
        "vwap_proximity": 0.80,
        "rsi":            0.90,
        "adx":            0.80,
        "volume_trend":   0.70,
    }


# ── Encoding helpers ───────────────────────────────────────────────────────────

class TestEncoders:
    def test_regime_trending(self):
        assert _encode_regime(_make_regime("TRENDING")) == 1

    def test_regime_ranging(self):
        assert _encode_regime(_make_regime("RANGING")) == 0

    def test_volatility_low_normal_high(self):
        assert _encode_volatility(_make_regime(volatility="low"))    == 0
        assert _encode_volatility(_make_regime(volatility="normal")) == 1
        assert _encode_volatility(_make_regime(volatility="high"))   == 2

    def test_direction_buy_sell(self):
        assert _encode_direction("BUY")  == 1
        assert _encode_direction("SELL") == 0


# ── Feature capture (Stage 1) ─────────────────────────────────────────────────

class TestRecordSignalFeatures:
    def test_saves_record_and_returns_id(self):
        """Should call save_ai_record and return a positive integer ID."""
        with patch("ai.dataset_builder.save_ai_record", return_value=42) as mock_save:
            rid = record_signal_features(
                signal          = _make_signal(),
                regime          = _make_regime(),
                selection       = _make_selection(),
                trade_id        = 99,
                score_breakdown = _make_score_breakdown(),
            )
        assert rid == 42
        mock_save.assert_called_once()

    def test_feature_record_contains_required_keys(self):
        """Saved record dict must contain all important feature keys."""
        captured = {}

        def capture_record(record):
            captured.update(record)
            return 1

        with patch("ai.dataset_builder.save_ai_record", side_effect=capture_record):
            record_signal_features(
                signal          = _make_signal(),
                regime          = _make_regime(),
                selection       = _make_selection(),
                trade_id        = 5,
                score_breakdown = _make_score_breakdown(),
            )

        required_keys = [
            "bullish_score", "bearish_score", "signal_strength",
            "vol_score", "trend_score", "rsi_score",
            "adx_val", "atr_val", "regime_encoded", "volatility_encoded",
            "oi", "strike", "direction_encoded", "feature_ts", "trade_id",
        ]
        for key in required_keys:
            assert key in captured, f"Missing key: {key}"

    def test_label_is_none_at_capture(self):
        """Label must be None at feature capture time (not yet known)."""
        captured = {}

        def capture_record(record):
            captured.update(record)
            return 1

        with patch("ai.dataset_builder.save_ai_record", side_effect=capture_record):
            record_signal_features(
                signal     = _make_signal(),
                regime     = _make_regime(),
                selection  = _make_selection(),
                trade_id   = 1,
            )

        assert captured.get("label") is None

    def test_leakage_guard_raises_when_feature_ts_after_entry(self):
        """
        Feature timestamp AFTER entry_ts should raise DataLeakageError.
        In practice this should never happen — it's a safeguard.
        """
        past_entry_ts = datetime.now() - timedelta(hours=1)  # already passed

        with patch("ai.dataset_builder.save_ai_record", return_value=1):
            with pytest.raises(DataLeakageError):
                record_signal_features(
                    signal     = _make_signal(),
                    regime     = _make_regime(),
                    selection  = _make_selection(),
                    trade_id   = 1,
                    entry_ts   = past_entry_ts,  # far in the past → feature_ts > entry_ts
                )

    def test_no_leakage_when_entry_ts_is_now(self):
        """Feature captured at 'now' with entry_ts=now should succeed."""
        with patch("ai.dataset_builder.save_ai_record", return_value=1) as mock:
            rid = record_signal_features(
                signal    = _make_signal(),
                regime    = _make_regime(),
                selection = _make_selection(),
                trade_id  = 1,
                entry_ts  = datetime.now() + timedelta(seconds=5),  # slightly in future
            )
        assert rid == 1


# ── Outcome labelling (Stage 2) ───────────────────────────────────────────────

class TestLabelTradeOutcome:
    def test_profitable_trade_gets_label_1(self):
        captured = {}

        def record_outcome(record_id, outcome):
            captured.update(outcome)

        with patch("ai.dataset_builder.update_ai_outcome", side_effect=record_outcome):
            result = label_trade_outcome(
                ai_record_id     = 1,
                pnl              = 500.0,
                entry_price      = 100.0,
                exit_price       = 120.0,
                qty              = 50,
                initial_risk     = 80.0,
                duration_candles = 5,
            )

        assert result is True
        assert captured["label"] == 1

    def test_losing_trade_gets_label_0(self):
        captured = {}
        with patch("ai.dataset_builder.update_ai_outcome", side_effect=lambda rid, o: captured.update(o)):
            label_trade_outcome(
                ai_record_id=1, pnl=-300.0,
                entry_price=100.0, exit_price=80.0,
                qty=50, initial_risk=80.0, duration_candles=3,
            )
        assert captured["label"] == 0

    def test_return_pct_clipped_to_50(self):
        """500% gain → clipped to +50%."""
        captured = {}
        with patch("ai.dataset_builder.update_ai_outcome", side_effect=lambda rid, o: captured.update(o)):
            label_trade_outcome(
                ai_record_id=1, pnl=5000.0,
                entry_price=10.0, exit_price=60.0,  # +500%
                qty=10, initial_risk=50.0, duration_candles=2,
            )
        assert captured["return_pct"] == 50.0, f"Expected clipped 50, got {captured['return_pct']}"

    def test_return_pct_clipped_to_minus_50(self):
        """−100% loss → clipped to −50%."""
        captured = {}
        with patch("ai.dataset_builder.update_ai_outcome", side_effect=lambda rid, o: captured.update(o)):
            label_trade_outcome(
                ai_record_id=1, pnl=-1000.0,
                entry_price=100.0, exit_price=0.01,  # ~-100%
                qty=10, initial_risk=50.0, duration_candles=2,
            )
        assert captured["return_pct"] == -50.0

    def test_log_return_computed_for_positive_return(self):
        """log_return = log(1 + return_pct/100) — must be positive for gains."""
        captured = {}
        with patch("ai.dataset_builder.update_ai_outcome", side_effect=lambda rid, o: captured.update(o)):
            label_trade_outcome(
                ai_record_id=1, pnl=200.0,
                entry_price=100.0, exit_price=120.0,  # +20%
                qty=10, initial_risk=50.0, duration_candles=4,
            )
        expected_log_return = math.log(1 + 20.0 / 100.0)
        assert abs(captured["log_return"] - expected_log_return) < 0.001

    def test_quality_filter_rejects_zero_duration(self):
        """Duration < 1 candle → skip labelling."""
        with patch("ai.dataset_builder.update_ai_outcome") as mock:
            result = label_trade_outcome(
                ai_record_id=1, pnl=300.0,
                entry_price=100.0, exit_price=110.0,
                qty=10, initial_risk=50.0, duration_candles=0,  # too short
            )
        assert result is False
        mock.assert_not_called()

    def test_quality_filter_rejects_below_noise_threshold(self):
        """PnL below noise threshold → skip."""
        with patch("ai.dataset_builder.update_ai_outcome") as mock:
            result = label_trade_outcome(
                ai_record_id=1, pnl=0.5,       # very tiny
                entry_price=100.0, exit_price=100.1,
                qty=10, initial_risk=50.0, duration_candles=3,
                noise_threshold=10.0,           # threshold = ₹10
            )
        assert result is False
        mock.assert_not_called()

    def test_r_multiple_computed_correctly(self):
        """R-multiple = pnl / (initial_risk × qty)."""
        captured = {}
        with patch("ai.dataset_builder.update_ai_outcome", side_effect=lambda rid, o: captured.update(o)):
            label_trade_outcome(
                ai_record_id=1, pnl=800.0,
                entry_price=100.0, exit_price=116.0,
                qty=50, initial_risk=8.0,         # 8.0 × 50 = 400 risk
                duration_candles=6,
            )
        expected_r = 800.0 / (8.0 * 50)   # = 2.0
        assert abs(captured["r_multiple"] - expected_r) < 0.001
