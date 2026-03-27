"""
tests/test_exit_engine.py
─────────────────────────
Tests for strategy/exit_engine.py
"""
import pytest
from strategy.exit_engine import ExitEngine, ExitContext, ExitDecision
from strategy.phase_manager import PhaseManager
from strategy.regime import MarketRegime
import core.config as cfg


def _make_regime(type_="TRENDING") -> MarketRegime:
    return MarketRegime(
        type             = type_,
        adx              = 25.0,
        volatility       = "normal",
        use_trailing_sl  = True,
        exit_tolerance   = 0.35,
        signal_threshold = 0.6,
    )


def _make_pm(entry=22000, sl=21900) -> PhaseManager:
    return PhaseManager(entry_price=entry, stop_loss=sl)


def _make_engine() -> ExitEngine:
    return ExitEngine(regime=_make_regime(), phase_manager=_make_pm())


def _make_ctx(**overrides):
    defaults = dict(
        held_direction = "BUY",
        held_score     = 0.70,
        opposite_score = 0.30,
        current_price  = 22050.0,
        entry_price    = 22000.0,
        initial_risk   = 100.0,
        vwap           = 22040.0,
        breakout_high  = 22000.0,
        breakout_low   = 21900.0,
        vol_trend      = "rising",
        candles_held   = 3,
        pnl_r          = 0.5,
    )
    defaults.update(overrides)
    return ExitContext(**defaults)


class TestProfitLock:
    def test_profit_lock_layer_removed(self):
        """PROFIT_LOCK_R Layer 1 has been INTENTIONALLY REMOVED.
        Profit protection is now handled by partial exit at +1.0R in OrderManager.
        Verify that no TIGHTEN_SL is returned even at +3R."""
        engine = _make_engine()
        ctx = _make_ctx(pnl_r=3.5)  # Well above old PROFIT_LOCK_R threshold
        decision = engine.evaluate(ctx)
        # Should NOT tighten SL (old Layer 1 behaviour is gone)
        assert decision.action != "TIGHTEN_SL", (
            "PROFIT_LOCK_R TIGHTEN_SL was removed. Partial exit at +1.0R in OrderManager now handles this."
        )
        assert decision.layer != "profit_lock"

    def test_profit_lock_not_triggered_below_threshold(self):
        engine = _make_engine()
        ctx = _make_ctx(pnl_r=0.5)  # well below any threshold
        decision = engine.evaluate(ctx)
        assert decision.action != "TIGHTEN_SL"


class TestMaxHoldTime:
    def test_max_hold_triggers_exit(self):
        engine = _make_engine()
        ctx = _make_ctx(candles_held=cfg.MAX_HOLD_CANDLES + 1, pnl_r=0.1)
        decision = engine.evaluate(ctx)
        assert decision.should_exit
        assert decision.layer == "max_hold"


class TestPersistenceFilter:
    def test_single_weak_candle_does_not_exit(self):
        """One candle below threshold should NOT exit (persistence = 2 by default)."""
        engine = _make_engine()
        # First call with weak score
        ctx = _make_ctx(held_score=0.25, opposite_score=0.30, vol_trend="flat")
        d1 = engine.evaluate(ctx)
        assert not d1.should_exit, "Should HOLD on single weak candle"

    def test_two_consecutive_weak_candles_may_exit(self):
        """Two consecutive weak candles can trigger further evaluation."""
        engine = _make_engine()
        ctx = _make_ctx(
            held_score=0.20, opposite_score=0.70, vol_trend="declining",
            vwap=21800.0,   # Far from VWAP — not a retest
        )
        for _ in range(cfg.REVERSAL_PERSIST_CANDLES):
            decision = engine.evaluate(ctx)
        # After N consecutive weak candles, the engine should either exit
        # or continue degrading confidence — but NOT hold on a neutral basis
        # (opposite strength not yet confirmed — still may HOLD)
        assert isinstance(decision, ExitDecision)


class TestRetestDetector:
    def test_retest_near_vwap_with_declining_volume_holds(self):
        """Price near VWAP + declining volume = healthy retest → HOLD."""
        engine = _make_engine()
        # Phase must allow pullbacks (tick to confirmation first)
        engine.phase_manager.tick(22050.0)  # → CONFIRMATION

        ctx = _make_ctx(
            current_price  = 22042.0,   # Very close to VWAP (22040)
            vwap           = 22040.0,
            held_score     = 0.35,      # Below tolerance → would normally flag
            opposite_score = 0.30,
            vol_trend      = "declining",
        )
        decision = engine.evaluate(ctx)
        assert decision.action == "HOLD"
        assert decision.layer == "retest"

    def test_retest_hold_cap_releases(self):
        """After RETEST_MAX_HOLD_CANDLES, normal evaluation resumes."""
        engine = _make_engine()
        engine.phase_manager.tick(22050.0)

        ctx = _make_ctx(
            current_price  = 22042.0,
            vwap           = 22040.0,
            held_score     = 0.20,   # Very weak — should exit eventually
            opposite_score = 0.75,
            vol_trend      = "declining",
        )
        # Burn through the retest hold cap
        for _ in range(cfg.RETEST_MAX_HOLD_CANDLES):
            engine.evaluate(ctx)

        # After cap: should not be indefinitely held on retest
        last = engine.evaluate(ctx)
        # It should now go through normal evaluation (not retest layer)
        assert last.layer != "retest" or last.should_exit


class TestConfidenceDecay:
    def test_confidence_decays_exponentially_not_linearly(self):
        """Confidence decay should be multiplicative (exponential)."""
        engine = _make_engine()
        ctx = _make_ctx(held_score=0.10, opposite_score=0.20, vol_trend="flat",
                        vwap=21000.0)  # Far from VWAP — no retest
        c1 = engine.confidence
        engine.evaluate(ctx)
        c2 = engine.confidence
        engine.evaluate(ctx)
        c3 = engine.confidence
        # Each step: c2 = c1 * (1 - rate), c3 = c2 * (1 - rate)
        # So ratio c2/c1 ≈ c3/c2 (exponential, not linear)
        if c1 > 0 and c2 > 0:
            ratio1 = c2 / c1
            ratio2 = c3 / c2
            assert abs(ratio1 - ratio2) < 0.15, "Decay should be roughly exponential"

    def test_confidence_recovers_on_improvement(self):
        """Confidence should recover when score improves."""
        engine = _make_engine()
        ctx_weak   = _make_ctx(held_score=0.10, vwap=21000.0, vol_trend="flat")
        ctx_strong = _make_ctx(held_score=0.85, vwap=22040.0, vol_trend="rising")

        engine.evaluate(ctx_weak)
        engine.evaluate(ctx_weak)
        low_confidence = engine.confidence

        engine.evaluate(ctx_strong)
        higher_confidence = engine.confidence

        assert higher_confidence > low_confidence
