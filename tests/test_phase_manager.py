"""
tests/test_phase_manager.py
───────────────────────────
Tests for strategy/phase_manager.py
"""
import pytest
from strategy.phase_manager import PhaseManager, TradePhase
import core.config as cfg


class TestPhaseManager:

    def _pm(self, entry=22000.0, sl=21900.0):
        return PhaseManager(entry_price=entry, stop_loss=sl)

    def test_starts_in_entry_phase(self):
        pm = self._pm()
        assert pm.phase == TradePhase.ENTRY

    def test_transitions_to_confirmation_at_half_r(self):
        pm = self._pm(entry=22000, sl=21900)   # risk = 100
        # Move to 0.5R profit: price = 22000 + 0.5*100 = 22050
        pm.tick(current_price=22050.0)
        assert pm.phase == TradePhase.CONFIRMATION

    def test_transitions_to_expansion_at_1_5_r(self):
        pm = self._pm(entry=22000, sl=21900)   # risk = 100
        # Move to 1.5R profit: price = 22000 + 1.5*100 = 22150
        pm.tick(current_price=22150.0)
        assert pm.phase == TradePhase.EXPANSION

    def test_stays_entry_below_confirm_threshold(self):
        pm = self._pm(entry=22000, sl=21900)
        pm.tick(current_price=22010.0)  # only 0.1R
        assert pm.phase == TradePhase.ENTRY

    def test_exit_tolerance_loosens_with_phase(self):
        pm = self._pm(entry=22000, sl=21900)
        entry_tol = pm.get_exit_tolerance()
        pm.tick(current_price=22050.0)  # → CONFIRMATION
        confirm_tol = pm.get_exit_tolerance()
        pm.tick(current_price=22150.0)  # → EXPANSION
        expand_tol = pm.get_exit_tolerance()
        # Tolerance should loosen (decrease) as phase advances
        assert entry_tol > confirm_tol > expand_tol

    def test_sl_multiplier_decreases_with_phase(self):
        """SL multiplier DECREASES as phase advances (entry=2.0, confirm=1.8, expansion=0.9).
        Wide at entry to breathe, tighter at expansion to lock profits.
        This is INTENTIONAL DESIGN: partial exit at +1R provides profit lock."""
        pm = self._pm(entry=22000, sl=21900)
        entry_m = pm.get_sl_multiplier()
        pm.tick(current_price=22050.0)   # → CONFIRMATION
        confirm_m = pm.get_sl_multiplier()
        pm.tick(current_price=22300.0)   # → EXPANSION (2+R)
        expand_m = pm.get_sl_multiplier()
        # Entry should have the widest multiplier, expansion the tightest
        assert entry_m > expand_m, (
            f"Entry({entry_m}) should be wider than Expansion({expand_m}) "
            "to give trades breathing room at the start."
        )

    def test_pullback_not_allowed_in_entry(self):
        pm = self._pm()
        assert pm.is_pullback_allowed() is False

    def test_pullback_allowed_in_confirmation(self):
        pm = self._pm(entry=22000, sl=21900)
        pm.tick(22050.0)
        assert pm.is_pullback_allowed() is True

    def test_pnl_r_computed_correctly(self):
        pm = self._pm(entry=22000, sl=21900)
        pm.tick(22100.0)   # 100 gain / 100 risk = 1.0R
        assert abs(pm.pnl_r - 1.0) < 0.001

    def test_zero_risk_guard(self):
        """Should not crash when entry == SL."""
        pm = PhaseManager(entry_price=22000, stop_loss=22000)
        pm.tick(22100)  # Should use initial_risk=1.0 fallback
