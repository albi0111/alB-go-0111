"""
strategy/phase_manager.py
─────────────────────────
Trade Phase Manager — tracks trade lifecycle and adjusts behaviour by phase.

Why phase management matters:
  A trade just entered behaves very differently from one that has already
  proven itself by reaching 1R profit. The early phase is fragile — a quick
  exit on invalidation is correct. The expansion phase deserves more room
  because the market has already rewarded the thesis.

  Time-based phases (e.g. "2 candles = confirmation") are unreliable because
  a fast-moving market can reach 1R in 1 candle or take 10 candles.
  This system uses R-based transitions instead.

Phases:
  ENTRY        PnL < 0.5R  → strict exit tolerance, 1× ATR trail
  CONFIRMATION PnL ≥ 0.5R  → moderate tolerance, 1.2× ATR trail
  EXPANSION    PnL ≥ 1.5R  → loose tolerance, aggressive 1.5× trail

R = initial risk = |entry_price - stop_loss|. PnL is tracked in units of R.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import core.config as cfg
from core.logger import log


class TradePhase(Enum):
    ENTRY        = "ENTRY"
    CONFIRMATION = "CONFIRMATION"  # 0.5R
    PARTIAL      = "PARTIAL"       # 1.0R (Partial exit done)
    EXPANSION    = "EXPANSION"     # 1.5R


@dataclass
class PhaseConfig:
    """Per-phase behavioural parameters."""
    exit_tolerance: float    # Minimum held-direction score before considering exit
    sl_multiplier: float     # ATR multiplier for trailing stop in this phase
    pullback_allowed: bool   # Whether the retest detector is active


# Phase configurations (tolerance loosens as trade matures)
PHASE_CONFIGS: dict[TradePhase, PhaseConfig] = {
    TradePhase.ENTRY:        PhaseConfig(exit_tolerance=0.45, sl_multiplier=2.0,  pullback_allowed=False), # Loose trail at entry
    TradePhase.CONFIRMATION: PhaseConfig(exit_tolerance=0.40, sl_multiplier=1.8,  pullback_allowed=True),
    TradePhase.PARTIAL:      PhaseConfig(exit_tolerance=0.35, sl_multiplier=1.5,  pullback_allowed=True),  # Runner breaths (was 2.5, too loose)
    TradePhase.EXPANSION:    PhaseConfig(exit_tolerance=0.30, sl_multiplier=0.9,  pullback_allowed=True),  # Tighten at +2R+ (was 1.2)
}


class PhaseManager:
    """
    Tracks the current trade phase and exposes phase-appropriate parameters
    to the exit engine and order manager.

    Usage:
        pm = PhaseManager(entry_price=22100, stop_loss=21900)
        pm.tick(current_price=22250)  # called each candle
        phase  = pm.phase
        config = pm.config
    """

    def __init__(self, entry_price: float, stop_loss: float) -> None:
        self.entry_price: float  = entry_price
        self.stop_loss: float    = stop_loss
        self.initial_risk: float = abs(entry_price - stop_loss)
        self.phase: TradePhase   = TradePhase.ENTRY
        self.candles_held: int   = 0
        self._pnl_r: float       = 0.0

        if self.initial_risk == 0:
            log.warning("PhaseManager: initial_risk=0 (entry == SL). Defaulting to 1.")
            self.initial_risk = 1.0

    @property
    def config(self) -> PhaseConfig:
        """Current phase's behavioural parameters."""
        return PHASE_CONFIGS[self.phase]

    @property
    def pnl_r(self) -> float:
        """Last known PnL in R-multiples."""
        return self._pnl_r

    def tick(self, current_price: float) -> TradePhase:
        """
        Update phase based on current price (R-based transitions).

        Args:
            current_price: Current Nifty spot price.

        Returns:
            The (possibly updated) TradePhase.
        """
        self.candles_held += 1

        # PnL in units of R (positive = profit, negative = loss)
        pnl = current_price - self.entry_price
        self._pnl_r = pnl / self.initial_risk

        old_phase = self.phase

        if self._pnl_r >= cfg.PHASE_EXPANSION_R:
            self.phase = TradePhase.EXPANSION
        elif self._pnl_r >= 1.0: # New PARTIAL phase
            self.phase = TradePhase.PARTIAL
        elif self._pnl_r >= cfg.PHASE_CONFIRM_R:
            self.phase = TradePhase.CONFIRMATION
        else:
            self.phase = TradePhase.ENTRY

        if self.phase != old_phase:
            log.info(
                "Trade phase: {old} → {new} | PnL={r:.2f}R (candle {c})",
                old=old_phase.value, new=self.phase.value,
                r=self._pnl_r, c=self.candles_held,
            )

        return self.phase

    def get_exit_tolerance(self) -> float:
        """Exit tolerance for the current phase."""
        return self.config.exit_tolerance

    def get_sl_multiplier(self) -> float:
        """ATR trailing stop multiplier for the current phase."""
        return self.config.sl_multiplier

    def is_pullback_allowed(self) -> bool:
        """Whether retest/pullback detection should hold the trade."""
        return self.config.pullback_allowed
