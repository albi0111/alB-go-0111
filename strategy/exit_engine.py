"""
strategy/exit_engine.py
────────────────────────
Multi-Layer Exit Engine — decides whether to exit, hold, or tighten SL.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE PROBLEM WITH SINGLE-CONDITION EXITS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-candle reversals, retests back to VWAP, and temporary indicator dips
are all normal market behavior in Nifty — not actual trade invalidation.
A single check_reversal() call can't distinguish between:
  (a) Noise — 1-candle RSI dip during a strong uptrend       → HOLD
  (b) Retest — price returns to VWAP on declining volume      → HOLD
  (c) Real reversal — all indicators flipping for 2+ candles  → EXIT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE SIX EXIT LAYERS (evaluated in order — first match wins)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Layer 1 — Profit Lock (TIGHTEN_SL):
  When pnl ≥ PROFIT_LOCK_R × initial_risk: lock profits by tightening
  stop aggressively. Highest priority — protects accumulated gains.

Layer 2 — Max Holding Time (EXIT):
  Force-exit after MAX_HOLD_CANDLES. Prevents capital lock in stalled trades.

Layer 3 — Retest Detector (HOLD):
  If: price near VWAP or breakout level AND volume declining AND trend valid
  → HOLD for up to RETEST_MAX_HOLD_CANDLES candles, then release.
  This is the key fix for healthy pullback premature exits.

Layer 4 — Persistence Filter (HOLD / EXIT):
  Weakness counter must reach REVERSAL_PERSIST_CANDLES before EXIT is allowed.
  Single-candle dips reset the counter → HOLD.

Layer 5 — Opposite Strength Confirmation (EXIT):
  True reversal requires: opposite_score > exit_tolerance AND has been rising
  for 2 consecutive candles (not just a spike).

Layer 6 — Confidence Decay Exponential (EXIT):
  confidence decays multiplicatively per weak candle:
    confidence = confidence × (1 - CONFIDENCE_DECAY_RATE)
  Recovers when held_score improves. Exit only at CONFIDENCE_EXIT_THRESHOLD.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import core.config as cfg
from core.logger import log
from strategy.regime import MarketRegime
from strategy.phase_manager import PhaseManager, TradePhase


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class ExitContext:
    """
    All information needed for one exit evaluation tick.
    Passed to ExitEngine.evaluate() every candle while a trade is open.
    """
    held_direction: str       # "BUY" or "SELL"
    held_score: float         # Composite score for our direction this candle
    opposite_score: float     # Composite score for the opposite direction
    current_price: float      # Nifty spot price
    entry_price: float
    initial_risk: float       # |entry_price - stop_loss|
    vwap: float
    breakout_high: float
    breakout_low: float
    vol_trend: str            # "rising" | "flat" | "declining"
    candles_held: int
    pnl_r: float              # Current PnL in R-multiples
    
    # ── Phase 4 Execution Realism ──
    current_ltp:    float = 0.0
    bid:            float = 0.0
    ask:            float = 0.0
    atr:            float = 0.0
    regime:         Optional[MarketRegime] = None


@dataclass
class ExitDecision:
    """
    Result of one exit evaluation.

    action:
      "HOLD"       — do nothing, trade still valid
      "EXIT"       — close position now
      "TIGHTEN_SL" — keep position but move SL aggressively closer
    """
    should_exit: bool = False
    action: str       = "HOLD"    # "HOLD" | "EXIT" | "TIGHTEN_SL"
    reason: str       = ""
    layer: str        = ""        # which layer triggered
    confidence: float = 1.0


# ── Exit Engine ────────────────────────────────────────────────────────────────

class ExitEngine:
    """
    Stateful multi-layer exit evaluator. One instance per open trade.

    Create at trade entry:
        engine = ExitEngine(regime=regime, phase_manager=pm)

    Call each candle:
        decision = engine.evaluate(ctx)
    """

    def __init__(self, regime: MarketRegime, phase_manager: PhaseManager) -> None:
        self.regime         = regime
        self.phase_manager  = phase_manager
        self.confidence     = 1.0
        self._weak_streak   = 0           # Consecutive candles below exit_tolerance
        self._retest_held   = 0           # Candles currently held as retest
        self._opp_history   = []          # Last 2 opposite scores (for confirmation)
        self._prev_opp_score: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, ctx: ExitContext) -> ExitDecision:
        """
        Evaluate all exit layers in order. First match that triggers returns.

        NOTE: Layer 1 (Profit Lock / TIGHTEN_SL) has been REMOVED.
        Partial exit at +1.0R and breakeven SL shift are handled in
        OrderManager._update_position() BEFORE this method is called.
        Keeping PROFIT_LOCK_R here caused winners to be cut prematurely
        by tightening the SL before partial exits executed.

        Args:
            ctx: ExitContext for this candle.

        Returns:
            ExitDecision with action, reason, and current confidence.
        """
        # Compute phase-appropriate exit tolerance
        exit_tol = max(
            self.phase_manager.get_exit_tolerance(),
            self.regime.exit_tolerance,
        )

        # ── Layer 1: Max Holding Time ─────────────────────────────────────────
        if ctx.candles_held >= cfg.MAX_HOLD_CANDLES:
            log.info("⏱ Max hold time reached ({n} candles). Exiting.", n=ctx.candles_held)
            return ExitDecision(
                should_exit = True,
                action      = "EXIT",
                reason      = f"Max hold time: {ctx.candles_held} candles",
                layer       = "max_hold",
                confidence  = self.confidence,
            )

        # ── Layer 3: Retest Detector (HOLD if pullback looks healthy) ─────────
        is_healthy_retest = self._is_healthy_retest(ctx, exit_tol)
        if is_healthy_retest and self.phase_manager.is_pullback_allowed():
            self._retest_held += 1
            if self._retest_held <= cfg.RETEST_MAX_HOLD_CANDLES:
                log.debug(
                    "Retest hold {n}/{max}: price near structure, volume declining.",
                    n=self._retest_held, max=cfg.RETEST_MAX_HOLD_CANDLES,
                )
                self._weak_streak = 0   # Reset weak streak — this is a retest, not weakness
                return ExitDecision(
                    should_exit = False,
                    action      = "HOLD",
                    reason      = f"Healthy retest (candle {self._retest_held}/{cfg.RETEST_MAX_HOLD_CANDLES})",
                    layer       = "retest",
                    confidence  = self.confidence,
                )
            else:
                # Retest hold cap reached — fall through to normal evaluation
                log.debug("Retest hold cap reached. Reverting to normal evaluation.")
        else:
            self._retest_held = 0

        # ── Layer 4: Persistence Filter ───────────────────────────────────────
        if ctx.held_score < exit_tol:
            self._weak_streak += 1
        else:
            self._weak_streak = 0   # Score recovered — reset
            self._update_confidence(ctx.held_score, exit_tol, improving=True)

        if self._weak_streak < cfg.REVERSAL_PERSIST_CANDLES:
            # Weakness not yet persistent — hold but decay confidence
            self._update_confidence(ctx.held_score, exit_tol, improving=False)
            self._track_opposite(ctx.opposite_score)
            return ExitDecision(
                should_exit = False,
                action      = "HOLD",
                reason      = f"Weak streak {self._weak_streak}/{cfg.REVERSAL_PERSIST_CANDLES} — waiting",
                layer       = "persistence",
                confidence  = self.confidence,
            )

        # ── Layer 5: Opposite Strength Confirmation ───────────────────────────
        opp_confirmed = self._is_opposite_confirmed(ctx.opposite_score, exit_tol)
        if not opp_confirmed and ctx.held_score >= exit_tol * 0.8:
            # Held score weakened but not collapsed, opposite not confirmed → hold
            self._track_opposite(ctx.opposite_score)
            self._update_confidence(ctx.held_score, exit_tol, improving=False)
            return ExitDecision(
                should_exit = False,
                action      = "HOLD",
                reason      = "Opposite direction not yet confirmed rising",
                layer       = "opposite_confirm",
                confidence  = self.confidence,
            )

        # ── Layer 6: Confidence Decay ─────────────────────────────────────────
        self._update_confidence(ctx.held_score, exit_tol, improving=False)
        self._track_opposite(ctx.opposite_score)

        if self.confidence >= cfg.CONFIDENCE_EXIT_THRESHOLD:
            return ExitDecision(
                should_exit = False,
                action      = "HOLD",
                reason      = f"Confidence {self.confidence:.3f} still above threshold",
                layer       = "confidence",
                confidence  = self.confidence,
            )

        # All layers cleared → EXIT
        log.warning(
            "🚨 EXIT triggered | Layer=confidence_decay | held={h:.3f} opp={o:.3f} "
            "conf={c:.3f} streak={s}",
            h=ctx.held_score, o=ctx.opposite_score,
            c=self.confidence, s=self._weak_streak,
        )
        return ExitDecision(
            should_exit = True,
            action      = "EXIT",
            reason      = (
                f"All exit layers cleared: held={ctx.held_score:.3f} "
                f"conf={self.confidence:.3f} streak={self._weak_streak}"
            ),
            layer       = "confidence_decay",
            confidence  = self.confidence,
        )

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _is_healthy_retest(self, ctx: ExitContext, exit_tol: float) -> bool:
        """
        A retest is 'healthy' if:
          1. Price is near VWAP or the breakout level (structural magnet)
          2. Volume is declining (not aggressive selling/buying — just test)
          3. Held score is not catastrophically low (still above 60% of tolerance)

        This pattern is extremely common in Nifty after a breakout — price
        returns to test the broken level before continuing in the trend direction.
        """
        near_vwap = (
            ctx.vwap > 0 and
            abs(ctx.current_price - ctx.vwap) / ctx.vwap <= cfg.VWAP_PROXIMITY_PCT * 2
        )
        near_breakout = False
        if ctx.held_direction == "BUY" and ctx.breakout_high > 0:
            near_breakout = abs(ctx.current_price - ctx.breakout_high) / ctx.breakout_high <= 0.003
        elif ctx.held_direction == "SELL" and ctx.breakout_low > 0:
            near_breakout = abs(ctx.current_price - ctx.breakout_low) / ctx.breakout_low <= 0.003

        at_structure   = near_vwap or near_breakout
        volume_ok      = ctx.vol_trend == "declining"
        score_not_dead = ctx.held_score >= exit_tol * 0.6

        return at_structure and volume_ok and score_not_dead

    def _is_opposite_confirmed(self, opp_score: float, exit_tol: float) -> bool:
        """
        Opposite direction is only 'confirmed' as a real reversal if:
          1. Its current score > exit_tolerance
          2. It has been rising for at least 2 consecutive candles
        """
        if len(self._opp_history) < 2:
            return False
        rising_2 = self._opp_history[-1] > self._opp_history[-2]
        above_tol = opp_score > exit_tol
        return above_tol and rising_2

    def _track_opposite(self, opp_score: float) -> None:
        """Keep rolling 2-element history of opposite direction scores."""
        self._opp_history.append(opp_score)
        if len(self._opp_history) > 2:
            self._opp_history.pop(0)

    def _update_confidence(self, held_score: float, exit_tol: float, improving: bool) -> None:
        """
        Update confidence using exponential decay when weakening.
        Recover linearly when improving (capped at 1.0).
        """
        if improving:
            # Gradual recovery capped at 1.0
            self.confidence = min(1.0, self.confidence + cfg.CONFIDENCE_DECAY_RATE)
        else:
            # Exponential decay: faster for sharp drops, slower for marginal weakness
            decay_scale = 1.0
            if held_score < exit_tol * 0.7:
                decay_scale = 1.5   # Bigger drop → faster decay
            rate = min(0.30, cfg.CONFIDENCE_DECAY_RATE * decay_scale)
            self.confidence = self.confidence * (1.0 - rate)
            self.confidence = max(0.0, self.confidence)
