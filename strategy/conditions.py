"""
strategy/conditions.py
───────────────────────
Directional signal scoring engine for Nifty breakout strategy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW DIRECTIONAL SCORING WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every indicator is evaluated TWICE — once for the bullish thesis (→ CE)
and once for the bearish thesis (→ PE). Each produces a score 0.0–1.0.

  Indicator        │  Bullish (CE) score high when…    │  Bearish (PE) score high when…
  ─────────────────┼───────────────────────────────────┼────────────────────────────────
  VWAP position    │  price > VWAP                     │  price < VWAP
  EMA alignment    │  EMA20 > EMA50                    │  EMA20 < EMA50
  Breakout         │  price breaks 30-min HIGH          │  price breaks 30-min LOW
  Volume spike     │  high volume validates direction   │  same (momentum confirmation)
  RSI zone         │  RSI 30–70 (not overbought)       │  RSI 30–70 (not oversold)

Both composite scores are computed. Whichever is higher (and ≥ threshold)
determines the trade direction:

  bullish_score > bearish_score AND bullish_score ≥ THRESHOLD → BUY CE
  bearish_score > bullish_score AND bearish_score ≥ THRESHOLD → BUY PE
  Both below threshold → No trade

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST-ENTRY REVERSAL MONITORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After entering a trade, check_reversal() re-scores every candle.

  If you're in a CE trade (bullish):
    - Monitor bullish_score — if it collapses below EXIT_THRESHOLD → exit
    - Monitor bearish_score — if it surges above REVERSAL_THRESHOLD → exit fast

  If you're in a PE trade (bearish):
    - Monitor bearish_score — if it collapses → exit
    - Monitor bullish_score — if it surges → exit fast

This catches reversals BEFORE price hits the stop-loss, because
indicators flip direction earlier than price does.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEIGHTS (must sum to 1.0):
  volume_spike  : 0.25  — momentum fuel
  breakout      : 0.20  — structural confirmation
  trend         : 0.25  — VWAP + EMA alignment (split 50/50 internally)
  vwap_proximity: 0.15  — entry quality (closer to VWAP = better R:R)
  rsi           : 0.15  — momentum health filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

import core.config as cfg
import core.constants as C
from core.logger import log


# ── Weights ────────────────────────────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "trend":           0.25,    # VWAP position + EMA alignment (internal 50/50)
    "volume_spike":    0.18,    # Momentum fuel (spike ratio)
    "breakout":        0.15,    # 30-min structural confirmation
    "rsi":             0.15,    # Momentum health filter
    "vwap_proximity":  0.12,    # Entry quality (closer = better R:R)
    "adx":             0.10,    # Trend strength (soft penalty when < threshold)
    "volume_trend":    0.05,    # Rising / flat / declining participation
}

# Score below which a held trade is considered "reversing" on next candle
REVERSAL_EXIT_THRESHOLD: float = 0.45
# Score above which the opposite direction signals a strong reversal
REVERSAL_ALERT_THRESHOLD: float = 0.65


# ── Signal Result ──────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """
    Full output of one strategy evaluation cycle.

    direction = "BUY"  → buy CE (Nifty expected to rise)
    direction = "SELL" → buy PE (Nifty expected to fall)
    direction = None   → no actionable signal this candle
    """
    direction: Optional[str]   = None    # "BUY" | "SELL" | None
    option_type: Optional[str] = None    # "CE"  | "PE"   | None

    # Composite scores for both directions (always populated, even if no signal)
    bullish_score: float = 0.0           # Score for CE entry thesis
    bearish_score: float = 0.0           # Score for PE entry thesis
    signal_strength: float = 0.0        # = max(bullish_score, bearish_score)

    # Indicator snapshot (for logging / AI dataset / monitoring)
    spot_price: float        = 0.0
    vwap: float              = 0.0
    rsi: float               = 0.0
    ema20: float             = 0.0
    ema50: float             = 0.0
    atr: float               = 0.0
    adx: float               = 0.0
    volume_spike_ratio: float = 0.0
    vwap_distance_pct: float  = 0.0
    ema_diff_pct: float       = 0.0
    breakout_high: float      = 0.0
    breakout_low: float       = 0.0
    # Diagnostic tracking (Phase 2)
    raw_score: float         = 0.0           # Score before any soft-penalties
    penalty_log: dict        = field(default_factory=dict) # Track which filters applied what penalty
    rejection_reason: Optional[str] = None   # Reason for signal filtering
    ml_quality_prob: float   = 0.0           # ML-predicted probability of R >= 0.5
    
    # Risk levels
    stop_loss: float = 0.0
    target: float    = 0.0

    # Per-condition breakdown for both directions (for debugging / display)
    bullish_scores: dict = field(default_factory=dict)
    bearish_scores: dict = field(default_factory=dict)


@dataclass
class ReversalSignal:
    """
    Result of a post-entry reversal check (called every candle while in a trade).
    """
    should_exit: bool = False
    reason: str = ""
    held_direction_score: float = 0.0    # Score of the direction we're holding
    opposite_direction_score: float = 0.0


# ── Individual Condition Scorers (direction-aware) ─────────────────────────────

def score_volume_spike(spike_ratio: float, threshold: float) -> float:
    """
    Volume spike score — direction-neutral momentum confirmation.

    Volume is the fuel. It doesn't care about direction — it validates
    whichever side is winning. Both bullish and bearish scores receive
    the same volume_spike score.

    Scale:
      ratio ≥ 2× threshold  → 1.0  (very strong)
      ratio = threshold      → 0.7  (just triggered)
      ratio < threshold      → 0.0–0.6 (partial credit)
    """
    if pd.isna(spike_ratio) or spike_ratio <= 0:
        return 0.0
    if spike_ratio >= threshold * 2:
        return 1.0
    if spike_ratio >= threshold:
        # Scale 0.7 → 1.0 between threshold and 2× threshold
        return 0.7 + ((spike_ratio - threshold) / threshold) * 0.3
    # Below threshold: partial credit
    return min(0.6, (spike_ratio / threshold) * 0.6)


def score_breakout(
    price: float,
    breakout_high: float,
    breakout_low: float,
    direction: str,
    margin_pct: float,
) -> float:
    """
    Breakout structural confirmation.

    For BULLISH (CE): price must break *above* the 30-min session high.
      - Above high:                       → 1.0 (confirmed breakout)
      - Within margin_pct below high:     → 0.7–1.0 (approaching, give partial)
      - Far below high:                   → 0.0–0.5 (price not near breakout zone)

    For BEARISH (PE): price must break *below* the 30-min session low.
      - Same logic inverted.

    If 30-min window data is unavailable (early morning): return 0.5 (neutral).
    """
    if direction == "BUY":
        if breakout_high <= 0:
            return 0.5   # Pre-9:45, no breakout reference yet — neutral
        
        # Buffer: allow entry if price is within 0.2% of high
        effective_high = breakout_high * 0.998
        dist = (price - effective_high) / breakout_high
        if dist >= 0:
            return 1.0                                   # Above effective high → confirmed
        if dist >= -margin_pct:
            return 0.7 + (dist / margin_pct) * 0.3      # Near high → approaching
        return max(0.0, 0.5 + dist / margin_pct)         # Far below → penalise

    else:  # SELL → PE
        if breakout_low <= 0:
            return 0.5
            
        # Buffer: allow entry if price is within 0.2% of low
        effective_low = breakout_low * 1.002
        dist = (effective_low - price) / breakout_low     # Positive when price < effective_low
        if dist >= 0:
            return 1.0                                   # Below effective low → confirmed
        if dist >= -margin_pct:
            return 0.7 + (dist / margin_pct) * 0.3
        return max(0.0, 0.5 + dist / margin_pct)


def score_trend(
    price: float,
    vwap: float,
    ema20: float,
    ema50: float,
    direction: str,
    ema_tolerance: float,
) -> float:
    """
    Trend alignment score — combination of VWAP position and EMA crossover.

    Split 50/50:
      VWAP position (50%):
        BULLISH: price > VWAP = 1.0; below VWAP penalised proportionally
        BEARISH: price < VWAP = 1.0; above VWAP penalised proportionally

      EMA alignment (50%):
        BULLISH: EMA20 > EMA50 = 1.0; flat EMAs = 0.5; EMA20 < EMA50 = 0.0
        BEARISH: EMA20 < EMA50 = 1.0; flat EMAs = 0.5; EMA20 > EMA50 = 0.0

    Why both matter:
      VWAP = intraday trend (faster, reacts to current session)
      EMA crossover = medium-term trend (slower, validates direction)
      Both agreeing = high conviction; only one agreeing = partial credit.
    """
    ema_diff     = ema20 - ema50
    ema_diff_pct = ema_diff / ema50 if ema50 > 0 else 0.0

    price_above_vwap = price > vwap
    ema_bullish = ema_diff_pct >  ema_tolerance   # EMA20 meaningfully above EMA50
    ema_bearish = ema_diff_pct < -ema_tolerance   # EMA20 meaningfully below EMA50
    ema_flat    = abs(ema_diff_pct) <= max(ema_tolerance, 0.003)  # EMAs converging

    if direction == "BUY":
        # VWAP component: full score above, scaled penalty below
        vwap_component = (
            1.0 if price_above_vwap
            else max(0.0, 1.0 - (vwap - price) / vwap * 30)
        )
        # EMA component: bullish=1.0, flat=0.5 (uncertain), bearish=0.0
        ema_component = 1.0 if ema_bullish else (0.5 if ema_flat else 0.0)

    else:  # SELL → PE
        # VWAP component: full score below, scaled penalty above
        vwap_component = (
            1.0 if not price_above_vwap
            else max(0.0, 1.0 - (price - vwap) / vwap * 30)
        )
        # EMA component: bearish=1.0, flat=0.5, bullish=0.0
        ema_component = 1.0 if ema_bearish else (0.5 if ema_flat else 0.0)

    return vwap_component * 0.5 + ema_component * 0.5


def score_vwap_proximity(price: float, vwap: float, proximity_pct: float) -> float:
    """
    Entry quality score — how close is price to VWAP?

    Closer to VWAP = better risk:reward for entry (less premium paid,
    tighter stop against VWAP). This is direction-neutral.

    Scoring:
      ≤ proximity_pct distance:  1.0 (ideal entry zone)
      ≤ 2× proximity_pct:        0.7–1.0 (acceptable)
      ≤ 4× proximity_pct:        0.3–0.7 (stretched)
      > 4× proximity_pct:        0.0 (too far from VWAP, chasing)
    """
    if vwap <= 0:
        return 0.5
    dist_pct = abs(price - vwap) / vwap
    if dist_pct <= proximity_pct:
        return 1.0
    if dist_pct <= proximity_pct * 2:
        return 0.7 - (dist_pct - proximity_pct) / proximity_pct * 0.2
    if dist_pct <= proximity_pct * 4:
        return 0.5 - (dist_pct - proximity_pct * 2) / (proximity_pct * 2) * 0.2
    return 0.0


def score_rsi(rsi_value: float, direction: str) -> float:
    """
    RSI momentum health filter.

    RSI tells us whether there is still room to move in the intended direction,
    or whether momentum is near exhaustion.

    For BULLISH (CE) entry:
      RSI 40–65:  ideal zone (momentum with room to grow) → 1.0
      RSI 65–75:  getting hot, caution → 0.7
      RSI > 75:   overbought, momentum likely exhausted → 0.3
      RSI < 30:   oversold — reversal possible but risky for calls → 0.4

    For BEARISH (PE) entry:
      RSI 35–60:  ideal short zone → 1.0
      RSI 25–35:  getting oversold, caution → 0.7
      RSI < 25:   deeply oversold, snap-back risk → 0.3
      RSI > 70:   overbought — puts risky in runaway rally → 0.4

    During MONITORING (reversal detection):
      For CE held: RSI dropping below 40 is early warning
      For PE held: RSI climbing above 60 is early warning
    """
    if pd.isna(rsi_value):
        return 0.7   # Neutral when RSI unavailable (early session data)

    if direction == "BUY":
        if   40 <= rsi_value <= 65: return 1.0
        elif 65 < rsi_value <= 75:  return 0.7
        elif rsi_value > 75:        return 0.3   # Overbought — entry risk
        elif rsi_value < 30:        return 0.4   # Oversold rally may fail
        else:                       return 0.8   # RSI 30–40: decent

    else:  # SELL → PE
        if   35 <= rsi_value <= 60: return 1.0
        elif 25 <= rsi_value < 35:  return 0.7
        elif rsi_value < 25:        return 0.3   # Oversold — PE entry risk
        elif rsi_value > 70:        return 0.4   # Overbought — puts risky
        else:                       return 0.8   # RSI 60–70: decent


def score_adx(adx_value: float, threshold: float) -> float:
    """
    ADX trend-strength score — MODIFIER, not a gate.

    ADX measures HOW STRONGLY the market is trending (not direction).
    Low ADX does NOT block entries — it softly penalises the composite
    score, naturally raising the quality bar for ranging-market entries.

    Scale:
      ADX ≥ 40:          1.0  (very strong trend)
      ADX 25–39:         0.85 (solid trend)
      ADX threshold–24:  0.70 (forming)
      ADX < threshold:   0.20–0.35 (ranging — penalised, never zero)
    """
    if pd.isna(adx_value) or adx_value <= 0:
        return 0.6   # Neutral during warmup
    if adx_value >= 30:
        return 1.0
    if adx_value >= 20:
        return 0.7
    return max(0.50, 0.65 * (adx_value / max(threshold, 1.0)))


def score_volume_trend(vol_trend: str) -> float:
    """
    Volume trend participation score — direction neutral.

    Rising volume = expanding market participation.
    Declining volume = retest or exhaustion.

    Scale:
      rising:   1.0
      flat:     0.7
      declining: 0.4
    """
    return {"rising": 1.0, "flat": 0.7, "declining": 0.4}.get(vol_trend, 0.7)


def score_acceleration(score_history: list) -> float:
    """
    Score acceleration — rate of change of composite score.

    Positive acceleration = building momentum → small bonus.
    Negative acceleration = decaying momentum → small penalty.

    Returns a float in [-0.10, +0.10] applied AFTER weighted sum
    (so weights still sum to 1.0, this is a capped additive modifier).
    """
    n = C.SCORE_ACCEL_LOOKBACK
    if len(score_history) < n + 1:
        return 0.0
    current    = score_history[-1]
    recent_avg = sum(score_history[-(n + 1):-1]) / n
    return max(-0.10, min(0.10, current - recent_avg))


# ── Breakout Level Calculation ─────────────────────────────────────────────────

def get_breakout_levels(df_3min: pd.DataFrame) -> tuple[float, float]:
    """
    Compute the 30-minute opening range (09:15–09:45) breakout levels.

    Returns:
        (breakout_high, breakout_low)
        Both are 0.0 if the 30-min window hasn't completed yet.
    """
    if df_3min.empty:
        return 0.0, 0.0

    df = df_3min.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    window = df[
        (df["ts"].dt.time >= C.MARKET_OPEN) &
        (df["ts"].dt.time <= C.BREAKOUT_WINDOW_END)
    ]
    if window.empty:
        return 0.0, 0.0
    return float(window["high"].max()), float(window["low"].min())


# ── Composite Directional Score Calculator ─────────────────────────────────────

def _compute_direction_score(
    direction: str,
    price: float,
    vwap: float,
    rsi_val: float,
    ema20: float,
    ema50: float,
    spike_ratio: float,
    breakout_high: float,
    breakout_low: float,
    adx_val: float = 0.0,
    vol_trend: str = "flat",
) -> tuple[float, dict]:
    """
    Compute the composite score for ONE direction (BUY or SELL).
    Returns (composite_score, per_condition_breakdown_dict).
    """
    s_vol      = score_volume_spike(spike_ratio, cfg.VOLUME_SPIKE_RATIO)
    s_breakout = score_breakout(price, breakout_high, breakout_low,
                                direction, cfg.BREAKOUT_MARGIN_PCT)
    s_trend    = score_trend(price, vwap, ema20, ema50,
                             direction, cfg.EMA_DIFF_TOLERANCE)
    s_vwap     = score_vwap_proximity(price, vwap, cfg.VWAP_PROXIMITY_PCT)
    s_rsi      = score_rsi(rsi_val, direction)
    s_adx      = score_adx(adx_val, cfg.ADX_TREND_MIN)
    s_vtrd     = score_volume_trend(vol_trend)

    breakdown = {
        "trend":           round(s_trend, 3),
        "volume_spike":    round(s_vol, 3),
        "breakout":        round(s_breakout, 3),
        "rsi":             round(s_rsi, 3),
        "vwap_proximity":  round(s_vwap, 3),
        "adx":             round(s_adx, 3),
        "volume_trend":    round(s_vtrd, 3),
    }
    composite = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)
    return round(composite, 4), breakdown


# ── Entry Signal Evaluation ────────────────────────────────────────────────────

def evaluate_signal(
    price: float,
    vwap: float,
    rsi_val: float,
    ema20: float,
    ema50: float,
    atr_val: float,
    spike_ratio: float,
    df_3min: pd.DataFrame,
    adx_val: float = 0.0,
    vol_trend: str = "flat",
    score_history: list | None = None,
    breakout_high: float | None = None,
    breakout_low: float | None = None,
) -> SignalResult:
    """
    Evaluate ALL conditions for BOTH directions and return a SignalResult.
    """
    if breakout_high is None or breakout_low is None:
        breakout_high, breakout_low = get_breakout_levels(df_3min)
    vwap_dist_pct = abs(price - vwap) / vwap if vwap > 0 else 0.0
    ema_diff_pct  = (ema20 - ema50) / ema50   if ema50 > 0 else 0.0

    # ── Compute scores for both directions ────────────────────────────────────
    bullish_score, bull_breakdown = _compute_direction_score(
        "BUY", price, vwap, rsi_val, ema20, ema50,
        spike_ratio, breakout_high, breakout_low,
        adx_val=adx_val, vol_trend=vol_trend,
    )
    bearish_score, bear_breakdown = _compute_direction_score(
        "SELL", price, vwap, rsi_val, ema20, ema50,
        spike_ratio, breakout_high, breakout_low,
        adx_val=adx_val, vol_trend=vol_trend,
    )

    # ── Score acceleration modifier (applied after weighted sum) ──────────────
    # Uses recent score history to detect building vs decaying momentum.
    # Clipped to ±0.10 so it doesn’t overwhelm the primary composite.
    accel = 0.0
    if score_history:
        accel = score_acceleration(score_history + [max(bullish_score, bearish_score)])
    bullish_score = round(max(0.0, min(1.0, bullish_score + accel)), 4)
    bearish_score = round(max(0.0, min(1.0, bearish_score + accel)), 4)

    log.debug(
        "Scores | BULL={b:.3f} [{bd}] | BEAR={s:.3f} [{sd}] | price={p:.2f}",
        b=bullish_score, bd=bull_breakdown,
        s=bearish_score, sd=bear_breakdown,
        p=price,
    )

    # ── Pick dominant direction ────────────────────────────────────────────────
    if bullish_score >= bearish_score: dominant, opt = "BUY",  "CE"
    else:                                dominant, opt = "SELL", "PE"

    initial_best_score = max(bullish_score, bearish_score)
    final_score = initial_best_score
    rejection_reason = None
    penalty_log = {}

    # ── Phase 2: Soft Penalties (Cumulative & Capped) ──────────────────────────
    multipliers = []
    
    # 1. Chop Penalty: if direction is not clear
    diff = abs(bullish_score - bearish_score)
    if diff < cfg.CHOP_FILTER_THRESHOLD:
        multipliers.append(("CHOP", 0.85))
        if rejection_reason is None: rejection_reason = "CHOP_FILTER"

    # 2. Breakout Validation (Exhaustion Filter)
    s_breakout = bull_breakdown["breakout"] if dominant == "BUY" else bear_breakdown["breakout"]
    if s_breakout > 0.7 and vol_trend == "declining":
        multipliers.append(("BREAKOUT", 0.90))
        if rejection_reason is None: rejection_reason = "BREAKOUT_INVALID"

    # 3. Confidence Penalty: if below confidence gate
    if initial_best_score < cfg.SIGNAL_CONFIDENCE_GATE:
        multipliers.append(("CONFIDENCE", 0.85))
        if rejection_reason is None: rejection_reason = "LOW_CONFIDENCE"

    # Apply Multipliers (Capped at 25% total reduction)
    total_multiplier = 1.0
    for name, m in multipliers:
        total_multiplier *= m
        penalty_log[name] = round(m, 2)
    
    # Cap total reduction at 25% (total_multiplier >= 0.75)
    total_multiplier = max(0.75, total_multiplier)
    final_score = initial_best_score * total_multiplier

    # ── Phase 3: Score Amplification (Expand Distribution) ─────────────────────
    # Push signals in the 0.50–0.55 range higher using Power Law (gamma=0.8)
    # This prevents the "clustering" issue where most signals are just below threshold.
    final_score = pow(final_score, 0.8)
    final_score = round(final_score, 4)

    # ── Phase 1: Debug Instrumentation (Upgraded) ─────────────────────────────
    if initial_best_score >= 0.45:
        threshold = cfg.SIGNAL_STRENGTH_THRESHOLD
    # ── Phase 4: ML + Rule Fusion (Soft Scoring) ──────────────────────────────
    from ai.inference import predict_quality_prob
    ml_prob = predict_quality_prob(result_temp := SignalResult(
        bullish_score=bullish_score, bearish_score=bearish_score, signal_strength=final_score,
        rsi=rsi_val, vwap_distance_pct=vwap_dist_pct, ema_diff_pct=ema_diff_pct, atr=atr_val, adx=adx_val
    ))
    
    # Trending (ADX > 25) -> ml_weight ~ 0.6
    # Ranging (ADX < 20) -> ml_weight ~ 0.35 (Empowering Standard Bin)
    ml_weight = 0.35 + (0.25 / (1 + np.exp(-cfg.FUSION_K * (adx_val - cfg.FUSION_ADX_THRESHOLD))))
    rule_weight = 1.0 - ml_weight
    
    rule_score  = final_score
    fused_score = round(
        (rule_weight * rule_score) + (ml_weight * ml_prob), 
        4
    )
    
    # ── Execution Gate Metrics ────────────────────────────────────────────────
    status = "TAKEN" if fused_score >= threshold else f"REJECTED ({rejection_reason or 'BELOW_THRESHOLD'})"
    log.info(
        "🔍 [Signal Debug] {dir} | {status} | Blended:{fused:.3f} (Rule:{rule:.3f}, ML:{ml:.3f}) | "
        "Thresh:{th:.2f} | Bull:{bull:.3f} Bear:{bear:.3f}",
        dir=dominant, status=status, fused=fused_score, rule=rule_score, ml=ml_prob,
        th=threshold, bull=bullish_score, bear=bearish_score
    )

    if fused_score >= 0.45:
        ml_decision = "ALLOW" if ml_prob >= cfg.ML_QUALITY_THRESHOLD else "REJECT"
        rule_decision = "ALLOW" if rule_score >= cfg.SIGNAL_STRENGTH_THRESHOLD else "REJECT"
        log.info(
            " [ML Fusion] Rule:{rd} | ML:{md} (P={p:.3f}) | Fused:{f:.3f} | Thresh:{th:.2f}",
            rd=rule_decision, md=ml_decision, p=ml_prob, f=fused_score, th=threshold
        )

    # ── Stop loss and target via ATR ───────────────────────────────────────────
    atr_sl = atr_val * C.ATR_SL_MULTIPLIER if not np.isnan(atr_val) else 0.0
    if dominant == "BUY":
        sl     = price - atr_sl
        target = price + atr_sl * C.RISK_REWARD_RATIO
    else:
        sl     = price + atr_sl
        target = price - atr_sl * C.RISK_REWARD_RATIO

    result = SignalResult(
        spot_price         = price,
        vwap               = vwap,
        rsi                = rsi_val,
        ema20              = ema20,
        ema50              = ema50,
        atr                = atr_val,
        volume_spike_ratio = spike_ratio,
        vwap_distance_pct  = round(vwap_dist_pct, 6),
        ema_diff_pct       = round(ema_diff_pct,   6),
        breakout_high      = float(breakout_high or 0.0),
        breakout_low       = float(breakout_low or 0.0),
        bullish_score      = bullish_score,
        bearish_score      = bearish_score,
        signal_strength    = fused_score,
        raw_score          = initial_best_score,
        adx                = adx_val,
        penalty_log        = penalty_log,
        rejection_reason   = rejection_reason,
        stop_loss          = round(sl, 2),
        target             = round(target, 2),
        bullish_scores     = bull_breakdown,
        bearish_scores     = bear_breakdown,
        ml_quality_prob     = ml_prob,
    )

    # ── Final Execution Gate (Scoring-based) ───────────────────────────────────
    if fused_score >= cfg.SIGNAL_STRENGTH_THRESHOLD:
        # Check ML selective filter (Optional secondary gate can be added here if needed)
        # But for now we use the fused_score directly.
        result.direction   = dominant
        result.option_type = opt
        log.info(
            "✅ Signal Captured: {dir} {opt} | fused_score={s:.3f} (rule={r:.3f}) | "
            "price={p:.2f} SL={sl:.2f} TGT={tgt:.2f}",
            dir=dominant, opt=opt, s=fused_score, r=rule_score,
            p=price, sl=sl, tgt=target,
        )

    return result


# ── Post-Entry Reversal Monitoring ─────────────────────────────────────────────

def check_reversal(
    held_direction: str,
    price: float,
    vwap: float,
    rsi_val: float,
    ema20: float,
    ema50: float,
    atr_val: float,
    spike_ratio: float,
    df_3min: pd.DataFrame,
) -> ReversalSignal:
    """
    Post-entry reversal check — called on EVERY candle while a trade is open.

    Re-evaluates the scoring for the direction we're holding AND its opposite.
    Warns to exit early if:
      (a) The held direction's score collapses below REVERSAL_EXIT_THRESHOLD (0.38)
      (b) The opposite direction's score surges above REVERSAL_ALERT_THRESHOLD (0.65)

    This catches reversals BEFORE price hits the stop-loss because indicators
    flip before price does.

    Args:
        held_direction: "BUY" (holding CE) or "SELL" (holding PE)

    Returns:
        ReversalSignal with should_exit=True and reason string if reversal detected.
    """
    breakout_high, breakout_low = get_breakout_levels(df_3min)
    opposite = "SELL" if held_direction == "BUY" else "BUY"

    # Score our current direction (should stay high while trade is healthy)
    held_score, held_bd = _compute_direction_score(
        held_direction, price, vwap, rsi_val, ema20, ema50,
        spike_ratio, breakout_high, breakout_low,
    )
    # Score the opposite direction (should stay low while trade is healthy)
    opp_score, opp_bd = _compute_direction_score(
        opposite, price, vwap, rsi_val, ema20, ema50,
        spike_ratio, breakout_high, breakout_low,
    )

    log.debug(
        "Reversal monitor [{held}]: held={h:.3f} opp={o:.3f}",
        held=held_direction, h=held_score, o=opp_score,
    )

    # ── Check exit conditions ─────────────────────────────────────────────────
    if held_score < REVERSAL_EXIT_THRESHOLD:
        reason = (
            f"Held {held_direction} score collapsed to {held_score:.3f} "
            f"(below exit threshold {REVERSAL_EXIT_THRESHOLD}). "
            f"Indicators no longer supporting {'bullish' if held_direction == 'BUY' else 'bearish'} thesis."
        )
        log.warning("⚠️  REVERSAL: {r}", r=reason)
        return ReversalSignal(
            should_exit              = True,
            reason                   = reason,
            held_direction_score     = held_score,
            opposite_direction_score = opp_score,
        )

    if opp_score > REVERSAL_ALERT_THRESHOLD:
        reason = (
            f"Opposite direction ({opposite}) score surged to {opp_score:.3f} "
            f"(above alert threshold {REVERSAL_ALERT_THRESHOLD}). "
            f"Strong {'bearish' if opposite == 'SELL' else 'bullish'} reversal forming."
        )
        log.warning("🚨 STRONG REVERSAL: {r}", r=reason)
        return ReversalSignal(
            should_exit              = True,
            reason                   = reason,
            held_direction_score     = held_score,
            opposite_direction_score = opp_score,
        )

    return ReversalSignal(
        should_exit              = False,
        reason                   = "No reversal detected",
        held_direction_score     = held_score,
        opposite_direction_score = opp_score,
    )
