"""
ai/inference.py
───────────────
Hybrid rule + dual-AI signal filter.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DUAL-MODEL FUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two models are trained separately and combined:

  Classifier:  P(profitable) = classifier_prob  ← 0 to 1
  Regressor:   predicted log_return → sigmoid → reg_score  ← 0 to 1

  ml_score = 0.6 × classifier_prob + 0.4 × sigmoid(reg_output)

Then blended with the rule-based score using dynamic weight (by dataset size):

  final_score = rule_w × rule_score + ai_w × ml_score

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FALLBACK CHAIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  No model file      → use rule_score directly
  Too few samples    → use rule_score directly
  Classifier P < 0.55 → use rule_score directly (model uncertain)
  If regressor missing → ml_score = classifier_prob alone
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCALER CONSISTENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
StandardScaler is embedded inside each Pipeline object (model.py).
Loading the Pipeline automatically loads the correct fitted scaler.
No separate scaler file needed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import math
from typing import Optional

from ai.model import (
    load_model, load_regressor, load_quality_model,
    FEATURE_COLS
)
from strategy.conditions import SignalResult
import core.config as cfg
from core.logger import log
from data.data_storage import get_session, AIDataset

# ── Globals (Model Cache) ───────────────────────────────────────────────────────
_CLF_CACHE_MAIN = None
_REG_CACHE      = None
_CLF_CACHE_QUAL  = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid: maps any real → (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def _dataset_size() -> int:
    """Return the number of labelled rows in ai_dataset."""
    with get_session() as session:
        return session.query(AIDataset).filter(AIDataset.label.isnot(None)).count()


def _ai_weight(n_samples: int) -> float:
    """
    Dynamic AI weight based on dataset maturity.
    Capped at HYBRID_AI_WEIGHT_MAX (default 0.5).
    """
    if n_samples < 100:
        raw = 0.20
    elif n_samples < 300:
        raw = 0.30
    elif n_samples < 1000:
        raw = 0.40
    else:
        raw = cfg.HYBRID_AI_WEIGHT_MAX
    return min(raw, cfg.HYBRID_AI_WEIGHT_MAX)


def _build_feature_vector(signal: SignalResult) -> list:
    """
    Build feature vector matching FEATURE_COLS order.
    Gracefully handles missing new fields with safe defaults.
    """
    return [[
        signal.bullish_score,
        signal.bearish_score,
        signal.signal_strength,
        # Individual scores (populated by evaluate_signal if score_breakdown stored)
        getattr(signal, "vol_score",         signal.signal_strength),
        getattr(signal, "trend_score",       signal.signal_strength),
        getattr(signal, "breakout_score",    signal.signal_strength),
        getattr(signal, "vwap_score",        signal.signal_strength),
        getattr(signal, "rsi_score",         signal.signal_strength),
        getattr(signal, "adx_score",         0.6),
        getattr(signal, "vol_trend_score",   0.7),
        # Raw indicators
        signal.rsi,
        signal.vwap_distance_pct,
        signal.ema_diff_pct,
        signal.atr,
        getattr(signal, "adx",               0.0),
        # Regime
        getattr(signal, "regime_encoded",    1),    # default TRENDING
        getattr(signal, "volatility_encoded", 1),   # default normal
        # Direction
        getattr(signal, "direction_encoded", 1),    # default BUY
    ]]


# ── Public API ─────────────────────────────────────────────────────────────────

def should_trade(signal: SignalResult) -> tuple[bool, float]:
    """
    Backward-compatible entry point.
    Returns (allow: bool, final_score: float).
    """
    return final_hybrid_score(signal)


def final_hybrid_score(signal: SignalResult) -> tuple[bool, float]:
    """
    Compute the hybrid rule + AI final score.

    Pipeline:
      1. Feature vector → classifier → P(profitable)
      2. Feature vector → regressor → log_return → sigmoid → reg_score
      3. ml_score = 0.6 × clf_prob + 0.4 × reg_score
      4. final_score = rule_w × rule_score + ai_w × ml_score

    Returns:
        (allow: bool, final_score: float)
    """
    rule_score = signal.signal_strength
    n_labelled = _dataset_size()

    # ── Not enough data ────────────────────────────────────────────────────────
    if n_labelled < cfg.AI_MIN_SAMPLES:
        log.info(
            "AI bypassed ({n}/{min} samples). Using rule score={r:.3f}.",
            n=n_labelled, min=cfg.AI_MIN_SAMPLES, r=rule_score,
        )
        return rule_score >= cfg.SIGNAL_STRENGTH_THRESHOLD, rule_score

    # ── Load models (Cached) ───────────────────────────────────────────────────
    global _CLF_CACHE_MAIN
    if _CLF_CACHE_MAIN is None:
        _CLF_CACHE_MAIN = load_model()
    
    clf = _CLF_CACHE_MAIN
    if clf is None:
        log.warning("No classifier found. Run `algo --train-ai`. Using rule score.")
        return rule_score >= cfg.SIGNAL_STRENGTH_THRESHOLD, rule_score

    # ── Build feature vector ───────────────────────────────────────────────────
    try:
        X = _build_feature_vector(signal)
        clf_prob = float(clf.predict_proba(X)[0][1])    # P(label=1 = profit)
    except Exception as exc:
        log.warning("Classifier inference error: {e}. Using rule score.", e=exc)
        return rule_score >= cfg.SIGNAL_STRENGTH_THRESHOLD, rule_score

    # ── Classifier uncertainty bypass ─────────────────────────────────────────
    if clf_prob < cfg.AI_CONFIDENCE_THRESHOLD:
        log.info(
            "🤖 Classifier uncertain (P={p:.3f} < {thr}). Using rule={r:.3f}.",
            p=clf_prob, thr=cfg.AI_CONFIDENCE_THRESHOLD, r=rule_score,
        )
        return rule_score >= cfg.SIGNAL_STRENGTH_THRESHOLD, rule_score

    # ── Regressor (optional — degrade gracefully if missing) ──────────────────
    global _REG_CACHE
    if _REG_CACHE is None:
        _REG_CACHE = load_regressor()
    
    reg = _REG_CACHE
    if reg is not None:
        try:
            reg_output = float(reg.predict(X)[0])     # predicted log_return
            reg_score  = _sigmoid(reg_output)          # normalise to (0, 1)
        except Exception as exc:
            log.debug("Regressor inference error: {e}. Using classifier only.", e=exc)
            reg_score = clf_prob
    else:
        reg_score = clf_prob   # Regressor not trained yet — fall back to clf

    # ── Dual-model fusion ────────────────────────────────────────────────────
    ml_score = 0.6 * clf_prob + 0.4 * reg_score

    # ── Dynamic rule / AI blend ──────────────────────────────────────────────
    ai_w   = _ai_weight(n_labelled)
    rule_w = 1.0 - ai_w
    final  = rule_w * rule_score + ai_w * ml_score
    allow  = final >= cfg.SIGNAL_STRENGTH_THRESHOLD

    log.info(
        "🤖 Hybrid | rule={r:.3f}×{rw:.0%} + ml={ml:.3f}×{aw:.0%} = {f:.3f} | "
        "clf={c:.3f} reg={rs:.3f} | n={n} | {'ALLOW' if allow else 'SKIP'}",
        r=rule_score, rw=rule_w, ml=ml_score, aw=ai_w,
        f=final, c=clf_prob, rs=reg_score, n=n_labelled,
    )

    return allow, final


def predict_quality_prob(signal: SignalResult) -> float:
    """
    Predict the probability of a signal resulting in a High Quality trade (R >= 0.5).
    ML-driven selective execution filter.
    """
    global _CLF_CACHE_QUAL
    if _CLF_CACHE_QUAL is None:
        _CLF_CACHE_QUAL = load_quality_model()
    
    clf = _CLF_CACHE_QUAL
    if clf is None:
        return 1.0  # Pass-through if model not trained

    try:
        X = _build_feature_vector(signal)
        prob = float(clf.predict_proba(X)[0][1])
        return prob
    except Exception as exc:
        log.warning("Quality classifier inference error: {e}", e=exc)
        return 1.0
