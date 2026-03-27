"""
ai/dataset_builder.py
---------------------
Two-stage ML dataset generation pipeline.
"""

from __future__ import annotations
import math
from datetime import datetime
from typing import Optional
from options.option_selector import OptionSelection
from strategy.conditions import SignalResult
from strategy.regime import MarketRegime
from data.data_storage import save_ai_record, update_ai_outcome
from core.logger import log
import core.config as cfg

class DataLeakageError(RuntimeError):
    """Raised when feature_ts > entry_ts."""

def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _encode_regime(regime: MarketRegime) -> int:
    return 1 if regime.type == "TRENDING" else 0

def _encode_volatility(regime: MarketRegime) -> int:
    return {"low": 0, "normal": 1, "high": 2}.get(regime.volatility, 1)

def _encode_direction(direction: str) -> int:
    return 1 if direction == "BUY" else 0

def record_signal_features(
    signal: SignalResult,
    regime: MarketRegime,
    selection: OptionSelection,
    trade_id: int,
    score_breakdown: Optional[dict] = None,
    feature_ts: Optional[datetime] = None,
    entry_ts: Optional[datetime] = None,
) -> int:
    """
    Save the full feature vector BEFORE the trade is live.

    Args:
        signal:          SignalResult from evaluate_signal().
        regime:          MarketRegime from detect_regime().
        selection:       OptionSelection with strike, OI, volume.
        trade_id:        Trade.id of the opened position.
        score_breakdown: Dict of per-condition scores from _compute_direction_score().
                         Keys: vol_score, trend_score, breakout_score, vwap_score,
                                rsi_score, adx_score, vol_trend_score
        feature_ts:      Timestamp of feature generation. Defaults to now().
        entry_ts:        Timestamp of trade entry. Defaults to now().

    Returns:
        ID of the inserted AIDataset row (stored on Position for Stage 2).

    Raises:
        DataLeakageError: If feature_ts > entry_ts (future data used).
    """
    feature_ts = feature_ts or datetime.now()
    entry_ts   = entry_ts or feature_ts

    # ── Leakage guard ─────────────────────────────────────────────────────────
    # Strip timezone info for comparison if present
    ft  = feature_ts.replace(tzinfo=None)
    et  = entry_ts.replace(tzinfo=None) if entry_ts.tzinfo else entry_ts
    if ft > et:
        raise DataLeakageError(
            f"Feature timestamp {ft} is AFTER entry timestamp {et}. "
            "This indicates future data leakage — check data collection loop."
        )

    # ── Unpack per-condition breakdown (fallback to signal_strength if absent) ─
    bd = score_breakdown or {}
    direction = signal.direction or "BUY"

    # ── OI change: difference vs previous snapshot (use 0 if unavailable) ─────
    oi_change = getattr(selection, "oi_change", 0.0) or 0.0

    record = {
        "trade_id":          trade_id,
        "feature_ts":        feature_ts,

        # Composite scores
        "bullish_score":     signal.bullish_score,
        "bearish_score":     signal.bearish_score,
        "signal_strength":   signal.signal_strength,

        # Individual condition scores
        "vol_score":         bd.get("volume_spike",   signal.signal_strength),
        "trend_score":       bd.get("trend",          signal.signal_strength),
        "breakout_score":    bd.get("breakout",       signal.signal_strength),
        "vwap_score":        bd.get("vwap_proximity", signal.signal_strength),
        "rsi_score":         bd.get("rsi",            signal.signal_strength),
        "adx_score":         bd.get("adx",            0.6),
        "vol_trend_score":   bd.get("volume_trend",   0.7),

        # Raw indicators
        "volume_spike_ratio": signal.volume_spike_ratio,
        "rsi":               signal.rsi,
        "vwap_distance_pct": signal.vwap_distance_pct,
        "ema_diff_pct":      signal.ema_diff_pct,
        "atr_val":           signal.atr,
        "adx_val":           getattr(signal, "adx", 0.0),

        # Regime (encoded)
        "regime_encoded":     _encode_regime(regime),
        "volatility_encoded": _encode_volatility(regime),
        "regime_at_entry":    regime.type,

        # Option chain
        "strike":        selection.strike,
        "oi":            getattr(selection, "oi", 0.0) or 0.0,
        "option_volume": getattr(selection, "volume", 0.0) or 0.0,
        "oi_change":     oi_change,

        # Direction
        "direction_encoded": _encode_direction(direction),

        # Outcome fields — all None until Stage 2
        "label":            None,
        "label_continuous": None,
        "sample_weight":    None,
        "weighted_label":   None,
        "slippage_cost":    None,
        "slippage_pct":     None,
    }

    row_id = save_ai_record(record)
    log.debug(
        "AI feature vector saved: trade_id={tid}, record_id={rid}, "
        "signal={s:.3f}, regime={reg}",
        tid=trade_id, rid=row_id,
        s=signal.signal_strength,
        reg=regime.type,
    )
    return row_id


# ── Stage 2: Outcome Labelling ─────────────────────────────────────────────────

def label_trade_outcome(
    ai_record_id: int,
    pnl: float,
    entry_price: float,
    exit_price: float,
    qty: int,
    initial_risk: float,
    duration_candles: int,
    noise_threshold: float | None = None,
    signal_strength: float = 0.6,
) -> bool:
    """
    Record outcome and generate labels for an existing AIDataset row.

    Args:
        ai_record_id:     Row ID returned by record_signal_features().
        pnl:              Realised PnL (positive = profit).
        entry_price:      Option LTP at entry.
        exit_price:       Option LTP at exit.
        qty:              Total quantity (lots × LOT_SIZE).
        initial_risk:     |spot_entry - stop_loss| (in spot points).
        duration_candles: Number of candles the trade was held for.
        noise_threshold:  Minimum |pnl| to record (default: cfg value or 0).
        signal_strength:  Strength of the signal at entry (for sample weighting).

    Returns:
        True if the record was labelled, False if filtered out.
    """
    noise_threshold = noise_threshold if noise_threshold is not None else getattr(
        cfg, "AI_NOISE_PNL_THRESHOLD", 0.0
    )

    # ── Quality filter ─────────────────────────────────────────────────────────
    if duration_candles < 1:
        log.debug("AI label skipped: duration < 1 candle (record_id={r})", r=ai_record_id)
        return False

    if abs(pnl) < noise_threshold:
        log.debug("AI label skipped: |pnl|={p:.2f} below noise threshold.", p=abs(pnl))
        return False

    # ── Outcome computation ────────────────────────────────────────────────────
    # return_pct relative to option entry price
    if entry_price > 0:
        raw_return_pct = (exit_price - entry_price) / entry_price * 100.0
    else:
        raw_return_pct = 0.0

    # Clip to [-50, +50] to prevent extreme option moves from destabilising training
    return_pct = max(-50.0, min(50.0, raw_return_pct))

    # Log-return: more stable for regression
    try:
        log_return = math.log(1.0 + return_pct / 100.0)
    except (ValueError, OverflowError):
        log_return = 0.0

    # R-multiple: normalises PnL relative to risk taken
    r_multiple_divisor = (initial_risk * qty) if (initial_risk > 0 and qty > 0) else 1.0
    r_multiple = pnl / r_multiple_divisor

    binary_label = 1 if pnl > 0 else 0

    # ── 3-Class Quality Label ─────────────────────────────────────────
    # 0 = Avoid  (R < 0.5)       — loser or tiny winner, don't replicate
    # 1 = Standard (0.5 ≤ R < 1.5) — acceptable, replicate with normal sizing
    # 2 = Premium  (R ≥ 1.5)       — high quality winner, upsize
    if r_multiple >= 1.5:
        quality_label = 2
    elif r_multiple >= 0.5:
        quality_label = 1
    else:
        quality_label = 0  # Includes all losses and near-breakeven

    quality_score = round(min(max(r_multiple, -3.0), 5.0), 4)  # Clipped continuous R
    # ── Advanced Labels & Weights ─────────────────────────────────────────────
    # weighted_label: sigmoid(r_multiple) ensures large winners/losers have clear labels
    weighted_label = sigmoid(r_multiple)
    
    # sample_weight: prioritized learning from high-conviction/magnitude trades
    # Clip R at 3.0 to avoid outlier dominance
    sample_weight = min(abs(r_multiple), 3.0) * signal_strength

    # ── Slippage Recording ────────────────────────────────────────────────────
    # Note: exit_price already contains slippage if passed from OrderManager._close()
    # entry_price already contains slippage if passed from OrderManager.try_open()
    # We don't re-calculate it here, we just record the result impact if desired.
    # slippage_cost can be inferred from the difference between raw and adjusted in OrderManager.

    outcome = {
        "entry_price":      entry_price,
        "exit_price":       exit_price,
        "pnl":              round(pnl, 2),
        "return_pct":       round(return_pct, 4),
        "log_return":       round(log_return, 6),
        "r_multiple":       round(r_multiple, 4),
        "label":            binary_label,
        "label_continuous": round(return_pct, 4),
        "weighted_label":   round(weighted_label, 6),
        "sample_weight":    round(sample_weight, 6),
        "quality_label":    quality_label,     # 0=Avoid, 1=Standard, 2=Premium
        "quality_score":    quality_score,     # Continuous R (clipped)
    }

    update_ai_outcome(ai_record_id, outcome)

    log.info(
        "AI label set: record_id={rid} | label={l} | R={rm:.2f} | Weight={w:.2f} "
        "| pnl=₹{pnl:.2f}",
        rid=ai_record_id, l=binary_label,
        rm=r_multiple, w=sample_weight, pnl=pnl,
    )
    return True
