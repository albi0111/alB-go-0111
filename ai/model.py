"""
ai/model.py
───────────
Dual-model training pipeline: Classifier + Regressor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Classifier (RandomForestClassifier)
   Target: label (1=profit, 0=loss)
   Handles class imbalance: class_weight="balanced"
   Outputs: P(profitable) → classifier_prob

2. Regressor (RandomForestRegressor)
   Target: log_return (natural log of 1 + return_pct/100)
   Why log_return: normalises option return distribution,
                   more stable training signal than raw %
   Outputs: predicted log_return → sigmoid → reg_score

Combined ML score in inference:
  ml_score = 0.6 × classifier_prob + 0.4 × sigmoid(reg_output)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL VERSIONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each training run saves:
  ai/model_clf_v{N}.joblib      — versioned classifier
  ai/model_reg_v{N}.joblib      — versioned regressor
  ai/model_clf_latest.joblib    — latest (loaded by inference.py)
  ai/model_reg_latest.joblib    — latest
  ai/model_v{N}_meta.json       — metrics + feature importances

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DRIFT PROTECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before promoting new model to "latest":
  If previous model exists and new_accuracy < previous_accuracy - 2%:
    → New model is NOT promoted (versioned copy saved, but latest unchanged)
    → User is warned via log.warning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCALER CONSISTENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
StandardScaler is embedded inside the sklearn Pipeline object.
The same fitted scaler is used at both training AND inference time.
No separate scaler file needed — load the Pipeline, it contains everything.

Usage:
    python -m ai.model --train
    python -m ai.model --evaluate
    python -m ai.model --versions
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

import core.config as cfg
from core.logger import log
from data.data_storage import get_training_dataframe


# ── Feature Columns ────────────────────────────────────────────────────────────
# Must match the columns saved by dataset_builder.py (Stage 1)
# Order is deterministic and must be consistent between training and inference.

FEATURE_COLS: list[str] = [
    # Composite scores
    "bullish_score",
    "bearish_score",
    "signal_strength",
    # Condition scores
    "vol_score",
    "trend_score",
    "breakout_score",
    "vwap_score",
    "rsi_score",
    "adx_score",
    "vol_trend_score",
    # Raw indicators
    "rsi",
    "vwap_distance_pct",
    "ema_diff_pct",
    "atr_val",
    "adx_val",
    # Regime
    "regime_encoded",
    "volatility_encoded",
    "direction_encoded",
]

# ── Model Paths ────────────────────────────────────────────────────────────────

def _model_dir() -> Path:
    return cfg.MODEL_PATH.parent


def _clf_versioned(n: int) -> Path:
    return _model_dir() / f"model_clf_v{n}.joblib"


def _reg_versioned(n: int) -> Path:
    return _model_dir() / f"model_reg_v{n}.joblib"


def _meta_path(n: int) -> Path:
    return _model_dir() / f"model_v{n}_meta.json"


CLF_LATEST = property(lambda _: _model_dir() / "model_clf_latest.joblib")
REG_LATEST = property(lambda _: _model_dir() / "model_reg_latest.joblib")


def clf_latest_path() -> Path:
    return _model_dir() / "model_clf_latest.joblib"


def reg_latest_path() -> Path:
    return _model_dir() / "model_reg_latest.joblib"


def clf_quality_latest_path() -> Path:
    return _model_dir() / "model_quality_clf_latest.joblib"


# ── Metrics Dataclass ──────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    version:       int
    n_samples:     int
    accuracy:      float
    precision:     float
    recall:        float
    f1:            float
    cv_mean:       float
    cv_std:        float
    top_features:  list   # [(feature_name, importance), ...]
    trained_at:    str
    promoted:      bool   # Was this model promoted to "latest"?


# ── Next Version Number ────────────────────────────────────────────────────────

def _next_version() -> int:
    """Return the next model version number (auto-increment based on saved files)."""
    existing = list(_model_dir().glob("model_clf_v*.joblib"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        try:
            # Handle model_clf_vN and model_quality_clf_vN
            parts = p.stem.split("_v")
            if len(parts) > 1:
                nums.append(int(parts[1]))
        except (IndexError, ValueError):
            pass
    return max(nums, default=0) + 1


def _previous_accuracy() -> Optional[float]:
    """Return the accuracy of the current 'latest' model from its meta JSON, or None."""
    metas = sorted(_model_dir().glob("model_v*_meta.json"), key=lambda p: p.stat().st_mtime)
    if not metas:
        return None
    try:
        with open(metas[-1]) as f:
            data = json.load(f)
        if data.get("promoted", False):
            return data.get("accuracy")
    except Exception:
        pass
    return None


# ── Dataset Loading ────────────────────────────────────────────────────────────

def _load_dataset(target_type: str = "binary") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load quality-filtered dataset from database.

    Args:
        target_type: "binary" (pnl > 0), "regressor" (log_return), or "quality" (R >= 0.5)

    Returns:
        (X, y_binary, y_continuous, sample_weights, df)
    """
    df = get_training_dataframe(min_duration_candles=1, min_pnl=0.0)

    if df.empty:
        raise ValueError("No labelled records found in ai_dataset. Run trading to generate data.")

    # Drop rows missing any feature or core labels
    df = df.dropna(subset=FEATURE_COLS + ["label", "log_return"])
    
    # ── Regime Balancing (Optional Undersampling) ──
    # If one regime (e.g. RANGING) heavily dominates, we undersample it to 
    # ensure the model learns trending patterns equally well.
    if cfg.AI_BALANCE_REGIMES and "regime_at_entry" in df.columns:
        counts = df["regime_at_entry"].value_counts()
        if len(counts) > 1:
            min_size = counts.min()
            log.info("Balancing regimes: Undersampling to {n} samples per regime.", n=min_size)
            df = df.groupby("regime_at_entry").apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)

    X           = df[FEATURE_COLS].values.astype(float)
    y_binary    = df["label"].values.astype(int)
    
    if target_type == "quality":
        # Target: y=1 if R >= 0.5 else 0
        y_binary = (df["r_multiple"] >= 0.5).astype(int).values

    y_continuous = df["log_return"].values.astype(float)
    
    # Load weights (fallback to 1.0 if not yet populated in old DBs)
    if "sample_weight" in df.columns:
        # Fill NaN weights with 1.0 (for legacy rows)
        sample_weights = df["sample_weight"].fillna(1.0).values.astype(float)
    else:
        sample_weights = np.ones(len(df))

    return X, y_binary, y_continuous, sample_weights, df


# ── Pipeline Builders ──────────────────────────────────────────────────────────

def _build_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators     = 200,
            max_depth        = 8,
            min_samples_leaf = 3,
            random_state     = 42,
            class_weight     = "balanced",
            n_jobs           = -1,
        )),
    ])


def _build_regressor() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators     = 200,
            max_depth        = 8,
            min_samples_leaf = 3,
            random_state     = 42,
            n_jobs           = -1,
        )),
    ])


# ── Feature Importance ─────────────────────────────────────────────────────────

def _top_features(pipeline: Pipeline, n: int = 5) -> list:
    """Extract top N features by importance from the trained pipeline."""
    importances = pipeline.named_steps["model"].feature_importances_
    pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    return [(name, round(float(imp), 4)) for name, imp in pairs[:n]]


# ── Training ───────────────────────────────────────────────────────────────────

def train(mode: str = "standard") -> Optional[ModelMetrics]:
    """
    Train both classifier and regressor on all available labelled data.
    mode: "standard" (binary + regressor) or "quality" (only high-quality classifier)
    """
    # ── Load dataset ──────────────────────────────────────────────────────────
    try:
        X, y_binary, y_continuous, weights, df = _load_dataset(target_type=mode)
    except ValueError as e:
        log.warning("{e}", e=e)
        return None

    n = len(y_binary)
    if n < cfg.AI_MIN_SAMPLES:
        log.warning(
            "Only {n}/{min} labelled samples. Skipping training.",
            n=n, min=cfg.AI_MIN_SAMPLES,
        )
        return None

    log.info("Training on {n} samples | features={f}", n=n, f=len(FEATURE_COLS))

    # ── Train/test split ────────────────────────────────────────────────────
    # Note: we split indices to keep weights aligned
    indices = np.arange(n)
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    X_train, X_test = X[idx_train], X[idx_test]
    yb_train, yb_test = y_binary[idx_train], y_binary[idx_test]
    yc_train, yc_test = y_continuous[idx_train], y_continuous[idx_test]
    w_train = weights[idx_train]

    # ── Classifier ────────────────────────────────────────────────────────────
    clf = _build_classifier()
    # cv_scores don't easily support sample_weight in cross_val_score without complex shim,
    # so we rely on test set metrics for drift protection.
    clf.fit(X_train, yb_train, model__sample_weight=w_train)
    yb_pred = clf.predict(X_test)

    accuracy  = accuracy_score(yb_test, yb_pred)
    precision = precision_score(yb_test, yb_pred, zero_division=0)
    recall    = recall_score(yb_test, yb_pred, zero_division=0)
    f1        = f1_score(yb_test, yb_pred, zero_division=0)

    log.info("Classifier report:\n{r}", r=classification_report(yb_test, yb_pred, zero_division=0))
    log.info(
        "Accuracy={acc:.3f} | Precision={p:.3f} | Recall={r:.3f} | F1={f:.3f}",
        acc=accuracy, p=precision, r=recall, f=f1
    )

    # ── Regressor ─────────────────────────────────────────────────────────────
    reg = _build_regressor()
    reg.fit(X_train, yc_train, model__sample_weight=w_train)

    # ── Feature importances ────────────────────────────────────────────────────
    top_feats = _top_features(clf)
    log.info("Top features: {f}", f=top_feats)

    # ── Drift protection ───────────────────────────────────────────────────────
    prev_acc    = _previous_accuracy()
    version     = _next_version()
    _model_dir().mkdir(parents=True, exist_ok=True)

    # Retrain on FULL dataset for final saved model
    clf_full = _build_classifier().fit(X, y_binary, model__sample_weight=weights)
    reg_full = _build_regressor().fit(X, y_continuous, model__sample_weight=weights)

    # Save versioned copies regardless
    joblib.dump(clf_full, _clf_versioned(version))
    joblib.dump(reg_full, _reg_versioned(version))

    promoted = True
    if prev_acc is not None and accuracy < prev_acc - 0.02:
        log.warning(
            "⚠️  Drift detected: new accuracy={new:.3f} < previous={old:.3f} - 2%. "
            "Model NOT promoted. Keeping previous version.",
            new=accuracy, old=prev_acc,
        )
        promoted = False
    else:
        # Promote to latest
        if mode == "quality":
            latest_clf = _model_dir() / f"model_quality_clf_v{version}.joblib"
            shutil.copy2(_clf_versioned(version), latest_clf) # Save vN locally first
            shutil.copy2(_clf_versioned(version), clf_quality_latest_path())
            log.info("✅ Quality Classifier v{v} promoted to latest.", v=version)
        else:
            shutil.copy2(_clf_versioned(version), clf_latest_path())
            shutil.copy2(_reg_versioned(version), reg_latest_path())
            log.info("✅ Model v{v} promoted to latest.", v=version)

    # ── Save metadata ──────────────────────────────────────────────────────────
    metrics = ModelMetrics(
        version      = version,
        n_samples    = n,
        accuracy     = round(accuracy,  4),
        precision    = round(precision, 4),
        recall       = round(recall,    4),
        f1           = round(f1,        4),
        cv_mean      = 0.0,  # CV skipped when using sample weights for simplicity
        cv_std       = 0.0,
        top_features = top_feats,
        trained_at   = datetime.now().isoformat(),
        promoted     = promoted,
    )
    with open(_meta_path(version), "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    log.info("Metadata saved → {p}", p=_meta_path(version))

    return metrics


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model() -> Pipeline | None:
    """Load the latest binary (profit/loss) classifier pipeline."""
    p = clf_latest_path()
    if not p.exists():
        return None
    return joblib.load(p)


def load_quality_model() -> Pipeline | None:
    """Load the latest quality (R >= 0.5) classifier pipeline."""
    p = clf_quality_latest_path()
    if not p.exists():
        return None
    return joblib.load(p)


def load_regressor() -> Pipeline | None:
    """Load the latest regressor pipeline. Returns None if not found."""
    p = reg_latest_path()
    if not p.exists():
        log.debug("No trained regressor found at {p}.", p=p)
        return None
    return joblib.load(p)


def list_model_versions() -> list[dict]:
    """Return metadata for all saved model versions, newest first."""
    metas = sorted(_model_dir().glob("model_v*_meta.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for m in metas:
        try:
            with open(m) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


# ── CLI entrypoint ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.logger import setup_logging
    from data.data_storage import init_db
    setup_logging()
    init_db()

    if "--train" in sys.argv:
        mode = "quality" if "--quality" in sys.argv else "standard"
        metrics = train(mode=mode)
        if metrics:
            print(f"\nModel v{metrics.version} ({mode}) trained.")
            print(f"  Accuracy:  {metrics.accuracy:.3f}")
            print(f"  Precision: {metrics.precision:.3f}")
            print(f"  Recall:    {metrics.recall:.3f}")
            print(f"  F1:        {metrics.f1:.3f}")
            print(f"  Top features: {metrics.top_features}")
            print(f"  Promoted:  {metrics.promoted}")
    elif "--versions" in sys.argv:
        versions = list_model_versions()
        for v in versions:
            promoted = "✅" if v.get("promoted") else "⏸"
            print(f"  v{v['version']} {promoted} | acc={v['accuracy']:.3f} | "
                  f"n={v['n_samples']} | {v['trained_at'][:10]}")
    elif "--evaluate" in sys.argv:
        clf = load_model()
        reg = load_regressor()
        print(f"Classifier: {'loaded' if clf else 'NOT FOUND'}")
        print(f"Regressor:  {'loaded' if reg else 'NOT FOUND'}")
    else:
        print("Usage: python -m ai.model --train | --versions | --evaluate")
