"""
core/config.py
──────────────
Central configuration loader. All runtime parameters are sourced from
environment variables (via .env file). No magic numbers elsewhere in the code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate and load the .env file from the project root
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _get(key: str, default=None, cast=str, required: bool = False):
    """Read an env var, optionally cast it, and raise if required but missing."""
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example → .env and fill in your values."
        )
    if value is not None and cast is not str:
        try:
            value = cast(value)
        except (ValueError, TypeError):
            raise ValueError(f"Config key '{key}' could not be cast to {cast.__name__}: got {value!r}")
    return value


# ── Upstox API ─────────────────────────────────────────────────────────────────
UPSTOX_API_KEY: str       = _get("UPSTOX_API_KEY",    required=False, default="")
UPSTOX_API_SECRET: str    = _get("UPSTOX_API_SECRET", required=False, default="")
# UPSTOX_ACCESS_TOKEN is the read-only token for data fetch.
# It is accepted under two env var names for compatibility: UPSTOX_ACCESS_TOKEN or UPSTOX_ANALYSIS_TOKEN.
UPSTOX_ANALYSIS_TOKEN: str = (
    _get("UPSTOX_ANALYSIS_TOKEN", required=False, default="")
    or _get("UPSTOX_ACCESS_TOKEN", required=False, default="")
)
UPSTOX_TRADING_TOKEN: str  = _get("UPSTOX_TRADING_TOKEN", required=False, default="")

# ── Database ───────────────────────────────────────────────────────────────────
_default_db = f"sqlite:///{_ROOT / 'database' / 'algo.db'}"
DB_URL: str = _get("DB_URL", default="") or _default_db


# ── Position Sizing & Risk ─────────────────────────────────────────────────────
LOT_SIZE: int            = _get("LOT_SIZE", default=65, cast=int)
MAX_LOTS: int            = _get("MAX_LOTS", default=6, cast=int)
MAX_OPEN_POSITIONS: int  = _get("MAX_OPEN_POSITIONS", default=3, cast=int)
DAILY_LOSS_LIMIT: float  = _get("DAILY_LOSS_LIMIT", default=5000.0, cast=float)

# ── Strategy Tolerances ────────────────────────────────────────────────────────
SIGNAL_STRENGTH_THRESHOLD: float = _get("SIGNAL_STRENGTH_THRESHOLD", default=0.54, cast=float)
SIGNAL_CONFIDENCE_GATE: float    = _get("SIGNAL_CONFIDENCE_GATE",    default=0.50, cast=float)
SIGNAL_EXIT_THRESHOLD: float     = _get("SIGNAL_EXIT_THRESHOLD",     default=0.45, cast=float)
CHOP_FILTER_THRESHOLD: float     = _get("CHOP_FILTER_THRESHOLD",     default=0.05, cast=float)

# ── Indicator Tolerances ───────────────────────────────────────────────────────
VOLUME_SPIKE_RATIO: float   = _get("VOLUME_SPIKE_RATIO",   default=2.5,  cast=float)
BREAKOUT_MARGIN_PCT: float  = _get("BREAKOUT_MARGIN_PCT",  default=0.001, cast=float)
EMA_DIFF_TOLERANCE: float   = _get("EMA_DIFF_TOLERANCE",   default=0.002, cast=float)
VWAP_PROXIMITY_PCT: float   = _get("VWAP_PROXIMITY_PCT",   default=0.005, cast=float)

# ── Slippage & Execution ───────────────────────────────────────────────────────
SLIPPAGE_K: float        = _get("SLIPPAGE_K",        default=0.05, cast=float)
MIN_SLIPPAGE: float      = _get("MIN_SLIPPAGE",      default=0.50, cast=float)

# ── ADX / Regime ────────────────────────────────────────────────────────────────
ADX_TREND_MIN: float = _get("ADX_TREND_MIN", default=20.0, cast=float)
# ADX below this = ranging market. Score is PENALISED (not blocked). Entries still
# possible if overall composite score exceeds SIGNAL_STRENGTH_THRESHOLD.

# ── Exit Engine ────────────────────────────────────────────────────────────────
REVERSAL_PERSIST_CANDLES: int  = _get("REVERSAL_PERSIST_CANDLES", default=2,    cast=int)
RETEST_MAX_HOLD_CANDLES: int   = _get("RETEST_MAX_HOLD_CANDLES",  default=2,    cast=int)
RETEST_VOLUME_DROP_PCT: float  = _get("RETEST_VOLUME_DROP_PCT",   default=0.30, cast=float)
MOMENTUM_DECAY_FAST: float     = _get("MOMENTUM_DECAY_FAST",      default=0.15, cast=float)
CONFIDENCE_DECAY_RATE: float   = _get("CONFIDENCE_DECAY_RATE",    default=0.05, cast=float)
CONFIDENCE_EXIT_THRESHOLD: float = _get("CONFIDENCE_EXIT_THRESHOLD", default=0.35, cast=float)
MAX_HOLD_CANDLES: int          = _get("MAX_HOLD_CANDLES",          default=30,   cast=int)
COOLDOWN_CANDLES: int          = _get("COOLDOWN_CANDLES",          default=3,    cast=int)

# ── Phase Manager ──────────────────────────────────────────────────────────────
PHASE_CONFIRM_R: float   = _get("PHASE_CONFIRM_R",   default=0.5, cast=float)
PHASE_EXPANSION_R: float = _get("PHASE_EXPANSION_R", default=2.0, cast=float)
# PROFIT_LOCK_R deprecated: replaced by partial exit at +1.0R in order_manager.py
PROFIT_LOCK_R: float     = _get("PROFIT_LOCK_R",     default=999.0, cast=float)  # Effectively disabled

# ── AI Filter ──────────────────────────────────────────────────────────────────
AI_CONFIDENCE_THRESHOLD: float = _get("AI_CONFIDENCE_THRESHOLD", default=0.55, cast=float)
AI_MIN_SAMPLES: int            = _get("AI_MIN_SAMPLES",           default=50,   cast=int)
# AI weight is DYNAMIC — computed at runtime based on dataset size.
# See ai/inference.py:_ai_weight(). HYBRID_AI_WEIGHT is the MAX weight (at ≥1000 samples).
HYBRID_AI_WEIGHT_MAX: float    = _get("HYBRID_AI_WEIGHT_MAX",     default=0.5,  cast=float)
AI_BALANCE_REGIMES: bool       = _get("AI_BALANCE_REGIMES",       default=False, cast=bool) # If True, undersample dominant regime to prevent bias
AI_NOISE_PNL_THRESHOLD: float  = _get("AI_NOISE_PNL_THRESHOLD",   default=0.0,  cast=float)
ML_QUALITY_FILTER_ACTIVE: bool = _get("ML_QUALITY_FILTER_ACTIVE", default=False, cast=bool)
ML_QUALITY_THRESHOLD: float    = _get("ML_QUALITY_THRESHOLD",     default=0.65, cast=float)
ML_FUSION_RULE_WEIGHT: float    = _get("ML_FUSION_RULE_WEIGHT",    default=0.7,  cast=float)
ML_FUSION_QUALITY_WEIGHT: float = _get("ML_FUSION_QUALITY_WEIGHT", default=0.3,  cast=float)


# ── Institutional Risk & Execution ───────────────────────────────────────────
MAX_MARGIN_PCT: float     = _get("MAX_MARGIN_PCT",     default=0.70,   cast=float)
RISK_PER_TRADE_PCT: float = _get("RISK_PER_TRADE_PCT", default=0.0125, cast=float)
LIQUIDITY_PENALTY: float  = _get("LIQUIDITY_PENALTY",  default=2.0,    cast=float)
PAPER_SLIPPAGE_K: float   = _get("PAPER_SLIPPAGE_K",   default=0.5,    cast=float) # ₹ per lot slippage
PAPER_EXEC_DELAY_MS: int  = _get("PAPER_EXEC_DELAY_MS", default=300,   cast=int)   # Simulated latency

# Sigmoid Fusion Tuning
FUSION_ADX_THRESHOLD: float = _get("FUSION_ADX_THRESHOLD", default=22.0, cast=float)
FUSION_K: float             = _get("FUSION_K",             default=0.5,  cast=float)

# Efficiency Stops: raised candles, lowered PnL bar (only kill truly stagnant trades)
ADVERSE_TIME_CANDLES: int = _get("ADVERSE_TIME_CANDLES", default=18,   cast=int)   # was 12
ADVERSE_PNL_R: float      = _get("ADVERSE_PNL_R",      default=-0.2,  cast=float) # was 0.4 — only exit if LOSING

# ── Lifecycle Maturation & Trailing SL ─────────────────────────────────────────
MIN_TRAIL_CANDLES: int     = _get("MIN_TRAIL_CANDLES",     default=5,    cast=int)
TRAIL_MATURATION_R: float  = _get("TRAIL_MATURATION_R",  default=0.5,  cast=float)
TRAIL_STEP_R: float        = _get("TRAIL_STEP_R",        default=0.3,  cast=float)

# ── Equity Curve Feedback ──────────────────────────────────────────────────────
DRAWDOWN_THRESHOLD_1: float = _get("DRAWDOWN_THRESHOLD_1", default=0.04, cast=float)
DRAWDOWN_THRESHOLD_2: float = _get("DRAWDOWN_THRESHOLD_2", default=0.08, cast=float)
RISK_SCALE_1: float         = _get("RISK_SCALE_1",         default=0.70, cast=float)
RISK_SCALE_2: float         = _get("RISK_SCALE_2",         default=0.40, cast=float)

# Multi-Position Logic
MAX_OPEN_POSITIONS: int   = _get("MAX_OPEN_POSITIONS", default=3,    cast=int)
MAX_DIRECTIONAL_RISK: float = _get("MAX_DIRECTIONAL_RISK", default=0.030, cast=float) # 3% per direction


# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("LOG_LEVEL", default="INFO").upper()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = _ROOT
LOGS_DIR: Path     = _ROOT / "logs"
DB_DIR: Path       = _ROOT / "database"
MODEL_PATH: Path         = _ROOT / "ai" / "model.joblib"
MODEL_QUALITY_PATH: Path = _ROOT / "ai" / "model_quality_clf_latest.joblib"
