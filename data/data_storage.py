"""
data/data_storage.py
────────────────────
SQLAlchemy ORM models and database helpers.

Tables:
  candles      — OHLCV bars (1-min and 3-min, index + futures)
  option_chain — Option chain snapshots
  trades       — All paper/live trades
  signals      — Every signal evaluated (whether traded or not)
  ai_dataset   — Rich feature vectors with dual labels for ML training
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import core.config as cfg
from core.logger import log


# ── ORM Base ───────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Models ─────────────────────────────────────────────────────────────────────

class Candle(Base):
    """OHLCV candle for any instrument and timeframe."""
    __tablename__ = "candles"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    instrument = Column(String(50), nullable=False, index=True)
    timeframe  = Column(Integer,    nullable=False, index=True)
    ts         = Column(DateTime,   nullable=False, index=True)
    open       = Column(Float, nullable=False)
    high       = Column(Float, nullable=False)
    low        = Column(Float, nullable=False)
    close      = Column(Float, nullable=False)
    volume     = Column(Float, nullable=False)


class OptionChainSnapshot(Base):
    """Single row per strike per snapshot time."""
    __tablename__ = "option_chain"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ts          = Column(DateTime, nullable=False, index=True)
    expiry      = Column(String(20), nullable=False)
    strike      = Column(Float, nullable=False)
    option_type = Column(String(2), nullable=False)
    ltp         = Column(Float)
    bid         = Column(Float)
    ask         = Column(Float)
    oi          = Column(Float)
    volume      = Column(Float)


class Signal(Base):
    """Every signal evaluation — stored regardless of whether a trade was taken."""
    __tablename__ = "signals"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    ts                 = Column(DateTime, nullable=False, index=True)
    direction          = Column(String(5))
    option_type        = Column(String(2))
    signal_strength    = Column(Float)
    spot_price         = Column(Float)
    vwap               = Column(Float)
    rsi                = Column(Float)
    ema20              = Column(Float)
    ema50              = Column(Float)
    atr                = Column(Float)
    volume_spike_ratio = Column(Float)
    vwap_distance_pct  = Column(Float)
    ema_diff_pct       = Column(Float)
    breakout_high      = Column(Float)
    breakout_low       = Column(Float)
    traded             = Column(Boolean, default=False)
    skip_reason        = Column(Text)
    mode               = Column(String(15))


class Trade(Base):
    """Full trade lifecycle record."""
    __tablename__ = "trades"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    signal_id        = Column(Integer, nullable=True)
    mode             = Column(String(15), nullable=False)
    direction        = Column(String(5),  nullable=False)
    option_type      = Column(String(2),  nullable=False)
    expiry           = Column(String(20))
    strike           = Column(Float)
    lots             = Column(Integer)
    entry_price      = Column(Float)
    exit_price       = Column(Float)
    stop_loss        = Column(Float)
    target           = Column(Float)
    entry_ts         = Column(DateTime)
    exit_ts          = Column(DateTime)
    pnl              = Column(Float)
    exit_reason      = Column(String(50))
    upstox_order_id  = Column(String(50))
    # Phase 4 Metrics
    bid_at_entry     = Column(Float)
    ask_at_entry     = Column(Float)
    slippage         = Column(Float)
    max_reached_r    = Column(Float)


class AIDataset(Base):
    """
    Rich labelled feature vector for ML training.

    Feature capture (at entry):
      - All 7 condition scores (vol, trend, breakout, vwap, rsi, adx, vol_trend)
      - Composite bullish/bearish scores
      - Regime features (ADX value, ATR, regime/volatility encoded)
      - Option chain features (OI, volume, OI change)
      - Direction (encoded integer)
      - feature_ts: timestamp when features were captured (leakage guard)

    Outcome (at close):
      - entry_price, exit_price, pnl
      - return_pct (clipped to [-50, +50])
      - log_return = log(1 + return_pct/100)
      - r_multiple = pnl / (initial_risk × qty)
      - label: binary (1=profit, 0=loss)
      - label_continuous: return_pct (clipped) — preferred for regression
    """
    __tablename__ = "ai_dataset"

    id                 = Column(Integer,  primary_key=True, autoincrement=True)
    trade_id           = Column(Integer,  nullable=True, index=True)

    # ── Leakage guard — when features were captured ──────────────────────────
    feature_ts         = Column(DateTime, nullable=True)

    # ── Composite scores ─────────────────────────────────────────────────────
    bullish_score      = Column(Float)
    bearish_score      = Column(Float)
    signal_strength    = Column(Float)

    # ── Individual condition scores ──────────────────────────────────────────
    vol_score          = Column(Float)
    trend_score        = Column(Float)
    breakout_score     = Column(Float)
    vwap_score         = Column(Float)
    rsi_score          = Column(Float)
    adx_score          = Column(Float)
    vol_trend_score    = Column(Float)

    # ── Raw indicators ───────────────────────────────────────────────────────
    volume_spike_ratio = Column(Float)
    rsi                = Column(Float)
    vwap_distance_pct  = Column(Float)
    ema_diff_pct       = Column(Float)
    atr_val            = Column(Float)
    adx_val            = Column(Float)

    # ── Regime (encoded for ML) ───────────────────────────────────────────────
    # regime_encoded:     0=RANGING, 1=TRENDING
    # volatility_encoded: 0=low, 1=normal, 2=high
    regime_encoded     = Column(Integer)
    volatility_encoded = Column(Integer)

    # ── Option chain ─────────────────────────────────────────────────────────
    strike             = Column(Float)
    oi                 = Column(Float)
    option_volume      = Column(Float)
    oi_change          = Column(Float)

    # ── Direction ────────────────────────────────────────────────────────────
    # direction_encoded: 0=PE (SELL), 1=CE (BUY)
    direction_encoded  = Column(Integer)
    regime_at_entry    = Column(String(20)) # TRENDING / RANGING

    # ── Outcome (filled at close) ─────────────────────────────────────────────
    entry_price        = Column(Float)
    exit_price         = Column(Float)
    slippage_cost      = Column(Float)
    slippage_pct       = Column(Float)
    pnl                = Column(Float)
    return_pct         = Column(Float)   # Clipped to [-50, +50]
    log_return         = Column(Float)   # log(1 + return_pct/100)
    r_multiple         = Column(Float)   # pnl / (initial_risk × qty)
    max_reached_r      = Column(Float)   # High-Water Mark in R units

    # ── Labels & Weights ──────────────────────────────────────────────────────
    label              = Column(Integer) # 1=profit, 0=loss, None=pending
    label_continuous   = Column(Float)   # return_pct (clipped)
    weighted_label     = Column(Float)   # sigmoid(r_multiple)
    sample_weight      = Column(Float)   # abs_r * strength (clipped & normalized)
    quality_label      = Column(Integer) # 0=Avoid, 1=Standard, 2=Premium (R-based)
    quality_score      = Column(Float)   # Continuous R-multiple (clipped -3 to +5)


# ── Database Engine & Session Factory ──────────────────────────────────────────

_engine = None
_SessionFactory = None


def init_db() -> None:
    """
    Initialise the database engine and create all tables.
    Also runs schema migration to add new columns to existing databases.
    Must be called once at application startup.
    """
    global _engine, _SessionFactory

    cfg.DB_DIR.mkdir(parents=True, exist_ok=True)

    connect_args = {}
    if cfg.DB_URL.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(cfg.DB_URL, connect_args=connect_args, echo=False)
    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)

    # Run migration to add any missing columns to ai_dataset
    _migrate_ai_dataset()

    log.info("Database initialised at {url}", url=cfg.DB_URL)


def _migrate_ai_dataset() -> None:
    """
    Safe migration: add missing columns to ai_dataset for existing databases.
    New columns are added with NULL default so existing rows are preserved.
    SQLite does not support ALTER TABLE ADD COLUMN IF NOT EXISTS — we check manually.
    """
    ai_columns = [
        ("feature_ts", "DATETIME"),
        ("bullish_score", "REAL"),
        ("bearish_score", "REAL"),
        ("signal_strength", "REAL"),
        ("vol_score", "REAL"),
        ("trend_score", "REAL"),
        ("breakout_score", "REAL"),
        ("vwap_score", "REAL"),
        ("rsi_score", "REAL"),
        ("adx_score", "REAL"),
        ("vol_trend_score", "REAL"),
        ("adx_val", "REAL"),
        ("atr_val", "REAL"),
        ("regime_encoded", "INTEGER"),
        ("volatility_encoded", "INTEGER"),
        ("strike", "REAL"),
        ("oi", "REAL"),
        ("option_volume", "REAL"),
        ("oi_change", "REAL"),
        ("direction_encoded", "INTEGER"),
        ("regime_at_entry", "TEXT"),
        ("entry_price", "REAL"),
        ("exit_price", "REAL"),
        ("slippage_cost", "REAL"),
        ("slippage_pct", "REAL"),
        ("pnl", "REAL"),
        ("return_pct", "REAL"),
        ("log_return", "REAL"),
        ("r_multiple", "REAL"),
        ("max_reached_r", "REAL"),
        ("label", "INTEGER"),
        ("label_continuous", "REAL"),
        ("weighted_label", "REAL"),
        ("sample_weight", "REAL"),
        ("quality_label", "INTEGER"),
        ("quality_score", "FLOAT"),
    ]
    
    with _engine.connect() as conn:
        for col_name, col_type in ai_columns:
            try:
                conn.execute(text(f"ALTER TABLE ai_dataset ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                pass # Already exists
        
        # Also migrate 'trades' table
        trade_columns = [
            ("bid_at_entry", "FLOAT"),
            ("ask_at_entry", "FLOAT"),
            ("slippage", "FLOAT"),
            ("max_reached_r", "FLOAT"),
        ]
        for col_name, col_type in trade_columns:
            try:
                conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                pass


def get_session() -> Session:
    """Return a new database session. Caller is responsible for closing it."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _SessionFactory()


# ── Convenience Helpers ────────────────────────────────────────────────────────

def save_candles(candles: list[dict]) -> None:
    """Bulk-insert candle dicts."""
    with get_session() as session:
        session.bulk_insert_mappings(Candle, candles)
        session.commit()


def save_signal(signal_data: dict) -> int:
    """Insert a Signal record and return its ID."""
    with get_session() as session:
        sig = Signal(**signal_data)
        session.add(sig)
        session.commit()
        return sig.id


def save_trade(trade_data: dict) -> int:
    """Insert a Trade record and return its ID."""
    with get_session() as session:
        trade = Trade(**trade_data)
        session.add(trade)
        session.commit()
        return trade.id


def update_trade(trade_id: int, **kwargs) -> None:
    """Update specific fields of an existing Trade record."""
    with get_session() as session:
        session.query(Trade).filter(Trade.id == trade_id).update(kwargs)
        session.commit()


def save_ai_record(record: dict) -> int:
    """
    Insert a new AIDataset row with feature snapshot (Stage 1 — at entry).
    Returns the inserted row ID for use in Stage 2 (label_trade_outcome).
    """
    with get_session() as session:
        row = AIDataset(**record)
        session.add(row)
        session.commit()
        return row.id


def update_ai_outcome(record_id: int, outcome: dict) -> None:
    """
    Update an existing AIDataset row with trade outcome (Stage 2 — at close).
    outcome dict should include: pnl, return_pct, log_return, r_multiple,
    label, label_continuous, entry_price, exit_price.
    """
    with get_session() as session:
        session.query(AIDataset).filter(AIDataset.id == record_id).update(outcome)
        session.commit()


def label_ai_record(trade_id: int, label: int) -> None:
    """Legacy helper — set binary label on row matching trade_id."""
    with get_session() as session:
        session.query(AIDataset).filter(AIDataset.trade_id == trade_id).update({"label": label})
        session.commit()


def get_labelled_dataset() -> list[dict]:
    """Return all labelled rows as list of dicts (legacy API — used by old model.py)."""
    with get_session() as session:
        rows = session.query(AIDataset).filter(AIDataset.label.isnot(None)).all()
        return [_row_to_dict(r) for r in rows]


def get_training_dataframe(min_duration_candles: int = 1, min_pnl: float = 0.0) -> pd.DataFrame:
    """
    Return a cleaned, quality-filtered DataFrame ready for sklearn.

    Quality filters applied:
      - label must be set (trade must be closed)
      - return_pct must not be None
      - Any custom filters (min duration, min |pnl|) are applied post-load
        since duration_candles is stored on the Trade table, not AIDataset.

    Returns:
        pd.DataFrame with all feature + label columns, NaN rows dropped.
    """
    with get_session() as session:
        rows = (
            session.query(AIDataset)
            .filter(AIDataset.label.isnot(None))
            .filter(AIDataset.return_pct.isnot(None))
            .all()
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([_row_to_dict(r) for r in rows])

    # Apply quality filters
    if "pnl" in df.columns:
        df = df[df["pnl"].abs() >= min_pnl]

    df = df.dropna(subset=_feature_cols())
    return df


def get_candles(instrument: str, timeframe: int, limit: int = 500) -> list[dict]:
    """Fetch the most recent N candles for an instrument/timeframe."""
    with get_session() as session:
        rows = (
            session.query(Candle)
            .filter(Candle.instrument == instrument, Candle.timeframe == timeframe)
            .order_by(Candle.ts.desc())
            .limit(limit)
            .all()
        )
        rows.reverse()
        return [
            {"ts": r.ts, "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "volume": r.volume}
            for r in rows
        ]


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _row_to_dict(r: AIDataset) -> dict:
    """Convert an AIDataset ORM row to a plain dict."""
    return {col.name: getattr(r, col.name) for col in AIDataset.__table__.columns}


def _feature_cols() -> list[str]:
    """Return the canonical feature column names (used for NaN dropping)."""
    from ai.model import FEATURE_COLS
    return FEATURE_COLS
