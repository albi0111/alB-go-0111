"""
core/logger.py
──────────────
Centralised loguru-based logging setup.

Features:
  - Daily rotating log files in logs/
  - Separate logs for trades, signals, and errors
  - Coloured, structured console output via Rich
  - Single call to `setup_logging()` from main entry point
"""

import sys
from pathlib import Path
from loguru import logger

import core.config as cfg


def setup_logging() -> None:
    """
    Configure loguru sinks.
    Call once at application startup before importing other modules.
    """
    log_dir: Path = cfg.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove the default stderr sink so we control formatting
    logger.remove()

    # ── Console sink ─────────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=cfg.LOG_LEVEL,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        backtrace=True,
        diagnose=False,   # Set True during development for full tracebacks
    )

    # ── General application log (daily rotation) ──────────────────────────────
    logger.add(
        log_dir / "algo_{time:YYYY-MM-DD}.log",
        level=cfg.LOG_LEVEL,
        rotation="00:00",           # New file each day at midnight
        retention="30 days",
        compression="zip",
        enqueue=True,               # Thread-safe async logging
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    )

    # ── Trade log — only trade entry/exit events ──────────────────────────────
    logger.add(
        log_dir / "trades_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation="00:00",
        retention="90 days",        # Keep trade records longer
        compression="zip",
        filter=lambda record: "TRADE" in record["extra"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    )

    # ── Error log ─────────────────────────────────────────────────────────────
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )

    logger.info("Logging initialised. Level={level}, LogDir={dir}", level=cfg.LOG_LEVEL, dir=str(log_dir))


def get_trade_logger():
    """Return a logger instance pre-bound with TRADE context for the trade sink."""
    return logger.bind(TRADE=True)


# Convenience aliases
log = logger
trade_log = get_trade_logger()
