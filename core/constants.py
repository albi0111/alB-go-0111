"""
core/constants.py
─────────────────
All domain constants for the trading system.
Instrument keys, market timings, expiry rules, and index definitions.
These are immutable facts about markets — not user-configurable parameters.
"""

from datetime import time

# ── Market Hours (IST) ──────────────────────────────────────────────────────────
MARKET_OPEN  = time(9, 15)   # NSE cash segment opens
MARKET_CLOSE = time(15, 30)  # NSE cash segment closes

# First 30-minute window used for breakout level calculation
BREAKOUT_WINDOW_END = time(9, 45)

# ── Candle Intervals ────────────────────────────────────────────────────────────
FETCH_INTERVAL_MIN    = 1    # Fetch 1-min candles from Upstox
ANALYSIS_INTERVAL_MIN = 3    # Aggregate to 3-min for strategy evaluation

# ── Nifty Index (cash, no volume — used only for spot price) ────────────────────
NIFTY_SPOT_KEY        = "NSE_INDEX|Nifty 50"      # Upstox instrument key
NIFTY_SPOT_EXCHANGE   = "NSE_INDEX"
NIFTY_SPOT_SYMBOL     = "Nifty 50"

# ── Nifty Futures (volume-bearing instrument — used for all volume analysis) ────
# Note: "NEAR" / "NEXT" month contracts are resolved at runtime by expiry logic.
NIFTY_FUT_EXCHANGE    = "NSE_FO"
NIFTY_FUT_SYMBOL_FMT  = "NIFTY{year}{month}FUT"   # e.g. NIFTY26MARFUT
MONTH_ABBR = {
    1: "JAN", 2: "FEB",  3: "MAR",  4: "APR",
    5: "MAY", 6: "JUN",  7: "JUL",  8: "AUG",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}

# ── Options Configuration ───────────────────────────────────────────────────────
STRIKE_STEP         = 50    # Nifty options are in multiples of ₹50
ATM_RANGE_STRIKES   = 2     # ±2 strikes (±₹100) from ATM to consider

# ── Expiry Rules ────────────────────────────────────────────────────────────────
# Futures: monthly expiry (last Thursday of month)
# If we are in the last 7 calendar days of the month, roll to next month's contract.
FUTURES_ROLLOVER_DAYS_BEFORE_EXPIRY = 7

# Options: weekly expiry (every Tuesday)
# If today is Monday or Tuesday, use the NEXT week's expiry.
OPTIONS_ROLLOVER_WEEKDAYS = {0, 1}  # 0 = Monday, 1 = Tuesday (Python weekday)

# ── Indicators ──────────────────────────────────────────────────────────────────
RSI_PERIOD              = 14
EMA_FAST_PERIOD         = 20
EMA_SLOW_PERIOD         = 50
ATR_PERIOD              = 14
ADX_PERIOD              = 14    # Average Directional Index
VOLUME_ROLLING_PERIOD   = 20   # Candles for rolling average volume

# ── Regime Detection ────────────────────────────────────────────────────────────
ADX_TREND_THRESHOLD     = 20    # ADX below = ranging (score penalty, NOT a gate)
ATR_HIGH_MULTIPLIER     = 1.5   # ATR > 1.5× median = high volatility
REGIME_LOOKBACK         = 20    # Candles to compute regime baselines
SCORE_ACCEL_LOOKBACK    = 3     # Candles for score acceleration calculation

# ── Risk / Reward ───────────────────────────────────────────────────────────────
RISK_REWARD_RATIO       = 1.5  # Target = SL_distance × 1.5
ATR_SL_MULTIPLIER       = 1.0  # Stop loss = entry ± (ATR × multiplier)
ATR_TRAIL_MULTIPLIER    = 0.5  # Trailing stop tightens at this ATR distance

# ── Upstox API Endpoints ────────────────────────────────────────────────────────
UPSTOX_BASE_URL         = "https://api.upstox.com/v2"
UPSTOX_WS_URL           = "wss://api.upstox.com/v2/feed/market-data-streamer"
