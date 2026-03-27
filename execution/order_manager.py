"""
execution/order_manager.py
──────────────────────────
High-level trade lifecycle management — decision-engine edition.

Modes:
  live       — Places real orders via upstox_client
  papertrade — Simulates orders using real-time prices, no real orders
  simulate   — Historical replay with virtual positions

New in this version:
  - ExitEngine (multi-layer exit logic) owned per position
  - PhaseManager (R-based ENTRY→CONFIRMATION→EXPANSION) owned per position
  - Cooldown counter: blocks new entries for N candles after any exit
  - MarketRegime passed into ExitEngine on each update()

Responsibilities:
  - Open / close positions with full risk management
  - Enforce single-trade concurrency (MAX_OPEN_POSITIONS)
  - Apply SL, target, trailing stop (regime-aware through ExitEngine)
  - Track daily PnL and enforce daily loss limit
  - Store all trades to database
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import core.config as cfg
import core.constants as C
from core.logger import log, trade_log
from data.data_storage import save_trade, update_trade, save_signal
from options.option_selector import OptionSelection
from strategy.conditions import SignalResult
from strategy.exit_engine import ExitEngine, ExitContext
from strategy.phase_manager import PhaseManager
from strategy.regime import MarketRegime


# ── Active Position ────────────────────────────────────────────────────────────

@dataclass
class Position:
    """Represents a single open trade position."""
    trade_id:         int
    mode:             str
    direction:        str           # "BUY" or "SELL"
    option_type:      str           # "CE" or "PE"
    instrument_key:   str
    expiry:           str
    strike:           float
    lots:             int
    qty:              int           # lots × LOT_SIZE
    entry_price:      float         # Spot price at entry
    option_ltp_entry: float         # Option premium at entry
    stop_loss:        float         # Spot price-derived SL (for info)
    target:           float         # Spot price-derived target (for info)
    trail_sl:         float         # Option price-derived SL (execution)
    initial_risk:     float         # |entry_price - stop_loss|
    candles_held:     int = 0
    ai_record_id:     Optional[int] = None
    upstox_order_id:  Optional[str] = None
    partial_exit_done: bool = False  # Track if 50% was closed at +1R
    lots_remaining:   int = 0        # Current open lots
    entry_ts:         datetime = field(default_factory=datetime.now)
    # ── Trailing SL Tracking ──────────────────────────────────────────
    high_water_mark:  float = 0.0        # Highest/Lowest option price seen
    last_trail_pnl_r: float = 0.0        # PnL_R when SL was last updated
    last_trail_update_candle: int = 0    # Candle count when SL last updated
    # ── Phase 4 Execution Realism ─────────────────────────────────────
    max_reached_r:    float = 0.0        # High-Water Mark in R units
    pending_action:   Optional[str] = None # "ENTRY" or "EXIT"
    is_active:        bool = True        # False if pending delay
    bid_at_entry:     float = 0.0
    ask_at_entry:     float = 0.0
    slippage_log:     float = 0.0
    last_quote:       dict = field(default_factory=dict)

    def __post_init__(self):
        if self.lots_remaining == 0:
            self.lots_remaining = self.lots


# ── Order Manager ──────────────────────────────────────────────────────────────

class OrderManager:
    """
    Manages the full lifecycle of trades in live, paper, or simulate mode.

    Usage:
        om = OrderManager(mode="papertrade")
        regime = detect_regime(df_3min)

        om.try_open(signal, option_selection, instrument_key, regime)
        om.update(current_ltp, current_spot, regime, indicators...)
        om.try_close(current_ltp, reason="EOD")
    """

    def __init__(self, mode: str, capital: float = 100000.0) -> None:
        assert mode in ("live", "papertrade", "simulate"), f"Invalid mode: {mode}"
        self.mode               = mode
        self.capital            = float(capital)
        self.hwm                = self.capital # High Water Mark
        self.positions: list[Position] = []
        self.daily_pnl: float   = 0.0
        self.trade_count: int   = 0

        # Cooldown: blocks new entries for N candles after any exit
        self._cooldown: int = 0

        # Per-trade decision engines (keyed by trade_id)
        self._exit_engines:   dict[int, ExitEngine]   = {}
        self._phase_managers: dict[int, PhaseManager] = {}

    # ── Position Gating ────────────────────────────────────────────────────────

    @property
    def has_open_position(self) -> bool:
        return len(self.positions) > 0

    def get_total_margin_used(self) -> float:
        """Sum of entry premiums for all open positions."""
        return sum(p.option_ltp_entry * p.qty for p in self.positions)

    def get_directional_risk(self, direction: str) -> float:
        """Sum of initial risk for all positions in a given direction."""
        return sum(p.initial_risk * p.qty for p in self.positions if p.direction == direction)

    def calculate_institutional_lots(
        self, 
        premium: float, 
        atr: float, 
        sl_points: float,
        ml_prob: float = 0.0
    ) -> int:
        """
        Calculate lots based on Capital-at-Risk (CaR) and ML quality.
        Formula: Lots = (Capital * Risk_Pct * ML_Mult) / (SL_Points * Lot_Size)
        """
        if sl_points <= 0: return 1 # Fallback
        
        # 1. Base Risk (default 1.25% of capital)
        risk_pct = cfg.RISK_PER_TRADE_PCT
        
        # 2. ML Quality Multiplier — NO GAPS in probability ranges
        ml_mult = 1.0
        if ml_prob >= 0.78:           # High Quality: Premium bin
            ml_mult = 1.5
            log.debug("💎 Premium quality signal. Sizing 1.5x.")
        elif ml_prob >= 0.60:         # Standard High Potential bin (was 0.70 - fixed gap)
            ml_mult = 1.25
            log.debug("⚡ Standard signal. Sizing 1.25x.")
        elif ml_prob >= 0.50:         # Marginal bin
            ml_mult = 0.85
            log.debug("⚠️  Marginal signal. Sizing 0.85x.")
        # else: ml_mult = 1.0 (weak but still above entry gate)
        
        # 3. Dynamic De-leveraging Multiplier
        drawdown_mult = self._get_risk_multiplier()
        
        target_risk_amount = self.capital * risk_pct * ml_mult * drawdown_mult
        
        # 4. Points-based risk per lot
        risk_per_lot = sl_points * cfg.LOT_SIZE
        
        if risk_per_lot <= 0: return 1
        
        lots = int(target_risk_amount // risk_per_lot)
        
        # Constraint: At least 1 lot, but respect Margin Cap
        max_allowed_by_margin = int((self.capital * cfg.MAX_MARGIN_PCT) // (premium * cfg.LOT_SIZE))
        
        final_lots = max(1, min(lots, max_allowed_by_margin))
        return final_lots

    def _get_risk_multiplier(self) -> float:
        """Dynamic de-leveraging based on drawdown from peak equity."""
        drawdown_pct = (self.hwm - self.capital) / self.hwm if self.hwm > 0 else 0.0
        if drawdown_pct >= cfg.DRAWDOWN_THRESHOLD_2: return cfg.RISK_SCALE_2
        if drawdown_pct >= cfg.DRAWDOWN_THRESHOLD_1: return cfg.RISK_SCALE_1
        return 1.0

    def reset_daily_pnl(self) -> None:
        """Reset intraday triggers but persist balance/HWM."""
        self.daily_pnl = 0.0
        self.trade_count = 0
        log.info("Intraday stats reset. Portfolio: Capital={c:.0f}, HWM={h:.0f}", 
                 c=self.capital, h=self.hwm)

    def is_within_daily_loss_limit(self) -> bool:
        return self.daily_pnl > -cfg.DAILY_LOSS_LIMIT

    def can_enter_trade(self, direction: str) -> bool:
        """Institutional gating for new entries."""
        if self._cooldown > 0:
            return False
            
        if len(self.positions) >= cfg.MAX_OPEN_POSITIONS:
            log.warning("Entry blocked — Max concurrent positions ({n}) reached.", n=cfg.MAX_OPEN_POSITIONS)
            return False

        # Directional Risk Cap (MDE)
        current_dir_risk = self.get_directional_risk(direction)
        if current_dir_risk >= self.capital * cfg.MAX_DIRECTIONAL_RISK:
            log.warning("Entry blocked — Max directional risk (3%) reached for {d}.", d=direction)
            return False

        # Margin Cap (70%)
        if self.get_total_margin_used() >= self.capital * cfg.MAX_MARGIN_PCT:
            log.warning("Entry blocked — Max margin utilization (70%) reached.")
            return False

        # Pyramiding Rule: Only add if existing positions in same direction are at +1R
        same_dir_pos = [p for p in self.positions if p.direction == direction]
        if same_dir_pos:
            # Check if all existing same-direction trades have reached +1R
            # (In our simple PhaseManager, CONFIRMATION starts at 0.5R, EXPANSION at 1.5R)
            # Implementation plan says +1R. Let's check the phase or PnL_R directly.
            for p in same_dir_pos:
                pm = self._phase_managers.get(p.trade_id)
                if pm and pm.pnl_r < 1.0:
                    log.warning("Entry blocked — Pyramiding requires existing {d} trades at +1R.", d=direction)
                    return False

        return True

    def tick_cooldown(self) -> None:
        """Decrement cooldown counter each candle (call from strategy loop)."""
        if self._cooldown > 0:
            self._cooldown -= 1

    # ── Open Trade ─────────────────────────────────────────────────────────────

    def try_open(
        self,
        signal: SignalResult,
        selection: OptionSelection,
        instrument_key: str,
        regime: Optional[MarketRegime] = None,
        score_breakdown: Optional[dict] = None,
        entry_ts: Optional[datetime] = None,
    ) -> bool:
        """
        Attempt to open a position with Volatility-Adjusted Notional Sizing (CaR).
        """
        if not self.can_enter_trade(signal.direction):
            return False

        if not self.is_within_daily_loss_limit():
            log.warning(
                "Trade rejected — daily loss limit reached (₹{lim:,.0f}). Daily PnL: ₹{pnl:,.0f}",
                lim=cfg.DAILY_LOSS_LIMIT, pnl=self.daily_pnl,
            )
            return False

        # ── Volatility-Adjusted Sizing (CaR) ──────────────────────────────────
        risk_per_trade_val = self.capital * cfg.RISK_PER_TRADE_PCT * self._get_risk_multiplier()
        initial_risk_points = abs(signal.spot_price - signal.stop_loss) or 1.0
        
        # ML Quality Scaling (Phase 3 Optimized)
        ml_mult = 1.0
        if signal.ml_quality_prob >= 0.75:
            ml_mult = 1.5
            log.debug("💎 Premium quality signal detected. Sizing scaled 1.5x.")
        elif signal.ml_quality_prob >= 0.70:
            ml_mult = 0.75
            log.debug("⚡ Standard High Potential signal detected. Sizing scaled 0.75x.")
        
        target_lots = int((risk_per_trade_val * ml_mult) / (initial_risk_points * cfg.LOT_SIZE))
        target_lots = max(1, target_lots) # Minimum 1 lot
        
        # Override selection lots with our calculated CaR lots
        qty = target_lots * cfg.LOT_SIZE
        order_id = None

        # ── Place actual order in live mode ───────────────────────────────────
        if self.mode == "live":
            from execution import upstox_client as uc
            order_id = uc.place_market_order(
                instrument_key=instrument_key,
                transaction_type="BUY",
                quantity=qty,
                tag="ALGO",
            )
            if order_id is None:
                log.error("Live order failed — position not opened.")
                return False

        # ── Execution Delay & Bid-Ask Fill (Paper / Live) ─────────────────────
        if self.mode in ("papertrade", "live"):
            # Set to pending — will be filled on next update() call with new quote
            p = Position(
                trade_id=self.trade_count + 1,
                mode=self.mode,
                direction=signal.direction,
                option_type=signal.option_type,
                instrument_key=instrument_key,
                expiry=selection.expiry,
                strike=selection.strike,
                lots=target_lots,
                qty=qty,
                entry_price=signal.spot_price,
                option_ltp_entry=selection.ltp, # provisional, subject to delay fill
                stop_loss=signal.stop_loss,
                target=signal.target or (signal.spot_price + initial_risk_points * 2),
                trail_sl=selection.ltp - (initial_risk_points * 0.5), # initial SL
                initial_risk=initial_risk_points,
                ai_record_id=signal.ai_record_id,
                upstox_order_id=order_id,
                pending_action="ENTRY",
                is_active=False
            )
            self.positions.append(p)
            self._exit_engines[p.trade_id] = ExitEngine(_default_regime(), PhaseManager())
            self._phase_managers[p.trade_id] = PhaseManager()
            log.info("⏳ Pending ENTRY initiated for {dir} {ot} {s}...", 
                     dir=p.direction, ot=p.option_type, s=p.strike)
            return True
        
        # ── Legacy Simulate Mode path ───────────
        # Apply Slippage at Entry (Execution Realism)
        slippage = self._calculate_slippage(signal.atr, selection.strike, signal.spot_price)
        
        # Spot entry is adjusted unfavorably based on direction
        if signal.direction == "BUY":
            entry_spot = signal.spot_price + slippage
        else:
            entry_spot = signal.spot_price - slippage
            
        # Option premium entry is ALWAYS higher (we are buying)
        entry_premium = selection.ltp + slippage

        # ── Calculate Option Stop Loss (Execution) ──────────────────────────
        # Simple delta-based approximation (0.5 for ATM/near-ATM)
        sl_dist_spot = abs(signal.spot_price - signal.stop_loss)
        option_sl = entry_premium - (sl_dist_spot * 0.5)
        option_sl = max(option_sl, cfg.MIN_SLIPPAGE) # Floor
        
        # ── Save to database ──────────────────────────────────────────────────
        trade_data = {
            "mode":            self.mode,
            "direction":       signal.direction,
            "option_type":     selection.option_type,
            "expiry":          selection.expiry,
            "strike":          selection.strike,
            "lots":            target_lots,
            "entry_price":     entry_premium,
            "stop_loss":       signal.stop_loss,
            "target":          signal.target,
            "entry_ts":        datetime.now(),
            "upstox_order_id": order_id,
        }
        trade_id = save_trade(trade_data)
        initial_risk = initial_risk_points

        new_pos = Position(
            trade_id         = trade_id,
            mode             = self.mode,
            direction        = signal.direction,
            option_type      = selection.option_type,
            instrument_key   = instrument_key,
            expiry           = selection.expiry,
            strike           = selection.strike,
            lots             = target_lots,
            qty              = qty,
            entry_price      = entry_spot,
            option_ltp_entry = entry_premium,
            stop_loss        = signal.stop_loss,
            target           = signal.target,
            trail_sl         = option_sl,
            initial_risk     = initial_risk,
            upstox_order_id  = order_id,
            entry_ts         = entry_ts or datetime.now()
        )
        self.positions.append(new_pos)

        # ── Create per-trade decision engine instances ────────────────────────
        pm = PhaseManager(
            entry_price = signal.spot_price,
            stop_loss   = signal.stop_loss,
        )
        self._phase_managers[trade_id] = pm
        
        default_regime_obj = regime or _default_regime()
        ee = ExitEngine(
            regime        = default_regime_obj,
            phase_manager = pm,
        )
        self._exit_engines[trade_id] = ee

        # ── Stage 1: Save AI feature snapshot ────────────────────────────────
        try:
            from ai.dataset_builder import record_signal_features
            ai_record_id = record_signal_features(
                signal         = signal,
                regime         = default_regime_obj,
                selection      = selection,
                trade_id       = trade_id,
                score_breakdown = score_breakdown,
                feature_ts     = new_pos.entry_ts,
                entry_ts       = new_pos.entry_ts,
            )
            new_pos.ai_record_id = ai_record_id
        except Exception as exc:
            log.warning("AI feature capture failed: {e}", e=exc)

        trade_log.info(
            "OPEN [{mode}] {dir} {ot} Strike={strike} | EntryPremium=₹{ltp:.2f} | "
            "OptionSL=₹{osl:.2f} | SpotSL={sl:.2f} TGT={tgt:.2f} | "
            "Lots={lots} (LOT_SIZE={ls}) | Qty={qty} | Cost=₹{cost:,.2f} | ML={ml:.2f}x | Regime={reg}",
            mode=self.mode,
            dir=signal.direction,
            ot=selection.option_type,
            strike=selection.strike,
            ltp=entry_premium,
            osl=option_sl,
            sl=signal.stop_loss,
            tgt=signal.target,
            lots=target_lots,
            ls=cfg.LOT_SIZE,
            qty=qty,
            cost=selection.total_cost,
            ml=ml_mult,
            reg=default_regime_obj.type if regime else "UNKNOWN",
        )
        return True

    # ── Update: ExitEngine + Price-Based Exits ─────────────────────────────────

    def update(
        self, 
        quotes: dict[str, dict], 
        signal: SignalResult,
        regime: Optional[MarketRegime] = None
    ) -> list[str]:
        """
        Update all open positions with latest prices and manage exits.
        Using SignalResult (from 3-min candle) to populate ExitContext scores.
        """
        reasons = []
        current_spot = signal.spot_price
        atr_value    = signal.atr
        
        for p in list(self.positions):
            # ── 1. Execution Delay Logic (Paper/Live) ─────────────────────────
            if not p.is_active:
                _handle_pending_delay(p, quotes)
                if not p.is_active: continue # Still pending

            # ── 2. Get realistic LTP / Bid / Ask ──────────────────────────────
            quote = quotes.get(p.instrument_key, {})
            p.last_quote = quote
            
            if self.mode in ("papertrade", "live") and quote:
                ltp = quote.get("ltp")
                bid = quote.get("bid")
                ask = quote.get("ask")
            else:
                # Use synthetic pricing as fallback (or for simulate mode)
                spot_move = current_spot - p.entry_price
                delta_sign = 1.0 if p.direction == "BUY" else -1.0
                ltp = max(p.option_ltp_entry + delta_sign * spot_move * 0.5, 1.0)
                bid = ask = ltp

            if ltp is None or bid is None or ask is None: 
                continue

            # Ensure they are floats for the next call
            try:
                ltp_f, bid_f, ask_f = float(ltp), float(bid), float(ask)
            except (ValueError, TypeError):
                continue

            # ── 3. Life-cycle Maturation & Max R Tracking ────────────────────
            risk_pts = p.initial_risk or 1.0
            pnl_r = (ltp_f - p.option_ltp_entry) / (risk_pts * 0.5) 
            if pnl_r > p.max_reached_r:
                p.max_reached_r = pnl_r

            reason = self._update_position(
                p, ltp_f, bid_f, ask_f, current_spot, atr_value, regime, signal
            )
            if reason:
                reasons.append(reason)

        return reasons

    def _update_position(
        self,
        p: Position,
        current_ltp: float,
        bid: float,
        ask: float,
        current_spot: float,
        atr_value: float,
        regime_obj: Optional[MarketRegime],
        signal: SignalResult,
    ) -> Optional[str]:
        """Update logic for a single position."""
        p.candles_held += 1
        pm = self._phase_managers.get(p.trade_id)
        ee = self._exit_engines.get(p.trade_id)

        if pm and current_spot > 0:
            pm.tick(current_spot)

        pnl_r = pm.pnl_r if pm else 0.0
        held_score     = signal.bullish_score if p.direction == "BUY" else signal.bearish_score
        opposite_score = signal.bearish_score if p.direction == "BUY" else signal.bullish_score

        # ── ExitEngine Layer ──────────────────────────────────────────────────
        if ee:
            ctx = ExitContext(
                held_direction   = p.direction,
                held_score       = held_score,
                opposite_score   = opposite_score,
                current_price    = current_spot,
                entry_price      = p.entry_price,
                initial_risk     = p.initial_risk,
                vwap             = signal.vwap,
                breakout_high    = signal.breakout_high,
                breakout_low     = signal.breakout_low,
                vol_trend        = getattr(signal, "volume_trend", "flat"),
                candles_held     = p.candles_held,
                pnl_r            = pnl_r,
                current_ltp      = current_ltp,
                bid              = bid,
                ask              = ask,
                atr              = atr_value,
                regime           = regime_obj or _default_regime(),
            )
            decision = ee.evaluate(ctx)

            if decision.action == "TIGHTEN_SL":
                _tighten_trail_sl(p, current_ltp, current_spot, atr_value, factor=0.5)
            elif decision.should_exit:
                return self._close(p, bid if self.mode in ("papertrade", "live") else current_ltp, f"EXIT_ENGINE:{decision.layer}")

        # ── Adverse Price-Time Stop (Efficiency) ─────────────────────────────
        if p.candles_held >= cfg.ADVERSE_TIME_CANDLES and pnl_r < cfg.ADVERSE_PNL_R:
            return self._close(p, bid if self.mode in ("papertrade", "live") else current_ltp, "ADVERSE_TIME_STOP")

        # ── Max Hold Time ────────────────────────────────────────────────────
        if p.candles_held >= cfg.MAX_HOLD_CANDLES:
            return self._close(p, bid if self.mode in ("papertrade", "live") else current_ltp, "MAX_HOLD_TIME")

        # ── Trailing Stop Update ─────────────────────────────────────────────
        if atr_value > 0:
            sl_mult = pm.get_sl_multiplier() if pm else 1.5
            _update_trail_sl(p, current_ltp, atr_value, sl_mult, pnl_r)

        # ── Step 1: Partial Exit at +1.0R (MUST run BEFORE any SL tightening) ──
        if not p.partial_exit_done and pnl_r >= 1.0 and p.lots_remaining > 1:
            lots_to_close = p.lots_remaining // 2
            self._close(p, bid if self.mode in ("papertrade", "live") else current_ltp, "PARTIAL_EXIT_1R", quantity_lots=lots_to_close)
            p.partial_exit_done = True
            p.trail_sl = max(p.trail_sl, p.option_ltp_entry)
            log.info("🔒 SL locked to breakeven: ₹{sl:.2f}", sl=p.trail_sl)

        # ── Step 2: Hard SL Hit (option-level) ──────────────────────────
        # Exit at BID with slippage penalty for realism
        if current_ltp <= p.trail_sl:
            return self._close(p, bid if self.mode in ("papertrade", "live") else current_ltp, "SL_HIT", apply_worst_case=True)

        # ── Step 3: Target Hit ────────────────────────────────────
        if current_ltp >= p.target:
            return self._close(p, bid if self.mode in ("papertrade", "live") else current_ltp, "TARGET_HIT")

        return None

    def _calculate_slippage(self, atr: float, strike: float, spot: float) -> float:
        """
        Advanced slippage model: ATR * k * liquidity_factor.
        liquidity_factor: ATM=1.0, Near OTM=1.2, Far OTM=1.5
        Enforces cfg.MIN_SLIPPAGE floor.
        """
        if atr <= 0:
            return cfg.MIN_SLIPPAGE
            
        # Liquidity factor based on distance from ATM
        diff = abs(spot - strike)
        if diff < 25:      factor = 1.0
        elif diff < 100:   factor = 1.2
        else:              factor = 1.5
        
        slippage = atr * cfg.SLIPPAGE_K * factor
        return max(slippage, cfg.MIN_SLIPPAGE)

    # ── Close Position ─────────────────────────────────────────────────────────

    def close_eod(self, current_spot: float, current_ltp: Optional[float] = None) -> list[str]:
        """Force-close all positions at end of day."""
        reasons = []
        for p in list(self.positions):
            ltp = current_ltp
            if ltp is None and self.mode == "simulate":
                spot_move = current_spot - p.entry_price
                delta_sign = 1.0 if p.direction == "BUY" else -1.0
                ltp = max(p.option_ltp_entry + delta_sign * spot_move * 0.5, 1.0)
            
            if ltp is not None:
                reasons.append(self._close(p, ltp, "EOD"))
        return reasons

    def _close(self, p: Position, exit_price_raw: float, reason: str, quantity_lots: Optional[int] = None, apply_worst_case: bool = False) -> str:
        """Internal close: calculate PnL, update DB, clear position."""
        is_partial = quantity_lots is not None and quantity_lots < p.lots_remaining
        close_lots = quantity_lots if is_partial else p.lots_remaining
        close_qty  = close_lots * cfg.LOT_SIZE

        # Liquidity Penalty for far OTM or low vol
        slippage = self._calculate_slippage(p.initial_risk / 1.5, p.strike, p.entry_price)
        if apply_worst_case:
            slippage *= 1.5 # Extra penalty for SL slippage
            
        exit_price = exit_price_raw - slippage
        p.slippage_log += slippage # Aggregate total slippage for this trade
        
        pnl = (exit_price - p.option_ltp_entry) * close_qty

        self.capital    += pnl
        if self.capital > self.hwm:
            self.hwm = self.capital

        self.daily_pnl  += pnl
        
        # Only increment trade count on full exit
        if not is_partial:
            self.trade_count += 1
            self._cooldown   = cfg.COOLDOWN_CANDLES

        # Database updates (Only for full close for simplicity in AIDataset, or update pnl)
        if not is_partial and p.ai_record_id is not None:
            try:
                from ai.dataset_builder import label_trade_outcome
                label_trade_outcome(
                    ai_record_id     = p.ai_record_id,
                    pnl              = pnl, # This should ideally be cumulative pnl
                    entry_price      = p.option_ltp_entry,
                    exit_price       = exit_price,
                    qty              = p.lots * cfg.LOT_SIZE, # Total original qty
                    initial_risk     = p.initial_risk,
                    duration_candles = p.candles_held,
                )
            except Exception as exc:
                log.warning("AI labelling failed: {e}", e=exc)

        update_trade(p.trade_id, 
                     exit_price=exit_price if not is_partial else None, 
                     exit_ts=datetime.now() if not is_partial else None, 
                     pnl=round(float(pnl), 2), 
                     exit_reason=reason if not is_partial else f"PARTIAL:{reason}",
                     slippage=p.slippage_log,
                     bid_at_entry=p.bid_at_entry,
                     ask_at_entry=p.ask_at_entry,
                     max_reached_r=p.max_reached_r)

        trade_log.info(
            "{type} {ot} {strike} | PnL=₹{pnl:,.2f} | Reason={reason} | Lots={l}/{tot} | Slippage=₹{sl:.2f} | MaxR={maxr:.2f}R",
            type="PARTIAL" if is_partial else "CLOSE",
            ot=p.option_type, strike=p.strike, pnl=pnl, reason=reason, 
            l=close_lots, tot=p.lots, sl=p.slippage_log, maxr=p.max_reached_r
        )

        # Remove or update active lists
        if is_partial:
            p.lots_remaining -= close_lots
            p.qty = p.lots_remaining * cfg.LOT_SIZE
        else:
            if p in self.positions:
                self.positions.remove(p)
            self._exit_engines.pop(p.trade_id, None)
            self._phase_managers.pop(p.trade_id, None)
            
        return reason

    def status_dict(self) -> dict:
        """Return summary of all open positions."""
        return {
            "mode":        self.mode,
            "open_count":  len(self.positions),
            "daily_pnl":   round(float(self.daily_pnl), 2),
            "margin_used": self.get_total_margin_used(),
            "positions": [
                {
                    "trade_id": p.trade_id,
                    "strike":   p.strike,
                    "dir":      p.direction,
                    "entry":    p.option_ltp_entry,
                    "pnl_r":    self._phase_managers[p.trade_id].pnl_r if p.trade_id in self._phase_managers else 0,
                } for p in self.positions
            ]
        }


# ── Private Helpers ────────────────────────────────────────────────────────────

def _update_trail_sl(
    p: Position,
    current_ltp: float,
    atr_value: float,
    multiplier: float,
    pnl_r: float = 0.0,
) -> None:
    """
    Standard trailing stop update (only moves in the favourable direction).
    Guards implemented to prevent premature exits:
    1. Min hold period: No trail for first N candles.
    2. Maturation: No SL tightening unless unrealized PnL >= 0.5R.
    3. Step increment: Only move SL in steps of 0.3R to reduce frequency.
    """
    if p.candles_held < cfg.MIN_TRAIL_CANDLES:
        return

    if pnl_r < cfg.TRAIL_MATURATION_R:
        return

    # Step increment gate: only update if pnl_r has moved significantly from last update
    if pnl_r < p.last_trail_pnl_r + cfg.TRAIL_STEP_R:
        return

    # High-water mark guard: Only update SL if price makes a new high (BUY) or new low (SELL)
    if p.direction == "BUY":
        if current_ltp > p.high_water_mark or p.high_water_mark == 0:
            p.high_water_mark = current_ltp
        else:
            return  # No new high, don't update trail
    else:  # SELL
        if current_ltp < p.high_water_mark or p.high_water_mark == 0:
            p.high_water_mark = current_ltp
        else:
            return  # No new low, don't update trail

    trail_dist = atr_value * multiplier
    updated = False

    if p.direction == "BUY":
        new_trail = current_ltp - trail_dist
        if new_trail > p.trail_sl:
            p.trail_sl = new_trail
            updated = True
    else:
        new_trail = current_ltp + trail_dist
        if new_trail < p.trail_sl:
            p.trail_sl = new_trail
            updated = True

    if updated:
        p.last_trail_pnl_r = pnl_r
        p.last_trail_update_candle = p.candles_held
        log.debug("Trail SL updated at {r:.2f}R → {sl:.2f}", r=pnl_r, sl=p.trail_sl)


def _tighten_trail_sl(
    p: Position,
    current_ltp: float,
    current_spot: float,
    atr_value: float,
    factor: float = 0.5,
) -> None:
    """Profit lock: trail SL with aggressively reduced ATR multiplier."""
    if atr_value > 0:
        _update_trail_sl(p, current_ltp, atr_value, factor)
    log.info("🔒 Profit lock SL tightened to {sl:.2f}", sl=p.trail_sl)


def _default_regime() -> MarketRegime:
    """Fallback regime when detect_regime() hasn't been called yet."""
    from strategy.regime import MarketRegime
    return MarketRegime(
        type             = "TRENDING",
        adx              = 0.0,
        volatility       = "normal",
        use_trailing_sl  = True,
        exit_tolerance   = 0.40,
        signal_threshold = cfg.SIGNAL_STRENGTH_THRESHOLD,
    )


def _handle_pending_delay(p: Position, quotes: dict[str, dict]) -> None:
    """Handle simulated latency for order fills (Paper/Live)."""
    quote = quotes.get(p.instrument_key, {})
    if not quote: return

    if p.pending_action == "ENTRY":
        # Fill at current ASK (we are BUYING)
        fill_price = quote.get("ask")
        if fill_price:
            p.option_ltp_entry = fill_price
            p.bid_at_entry = quote.get("bid")
            p.ask_at_entry = quote.get("ask")
            p.is_active = True
            p.pending_action = None
            log.info("📌 Pending ENTRY filled | Side: {dir} | Price: ₹{p:.2f} (ASK) | Spread: ₹{s:.2f}",
                     dir=p.direction, p=fill_price, s=p.ask_at_entry - p.bid_at_entry)
