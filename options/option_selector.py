"""
options/option_selector.py
──────────────────────────
Selects the optimal option strike and contract to trade.

Logic:
  1. Determine ATM strike from spot price (round to nearest STRIKE_STEP)
  2. Filter option chain to ATM ± ATM_RANGE_STRIKES strikes
  3. Filter to the relevant option type (CE for BUY, PE for SELL)
  4. Check capital fit: premium × LOT_SIZE × lots ≤ available_capital
  5. Rank by volume (primary) then OI (secondary)
  6. Return the best strike that fits within capital
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

import core.config as cfg
import core.constants as C
from core.logger import log


@dataclass
class OptionSelection:
    """Result of the option selection process."""
    strike:      float
    option_type: str       # "CE" or "PE"
    expiry:      str
    ltp:         float
    bid:         float
    ask:         float
    oi:          float
    volume:      float
    lots:        int
    total_cost:  float     # lots × LOT_SIZE × mid_price


def select_option(
    option_chain_df: pd.DataFrame,
    spot_price: float,
    option_type: str,            # "CE" or "PE"
    available_capital: float,
) -> Optional[OptionSelection]:
    """
    Select the best option strike for trading.

    Args:
        option_chain_df: DataFrame from data_fetcher.fetch_option_chain().
        spot_price:      Current Nifty 50 index price.
        option_type:     "CE" for CALL, "PE" for PUT.
        available_capital: Capital available for this trade (INR).

    Returns:
        OptionSelection if a valid strike is found, else None.
    """
    if option_chain_df.empty:
        log.warning("Option chain is empty — cannot select strike.")
        return None

    # ── Step 1: Identify ATM ────────────────────────────────────────────────
    atm = round(spot_price / C.STRIKE_STEP) * C.STRIKE_STEP

    # ── Step 2: Filter to relevant option type and strike range ────────────
    df = option_chain_df[option_chain_df["option_type"] == option_type].copy()
    df = df[abs(df["strike"] - atm) <= C.ATM_RANGE_STRIKES * C.STRIKE_STEP]

    if df.empty:
        log.warning(
            "No {ot} options found within ±{r} strikes of ATM {atm}",
            ot=option_type, r=C.ATM_RANGE_STRIKES, atm=atm,
        )
        return None

    # ── Step 3: Filter to strikes within capital for at least 1 lot ────────
    lot_size = cfg.LOT_SIZE
    df = df[df["ltp"] > 0]   # Ignore zero-premium dead strikes

    affordable = []
    for _, row in df.iterrows():
        # Use mid-price for realistic cost estimate
        mid_price = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 else row["ltp"]
        cost_1_lot = mid_price * lot_size
        cost_2_lot = mid_price * lot_size * 2
        lots = 0
        cost = 0.0

        if cost_2_lot <= available_capital and cfg.MAX_LOTS >= 2:
            lots = 2
            cost = cost_2_lot
        elif cost_1_lot <= available_capital:
            lots = 1
            cost = cost_1_lot

        if lots > 0:
            affordable.append({**row.to_dict(), "lots": lots, "total_cost": cost})

    if not affordable:
        log.warning(
            "No {ot} options fit within capital ₹{cap:,.0f} near ATM {atm}. "
            "Cheapest LTP: {ltp:.2f}",
            ot=option_type,
            cap=available_capital,
            atm=atm,
            ltp=df["ltp"].min() if not df.empty else 0,
        )
        return None

    # ── Step 4: Rank by volume (primary), OI (secondary) ───────────────────
    ranked = sorted(affordable, key=lambda x: (-x["volume"], -x["oi"]))
    best   = ranked[0]

    selection = OptionSelection(
        strike      = best["strike"],
        option_type = option_type,
        expiry      = best.get("expiry", ""),
        ltp         = best["ltp"],
        bid         = best.get("bid", 0),
        ask         = best.get("ask", 0),
        oi          = best.get("oi", 0),
        volume      = best.get("volume", 0),
        lots        = best["lots"],
        total_cost  = round(best["total_cost"], 2),
    )

    log.info(
        "📌 Option selected: {ot} {strike} | LTP={ltp:.2f} | "
        "Lots={lots} | Cost=₹{cost:,.2f} | Vol={vol:.0f} | OI={oi:.0f}",
        ot=option_type,
        strike=selection.strike,
        ltp=selection.ltp,
        lots=selection.lots,
        cost=selection.total_cost,
        vol=selection.volume,
        oi=selection.oi,
    )

    return selection
