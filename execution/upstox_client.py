"""
execution/upstox_client.py
───────────────────────────
Low-level Upstox v2 API wrapper.

Responsibilities:
  - Authentication (trading token)
  - Place / modify / cancel orders
  - Fetch positions, order status, account funds
  - Thin layer — no business logic here; only API calls

Trading token must be set in cfg.UPSTOX_TRADING_TOKEN before calling
any order-placement function. Market data calls use the analysis token.
"""

from __future__ import annotations

from typing import Optional

import requests

import core.config as cfg
import core.constants as C
from core.logger import log


# ── Internal HTTP helpers ───────────────────────────────────────────────────────

def _trading_headers() -> dict:
    return {
        "Authorization": f"Bearer {cfg.UPSTOX_TRADING_TOKEN}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{C.UPSTOX_BASE_URL}{endpoint}"
    resp = requests.get(url, headers=_trading_headers(), params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _post(endpoint: str, payload: dict) -> dict:
    url = f"{C.UPSTOX_BASE_URL}{endpoint}"
    resp = requests.post(url, headers=_trading_headers(), json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _put(endpoint: str, payload: dict) -> dict:
    url = f"{C.UPSTOX_BASE_URL}{endpoint}"
    resp = requests.put(url, headers=_trading_headers(), json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _delete(endpoint: str, params: dict = None) -> dict:
    url = f"{C.UPSTOX_BASE_URL}{endpoint}"
    resp = requests.delete(url, headers=_trading_headers(), params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Order Placement ─────────────────────────────────────────────────────────────

def place_market_order(
    instrument_key: str,
    transaction_type: str,    # "BUY" or "SELL"
    quantity: int,
    tag: str = "ALGO",
) -> Optional[str]:
    """
    Place a market order. Returns order_id on success, None on failure.

    Args:
        instrument_key:   Upstox instrument key (e.g. NSE_FO|NIFTY2526APR24000CE)
        transaction_type: "BUY" or "SELL"
        quantity:         Total quantity (lots × lot_size)
        tag:              Order tag visible in Upstox console
    """
    payload = {
        "quantity":         quantity,
        "product":          "D",       # Intraday (D = Delivery-like; use "I" for MIS)
        "validity":         "DAY",
        "price":            0,
        "tag":              tag,
        "instrument_token": instrument_key,
        "order_type":       "MARKET",
        "transaction_type": transaction_type,
        "disclosed_quantity": 0,
        "trigger_price":    0,
        "is_amo":           False,
    }
    try:
        resp = _post("/order/place", payload)
        order_id = resp.get("data", {}).get("order_id")
        log.info(
            "Order placed: {tx} {qty} × {key} | order_id={oid}",
            tx=transaction_type, qty=quantity, key=instrument_key, oid=order_id,
        )
        return order_id
    except Exception as exc:
        log.error("Order placement failed: {err}", err=exc)
        return None


def place_limit_order(
    instrument_key: str,
    transaction_type: str,
    quantity: int,
    price: float,
    tag: str = "ALGO",
) -> Optional[str]:
    """Place a limit order. Returns order_id on success."""
    payload = {
        "quantity":         quantity,
        "product":          "D",
        "validity":         "DAY",
        "price":            round(price, 2),
        "tag":              tag,
        "instrument_token": instrument_key,
        "order_type":       "LIMIT",
        "transaction_type": transaction_type,
        "disclosed_quantity": 0,
        "trigger_price":    0,
        "is_amo":           False,
    }
    try:
        resp = _post("/order/place", payload)
        order_id = resp.get("data", {}).get("order_id")
        log.info(
            "Limit order placed: {tx} {qty} × {key} @ {price} | order_id={oid}",
            tx=transaction_type, qty=quantity, key=instrument_key,
            price=price, oid=order_id,
        )
        return order_id
    except Exception as exc:
        log.error("Limit order failed: {err}", err=exc)
        return None


def cancel_order(order_id: str) -> bool:
    """Cancel an open order. Returns True on success."""
    try:
        _delete("/order/cancel", params={"order_id": order_id})
        log.info("Order cancelled: {oid}", oid=order_id)
        return True
    except Exception as exc:
        log.error("Order cancel failed [{oid}]: {err}", oid=order_id, err=exc)
        return False


# ── Order & Position Status ─────────────────────────────────────────────────────

def get_order_status(order_id: str) -> Optional[dict]:
    """Fetch the current status of an order by ID."""
    try:
        resp = _get("/order/details", params={"order_id": order_id})
        return resp.get("data", {})
    except Exception as exc:
        log.error("Failed to get order status [{oid}]: {err}", oid=order_id, err=exc)
        return None


def get_positions() -> list[dict]:
    """Return all open intraday positions."""
    try:
        resp = _get("/portfolio/short-term-positions")
        return resp.get("data", [])
    except Exception as exc:
        log.error("Failed to fetch positions: {err}", err=exc)
        return []


def get_funds() -> dict:
    """Return available margin and funds."""
    try:
        resp = _get("/user/get-funds-and-margin")
        return resp.get("data", {}).get("equity", {})
    except Exception as exc:
        log.error("Failed to fetch funds: {err}", err=exc)
        return {}
