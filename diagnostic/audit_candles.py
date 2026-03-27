"""
diagnostic/audit_candles.py
────────────────────────────
Data Integrity Audit: Verifies candle completeness for all trading sessions.

Outputs:
  - Per-day row count, first/last timestamp, gap analysis
  - Overall pass/fail summary
  - Any days with < expected candles are flagged for rebuild

Run:
    cd /home/albin/algo-0111
    python diagnostic/audit_candles.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta, datetime, time as dtime
from collections import defaultdict

from data.data_storage import init_db, get_session, Candle
from core.logger import setup_logging

setup_logging()

# ── Constants ──────────────────────────────────────────────────────────────────
AUDIT_START = date(2026, 1, 1)
AUDIT_END   = date(2026, 3, 27)

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 29)

# Expected 1-min candle count for a full session (9:15 to 15:29 = 375 candles)
EXPECTED_CANDLES = 375
WARN_BELOW       = int(EXPECTED_CANDLES * 0.90)   # <337 = warn
FAIL_BELOW       = int(EXPECTED_CANDLES * 0.50)   # <187 = fail

INSTRUMENTS = ["NIFTY50", "NIFTY26APRFUT", "NIFTY26MARFUT", "NIFTY26FEBFUT", "NIFTY26JANFUT"]
GAP_THRESHOLD_MIN = 5  # Flag gaps > 5 minutes


def is_trading_day(dt: date) -> bool:
    """Skip weekends. (Public holidays not yet listed — warn separately)."""
    return dt.weekday() < 5  # Mon=0 … Fri=4


def get_trading_days(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def audit_day(session, trade_date: date) -> dict:
    """Return audit info for one trading day across all instruments."""
    result = {
        "date": trade_date,
        "instrument": None,
        "rows": 0,
        "first_ts": None,
        "last_ts": None,
        "max_gap_min": 0,
        "gaps": [],
        "status": "NO_DATA",
    }

    # Try instruments in priority order
    for instr in INSTRUMENTS:
        rows = (
            session.query(Candle)
            .filter(
                Candle.instrument == instr,
                Candle.timeframe == 1,
                Candle.ts >= datetime.combine(trade_date, dtime.min),
                Candle.ts <  datetime.combine(trade_date + timedelta(days=1), dtime.min),
            )
            .order_by(Candle.ts)
            .all()
        )
        if rows:
            result["instrument"] = instr
            result["rows"] = len(rows)
            result["first_ts"] = rows[0].ts
            result["last_ts"]  = rows[-1].ts

            # Gap analysis
            gaps = []
            for i in range(1, len(rows)):
                delta = (rows[i].ts - rows[i-1].ts).total_seconds() / 60.0
                if delta > GAP_THRESHOLD_MIN:
                    gaps.append((rows[i-1].ts, rows[i].ts, round(delta, 1)))
            result["gaps"] = gaps
            result["max_gap_min"] = max((g[2] for g in gaps), default=0)
            break

    # Assign status
    if result["rows"] == 0:
        result["status"] = "MISSING"
    elif result["rows"] < FAIL_BELOW:
        result["status"] = "FAIL"
    elif result["rows"] < WARN_BELOW:
        result["status"] = "WARN"
    elif result["max_gap_min"] > GAP_THRESHOLD_MIN:
        result["status"] = "GAP"
    else:
        result["status"] = "OK"

    return result


def main():
    init_db()
    trading_days = get_trading_days(AUDIT_START, AUDIT_END)
    
    print(f"\n{'='*80}")
    print(f"  NIFTY ML-FUSION — DATA INTEGRITY AUDIT")
    print(f"  Period: {AUDIT_START} to {AUDIT_END}")
    print(f"  Trading days requested: {len(trading_days)}")
    print(f"{'='*80}\n")
    
    print(f"{'Date':<12} {'Instr':<18} {'Rows':>5} {'First':>8} {'Last':>8} {'MaxGap':>7} {'Status':>8}")
    print("-" * 80)

    results = []
    with get_session() as session:
        for td in trading_days:
            r = audit_day(session, td)
            results.append(r)

            status_icon = {
                "OK":      "✅",
                "WARN":    "⚠️ ",
                "GAP":     "🔸",
                "FAIL":    "❌",
                "MISSING": "🚫",
            }.get(r["status"], "?")

            first = r["first_ts"].strftime("%H:%M") if r["first_ts"] else "—"
            last  = r["last_ts"].strftime("%H:%M")  if r["last_ts"]  else "—"
            instr = r["instrument"] or "—"
            
            print(
                f"{str(r['date']):<12} {instr:<18} {r['rows']:>5} "
                f"{first:>8} {last:>8} {r['max_gap_min']:>6.0f}m "
                f"  {status_icon} {r['status']}"
            )
            if r["gaps"]:
                for g_start, g_end, g_min in r["gaps"][:3]:  # Show first 3 gaps
                    print(f"   ⚠ Gap: {g_start.strftime('%H:%M')} → {g_end.strftime('%H:%M')} ({g_min:.0f} min)")

    # ── Summary ────────────────────────────────────────────────────────────────
    total    = len(results)
    ok       = sum(1 for r in results if r["status"] == "OK")
    warn     = sum(1 for r in results if r["status"] == "WARN")
    gaps     = sum(1 for r in results if r["status"] == "GAP")
    fail     = sum(1 for r in results if r["status"] == "FAIL")
    missing  = sum(1 for r in results if r["status"] == "MISSING")
    
    simulated = ok + warn + gaps + fail  # Days that had at least some data
    
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Total trading days requested : {total}")
    print(f"  Days with data (will simulate): {simulated}  ({100*simulated/total:.1f}%)")
    print(f"  ✅ OK (≥90% candles, no gaps) : {ok}")
    print(f"  ⚠️  WARN (80–90% candles)     : {warn}")
    print(f"  🔸 GAP (gap > {GAP_THRESHOLD_MIN} min)          : {gaps}")
    print(f"  ❌ FAIL (<50% candles)        : {fail}")
    print(f"  🚫 MISSING (no data at all)  : {missing}")
    
    reliability_pct = 100.0 * ok / total if total > 0 else 0.0
    print(f"\n  Data Reliability Score: {reliability_pct:.1f}%")
    
    if missing > 0:
        print(f"\n  🔴 ACTION REQUIRED: {missing} days with NO data.")
        print(f"  Run the historical fetcher for these dates:")
        for r in results:
            if r["status"] == "MISSING":
                print(f"    → {r['date']}")
    elif fail > 0:
        print(f"\n  🟠 WARNING: {fail} days with critically incomplete data.")
    else:
        print(f"\n  ✅ Data pipeline integrity is ACCEPTABLE for simulation.")

    print(f"\n  Simulated days vs Requested: {simulated}/{total} = {100*simulated/total:.1f}%")
    if simulated < total * 0.90:
        print(f"  🔴 CRITICAL: Less than 90% of days have data. Rebuild required before optimization.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
