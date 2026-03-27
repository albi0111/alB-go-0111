"""
diagnostic/regime_analysis.py
───────────────────────────────
ADX-Regime Performance Segmentation.

Segments all labelled trades into ADX buckets:
  - ADX < 20  (Ranging/Choppy)
  - ADX 20–25 (Transitional)
  - ADX > 25  (Trending)

Reports per-bucket: count, win rate, avg win R, avg loss R, expectancy.
Identifies regimes where a hard gate (skip entry) may improve expectancy.

Run:
    cd /home/albin/algo-0111
    python diagnostic/regime_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.data_storage import init_db, get_training_dataframe
from core.logger import setup_logging

setup_logging()


def analyze_bucket(name: str, subset):
    if subset.empty:
        print(f"\n  [{name}]  No trades in this bucket.")
        return

    r    = subset["r_multiple"].dropna()
    pnl  = subset["pnl"].dropna() if "pnl" in subset.columns else None
    
    wins   = r[r > 0]
    losses = r[r < 0]
    
    win_rate   = 100.0 * len(wins) / len(r) if len(r) > 0 else 0
    avg_win    = wins.mean()   if len(wins)   > 0 else 0
    avg_loss   = losses.mean() if len(losses) > 0 else 0
    rr         = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")
    expectancy = avg_win * (win_rate/100) + avg_loss * (1 - win_rate/100)
    
    icon = "✅" if expectancy > 0 else "❌"
    
    print(f"\n  ┌─ {name.upper()}")
    print(f"  │  Trades     : {len(r)}")
    print(f"  │  Win Rate   : {win_rate:.1f}%")
    print(f"  │  Avg Win R  : +{avg_win:.3f}R")
    print(f"  │  Avg Loss R : {avg_loss:.3f}R")
    print(f"  │  R/R Ratio  : {rr:.2f}")
    
    if pnl is not None and len(pnl) > 0:
        pnl_exp = pnl.mean()
        print(f"  │  PnL / trade: ₹{pnl_exp:,.2f}  {icon}")
    else:
        print(f"  │  Expectancy : {expectancy:.3f}R  {icon}")
    
    if expectancy < 0:
        print(f"  │  ⚠️  NEGATIVE expectancy — consider gating this regime")
    print(f"  └{'─'*50}")


def main():
    init_db()
    df = get_training_dataframe(min_duration_candles=1, min_pnl=0.0)
    
    if df.empty:
        print("No labelled trade records found.")
        return

    if "adx_val" not in df.columns or "r_multiple" not in df.columns:
        print("Missing adx_val or r_multiple columns. Ensure dataset is populated.")
        return

    df = df.dropna(subset=["adx_val", "r_multiple"])
    
    print(f"\n{'='*60}")
    print(f"  ADX-REGIME PERFORMANCE DIAGNOSTIC")
    print(f"  Total trades analysed: {len(df)}")
    print(f"{'='*60}")
    
    # Also show missing-opportunity analysis
    print(f"\n  NOTE: This analysis uses trades that WERE taken.")
    print(f"  For EXISTING_POSITION rejections, see missed_opportunity report.")
    
    buckets = [
        ("ADX < 20  (Choppy/Ranging)",   df[df["adx_val"] < 20]),
        ("ADX 20–25 (Transitional)",      df[(df["adx_val"] >= 20) & (df["adx_val"] < 25)]),
        ("ADX ≥ 25  (Trending)",          df[df["adx_val"] >= 25]),
    ]
    
    for name, subset in buckets:
        analyze_bucket(name, subset)
    
    # ── Regime recommendation ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION")
    print(f"{'='*60}")
    
    choppy_r = df[df["adx_val"] < 20]["r_multiple"].dropna()
    if len(choppy_r) > 10:
        choppy_exp = choppy_r[choppy_r > 0].mean() * (choppy_r > 0).mean() + \
                     choppy_r[choppy_r < 0].mean() * (choppy_r < 0).mean()
        if choppy_exp < 0:
            print(f"  🔴 ADX < 20 regime has NEGATIVE expectancy ({choppy_exp:.3f}R).")
            print(f"     Consider a hard gate: skip ALL entries when ADX < 20.")
        else:
            print(f"  ✅ ADX < 20 regime is marginally positive ({choppy_exp:.3f}R). Monitor.")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
