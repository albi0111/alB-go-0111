"""
diagnostic/r_distribution.py
──────────────────────────────
R-Multiple Distribution & Exit Efficiency Analysis.

Reads the ai_dataset table and computes:
  - Histogram of R-multiples (bins: <0, 0–0.5, 0.5–1.0, 1.0–1.5, 1.5–2.0, >2.0)
  - Avg Win R, Avg Loss R, Reward/Risk Ratio
  - Trades reaching >1R, >2R (efficiency metric)
  - Trades that might have been cut prematurely (0.01–0.3R = likely premature exit)

Run:
    cd /home/albin/algo-0111
    python diagnostic/r_distribution.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.data_storage import init_db, get_training_dataframe
from core.logger import setup_logging

setup_logging()


def main():
    init_db()
    df = get_training_dataframe(min_duration_candles=1, min_pnl=0.0)
    
    if df.empty:
        print("No labelled trade records found in ai_dataset.")
        return

    if "r_multiple" not in df.columns:
        print("r_multiple column missing from dataset. Ensure trades have been labelled.")
        return

    r = df["r_multiple"].dropna()
    pnl = df["pnl"].dropna() if "pnl" in df.columns else None
    
    print(f"\n{'='*65}")
    print(f"  R-MULTIPLE DISTRIBUTION & EXIT EFFICIENCY ANALYSIS")
    print(f"  Total labelled trades: {len(r)}")
    print(f"{'='*65}\n")
    
    # ── Histogram ─────────────────────────────────────────────────────────
    bins = [
        ("< -1.0R  (blow-up loss)",     r < -1.0),
        ("-1.0R – 0.0R (normal loss)", (r >= -1.0) & (r < 0)),
        ("0.0R – 0.5R  (break-even)",  (r >= 0.0)  & (r < 0.5)),
        ("0.5R – 1.0R  (small win)",   (r >= 0.5)  & (r < 1.0)),
        ("1.0R – 1.5R  (good win)",    (r >= 1.0)  & (r < 1.5)),
        ("1.5R – 2.0R  (great win)",   (r >= 1.5)  & (r < 2.0)),
        (">  2.0R      (runner)",       r >= 2.0),
    ]
    
    print(f"  {'Bucket':<35} {'Count':>6}  {'% of Total':>10}  Bar")
    print(f"  {'-'*72}")
    total_n = len(r)
    for label, mask in bins:
        n = mask.sum()
        pct = 100.0 * n / total_n if total_n > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<35} {n:>6}  {pct:>9.1f}%  {bar}")
    
    # ── Key Metrics ────────────────────────────────────────────────────────
    wins   = r[r > 0]
    losses = r[r < 0]
    
    avg_win_r  = wins.mean()   if len(wins)   > 0 else 0
    avg_loss_r = losses.mean() if len(losses) > 0 else 0
    rr_ratio   = avg_win_r / abs(avg_loss_r) if avg_loss_r != 0 else float("inf")
    
    pct_reach_1r  = 100.0 * (r >= 1.0).sum() / total_n
    pct_reach_2r  = 100.0 * (r >= 2.0).sum() / total_n
    pct_premature = 100.0 * ((r >= 0.01) & (r < 0.35)).sum() / total_n  # Likely premature exits
    
    print(f"\n  {'─'*65}")
    print(f"  METRICS")
    print(f"  {'─'*65}")
    print(f"  Win Rate          : {100.0 * len(wins)/total_n:.1f}%  ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Avg Win R         : +{avg_win_r:.3f}R")
    print(f"  Avg Loss R        : {avg_loss_r:.3f}R")
    print(f"  Reward/Risk Ratio : {rr_ratio:.2f}  {'✅' if rr_ratio >= 1.2 else '❌ BELOW 1.2'}")
    
    print(f"\n  EXIT EFFICIENCY")
    print(f"  {'─'*40}")
    print(f"  Trades reaching >1.0R : {pct_reach_1r:.1f}%  {'✅' if pct_reach_1r >= 20 else '⚠️'}")
    print(f"  Trades reaching >2.0R : {pct_reach_2r:.1f}%  {'✅' if pct_reach_2r >= 8 else '⚠️'}")
    print(f"  Likely premature exits: {pct_premature:.1f}%  (0.01–0.35R wins — possible round-trips)")
    
    if rr_ratio < 1.2:
        print(f"\n  🔴 R/R below 1.2: Winners are being cut too early or losers run too long.")
    if pct_premature > 25:
        print(f"  🔴 >25% trades in 0.01–0.35R bucket: Strong evidence of premature profit-taking.")
    
    # ── PnL Summary ────────────────────────────────────────────────────────
    if pnl is not None:
        print(f"\n  PnL SUMMARY (from DB)")
        print(f"  {'─'*40}")
        print(f"  Total PnL    : ₹{pnl.sum():,.2f}")
        print(f"  Avg per trade: ₹{pnl.mean():,.2f}  {'✅' if pnl.mean() >= 200 else '❌ BELOW ₹200 target'}")
        print(f"  Expectancy   : ₹{pnl.mean():.2f} per trade")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
