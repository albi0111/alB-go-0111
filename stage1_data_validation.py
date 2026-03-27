
import os
import pandas as pd
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, text
import sys

# Add project root to path
sys.path.append("/home/albin/algo-0111")

from core.config import DB_URL
from core.constants import MARKET_OPEN, MARKET_CLOSE

def validate_data():
    engine = create_engine(DB_URL)
    
    print("--- STAGE 1: DATA INTEGRITY VALIDATION ---")
    
    # 1. Check date range and completeness
    query = """
    SELECT date(ts) as day, count(*) as candle_count 
    FROM candles 
    WHERE instrument = 'NIFTY50' AND timeframe = 1
    AND ts >= '2026-01-01' AND ts <= '2026-03-31'
    GROUP BY day
    ORDER BY day
    """
    df_counts = pd.read_sql(query, engine)
    
    if df_counts.empty:
        print("❌ CRITICAL: No NIFTY50 1-min candles found for Jan-March 2026.")
        return False
    
    start_date = df_counts['day'].iloc[0]
    end_date = df_counts['day'].iloc[-1]
    print(f"Data found from {start_date} to {end_date}")
    
    # Identify expected trading days (mon-fri)
    expected_days = []
    curr = datetime.strptime(start_date, '%Y-%m-%d').date()
    target_end = datetime.strptime(end_date, '%Y-%m-%d').date()
    while curr <= target_end:
        if curr.weekday() < 5:
            expected_days.append(str(curr))
        curr += timedelta(days=1)
    
    found_days = set(df_counts['day'].tolist())
    missing_days = [d for d in expected_days if d not in found_days]
    
    print(f"Total trading days expected: {len(expected_days)}")
    print(f"Total days with data: {len(df_counts)}")
    if missing_days:
        print(f"⚠️ Missing days (holidays or missing data): {missing_days}")
    
    # Check for partial sessions
    EXPECTED_CANDLES = 375 # 09:15 to 15:30 is 375 minutes
    partial_days = df_counts[df_counts['candle_count'] < 370] # allowing some small buffer
    if not partial_days.empty:
        print(f"⚠️ Days with partial data (<370 candles):")
        print(partial_days.to_string(index=False))
    
    # 2. Alignment and Gaps
    print("\nChecking for intra-day gaps...")
    gap_query = """
    SELECT ts, instrument 
    FROM candles 
    WHERE instrument = 'NIFTY50' AND timeframe = 1
    AND ts >= '2026-01-01' AND ts <= '2026-03-31'
    ORDER BY ts
    """
    df_all = pd.read_sql(gap_query, engine)
    df_all['ts'] = pd.to_datetime(df_all['ts'])
    
    df_all['diff'] = df_all['ts'].diff()
    # Gaps within the same day
    intra_day_gaps = df_all[(df_all['diff'] > timedelta(minutes=1)) & (df_all['ts'].dt.date == df_all['ts'].shift(1).dt.date)]
    
    if not intra_day_gaps.empty:
        print(f"❌ Intra-day gaps detected:")
        print(intra_day_gaps[['ts', 'diff']].head(20).to_string(index=False))
    else:
        print("✅ No intra-day gaps detected.")

    # 3. NaN Leakage / Nulls
    print("\nChecking for NULL values in candles...")
    null_query = """
    SELECT count(*) as null_count FROM candles 
    WHERE (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL)
    AND instrument = 'NIFTY50'
    """
    null_count = pd.read_sql(null_query, engine).iloc[0]['null_count']
    if null_count > 0:
        print(f"❌ {null_count} candles have NULL values.")
    else:
        print("✅ No NULL values in candle data.")

    # 4. Options Data Check
    print("\nChecking for Option Chain data...")
    opt_query = "SELECT count(*) as count FROM option_chain"
    opt_count = pd.read_sql(opt_query, engine).iloc[0]['count']
    print(f"Total Option Chain snapshots: {opt_count}")

    # Reliability Score
    reliability = (len(df_counts) / len(expected_days)) * 100
    # Penalty for partial days
    reliability -= (len(partial_days) / len(expected_days)) * 50 
    # Penalty for intra-day gaps
    if not intra_day_gaps.empty:
        reliability -= 20
        
    reliability = max(0, min(100, reliability))
    print(f"\nData Reliability Score: {reliability:.1f}%")
    
    if reliability < 85:
        print("Recommendation: FIX data before proceeding.")
    else:
        print("Recommendation: PROCEED to validation.")

    return True

if __name__ == "__main__":
    validate_data()
