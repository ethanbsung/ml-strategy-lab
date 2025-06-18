"""
Build Features - Python wrapper for materializing SQL feature views

This script:
1. Loads SQL view definitions
2. Executes feature engineering queries 
3. Saves results as Parquet for fast model training
4. Performs validation checks
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

def setup_connection():
    """Connect to DuckDB and load SQL views"""
    
    DB_PATH = "lake/market.db"
    
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
        
    con = duckdb.connect(DB_PATH)
    
    # Load SQL view definitions
    sql_files = [
        "pipelines/feature_daily.sql",
        "pipelines/feature_5m.sql"
    ]
    
    for sql_file in sql_files:
        if Path(sql_file).exists():
            print(f"Loading SQL view: {sql_file}")
            with open(sql_file, 'r') as f:
                con.execute(f.read())
        else:
            print(f"Warning: SQL file not found: {sql_file}")
    
    return con

def materialize_daily_features(con):
    """Execute daily feature view and save as Parquet"""
    
    print("🔄 Materializing daily features...")
    start_time = time.time()
    
    try:
        # Execute feature query
        df = con.execute("SELECT * FROM daily_features ORDER BY symbol, ts_utc").df()
        
        # Ensure lake directory exists
        Path("lake").mkdir(exist_ok=True)
        
        # Save as Parquet for fast loading
        output_path = "lake/daily_features.parquet"
        df.to_parquet(output_path, index=False)
        
        elapsed = time.time() - start_time
        print(f"✅ Daily features saved: {len(df):,} rows in {elapsed:.1f}s")
        print(f"   Output: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error materializing daily features: {e}")
        return None

def materialize_5m_features(con):
    """Execute 5-minute feature view and save as Parquet"""
    
    print("🔄 Materializing 5-minute features...")
    start_time = time.time()
    
    try:
        # Check if 5m data exists
        count = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        if count == 0:
            print("⚠️  No 5-minute data found, skipping...")
            return None
            
        # Execute feature query
        df = con.execute("SELECT * FROM fivemin_features ORDER BY symbol, ts_utc").df()
        
        # Save as Parquet
        output_path = "lake/fivemin_features.parquet"
        df.to_parquet(output_path, index=False)
        
        elapsed = time.time() - start_time
        print(f"✅ 5-min features saved: {len(df):,} rows in {elapsed:.1f}s")
        print(f"   Output: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error materializing 5-min features: {e}")
        return None

def validate_features(daily_df=None, fivemin_df=None):
    """Perform validation checks on generated features"""
    
    print("\n📊 Feature Validation:")
    
    if daily_df is not None:
        print("\n📈 Daily Features:")
        print(f"  Rows: {len(daily_df):,}")
        print(f"  Symbols: {daily_df['symbol'].nunique()}")
        print(f"  Date range: {daily_df['ts_utc'].min().date()} to {daily_df['ts_utc'].max().date()}")
        
        # Check for null values in key features
        key_features = ['ln_ret_1', 'mom_20', 'vol_20', 'gap_pc']
        null_counts = {}
        
        for feature in key_features:
            if feature in daily_df.columns:
                null_count = daily_df[feature].isnull().sum()
                null_pct = 100 * null_count / len(daily_df)
                null_counts[feature] = (null_count, null_pct)
                
        print("  Null values in key features:")
        for feature, (count, pct) in null_counts.items():
            print(f"    {feature}: {count:,} ({pct:.1f}%)")
        
        # Check data completeness
        latest_timestamp = daily_df.groupby('symbol')['ts_utc'].max()
        print(f"  Latest timestamps per symbol (showing first 5):")
        for symbol, ts in latest_timestamp.head().items():
            print(f"    {symbol}: {ts.date()}")
    
    if fivemin_df is not None:
        print("\n📊 5-Minute Features:")
        print(f"  Rows: {len(fivemin_df):,}")
        print(f"  Symbols: {fivemin_df['symbol'].nunique()}")
        print(f"  Datetime range: {fivemin_df['ts_utc'].min()} to {fivemin_df['ts_utc'].max()}")
        
        # Check intraday patterns
        hour_dist = fivemin_df['hour'].value_counts().sort_index()
        print(f"  Hour distribution (top 5):")
        for hour, count in hour_dist.head().items():
            print(f"    {hour:02d}:00: {count:,} bars")

def run_leak_test(con):
    """Test for data leakage by checking if latest feature timestamp < raw data timestamp"""
    
    print("\n🔍 Leak Detection Test:")
    
    try:
        # Check daily features
        daily_check = con.execute("""
            WITH raw_latest AS (
                SELECT symbol, MAX(ts_utc) as latest_raw
                FROM bars_daily 
                GROUP BY symbol
            ),
            feature_latest AS (
                SELECT symbol, MAX(ts_utc) as latest_feature
                FROM daily_features
                GROUP BY symbol
            )
            SELECT 
                r.symbol,
                r.latest_raw,
                f.latest_feature,
                r.latest_raw > f.latest_feature AS is_leak_free
            FROM raw_latest r
            JOIN feature_latest f ON r.symbol = f.symbol
            WHERE r.latest_raw <= f.latest_feature
        """).fetchall()
        
        if len(daily_check) == 0:
            print("✅ Daily features: No leakage detected")
        else:
            print(f"⚠️  Daily features: Potential leakage in {len(daily_check)} symbols")
            for symbol, raw_ts, feat_ts, _ in daily_check[:3]:
                print(f"    {symbol}: raw={raw_ts}, feature={feat_ts}")
        
        # Similar check for 5m if data exists
        fivemin_count = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        if fivemin_count > 0:
            fivemin_check = con.execute("""
                WITH raw_latest AS (
                    SELECT symbol, MAX(ts_utc) as latest_raw
                    FROM bars_5m 
                    GROUP BY symbol
                ),
                feature_latest AS (
                    SELECT symbol, MAX(ts_utc) as latest_feature
                    FROM fivemin_features
                    GROUP BY symbol
                )
                SELECT COUNT(*) as leak_count
                FROM raw_latest r
                JOIN feature_latest f ON r.symbol = f.symbol
                WHERE r.latest_raw <= f.latest_feature
            """).fetchone()[0]
            
            if fivemin_check == 0:
                print("✅ 5-min features: No leakage detected")
            else:
                print(f"⚠️  5-min features: Potential leakage in {fivemin_check} cases")
        
    except Exception as e:
        print(f"❌ Leak test failed: {e}")

def main():
    """Main execution function"""
    
    print("🚀 Building Features from SQL Views...")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        # Setup connection and load views
        con = setup_connection()
        
        # Materialize features
        daily_df = materialize_daily_features(con)
        fivemin_df = materialize_5m_features(con)
        
        # Validation
        validate_features(daily_df, fivemin_df)
        
        # Leak detection
        run_leak_test(con)
        
        con.close()
        
        print("\n✅ Feature building complete!")
        print("\nNext steps:")
        print("  1. Check lake/*.parquet files")
        print("  2. Run validation notebook")
        print("  3. Begin model training")
        
    except Exception as e:
        print(f"❌ Feature building failed: {e}")
        raise

if __name__ == "__main__":
    main() 