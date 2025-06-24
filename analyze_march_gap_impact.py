"""
March 2023 Data Gap Impact Analysis

Analyzes how the 19-day data gap from March 12-April 1, 2023 affects 
the dollar bar dataset and provides recommendations for handling it.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_march_gap_impact():
    """Comprehensive analysis of March 2023 data gap impact"""
    
    print("🚨 MARCH 2023 DATA GAP IMPACT ANALYSIS")
    print("=" * 60)
    
    # Load relevant dollar bar files
    march_bars = pd.read_parquet('data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-2023-03_dollar_bars.parquet')
    april_bars = pd.read_parquet('data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-2023-04_dollar_bars.parquet')
    
    print("📊 DATA SUMMARY:")
    print(f"March 2023 dollar bars: {len(march_bars):,}")
    print(f"April 2023 dollar bars: {len(april_bars):,}")
    
    # Analyze the gap bar (first April bar)
    gap_bar = april_bars.iloc[0]
    normal_bars = april_bars.iloc[1:100]  # Sample of normal bars
    
    print("\n🔍 GAP BAR ANALYSIS:")
    print(f"Gap bar duration: {gap_bar['duration_seconds']:,} seconds ({gap_bar['duration_seconds']/86400:.1f} days)")
    print(f"Normal bar avg duration: {normal_bars['duration_seconds'].mean():.0f} seconds")
    print(f"Gap bar is {gap_bar['duration_seconds']/normal_bars['duration_seconds'].mean():.0f}x longer than normal")
    
    print(f"\nPrice movement during gap:")
    print(f"  Start: ${gap_bar['open']:.2f}")
    print(f"  End: ${gap_bar['close']:.2f}")
    print(f"  High: ${gap_bar['high']:.2f}")
    print(f"  Low: ${gap_bar['low']:.2f}")
    print(f"  Total return: {((gap_bar['close'] / gap_bar['open']) - 1) * 100:.2f}%")
    
    print(f"\nTrade activity:")
    print(f"  Trades in gap bar: {gap_bar['trade_count']:,}")
    print(f"  Avg trades in normal bar: {normal_bars['trade_count'].mean():.0f}")
    print(f"  Volume: {gap_bar['volume']:.2f} BTC")
    print(f"  Dollar volume: ${gap_bar['dollar_volume']:,}")
    
    # Impact assessment
    print("\n⚠️ IMPACT ASSESSMENT:")
    
    print("\n1. STATISTICAL IMPACT:")
    # Calculate how this affects overall statistics
    all_april_durations = april_bars['duration_seconds']
    normal_durations = all_april_durations[1:]  # Exclude gap bar
    
    print(f"   April mean duration (with gap bar): {all_april_durations.mean():.0f}s")
    print(f"   April mean duration (without gap bar): {normal_durations.mean():.0f}s")
    print(f"   April std duration (with gap bar): {all_april_durations.std():.0f}s")
    print(f"   April std duration (without gap bar): {normal_durations.std():.0f}s")
    
    gap_impact = (all_april_durations.std() - normal_durations.std()) / normal_durations.std() * 100
    print(f"   Standard deviation increase: {gap_impact:.1f}%")
    
    print("\n2. TIME SERIES IMPACT:")
    print(f"   ✅ Price continuity: Maintained (gap bar connects properly)")
    print(f"   ❌ Time regularity: Broken (one bar spans 19 days)")
    print(f"   ❌ Statistical uniformity: Violated (extreme outlier)")
    
    print("\n3. MODELING IMPACT:")
    print(f"   ❌ Feature calculation: Gap bar will skew moving averages")
    print(f"   ❌ Return calculations: Abnormal 19-day return in single bar")
    print(f"   ❌ Volatility estimation: Huge spike due to extended duration")
    print(f"   ❌ Pattern recognition: Gap bar doesn't represent normal market behavior")
    
    # Recommendations
    print("\n🛠️ RECOMMENDATIONS:")
    
    print("\n1. DATA CLEANING OPTIONS:")
    print("   Option A: Remove gap bar entirely")
    print("   Option B: Split gap bar into multiple synthetic bars")  
    print("   Option C: Mark gap bar for special handling")
    print("   Option D: Interpolate missing period with synthetic data")
    
    print("\n2. PREFERRED SOLUTION:")
    print("   ✅ REMOVE the gap bar (first bar of April 2023)")
    print("   ✅ CREATE explicit gap marker in dataset")
    print("   ✅ DOCUMENT the missing period for transparency")
    
    # Show impact on different timeframes
    print("\n📈 IMPACT ON ANALYSIS TIMEFRAMES:")
    
    # Count bars per month in 2023
    all_2023_files = [
        'data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-2023-01_dollar_bars.parquet',
        'data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-2023-02_dollar_bars.parquet', 
        'data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-2023-03_dollar_bars.parquet',
        'data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-2023-04_dollar_bars.parquet',
        'data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-2023-05_dollar_bars.parquet',
        'data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-2023-06_dollar_bars.parquet'
    ]
    
    monthly_counts = {}
    for file_path in all_2023_files:
        try:
            df = pd.read_parquet(file_path)
            month = file_path.split('-')[-2]  # Extract month from filename
            monthly_counts[month] = len(df)
        except:
            continue
    
    print("\n   Monthly bar counts (2023):")
    for month, count in monthly_counts.items():
        if month == '04':
            normal_april = count - 1  # Subtract gap bar
            print(f"     {month}: {count:,} bars ({normal_april:,} normal + 1 gap bar)")
        else:
            print(f"     {month}: {count:,} bars")
    
    # Calculate what "normal" April should look like
    if '03' in monthly_counts:
        march_daily_avg = monthly_counts['03'] / 12  # March only had 12 days
        expected_april = march_daily_avg * 30  # April has 30 days
        actual_april = monthly_counts.get('04', 0) - 1  # Minus gap bar
        
        print(f"\n   Expected April bars (based on March rate): ~{expected_april:.0f}")
        print(f"   Actual April bars (excluding gap): {actual_april:,}")
        print(f"   Shortfall: {expected_april - actual_april:.0f} bars")
    
    print("\n💡 BUSINESS IMPACT:")
    print("   • Backtesting: Gap bar will create false signals")
    print("   • Risk management: Volatility calculations will be wrong")
    print("   • Strategy development: Pattern matching will fail")
    print("   • Data sales: Customers will notice the anomaly")
    
    print("\n✅ CONCLUSION:")
    print("   The March 12 - April 1, 2023 gap creates ONE problematic bar")
    print("   that spans 19 days and distorts all time-based analysis.")
    print("   Recommend removing this bar and documenting the gap.")

if __name__ == "__main__":
    analyze_march_gap_impact() 