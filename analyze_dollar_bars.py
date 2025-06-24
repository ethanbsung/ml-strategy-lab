"""
Dollar Bar Analysis

Demonstrates the advantages of dollar bars over time-based bars and
provides insights into the generated dollar bar dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import seaborn as sns

def analyze_dollar_bars():
    """Analyze the generated dollar bar dataset"""
    
    print("📊 DOLLAR BAR ANALYSIS")
    print("=" * 40)
    
    # Load recent data (2025-01 for speed)
    file_path = 'data/BTCUSDT/dollar_bars_5M/BTCUSDT-trades-2025-01_dollar_bars.parquet'
    df = pd.read_parquet(file_path)
    
    print(f"Dataset: {file_path}")
    print(f"Date range: {df['bar_start_time'].min().date()} to {df['bar_end_time'].max().date()}")
    print(f"Total bars: {len(df):,}")
    print()
    
    # Key statistics
    print("🔍 KEY STATISTICS:")
    print(f"Average bar duration: {df['duration_seconds'].mean():.1f} seconds")
    print(f"Duration range: {df['duration_seconds'].min():.1f}s - {df['duration_seconds'].max():.1f}s")
    print(f"Average trades per bar: {df['trade_count'].mean():.0f}")
    print(f"Trade count range: {df['trade_count'].min()} - {df['trade_count'].max()}")
    print(f"Average volume per bar: {df['volume'].mean():.2f} BTC")
    print(f"Dollar volume consistency: ${df['dollar_volume'].mean():,.0f} ± ${df['dollar_volume'].std():,.0f}")
    print()
    
    # Market microstructure insights
    print("🧠 MARKET MICROSTRUCTURE INSIGHTS:")
    
    # Buy/sell balance
    avg_buy_ratio = df['buy_volume_ratio'].mean()
    print(f"Average buy volume ratio: {avg_buy_ratio:.1%}")
    
    # VWAP vs Close relationship
    vwap_close_corr = np.corrcoef(df['vwap'], df['close'])[0,1]
    print(f"VWAP-Close correlation: {vwap_close_corr:.4f}")
    
    # Volatility analysis (high-low ratio)
    avg_volatility = df['hl_ratio'].mean()
    print(f"Average intrabar volatility: {avg_volatility:.2%}")
    
    # Dollar bar efficiency (adapts to market activity)
    print(f"\\n⚡ ADAPTIVE TIME INTERVALS:")
    print(f"Fastest bar (high activity): {df['duration_seconds'].min():.1f} seconds")
    print(f"Slowest bar (low activity): {df['duration_seconds'].max():.1f} seconds")
    print(f"Adaptation ratio: {df['duration_seconds'].max() / df['duration_seconds'].min():.1f}x")
    print()
    
    # Price movement analysis
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = df['price_change'] / df['open'] * 100
    
    print("📈 PRICE MOVEMENT ANALYSIS:")
    print(f"Average price change per bar: ${df['price_change'].mean():.2f}")
    print(f"Price change std: ${df['price_change'].std():.2f}")
    print(f"Directional accuracy: {(df['price_change'] > 0).mean():.1%} bullish bars")
    print()
    
    # Volume-price relationship
    df['volume_intensity'] = df['volume'] / (df['duration_seconds'] / 60)  # BTC per minute
    volume_price_corr = np.corrcoef(df['volume_intensity'], np.abs(df['price_change_pct']))[0,1]
    print(f"Volume intensity vs price change correlation: {volume_price_corr:.4f}")
    print()
    
    # Order flow analysis
    print("🔄 ORDER FLOW ANALYSIS:")
    
    # Aggressive buying vs selling periods
    aggressive_buying = df[df['buy_volume_ratio'] > 0.6]
    aggressive_selling = df[df['buy_volume_ratio'] < 0.4]
    
    print(f"Aggressive buying periods: {len(aggressive_buying)} bars ({len(aggressive_buying)/len(df):.1%})")
    print(f"Aggressive selling periods: {len(aggressive_selling)} bars ({len(aggressive_selling)/len(df):.1%})")
    
    if len(aggressive_buying) > 0:
        avg_buying_return = aggressive_buying['price_change_pct'].mean()
        print(f"Avg return during aggressive buying: {avg_buying_return:.3f}%")
    
    if len(aggressive_selling) > 0:
        avg_selling_return = aggressive_selling['price_change_pct'].mean()
        print(f"Avg return during aggressive selling: {avg_selling_return:.3f}%")
    
    print()
    
    # Trading insights
    print("💡 TRADING INSIGHTS:")
    
    # Large vs small bars
    large_bars = df[df['trade_count'] > df['trade_count'].quantile(0.8)]
    small_bars = df[df['trade_count'] < df['trade_count'].quantile(0.2)]
    
    print(f"High activity bars (>80th percentile): {len(large_bars)} bars")
    print(f"  Avg duration: {large_bars['duration_seconds'].mean():.1f}s")
    print(f"  Avg price change: {large_bars['price_change_pct'].mean():.3f}%")
    
    print(f"Low activity bars (<20th percentile): {len(small_bars)} bars")
    print(f"  Avg duration: {small_bars['duration_seconds'].mean():.1f}s") 
    print(f"  Avg price change: {small_bars['price_change_pct'].mean():.3f}%")
    
    print()
    
    # Temporal patterns
    df['hour'] = df['bar_start_time'].dt.hour
    hourly_activity = df.groupby('hour')['duration_seconds'].mean()
    
    most_active_hour = hourly_activity.idxmin()  # Shortest average duration = most active
    least_active_hour = hourly_activity.idxmax()  # Longest average duration = least active
    
    print(f"Most active hour: {most_active_hour}:00 UTC (avg {hourly_activity[most_active_hour]:.1f}s per bar)")
    print(f"Least active hour: {least_active_hour}:00 UTC (avg {hourly_activity[least_active_hour]:.1f}s per bar)")
    
    print()
    print("🎯 ADVANTAGES OF DOLLAR BARS:")
    print("✅ Adaptive to market activity (faster during high volume)")
    print("✅ Consistent economic significance ($5M per bar)")
    print("✅ Better for ML models (removes time-based noise)")
    print("✅ Captures true market microstructure")
    print("✅ Superior for backtesting (realistic execution)")
    
    return df

def quick_visualization(df):
    """Create simple visualizations of dollar bar properties"""
    
    print("\\n📊 Creating quick visualizations...")
    
    # Simple stats without plotting (since we're in terminal)
    print("\\nDuration Distribution:")
    duration_bins = pd.cut(df['duration_seconds'], bins=5)
    duration_counts = duration_bins.value_counts().sort_index()
    
    for interval, count in duration_counts.items():
        print(f"  {interval}: {count} bars ({count/len(df)*100:.1f}%)")
    
    print("\\nTrade Count Distribution:")
    trade_bins = pd.cut(df['trade_count'], bins=5)
    trade_counts = trade_bins.value_counts().sort_index()
    
    for interval, count in trade_counts.items():
        print(f"  {interval}: {count} bars ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    df = analyze_dollar_bars()
    quick_visualization(df) 