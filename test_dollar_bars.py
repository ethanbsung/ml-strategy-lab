"""
Test Dollar Bar Generation

Quick test to validate the dollar bar logic before processing the full dataset.
"""

import pandas as pd
from data.create_dollar_bars import DollarBarGenerator

def test_dollar_bars():
    """Test dollar bar generation with a small sample"""
    
    print("🧪 TESTING DOLLAR BAR GENERATION")
    print("=" * 40)
    
    # Test with recent data (smaller file)
    test_file = 'data/BTCUSDT/parquet/BTCUSDT-trades-2025-01.parquet'
    
    print(f"Loading test data from: {test_file}")
    df = pd.read_parquet(test_file)
    
    print(f"Sample data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    # Take first 100K trades for quick testing
    sample_df = df.head(100000).copy()
    print(f"\\nTesting with first {len(sample_df):,} trades...")
    
    # Calculate total dollar volume in sample
    sample_dollar_volume = (sample_df['price'] * sample_df['qty']).sum()
    print(f"Total dollar volume in sample: ${sample_dollar_volume:,.0f}")
    
    # Test with $1M threshold (should create multiple bars from sample)
    threshold = 1_000_000
    expected_bars = int(sample_dollar_volume / threshold)
    print(f"Expected ~{expected_bars} bars with ${threshold:,} threshold")
    
    # Initialize generator
    generator = DollarBarGenerator(
        dollar_threshold=threshold,
        output_dir='data/BTCUSDT/test_dollar_bars'
    )
    
    # Process the sample
    print("\\nProcessing sample...")
    bars = generator.process_trades_chunk(sample_df)
    
    print(f"\\n📊 RESULTS:")
    print(f"Bars created: {len(bars)}")
    
    if len(bars) > 0:
        bars_df = pd.DataFrame(bars)
        
        print("\\n🔍 Sample bars:")
        for i, bar in enumerate(bars_df.head(3).iterrows()):
            bar_data = bar[1]
            print(f"  Bar {i+1}:")
            print(f"    Time: {bar_data['bar_start_time']} -> {bar_data['bar_end_time']}")
            print(f"    OHLC: {bar_data['open']:.2f} / {bar_data['high']:.2f} / {bar_data['low']:.2f} / {bar_data['close']:.2f}")
            print(f"    Volume: {bar_data['volume']:.3f} BTC")
            print(f"    Dollar Volume: ${bar_data['dollar_volume']:,.0f}")
            print(f"    Trades: {bar_data['trade_count']}")
            print(f"    Duration: {bar_data['duration_seconds']:.1f} seconds")
            print(f"    VWAP: ${bar_data['vwap']:.2f}")
            print(f"    Buy Volume %: {bar_data['buy_volume_ratio']*100:.1f}%")
            print()
        
        # Validate dollar thresholds
        print("✅ VALIDATION CHECKS:")
        dollar_volumes = bars_df['dollar_volume']
        min_volume = dollar_volumes.min()
        max_volume = dollar_volumes.max()
        
        print(f"  Dollar volume range: ${min_volume:,.0f} - ${max_volume:,.0f}")
        
        # Most bars should be close to threshold (last bar may be partial)
        within_threshold = ((dollar_volumes >= threshold * 0.95) & (dollar_volumes <= threshold * 1.05)).sum()
        print(f"  Bars within 5% of threshold: {within_threshold}/{len(bars)} ({100*within_threshold/len(bars):.1f}%)")
        
        # All bars should have positive values
        assert (bars_df['volume'] > 0).all(), "Some bars have zero volume"
        assert (bars_df['trade_count'] > 0).all(), "Some bars have zero trades" 
        assert (bars_df['dollar_volume'] > 0).all(), "Some bars have zero dollar volume"
        print(f"  ✅ All bars have positive volume, trades, and dollar volume")
        
        # OHLC consistency
        ohlc_valid = ((bars_df['low'] <= bars_df['open']) & 
                     (bars_df['low'] <= bars_df['close']) &
                     (bars_df['high'] >= bars_df['open']) & 
                     (bars_df['high'] >= bars_df['close'])).all()
        assert ohlc_valid, "OHLC relationships invalid"
        print(f"  ✅ OHLC relationships valid")
        
        print(f"\\n🎉 Dollar bar generation test PASSED!")
        return True
    
    else:
        print("❌ No bars generated - check threshold or data")
        return False

if __name__ == "__main__":
    test_dollar_bars() 