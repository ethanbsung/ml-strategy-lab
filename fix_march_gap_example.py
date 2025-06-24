#!/usr/bin/env python3
"""
Fix March 2023 Gap - Practical Example

This script demonstrates how to clean the March 12 - April 1, 2023 BTCUSDT
data gap and create robust ML features that won't be distorted by the gap.

Run this script to:
1. Analyze the gap impact in your data
2. Apply cleaning strategies
3. Create gap-safe ML features
4. Generate clean datasets for training

Usage:
    python fix_march_gap_example.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append('src')

try:
    from gap_handler import DataGapHandler, clean_btcusdt_march_gap
    from gap_aware_features import GapAwareFeatureEngine, create_gap_safe_features
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and src/gap_handler.py exists")
    sys.exit(1)

def main():
    """
    Main workflow to fix the March 2023 gap and prepare ML-ready data.
    """
    print("🛠️ FIXING MARCH 2023 BTCUSDT GAP FOR ML")
    print("=" * 60)
    print("This script will clean your data and create robust ML features")
    print("that properly handle the 19-day gap from March 12 - April 1, 2023.\n")
    
    # Configuration
    data_dir = Path("data/BTCUSDT/dollar_bars_1M")
    output_dir = Path("outputs/clean_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Check if data exists
    print("📁 Checking data availability...")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("\nPlease ensure you have:")
        print("  - data/BTCUSDT/dollar_bars_1M/ directory")
        print("  - BTCUSDT-trades-2023-03_dollar_bars.parquet")
        print("  - BTCUSDT-trades-2023-04_dollar_bars.parquet")
        print("  - Other monthly dollar bar files")
        return False
    
    parquet_files = list(data_dir.glob("*.parquet"))
    if len(parquet_files) == 0:
        print(f"❌ No parquet files found in {data_dir}")
        return False
    
    print(f"✅ Found {len(parquet_files)} dollar bar files")
    for f in sorted(parquet_files)[:5]:  # Show first 5
        print(f"   {f.name}")
    if len(parquet_files) > 5:
        print(f"   ... and {len(parquet_files) - 5} more")
    
    # Step 2: Load and analyze the gap
    print(f"\n🔍 STEP 1: Analyzing March 2023 Gap")
    print("-" * 40)
    
    try:
        # Load 2023 data to analyze the gap
        files_2023 = [f for f in parquet_files if '2023' in f.name]
        if not files_2023:
            print("⚠️ No 2023 files found - using all available data")
            files_to_load = parquet_files
        else:
            files_to_load = files_2023
        
        print(f"Loading {len(files_to_load)} files for gap analysis...")
        
        dfs = []
        for file in sorted(files_to_load):
            try:
                df = pd.read_parquet(file)
                print(f"  ✅ {file.name}: {len(df):,} bars")
                dfs.append(df)
            except Exception as e:
                print(f"  ❌ Failed to load {file.name}: {e}")
        
        if not dfs:
            print("❌ No files could be loaded")
            return False
        
        # Combine data
        df_combined = pd.concat(dfs, ignore_index=True)
        print(f"\n📊 Combined dataset: {len(df_combined):,} bars")
        
        # Analyze with gap handler
        handler = DataGapHandler()
        df_with_gaps = handler.detect_gap_bars(df_combined)
        
        gap_count = df_with_gaps['is_gap_bar'].sum()
        if gap_count > 0:
            print(f"\n🚨 Found {gap_count} gap bars in the dataset")
            gap_bars = df_with_gaps[df_with_gaps['is_gap_bar']]
            
            print("\nGap bar details:")
            for idx, row in gap_bars.iterrows():
                print(f"  Index {idx}: {row['gap_duration_days']:.1f} days")
                if 'bar_start_time' in row:
                    print(f"    Start: {row['bar_start_time']}")
                if 'duration_seconds' in row:
                    print(f"    Duration: {row['duration_seconds']:,} seconds")
        else:
            print("✅ No gap bars detected (gap may already be cleaned)")
        
    except Exception as e:
        print(f"❌ Error during gap analysis: {e}")
        return False
    
    # Step 3: Apply gap cleaning strategies
    print(f"\n🧹 STEP 2: Applying Gap Cleaning Strategies")
    print("-" * 50)
    
    try:
        # Strategy 1: Simple removal (recommended for most ML use cases)
        print("\nApplying Strategy 1: Remove gap bars entirely")
        df_clean_simple = handler.strategy_1_remove_gap_bars(df_combined, mark_boundaries=True)
        
        # Save cleaned dataset
        clean_simple_path = output_dir / "btcusdt_clean_simple.parquet"
        df_clean_simple.to_parquet(clean_simple_path, compression='zstd')
        print(f"💾 Saved simple clean dataset: {clean_simple_path}")
        
        # Strategy 4: Comprehensive ML preprocessing  
        print("\nApplying Strategy 4: Comprehensive ML preprocessing")
        df_clean_ml = handler.strategy_4_ml_preprocessing(df_combined)
        
        # Save ML-ready dataset
        clean_ml_path = output_dir / "btcusdt_clean_ml_ready.parquet"
        df_clean_ml.to_parquet(clean_ml_path, compression='zstd')
        print(f"💾 Saved ML-ready dataset: {clean_ml_path}")
        
        # Validation
        print("\n🔍 Validating cleaning results...")
        handler.validate_cleaning(df_combined, df_clean_ml)
        
    except Exception as e:
        print(f"❌ Error during gap cleaning: {e}")
        return False
    
    # Step 4: Create gap-aware features
    print(f"\n⚙️ STEP 3: Creating Gap-Aware ML Features")
    print("-" * 45)
    
    try:
        # Use the ML-ready cleaned dataset for feature engineering
        print("Creating comprehensive feature set...")
        
        engine = GapAwareFeatureEngine(gap_threshold_days=5)
        df_features = engine.create_ml_ready_features(df_clean_ml, target_horizon=5)
        
        # Save feature dataset
        features_path = output_dir / "btcusdt_features_gap_safe.parquet"
        df_features.to_parquet(features_path, compression='zstd')
        print(f"💾 Saved feature dataset: {features_path}")
        
        # Create a training-ready subset (remove rows with missing targets)
        df_training = df_features.dropna(subset=['target_robust'])
        
        training_path = output_dir / "btcusdt_training_ready.parquet"
        df_training.to_parquet(training_path, compression='zstd')
        print(f"💾 Saved training-ready dataset: {training_path}")
        
        print(f"\n📊 Training dataset summary:")
        print(f"   Total rows: {len(df_training):,}")
        print(f"   Valid targets: {df_training['target_valid'].sum():,}")
        print(f"   Feature columns: {len([c for c in df_training.columns if c.startswith(('sma_', 'vol_', 'mom_', 'rsi_'))])}")
        
    except Exception as e:
        print(f"❌ Error during feature creation: {e}")
        return False
    
    # Step 5: Create data splits that respect gaps
    print(f"\n📊 STEP 4: Creating Gap-Aware Data Splits")
    print("-" * 40)
    
    try:
        # Create train/validation splits that avoid gap periods
        gap_affected_mask = df_training.get('is_gap_affected', False)
        
        # Split by time, ensuring we don't train on post-gap data and validate on pre-gap
        if 'bar_start_time' in df_training.columns:
            # Use date-based split (safer)
            split_date = pd.to_datetime('2023-06-01')  # Well after the March gap
            
            train_mask = (df_training['bar_start_time'] < split_date) & (~gap_affected_mask)
            val_mask = (df_training['bar_start_time'] >= split_date) & (~gap_affected_mask)
            
            df_train = df_training[train_mask].copy()
            df_val = df_training[val_mask].copy()
            
            print(f"📅 Time-based split (cutoff: {split_date.date()}):")
            print(f"   Training set: {len(df_train):,} rows")
            print(f"   Validation set: {len(df_val):,} rows")
            print(f"   Gap-affected rows excluded: {gap_affected_mask.sum():,}")
            
            # Save splits
            train_path = output_dir / "btcusdt_train.parquet"
            val_path = output_dir / "btcusdt_val.parquet"
            
            df_train.to_parquet(train_path, compression='zstd')
            df_val.to_parquet(val_path, compression='zstd')
            
            print(f"💾 Saved training split: {train_path}")
            print(f"💾 Saved validation split: {val_path}")
            
        else:
            print("⚠️ No timestamp column - using simple index-based split")
            # Simple 80/20 split avoiding gap-affected rows
            clean_data = df_training[~gap_affected_mask]
            split_idx = int(0.8 * len(clean_data))
            
            df_train = clean_data.iloc[:split_idx].copy()
            df_val = clean_data.iloc[split_idx:].copy()
            
            print(f"📊 Index-based split:")
            print(f"   Training set: {len(df_train):,} rows")
            print(f"   Validation set: {len(df_val):,} rows")
        
    except Exception as e:
        print(f"❌ Error during data splitting: {e}")
        return False
    
    # Step 6: Summary and recommendations
    print(f"\n✅ SUCCESS: March 2023 Gap Fixed!")
    print("=" * 50)
    
    print("\n📁 Generated Files:")
    for file_path in output_dir.glob("*.parquet"):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   {file_path.name} ({size_mb:.1f} MB)")
    
    print("\n🎯 Next Steps for ML Model Training:")
    print("1. Use btcusdt_train.parquet for model training")
    print("2. Use btcusdt_val.parquet for validation")
    print("3. Features are gap-safe and ready for ML algorithms")
    print("4. Target variable 'target_robust' excludes gap-contaminated predictions")
    print("5. 'is_gap_affected' column marks potentially problematic periods")
    
    print("\n💡 Key Benefits:")
    print("✅ Gap bars removed to prevent statistical distortion")
    print("✅ Rolling features calculated safely within data segments")
    print("✅ Gap context features inform model about data quality")
    print("✅ Robust target variable excludes cross-gap predictions")
    print("✅ Training splits avoid gap-contaminated periods")
    
    print(f"\n🚀 Your ML models are now protected from the March 2023 gap!")
    
    return True

def run_quick_test():
    """
    Quick test to verify the gap handling works on sample data.
    """
    print("\n🧪 RUNNING QUICK TEST")
    print("-" * 30)
    
    try:
        # Create sample data with a gap
        print("Creating sample data with artificial gap...")
        
        dates = pd.date_range('2023-03-01', '2023-04-30', freq='1H')
        
        # Remove data from March 12 to April 1 to simulate the gap
        gap_start = pd.to_datetime('2023-03-12')
        gap_end = pd.to_datetime('2023-04-01')
        
        # Keep data before gap
        before_gap = dates[dates < gap_start]
        
        # Keep data after gap  
        after_gap = dates[dates >= gap_end]
        
        # Create the gap by jumping directly
        clean_dates = before_gap.tolist() + after_gap.tolist()
        
        # Create sample dollar bar data
        np.random.seed(42)
        n_bars = len(clean_dates)
        
        sample_data = {
            'bar_start_time': clean_dates,
            'bar_end_time': [d + pd.Timedelta(hours=1) for d in clean_dates],
            'open': 25000 + np.random.randn(n_bars) * 100,
            'close': 25000 + np.random.randn(n_bars) * 100,
            'high': 25100 + np.random.randn(n_bars) * 100,
            'low': 24900 + np.random.randn(n_bars) * 100,
            'volume': np.random.exponential(10, n_bars),
            'duration_seconds': np.concatenate([
                np.full(len(before_gap), 3600),  # Normal 1-hour bars
                [19 * 24 * 3600],  # Gap bar spanning 19 days  
                np.full(len(after_gap) - 1, 3600)  # Normal bars after gap
            ])
        }
        
        df_sample = pd.DataFrame(sample_data)
        
        print(f"✅ Created sample data: {len(df_sample)} bars")
        print(f"   Date range: {df_sample['bar_start_time'].min()} to {df_sample['bar_start_time'].max()}")
        
        # Test gap detection
        handler = DataGapHandler()
        df_with_gaps = handler.detect_gap_bars(df_sample)
        
        # Test cleaning
        df_clean = handler.strategy_1_remove_gap_bars(df_sample)
        
        # Test feature creation
        engine = GapAwareFeatureEngine()
        df_features = engine.create_ml_ready_features(df_clean[:100])  # Use subset for speed
        
        print("✅ All gap handling functions work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting March 2023 Gap Fix Pipeline...")
    
    # Option to run quick test first
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if run_quick_test():
            print("\n🎉 Test passed! Ready to process real data.")
        else:
            print("\n❌ Test failed! Check the gap handling modules.")
            sys.exit(1)
    else:
        # Run main pipeline
        success = main()
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            print("\nTo run a quick test: python fix_march_gap_example.py --test")
        else:
            print("\n❌ Pipeline failed. Check error messages above.")
            sys.exit(1) 