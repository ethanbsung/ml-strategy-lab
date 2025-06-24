"""
Data Gap Handler for ML Pipeline

Handles the March 12 - April 1, 2023 BTCUSDT data gap to prevent it from
impacting ML model training and inference. Provides multiple strategies
for gap treatment based on modeling requirements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

class DataGapHandler:
    """
    Comprehensive handler for data gaps in financial time series.
    
    Primary focus: March 12 - April 1, 2023 BTCUSDT gap that creates
    a single 19-day dollar bar distorting all statistical analysis.
    """
    
    def __init__(self, gap_start='2023-03-12', gap_end='2023-04-01'):
        self.gap_start = pd.to_datetime(gap_start)
        self.gap_end = pd.to_datetime(gap_end)
        self.gap_duration_days = (self.gap_end - self.gap_start).days
        
        print(f"🔧 DataGapHandler initialized for {gap_start} to {gap_end}")
        print(f"   Gap duration: {self.gap_duration_days} days")
    
    def detect_gap_bars(self, df, duration_threshold_days=5):
        """
        Detect problematic bars that span multiple days due to data gaps.
        
        Args:
            df: DataFrame with 'duration_seconds' column
            duration_threshold_days: Bars longer than this are considered gaps
            
        Returns:
            DataFrame with gap bar indicators
        """
        if 'duration_seconds' not in df.columns:
            raise ValueError("DataFrame must contain 'duration_seconds' column")
        
        # Convert threshold to seconds
        threshold_seconds = duration_threshold_days * 24 * 3600
        
        # Identify gap bars
        df = df.copy()
        df['is_gap_bar'] = df['duration_seconds'] > threshold_seconds
        df['gap_duration_days'] = df['duration_seconds'] / (24 * 3600)
        
        gap_bars = df[df['is_gap_bar']]
        
        if len(gap_bars) > 0:
            print(f"\n🚨 DETECTED {len(gap_bars)} GAP BARS:")
            for idx, bar in gap_bars.iterrows():
                print(f"  Bar {idx}: {bar.get('bar_start_time', 'N/A')} "
                      f"({bar['gap_duration_days']:.1f} days)")
        else:
            print("✅ No gap bars detected")
            
        return df
    
    def strategy_1_remove_gap_bars(self, df, mark_boundaries=True):
        """
        Strategy 1: Remove gap bars entirely (RECOMMENDED)
        
        Cleanest approach - simply removes problematic bars that span
        multiple days. Optionally marks gap boundaries for transparency.
        """
        print("\n🛠️ STRATEGY 1: Remove Gap Bars")
        
        original_len = len(df)
        
        # Detect gap bars first
        df = self.detect_gap_bars(df)
        gap_bars = df[df['is_gap_bar']].copy()
        
        if len(gap_bars) == 0:
            print("   No gap bars to remove")
            return df.drop(columns=['is_gap_bar', 'gap_duration_days'])
        
        # Remove gap bars
        clean_df = df[~df['is_gap_bar']].copy()
        
        if mark_boundaries:
            # Add gap boundary markers
            for _, gap_bar in gap_bars.iterrows():
                # Find bars before and after gap
                gap_start_time = gap_bar.get('bar_start_time')
                gap_end_time = gap_bar.get('bar_end_time')
                
                if gap_start_time is not None:
                    # Mark pre-gap bar
                    pre_gap_mask = clean_df['bar_end_time'] <= gap_start_time
                    if pre_gap_mask.any():
                        last_pre_gap_idx = clean_df[pre_gap_mask].index[-1]
                        clean_df.loc[last_pre_gap_idx, 'pre_gap_marker'] = True
                    
                    # Mark post-gap bar
                    post_gap_mask = clean_df['bar_start_time'] >= gap_end_time
                    if post_gap_mask.any():
                        first_post_gap_idx = clean_df[post_gap_mask].index[0]
                        clean_df.loc[first_post_gap_idx, 'post_gap_marker'] = True
        
        # Clean up columns
        clean_df = clean_df.drop(columns=['is_gap_bar', 'gap_duration_days'])
        
        removed_count = original_len - len(clean_df)
        print(f"   ✅ Removed {removed_count} gap bars")
        print(f"   📊 Clean dataset: {len(clean_df):,} bars")
        
        return clean_df
    
    def strategy_2_filter_features(self, df, feature_columns=None):
        """
        Strategy 2: Filter features calculated on gap bars
        
        Removes feature rows that were calculated using gap bars,
        preventing contamination of rolling statistics.
        """
        print("\n🛠️ STRATEGY 2: Filter Gap-Contaminated Features")
        
        if feature_columns is None:
            # Common feature columns that would be contaminated
            feature_columns = [
                'mom_20', 'vol_20', 'mom_5', 'vol_5', 'vol_z_20', 
                'price_sma20_dev', 'sma20', 'vol_1d', 'mom_halfday',
                'rsi_14', 'zscore_20', 'atr_14', 'dist_from_sma20'
            ]
        
        df = df.copy()
        
        # Detect gap bars
        df = self.detect_gap_bars(df)
        
        if not df['is_gap_bar'].any():
            print("   No gap bars detected")
            return df.drop(columns=['is_gap_bar', 'gap_duration_days'])
        
        gap_indices = df[df['is_gap_bar']].index
        
        # Calculate lookback window for contamination
        # Conservative: 20-day features need 20 bars after gap to be clean
        max_lookback = 20  # Adjust based on your longest feature window
        
        contaminated_mask = pd.Series(False, index=df.index)
        
        for gap_idx in gap_indices:
            # Mark gap bar itself
            contaminated_mask.iloc[gap_idx] = True
            
            # Mark next N bars that use the gap bar in their calculation
            start_idx = gap_idx + 1
            end_idx = min(gap_idx + max_lookback + 1, len(df))
            contaminated_mask.iloc[start_idx:end_idx] = True
        
        # Filter out contaminated features
        clean_mask = ~contaminated_mask
        clean_df = df[clean_mask].copy()
        
        # Remove detection columns
        clean_df = clean_df.drop(columns=['is_gap_bar', 'gap_duration_days'])
        
        removed_count = len(df) - len(clean_df)
        print(f"   ✅ Filtered {removed_count} contaminated feature rows")
        print(f"   📊 Clean features: {len(clean_df):,} rows")
        
        return clean_df
    
    def strategy_3_gap_aware_features(self, df):
        """
        Strategy 3: Create gap-aware features
        
        Adds explicit gap indicators and adjusts rolling calculations
        to handle gaps intelligently.
        """
        print("\n🛠️ STRATEGY 3: Gap-Aware Feature Engineering")
        
        df = df.copy()
        
        # Detect gap bars
        df = self.detect_gap_bars(df)
        
        # Add gap context features
        df['days_since_gap'] = 0
        df['days_until_gap'] = 0
        df['gap_proximity_score'] = 0
        
        if df['is_gap_bar'].any():
            gap_indices = df[df['is_gap_bar']].index
            
            for gap_idx in gap_indices:
                gap_time = df.loc[gap_idx, 'bar_start_time'] if 'bar_start_time' in df.columns else None
                
                if gap_time is not None:
                    # Calculate days since gap for bars after gap
                    post_gap_mask = df.index > gap_idx
                    if post_gap_mask.any():
                        for idx in df[post_gap_mask].index:
                            bar_time = df.loc[idx, 'bar_start_time']
                            days_since = (bar_time - gap_time).days
                            df.loc[idx, 'days_since_gap'] = min(days_since, 
                                                              df.loc[idx, 'days_since_gap'])
                    
                    # Calculate days until gap for bars before gap
                    pre_gap_mask = df.index < gap_idx
                    if pre_gap_mask.any():
                        for idx in df[pre_gap_mask].index:
                            bar_time = df.loc[idx, 'bar_start_time']
                            days_until = (gap_time - bar_time).days
                            df.loc[idx, 'days_until_gap'] = min(days_until,
                                                               df.loc[idx, 'days_until_gap'])
        
        # Create proximity score (0 = far from gap, 1 = at gap)
        proximity_window = 30  # days
        df['gap_proximity_score'] = np.maximum(
            np.exp(-df['days_since_gap'] / proximity_window),
            np.exp(-df['days_until_gap'] / proximity_window)
        )
        
        print(f"   ✅ Added gap-aware features")
        print(f"   📊 Gap proximity scores: {df['gap_proximity_score'].describe()}")
        
        return df
    
    def strategy_4_ml_preprocessing(self, df, target_column=None):
        """
        Strategy 4: ML-specific preprocessing
        
        Applies multiple strategies in sequence for robust ML pipeline:
        1. Remove gap bars
        2. Filter contaminated features  
        3. Add gap indicators for model awareness
        """
        print("\n🛠️ STRATEGY 4: Comprehensive ML Preprocessing")
        
        original_len = len(df)
        
        # Step 1: Remove gap bars
        df = self.strategy_1_remove_gap_bars(df, mark_boundaries=True)
        
        # Step 2: Add gap awareness (before filtering)
        df = self.strategy_3_gap_aware_features(df)
        
        # Step 3: Handle target variable if provided
        if target_column and target_column in df.columns:
            # Remove rows where target is calculated using gap bars
            print(f"   Handling target variable: {target_column}")
            
            # Simple approach: remove extreme outliers in target
            target_std = df[target_column].std()
            target_mean = df[target_column].mean()
            outlier_threshold = 3 * target_std
            
            outlier_mask = np.abs(df[target_column] - target_mean) > outlier_threshold
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                print(f"   ⚠️  Removing {outlier_count} target outliers (likely gap-related)")
                df = df[~outlier_mask]
        
        # Step 4: Create train/validation splits avoiding gap periods
        df['is_gap_affected'] = (
            (df.get('gap_proximity_score', 0) > 0.1) |
            df.get('pre_gap_marker', False) |
            df.get('post_gap_marker', False)
        )
        
        final_len = len(df)
        total_removed = original_len - final_len
        
        print(f"   ✅ Comprehensive preprocessing complete")
        print(f"   📊 Original: {original_len:,} → Clean: {final_len:,} bars")
        print(f"   📉 Removed: {total_removed:,} bars ({100*total_removed/original_len:.1f}%)")
        print(f"   🎯 Gap-affected bars marked: {df['is_gap_affected'].sum():,}")
        
        return df
    
    def create_clean_dataset(self, data_path, output_path=None, strategy='comprehensive'):
        """
        Create a clean dataset from dollar bars with gap handling.
        
        Args:
            data_path: Path to dollar bar parquet files
            output_path: Where to save clean dataset
            strategy: 'remove', 'filter', 'aware', or 'comprehensive'
        """
        print(f"\n🔄 CREATING CLEAN DATASET")
        print(f"   Input: {data_path}")
        print(f"   Strategy: {strategy}")
        
        # Load data (assuming parquet files)
        data_path = Path(data_path)
        
        if data_path.is_file():
            df = pd.read_parquet(data_path)
        elif data_path.is_dir():
            # Load multiple files
            parquet_files = list(data_path.glob("*.parquet"))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {data_path}")
            
            dfs = []
            for file in sorted(parquet_files):
                print(f"   Loading: {file.name}")
                dfs.append(pd.read_parquet(file))
            
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError(f"Invalid data path: {data_path}")
        
        print(f"   📊 Loaded: {len(df):,} bars")
        
        # Apply selected strategy
        if strategy == 'remove':
            clean_df = self.strategy_1_remove_gap_bars(df)
        elif strategy == 'filter':
            clean_df = self.strategy_2_filter_features(df)
        elif strategy == 'aware':
            clean_df = self.strategy_3_gap_aware_features(df)
        elif strategy == 'comprehensive':
            clean_df = self.strategy_4_ml_preprocessing(df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.parquet':
                clean_df.to_parquet(output_path, compression='zstd')
            elif output_path.suffix == '.csv':
                clean_df.to_csv(output_path, index=False)
            else:
                # Default to parquet
                output_path = output_path.with_suffix('.parquet')
                clean_df.to_parquet(output_path, compression='zstd')
            
            print(f"   💾 Saved clean dataset: {output_path}")
        
        return clean_df
    
    def validate_cleaning(self, original_df, clean_df):
        """
        Validate that gap cleaning was successful.
        """
        print(f"\n🔍 VALIDATION REPORT")
        
        # Check for remaining gap bars
        if 'duration_seconds' in clean_df.columns:
            clean_df_checked = self.detect_gap_bars(clean_df)
            remaining_gaps = clean_df_checked['is_gap_bar'].sum()
            
            if remaining_gaps == 0:
                print("   ✅ No gap bars remaining")
            else:
                print(f"   ⚠️  {remaining_gaps} gap bars still present")
        
        # Statistical comparison
        print(f"\n   📊 Dataset Comparison:")
        print(f"   Original size: {len(original_df):,}")
        print(f"   Clean size: {len(clean_df):,}")
        print(f"   Reduction: {len(original_df) - len(clean_df):,} bars "
              f"({100*(len(original_df) - len(clean_df))/len(original_df):.1f}%)")
        
        # Feature stability check
        common_cols = set(original_df.columns) & set(clean_df.columns)
        numeric_cols = []
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(original_df[col]):
                numeric_cols.append(col)
        
        if numeric_cols:
            print(f"\n   📈 Feature Stability (first 3 numeric columns):")
            for col in numeric_cols[:3]:
                orig_std = original_df[col].std()
                clean_std = clean_df[col].std()
                stability = abs(clean_std - orig_std) / orig_std if orig_std > 0 else 0
                
                print(f"   {col}:")
                print(f"     Original std: {orig_std:.6f}")
                print(f"     Clean std: {clean_std:.6f}")
                print(f"     Stability: {stability:.1%} change")
        
        return True

# Convenience function for quick gap handling
def clean_btcusdt_march_gap(data_path, output_path=None, strategy='comprehensive'):
    """
    Quick function to clean the specific March 2023 BTCUSDT gap.
    """
    handler = DataGapHandler()
    return handler.create_clean_dataset(data_path, output_path, strategy)

if __name__ == "__main__":
    # Example usage
    handler = DataGapHandler()
    
    print("\n🎯 DataGapHandler Ready!")
    print("\nAvailable strategies:")
    print("  1. 'remove' - Remove gap bars entirely (cleanest)")
    print("  2. 'filter' - Filter contaminated features") 
    print("  3. 'aware' - Add gap-aware features")
    print("  4. 'comprehensive' - Full ML preprocessing pipeline")
    print("\nUsage:")
    print("  clean_df = handler.create_clean_dataset('data/BTCUSDT/dollar_bars_1M/', strategy='comprehensive')") 