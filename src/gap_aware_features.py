"""
Gap-Aware Feature Engineering

Creates robust ML features that properly handle data gaps, specifically 
the March 12 - April 1, 2023 BTCUSDT gap. Ensures rolling calculations
don't span across gaps and maintains feature integrity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

class GapAwareFeatureEngine:
    """
    Feature engineering that respects data gaps and ensures no look-ahead bias.
    
    Key principles:
    1. Never calculate rolling features across data gaps
    2. Mark features calculated near gaps for special handling
    3. Create gap-context features for model awareness
    4. Maintain full traceability of feature calculations
    """
    
    def __init__(self, gap_threshold_days=5):
        self.gap_threshold_days = gap_threshold_days
        self.gap_threshold_seconds = gap_threshold_days * 24 * 3600
        
    def identify_gaps(self, df, time_col='bar_start_time', duration_col='duration_seconds'):
        """
        Identify all data gaps in the time series.
        """
        if duration_col not in df.columns:
            # Calculate duration from timestamps if not provided
            if time_col in df.columns:
                df = df.copy()
                df['calc_duration'] = df[time_col].diff().dt.total_seconds()
                duration_col = 'calc_duration'
            else:
                raise ValueError("Need either duration_seconds or timestamp column")
        
        # Find gaps
        gap_mask = df[duration_col] > self.gap_threshold_seconds
        gap_indices = df.index[gap_mask].tolist()
        
        if gap_indices:
            print(f"🔍 Found {len(gap_indices)} data gaps:")
            for idx in gap_indices:
                gap_days = df.loc[idx, duration_col] / (24 * 3600)
                gap_time = df.loc[idx, time_col] if time_col in df.columns else f"Index {idx}"
                print(f"  Gap at {gap_time}: {gap_days:.1f} days")
        
        return gap_indices
    
    def create_safe_rolling_features(self, df, price_col='close', volume_col='volume'):
        """
        Create rolling features that respect data gaps.
        
        Strategy: Reset rolling calculations after each gap to prevent
        contamination from pre-gap data.
        """
        print("\n📊 Creating gap-safe rolling features...")
        
        df = df.copy()
        
        # Identify gaps
        gap_indices = self.identify_gaps(df)
        
        # Calculate log returns
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Create segment boundaries (after each gap)
        segment_starts = [0] + [idx + 1 for idx in gap_indices if idx + 1 < len(df)]
        segment_ends = gap_indices + [len(df)]
        
        # Initialize feature columns
        feature_cols = [
            'sma_5', 'sma_20', 'sma_50',
            'vol_5', 'vol_20', 'vol_50', 
            'mom_5', 'mom_20',
            'rsi_14', 'bb_upper', 'bb_lower',
            'volume_sma_20', 'volume_ratio'
        ]
        
        for col in feature_cols:
            df[col] = np.nan
        
        # Add gap context columns
        df['segment_id'] = np.nan
        df['bars_since_gap'] = np.nan
        df['is_post_gap'] = False
        
        # Calculate features for each segment
        for segment_id, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            if start >= len(df):
                continue
                
            end = min(end, len(df))
            segment_df = df.iloc[start:end].copy()
            
            if len(segment_df) < 5:  # Skip tiny segments
                continue
            
            print(f"  Processing segment {segment_id}: rows {start}-{end} ({len(segment_df)} bars)")
            
            # Mark segment
            df.loc[start:end-1, 'segment_id'] = segment_id
            df.loc[start:end-1, 'bars_since_gap'] = range(len(segment_df))
            
            # Mark post-gap bars (first 20 bars after gap)
            if segment_id > 0:  # Not the first segment
                post_gap_end = min(start + 20, end)
                df.loc[start:post_gap_end-1, 'is_post_gap'] = True
            
            # Calculate rolling features for this segment
            segment_prices = segment_df[price_col]
            segment_returns = segment_df['log_return'].dropna()
            segment_volume = segment_df[volume_col] if volume_col in segment_df else None
            
            # Price-based features
            df.loc[start:end-1, 'sma_5'] = segment_prices.rolling(5, min_periods=5).mean()
            df.loc[start:end-1, 'sma_20'] = segment_prices.rolling(20, min_periods=20).mean()
            df.loc[start:end-1, 'sma_50'] = segment_prices.rolling(50, min_periods=50).mean()
            
            # Volatility features (using returns)
            vol_5 = segment_returns.rolling(5, min_periods=5).std() * np.sqrt(252)
            vol_20 = segment_returns.rolling(20, min_periods=20).std() * np.sqrt(252)
            vol_50 = segment_returns.rolling(50, min_periods=50).std() * np.sqrt(252)
            
            # Align volatility with main dataframe
            vol_start = start + 1  # Returns start from index 1
            df.loc[vol_start:end-1, 'vol_5'] = vol_5.values[:end-vol_start]
            df.loc[vol_start:end-1, 'vol_20'] = vol_20.values[:end-vol_start]
            df.loc[vol_start:end-1, 'vol_50'] = vol_50.values[:end-vol_start]
            
            # Momentum features
            mom_5 = segment_returns.rolling(5, min_periods=5).sum()
            mom_20 = segment_returns.rolling(20, min_periods=20).sum()
            
            df.loc[vol_start:end-1, 'mom_5'] = mom_5.values[:end-vol_start]
            df.loc[vol_start:end-1, 'mom_20'] = mom_20.values[:end-vol_start]
            
            # RSI calculation
            gains = segment_returns.where(segment_returns > 0, 0)
            losses = -segment_returns.where(segment_returns < 0, 0)
            
            avg_gains = gains.rolling(14, min_periods=14).mean()
            avg_losses = losses.rolling(14, min_periods=14).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            df.loc[vol_start:end-1, 'rsi_14'] = rsi.values[:end-vol_start]
            
            # Bollinger Bands
            sma_20 = segment_prices.rolling(20, min_periods=20).mean()
            std_20 = segment_prices.rolling(20, min_periods=20).std()
            
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)
            
            df.loc[start:end-1, 'bb_upper'] = bb_upper
            df.loc[start:end-1, 'bb_lower'] = bb_lower
            
            # Volume features (if available)
            if segment_volume is not None:
                vol_sma_20 = segment_volume.rolling(20, min_periods=20).mean()
                df.loc[start:end-1, 'volume_sma_20'] = vol_sma_20
                df.loc[start:end-1, 'volume_ratio'] = segment_volume / vol_sma_20
        
        print(f"✅ Created {len(feature_cols)} gap-safe rolling features")
        return df
    
    def create_gap_context_features(self, df):
        """
        Create features that explicitly encode gap context for the model.
        """
        print("\n🎯 Creating gap context features...")
        
        df = df.copy()
        gap_indices = self.identify_gaps(df)
        
        # Initialize gap context features
        df['has_recent_gap'] = False
        df['days_since_last_gap'] = np.inf
        df['days_until_next_gap'] = np.inf
        df['gap_proximity_score'] = 0.0
        df['volatility_regime'] = 'normal'
        
        for gap_idx in gap_indices:
            # Calculate days since gap for subsequent bars
            for idx in range(gap_idx + 1, len(df)):
                if 'bar_start_time' in df.columns:
                    gap_time = df.loc[gap_idx, 'bar_start_time']
                    bar_time = df.loc[idx, 'bar_start_time']
                    days_since = (bar_time - gap_time).days
                else:
                    days_since = idx - gap_idx
                
                # Update days_since_last_gap to minimum (closest gap)
                df.loc[idx, 'days_since_last_gap'] = min(
                    df.loc[idx, 'days_since_last_gap'], days_since
                )
                
                # Mark recent gap (within 30 days)
                if days_since <= 30:
                    df.loc[idx, 'has_recent_gap'] = True
            
            # Calculate days until gap for preceding bars  
            for idx in range(0, gap_idx):
                if 'bar_start_time' in df.columns:
                    gap_time = df.loc[gap_idx, 'bar_start_time']
                    bar_time = df.loc[idx, 'bar_start_time']
                    days_until = (gap_time - bar_time).days
                else:
                    days_until = gap_idx - idx
                
                # Update days_until_next_gap to minimum (closest gap)
                df.loc[idx, 'days_until_next_gap'] = min(
                    df.loc[idx, 'days_until_next_gap'], days_until
                )
        
        # Create proximity score (higher = closer to gap)
        proximity_decay = 30  # days
        
        gap_proximity = np.minimum(
            np.exp(-df['days_since_last_gap'] / proximity_decay),
            np.exp(-df['days_until_next_gap'] / proximity_decay)
        )
        
        df['gap_proximity_score'] = gap_proximity
        
        # Create volatility regime based on gap proximity
        df.loc[df['gap_proximity_score'] > 0.5, 'volatility_regime'] = 'high'
        df.loc[df['gap_proximity_score'] > 0.8, 'volatility_regime'] = 'extreme'
        
        print(f"✅ Gap context features created")
        print(f"   Bars with recent gaps: {df['has_recent_gap'].sum()}")
        print(f"   High volatility regime: {(df['volatility_regime'] == 'high').sum()}")
        print(f"   Extreme volatility regime: {(df['volatility_regime'] == 'extreme').sum()}")
        
        return df
    
    def create_stable_features(self, df):
        """
        Create features that are inherently stable across gaps.
        
        These features don't rely on rolling windows and are safe to use
        even around data gaps.
        """
        print("\n🛡️ Creating gap-stable features...")
        
        df = df.copy()
        
        # Price-based features (no rolling window needed)
        if 'open' in df.columns and 'close' in df.columns:
            df['intrabar_return'] = (df['close'] - df['open']) / df['open']
        
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Time-based features (inherently stable)
        if 'bar_start_time' in df.columns:
            df['hour'] = df['bar_start_time'].dt.hour
            df['day_of_week'] = df['bar_start_time'].dt.dayofweek
            df['month'] = df['bar_start_time'].dt.month
            df['quarter'] = df['bar_start_time'].dt.quarter
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Volume-based stable features
        if 'volume' in df.columns and 'duration_seconds' in df.columns:
            df['volume_per_second'] = df['volume'] / df['duration_seconds']
            
        if 'trade_count' in df.columns and 'volume' in df.columns:
            df['avg_trade_size'] = df['volume'] / df['trade_count']
            
        if 'dollar_volume' in df.columns and 'duration_seconds' in df.columns:
            df['dollar_flow_rate'] = df['dollar_volume'] / df['duration_seconds']
        
        # Microstructure features (stable)
        if 'buy_volume_ratio' in df.columns:
            df['order_flow_imbalance'] = 2 * df['buy_volume_ratio'] - 1  # Range [-1, 1]
            
        if 'vwap' in df.columns and 'close' in df.columns:
            df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        stable_feature_count = sum(1 for col in df.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'bar_start_time', 'bar_end_time'
        ])
        
        print(f"✅ Created {stable_feature_count} gap-stable features")
        return df
    
    def create_robust_target(self, df, target_horizon=5, price_col='close'):
        """
        Create target variable that properly handles gaps.
        
        Ensures forward returns don't span across data gaps.
        """
        print(f"\n🎯 Creating robust {target_horizon}-bar forward target...")
        
        df = df.copy()
        gap_indices = self.identify_gaps(df)
        
        # Calculate standard forward return
        df['target_price'] = df[price_col].shift(-target_horizon)
        df['forward_return'] = (df['target_price'] / df[price_col]) - 1
        
        # Mark targets that span gaps as invalid
        df['target_valid'] = True
        
        for gap_idx in gap_indices:
            # Invalidate targets that would look across this gap
            lookback_start = max(0, gap_idx - target_horizon)
            df.loc[lookback_start:gap_idx, 'target_valid'] = False
            
            print(f"  Invalidated targets around gap at index {gap_idx}")
        
        # Create classification target (for models that prefer it)
        df['target_direction'] = np.where(df['forward_return'] > 0, 1, 0)
        
        # Robust target (only valid targets)
        df['target_robust'] = np.where(df['target_valid'], df['forward_return'], np.nan)
        
        valid_targets = df['target_valid'].sum()
        total_possible = len(df) - target_horizon
        
        print(f"✅ Target created: {valid_targets}/{total_possible} valid targets "
              f"({100*valid_targets/total_possible:.1f}%)")
        
        return df
    
    def create_ml_ready_features(self, df, target_horizon=5):
        """
        Create a complete set of ML-ready features with proper gap handling.
        
        This is the main method that combines all feature creation strategies.
        """
        print("🚀 Creating complete ML-ready feature set...")
        print("="*60)
        
        # Step 1: Create gap-safe rolling features
        df = self.create_safe_rolling_features(df)
        
        # Step 2: Add gap context features  
        df = self.create_gap_context_features(df)
        
        # Step 3: Add stable features
        df = self.create_stable_features(df)
        
        # Step 4: Create robust target
        df = self.create_robust_target(df, target_horizon)
        
        # Step 5: Feature quality assessment
        self._assess_feature_quality(df)
        
        print("✅ ML-ready features complete!")
        return df
    
    def _assess_feature_quality(self, df):
        """
        Assess the quality of created features.
        """
        print(f"\n📊 Feature Quality Assessment:")
        
        # Count features by type
        feature_cols = [col for col in df.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'bar_start_time', 
            'bar_end_time', 'duration_seconds', 'trade_count'
        ]]
        
        # Check for missing values
        missing_counts = df[feature_cols].isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.1]  # >10% missing
        
        print(f"   Total features: {len(feature_cols)}")
        print(f"   High missing value features: {len(high_missing)}")
        
        if len(high_missing) > 0:
            print("   Features with >10% missing values:")
            for feature, count in high_missing.items():
                pct = 100 * count / len(df)
                print(f"     {feature}: {count} ({pct:.1f}%)")
        
        # Check for infinite values
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        high_inf = inf_counts[inf_counts > 0]
        
        if len(high_inf) > 0:
            print(f"   Features with infinite values: {len(high_inf)}")
            for feature, count in high_inf.items():
                print(f"     {feature}: {count}")
        
        # Check target validity
        if 'target_valid' in df.columns:
            valid_pct = 100 * df['target_valid'].mean()
            print(f"   Target validity: {valid_pct:.1f}%")

def create_gap_safe_features(data_path, output_path=None, target_horizon=5):
    """
    Convenience function to create gap-safe features from dollar bar data.
    """
    from pathlib import Path
    
    # Load data
    data_path = Path(data_path)
    if data_path.is_file():
        df = pd.read_parquet(data_path)
    elif data_path.is_dir():
        parquet_files = list(data_path.glob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    # Create features
    engine = GapAwareFeatureEngine()
    df_features = engine.create_ml_ready_features(df, target_horizon)
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(output_path, compression='zstd')
        print(f"💾 Saved features to: {output_path}")
    
    return df_features

if __name__ == "__main__":
    print("🔧 Gap-Aware Feature Engine Ready!")
    print("\nUsage:")
    print("  engine = GapAwareFeatureEngine()")
    print("  df_with_features = engine.create_ml_ready_features(df)")
    print("\nOr use convenience function:")
    print("  df_features = create_gap_safe_features('data/BTCUSDT/dollar_bars_1M/')") 