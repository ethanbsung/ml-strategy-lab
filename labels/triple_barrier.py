"""
Triple Barrier Labeling System for Walk-Forward Backtesting
Implementation based on López-de-Prado methodology
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define a no-op decorator when numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
import warnings
warnings.filterwarnings('ignore')

import config


@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def find_barrier_hit(prices: np.ndarray, start_idx: int, profit_thresh: float, 
                     stop_thresh: float, timeout: int) -> Tuple[int, int]:
    """
    Fast numba implementation to find which barrier is hit first
    Returns: (hit_type, hit_index)
    hit_type: 1=profit, -1=stop, 0=timeout
    """
    entry_price = prices[start_idx]
    max_idx = min(start_idx + timeout, len(prices) - 1)
    
    for i in range(start_idx + 1, max_idx + 1):
        current_price = prices[i]
        ret = (current_price / entry_price) - 1
        
        # Check profit target
        if ret >= profit_thresh:
            return 1, i
        
        # Check stop loss
        if ret <= -stop_thresh:
            return -1, i
    
    # Timeout
    return 0, max_idx


def create_triple_barrier_labels(df: pd.DataFrame, 
                                profit_target: float = None,
                                stop_target: float = None,
                                timeout_bars: int = None) -> pd.DataFrame:
    """
    Create triple barrier labels for the dataset (Optimized for large datasets)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with close prices and timestamp index
    profit_target : float
        Profit target as decimal (e.g., 0.002 for 0.2%)
    stop_target : float
        Stop loss target as decimal (e.g., 0.0015 for 0.15%)
    timeout_bars : int
        Maximum number of bars to hold position
    
    Returns:
    --------
    pd.DataFrame with labels and barrier information
    """
    if profit_target is None:
        profit_target = config.PROFIT_TGT
    if stop_target is None:
        stop_target = config.STOP_TGT
    if timeout_bars is None:
        timeout_bars = config.TIMEOUT
    
    df = df.copy()
    prices = df['close'].values
    n_bars = len(prices)
    
    # Initialize arrays for results
    labels = np.full(n_bars, np.nan)
    hit_types = np.full(n_bars, np.nan)
    hit_indices = np.full(n_bars, np.nan)
    returns = np.full(n_bars, np.nan)
    hold_periods = np.full(n_bars, np.nan)
    
    print(f"Creating triple barrier labels for {n_bars:,} bars...")
    print(f"Profit target: {profit_target:.4f}, Stop target: {stop_target:.4f}, Timeout: {timeout_bars}")
    
    # Process each bar (except last few where we can't complete the barrier)
    valid_bars = n_bars - timeout_bars
    print_interval = max(10000, valid_bars // 20)  # Print at most 20 times
    
    for i in range(valid_bars):
        if i % print_interval == 0:
            print(f"Processing bar {i:,}/{valid_bars:,} ({i/valid_bars*100:.1f}%)")
        
        hit_type, hit_idx = find_barrier_hit(prices, i, profit_target, stop_target, timeout_bars)
        
        labels[i] = hit_type
        hit_types[i] = hit_type
        hit_indices[i] = hit_idx
        
        # Calculate actual return
        entry_price = prices[i]
        exit_price = prices[hit_idx]
        returns[i] = (exit_price / entry_price) - 1
        hold_periods[i] = hit_idx - i
    
    # Create results dataframe
    result_df = pd.DataFrame({
        'label': labels,
        'hit_type': hit_types,
        'hit_index': hit_indices,
        'forward_return': returns,
        'hold_period': hold_periods
    }, index=df.index)
    
    # Remove NaN rows
    result_df = result_df.dropna()
    
    # Print label distribution
    label_counts = result_df['label'].value_counts().sort_index()
    print(f"\nLabel distribution:")
    print(f"Stop (-1): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/len(result_df)*100:.1f}%)")
    print(f"Timeout (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(result_df)*100:.1f}%)")
    print(f"Profit (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(result_df)*100:.1f}%)")
    
    print(f"Average hold period: {result_df['hold_period'].mean():.1f} bars")
    print(f"Average return: {result_df['forward_return'].mean():.4f}")
    
    return result_df


def align_labels_with_features(X: pd.DataFrame, labels_df: pd.DataFrame, 
                             lookback: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align labels with features considering the lookback window
    Labels should be aligned to predict the future return of the current bar
    """
    if lookback is None:
        lookback = config.LOOKBACK
    
    # Ensure both dataframes are sorted by index
    X = X.sort_index()
    labels_df = labels_df.sort_index()
    
    # Find common time period
    start_date = max(X.index.min(), labels_df.index.min())
    end_date = min(X.index.max(), labels_df.index.max())
    
    # Filter to common period
    X_aligned = X.loc[start_date:end_date].copy()
    y_aligned = labels_df.loc[start_date:end_date].copy()
    
    # Ensure indices match exactly
    common_index = X_aligned.index.intersection(y_aligned.index)
    X_aligned = X_aligned.loc[common_index]
    y_aligned = y_aligned.loc[common_index]
    
    print(f"Aligned features and labels: {len(X_aligned):,} samples")
    print(f"Date range: {X_aligned.index.min()} to {X_aligned.index.max()}")
    
    return X_aligned, y_aligned


def create_sequences_for_lstm(X: pd.DataFrame, y: pd.DataFrame, 
                            lookback: int = None) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Create sequences for LSTM model training
    Each sample contains lookback bars of features to predict the next bar's label
    """
    if lookback is None:
        lookback = config.LOOKBACK
    
    X_vals = X.values
    y_vals = y['label'].values
    
    X_sequences = []
    y_sequences = []
    indices = []
    
    for i in range(lookback, len(X_vals)):
        X_sequences.append(X_vals[i-lookback:i])
        y_sequences.append(y_vals[i])
        indices.append(X.index[i])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    indices = pd.Index(indices)
    
    print(f"Created {len(X_sequences):,} sequences with lookback={lookback}")
    print(f"X_sequences shape: {X_sequences.shape}")
    print(f"y_sequences shape: {y_sequences.shape}")
    
    return X_sequences, y_sequences, indices


if __name__ == "__main__":
    # Test triple barrier labeling
    from features.engineer import engineer_features
    
    print("Loading features...")
    X, _ = engineer_features()
    
    print("\nCreating triple barrier labels...")
    # Create a simple DataFrame with close prices for labeling
    price_df = pd.DataFrame({'close': np.exp(X['log_ret'].cumsum())}, index=X.index)
    price_df['close'] = price_df['close'] * 50000  # Scale to realistic BTC prices
    
    labels_df = create_triple_barrier_labels(price_df)
    
    print("\nAligning features with labels...")
    X_aligned, y_aligned = align_labels_with_features(X, labels_df)
    
    print("\nCreating sequences for LSTM...")
    X_seq, y_seq, seq_indices = create_sequences_for_lstm(X_aligned, y_aligned)
    
    print("\nTriple barrier labeling completed successfully!")
    
    # Save results
    output_path = config.OUTPUT_PATH / "labels.parquet"
    y_aligned.to_parquet(output_path)
    print(f"Labels saved to {output_path}") 