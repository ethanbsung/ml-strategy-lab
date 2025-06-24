"""
Feature Engineering Module for Bitcoin Walk-Forward Backtesting
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
# import pandas_ta as ta  # Temporarily disabled due to numpy compatibility issues
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import config


def load_dollar_bars(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all dollar bar parquet files and combine into single DataFrame
    """
    if data_path is None:
        data_path = config.DATA_PATH
    
    # Get all parquet files
    parquet_files = sorted(glob.glob(str(data_path / "*.parquet")))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    
    # Load and combine all files
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Set bar_end_time as index
    combined_df['bar_end_ts'] = pd.to_datetime(combined_df['bar_end_time'])
    combined_df = combined_df.set_index('bar_end_ts').sort_index()
    
    print(f"Loaded {len(combined_df):,} dollar bars from {len(parquet_files)} files")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    return combined_df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns and volatility-adjusted returns"""
    df = df.copy()
    
    # Log returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling volatility
    df['sigma_50'] = df['log_ret'].rolling(config.VOLATILITY_PERIOD).std()
    
    # Volatility-adjusted returns
    df['vol_adj_ret'] = df['log_ret'] / df['sigma_50']
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI manually"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    return k_percent

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bb_position

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate CCI manually"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate EMA manually"""
    return prices.ewm(span=period).mean()

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate COMPREHENSIVE technical analysis indicators"""
    df = df.copy()
    
    print("Calculating comprehensive technical indicators...")
    
    # Multiple RSI periods
    df['rsi_7'] = calculate_rsi(df['close'], config.RSI_PERIOD_SHORT)
    df['rsi_14'] = calculate_rsi(df['close'], config.RSI_PERIOD)
    df['rsi_21'] = calculate_rsi(df['close'], config.RSI_PERIOD_LONG)
    
    # RSI z-scores with shorter lookback for faster adaptation
    for period in [7, 14, 21]:
        col = f'rsi_{period}'
        rsi_mean = df[col].expanding(min_periods=config.ZSCORE_LOOKBACK).mean().shift(1)
        rsi_std = df[col].expanding(min_periods=config.ZSCORE_LOOKBACK).std().shift(1)
        df[f'{col}_zscore'] = (df[col] - rsi_mean) / rsi_std
    
    # Multiple CCI periods
    df['cci_10'] = calculate_cci(df['high'], df['low'], df['close'], config.CCI_PERIOD_SHORT)
    df['cci_20'] = calculate_cci(df['high'], df['low'], df['close'], config.CCI_PERIOD)
    
    # CCI z-scores
    for period in [10, 20]:
        col = f'cci_{period}'
        cci_mean = df[col].expanding(min_periods=config.ZSCORE_LOOKBACK).mean().shift(1)
        cci_std = df[col].expanding(min_periods=config.ZSCORE_LOOKBACK).std().shift(1)
        df[f'{col}_zscore'] = (df[col] - cci_mean) / cci_std
    
    # Stochastic Oscillator
    df['stoch_k'] = calculate_stochastic(df['high'], df['low'], df['close'], config.STOCH_PERIOD)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()  # %D line
    stoch_mean = df['stoch_k'].expanding(min_periods=config.ZSCORE_LOOKBACK).mean().shift(1)
    stoch_std = df['stoch_k'].expanding(min_periods=config.ZSCORE_LOOKBACK).std().shift(1)
    df['stoch_zscore'] = (df['stoch_k'] - stoch_mean) / stoch_std
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(
        df['close'], config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
    )
    
    # MACD z-scores
    for col in ['macd', 'macd_histogram']:
        col_mean = df[col].expanding(min_periods=config.ZSCORE_LOOKBACK).mean().shift(1)
        col_std = df[col].expanding(min_periods=config.ZSCORE_LOOKBACK).std().shift(1)
        df[f'{col}_zscore'] = (df[col] - col_mean) / col_std
    
    # Bollinger Bands
    df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger_bands(
        df['close'], config.BOLLINGER_PERIOD
    )
    bb_pos_mean = df['bb_position'].expanding(min_periods=config.ZSCORE_LOOKBACK).mean().shift(1)
    bb_pos_std = df['bb_position'].expanding(min_periods=config.ZSCORE_LOOKBACK).std().shift(1)
    df['bb_position_zscore'] = (df['bb_position'] - bb_pos_mean) / bb_pos_std
    
    # Average True Range
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], config.ATR_PERIOD)
    df['atr_normalized'] = df['atr'] / df['close']  # Normalized by price
    
    # Multiple EMAs and their slopes
    for period in config.EMA_PERIODS:
        ema_col = f'ema_{period}'
        slope_col = f'ema_{period}_slope'
        
        df[ema_col] = calculate_ema(df['close'], period)
        df[slope_col] = (df[ema_col] / df[ema_col].shift(5) - 1)  # 5-bar slope
        
        # EMA slope z-scores
        slope_mean = df[slope_col].expanding(min_periods=config.ZSCORE_LOOKBACK).mean().shift(1)
        slope_std = df[slope_col].expanding(min_periods=config.ZSCORE_LOOKBACK).std().shift(1)
        df[f'{slope_col}_zscore'] = (df[slope_col] - slope_mean) / slope_std
        
        # Price distance from EMA
        df[f'price_ema_{period}_dist'] = (df['close'] - df[ema_col]) / df[ema_col]
    
    # Multiple SMAs
    for period in config.SMA_PERIODS:
        sma_col = f'sma_{period}'
        df[sma_col] = calculate_sma(df['close'], period)
        df[f'price_sma_{period}_dist'] = (df['close'] - df[sma_col]) / df[sma_col]
    
    # EMA crossovers (trend signals)
    df['ema_5_10_cross'] = np.where(df['ema_5'] > df['ema_10'], 1, -1)
    df['ema_10_21_cross'] = np.where(df['ema_10'] > df['ema_21'], 1, -1)
    df['ema_21_55_cross'] = np.where(df['ema_21'] > df['ema_55'], 1, -1)
    
    # Price momentum at multiple timeframes
    for period in [3, 5, 10, 15, 20, 30]:
        df[f'price_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
    
    # Volatility measures
    df['vol_short'] = df['log_ret'].rolling(config.VOLATILITY_PERIOD_SHORT).std()
    df['vol_long'] = df['log_ret'].rolling(config.VOLATILITY_PERIOD).std()
    df['vol_ratio'] = df['vol_short'] / df['vol_long']  # Short vs long-term vol
    
    print("✓ Comprehensive technical indicators calculated")
    return df


def calculate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate COMPREHENSIVE microstructure features"""
    df = df.copy()
    
    print("Calculating comprehensive microstructure features...")
    
    # Basic order flow metrics
    df['dollar_imbalance'] = (df['buyer_initiated_volume'] - df['seller_initiated_volume']) / df['volume']
    df['signed_dollar_vol'] = df['dollar_volume'] * np.where(df['buy_volume_ratio'] > 0.5, 1, -1)
    
    # VPIN at multiple timeframes
    df['abs_imbalance'] = np.abs(df['buyer_initiated_volume'] - df['seller_initiated_volume'])
    
    # Short-term VPIN
    vpin_num_short = df['abs_imbalance'].rolling(config.VPIN_PERIOD_SHORT, min_periods=config.VPIN_PERIOD_SHORT).sum()
    vpin_den_short = df['volume'].rolling(config.VPIN_PERIOD_SHORT, min_periods=config.VPIN_PERIOD_SHORT).sum()
    df['vpin_short'] = vpin_num_short / vpin_den_short
    
    # Long-term VPIN
    vpin_num_long = df['abs_imbalance'].rolling(config.VPIN_PERIOD, min_periods=config.VPIN_PERIOD).sum()
    vpin_den_long = df['volume'].rolling(config.VPIN_PERIOD, min_periods=config.VPIN_PERIOD).sum()
    df['vpin_long'] = vpin_num_long / vpin_den_long
    
    # VPIN ratio (short vs long term)
    df['vpin_ratio'] = df['vpin_short'] / df['vpin_long']
    
    # Volume intensity at multiple timeframes
    volume_baseline_short = df['volume'].expanding(min_periods=20).mean().shift(1)
    volume_baseline_long = df['volume'].expanding(min_periods=100).mean().shift(1)
    df['volume_intensity_short'] = df['volume'] / volume_baseline_short
    df['volume_intensity_long'] = df['volume'] / volume_baseline_long
    
    # Trade intensity
    trade_baseline_short = df['trade_count'].expanding(min_periods=20).mean().shift(1)
    trade_baseline_long = df['trade_count'].expanding(min_periods=100).mean().shift(1)
    df['trade_intensity_short'] = df['trade_count'] / trade_baseline_short
    df['trade_intensity_long'] = df['trade_count'] / trade_baseline_long
    
    # Average trade size
    df['avg_trade_size'] = df['volume'] / df['trade_count']
    avg_trade_baseline = df['avg_trade_size'].expanding(min_periods=50).mean().shift(1)
    df['avg_trade_size_intensity'] = df['avg_trade_size'] / avg_trade_baseline
    
    # High-low ratio features
    hl_ratio_ma_short = df['hl_ratio'].expanding(min_periods=20).mean().shift(1)
    hl_ratio_std_short = df['hl_ratio'].expanding(min_periods=20).std().shift(1)
    df['hl_ratio_zscore_short'] = (df['hl_ratio'] - hl_ratio_ma_short) / hl_ratio_std_short
    
    hl_ratio_ma_long = df['hl_ratio'].expanding(min_periods=100).mean().shift(1)
    hl_ratio_std_long = df['hl_ratio'].expanding(min_periods=100).std().shift(1)
    df['hl_ratio_zscore_long'] = (df['hl_ratio'] - hl_ratio_ma_long) / hl_ratio_std_long
    
    # Buy volume ratio features
    df['buy_volume_ratio_zscore'] = (df['buy_volume_ratio'] - 0.5) / df['buy_volume_ratio'].expanding(min_periods=50).std().shift(1)
    
    # Rolling buy volume ratio
    df['buy_volume_ratio_ma_5'] = df['buy_volume_ratio'].rolling(5).mean()
    df['buy_volume_ratio_ma_20'] = df['buy_volume_ratio'].rolling(20).mean()
    df['buy_volume_ratio_trend'] = df['buy_volume_ratio_ma_5'] - df['buy_volume_ratio_ma_20']
    
    # Dollar volume intensity
    dollar_vol_baseline = df['dollar_volume'].expanding(min_periods=50).mean().shift(1)
    df['dollar_volume_intensity'] = df['dollar_volume'] / dollar_vol_baseline
    
    # Price impact proxy (return per unit volume)
    df['price_impact_proxy'] = np.abs(df['log_ret']) / (df['volume'] + 1e-8)
    price_impact_baseline = df['price_impact_proxy'].expanding(min_periods=50).mean().shift(1)
    df['price_impact_intensity'] = df['price_impact_proxy'] / price_impact_baseline
    
    # Volume-weighted price metrics
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Rolling correlations between volume and returns
    for window in [10, 20, 50]:
        df[f'vol_ret_corr_{window}'] = df['volume'].rolling(window).corr(df['log_ret'].abs())
    
    # Tick direction and momentum
    df['price_direction'] = np.sign(df['close'] - df['close'].shift(1))
    df['price_direction_ma_5'] = df['price_direction'].rolling(5).mean()
    df['price_direction_ma_20'] = df['price_direction'].rolling(20).mean()
    
    print("✓ Comprehensive microstructure features calculated")
    return df


def select_features(df: pd.DataFrame) -> list:
    """Define the feature columns to use for modeling"""
    feature_cols = [
        'log_ret', 'vol_adj_ret', 'sigma_50',
        'rsi_zscore', 'cci_zscore',
        'ema_10_slope_zscore', 'ema_21_slope_zscore', 'ema_55_slope_zscore',
        'dollar_imbalance', 'signed_dollar_vol', 'vpin',
        'volume_intensity', 'trade_intensity',
        'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
        'hl_ratio_zscore', 'buy_volume_ratio'
    ]
    
    # Verify all features exist
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    return available_features


def engineer_features(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main feature engineering pipeline
    Returns: (X_features, y_placeholder)
    """
    print("Loading dollar bars...")
    df = load_dollar_bars(data_path)
    
    print("Calculating returns...")
    df = calculate_returns(df)
    
    print("Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    print("Calculating microstructure features...")
    df = calculate_microstructure_features(df)
    
    print("Selecting features...")
    feature_cols = select_features(df)
    
    # Create feature matrix
    X = df[feature_cols].copy()
    
    # Drop rows with NaNs
    initial_count = len(X)
    X = X.dropna()
    y_placeholder = pd.DataFrame(index=X.index, columns=['label'])
    
    final_count = len(X)
    print(f"Dropped {initial_count - final_count:,} rows with NaNs")
    print(f"Final dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y_placeholder


if __name__ == "__main__":
    # Test feature engineering
    X, y = engineer_features()
    print("\nFeature engineering completed successfully!")
    print(f"X shape: {X.shape}")
    print(f"Date range: {X.index.min()} to {X.index.max()}")
    
    # Save features
    output_path = config.OUTPUT_PATH / "features.parquet"
    X.to_parquet(output_path)
    print(f"Features saved to {output_path}") 