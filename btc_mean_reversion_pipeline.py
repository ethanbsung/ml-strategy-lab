"""
BTCUSDT Dollar Bars Mean Reversion ML Pipeline

Tests the hypothesis that BTCUSDT dollar bars mean-revert after deviating 
significantly from their local rolling mean using a comprehensive feature 
engineering approach and LightGBM classification.

Author: ML Strategy Lab
Date: $(date)
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML and evaluation
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_score
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def load_dollar_bars_data(data_dir="data/BTCUSDT/dollar_bars_5M", limit_months=None):
    """
    Load BTCUSDT dollar bars data from Parquet files.
    
    Args:
        data_dir (str): Directory containing dollar bar parquet files
        limit_months (int): Limit to most recent N months for faster testing
        
    Returns:
        pd.DataFrame: Combined dollar bars data sorted by timestamp
    """
    print("📊 LOADING BTCUSDT DOLLAR BARS DATA")
    print("=" * 50)
    
    # Find all parquet files
    parquet_files = sorted(glob.glob(f"{data_dir}/*.parquet"))
    
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    # Limit to most recent months if specified
    if limit_months:
        parquet_files = parquet_files[-limit_months:]
    
    print(f"Loading {len(parquet_files)} files...")
    for f in parquet_files:
        print(f"  {Path(f).name}")
    
    # Load and combine all files
    dfs = []
    total_bars = 0
    
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            
            # Rename timestamp column for consistency
            if 'bar_start_time' in df.columns:
                df['timestamp'] = df['bar_start_time']
            
            dfs.append(df)
            total_bars += len(df)
            print(f"  ✅ {Path(file_path).name}: {len(df):,} bars")
            
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
    
    # Combine and sort
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n✅ Total loaded: {len(combined_df):,} bars")
    print(f"Date range: {combined_df['timestamp'].min().date()} to {combined_df['timestamp'].max().date()}")
    
    return combined_df

def create_basic_features(df):
    """
    Create basic time-series features from OHLCV data.
    
    Args:
        df (pd.DataFrame): Dollar bars data
        
    Returns:
        pd.DataFrame: Data with basic features added
    """
    print("\n🔧 CREATING BASIC FEATURES")
    print("-" * 30)
    
    df = df.copy()
    
    # Basic price features
    df['pct_change'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # VWAP-close spread
    df['vwap_close_spread'] = df['vwap'] - df['close']
    
    # High-low ratio (already exists as hl_ratio, but create our own for clarity)
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Buyer/seller initiated volume ratio (already exists as buy_volume_ratio)
    if 'buy_volume_ratio' not in df.columns:
        df['buy_volume_ratio'] = df['buyer_initiated_volume'] / df['volume']
    
    print("✅ Basic features created:")
    print("  • pct_change, log_return")
    print("  • vwap_close_spread")
    print("  • hl_ratio")
    print("  • buy_volume_ratio")
    
    return df

def create_rolling_features(df, window=20):
    """
    Create rolling statistical features for mean reversion analysis.
    
    Args:
        df (pd.DataFrame): Data with basic features
        window (int): Rolling window size (default 20 bars)
        
    Returns:
        pd.DataFrame: Data with rolling features added
    """
    print(f"\n📈 CREATING ROLLING FEATURES (window={window})")
    print("-" * 40)
    
    df = df.copy()
    
    # Rolling mean and std for z-score calculation
    df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
    df[f'close_rolling_std_{window}'] = df['close'].rolling(window=window).std()
    
    # Z-score: (close - rolling_mean) / rolling_std
    df[f'zscore_{window}'] = (df['close'] - df[f'close_rolling_mean_{window}']) / df[f'close_rolling_std_{window}']
    
    # Rolling volatility features
    df[f'vol_rolling_{window}'] = df['log_return'].rolling(window=window).std()
    df[f'vol_zscore_{window}'] = (df[f'vol_rolling_{window}'] - df[f'vol_rolling_{window}'].rolling(window=window).mean()) / df[f'vol_rolling_{window}'].rolling(window=window).std()
    
    # Rolling volume features
    df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()
    df[f'volume_zscore_{window}'] = (df['volume'] - df[f'volume_rolling_mean_{window}']) / df['volume'].rolling(window=window).std()
    
    print(f"✅ Rolling features created:")
    print(f"  • close_rolling_mean_{window}, close_rolling_std_{window}")
    print(f"  • zscore_{window}")
    print(f"  • vol_rolling_{window}, vol_zscore_{window}")
    print(f"  • volume_rolling_mean_{window}, volume_zscore_{window}")
    
    return df

def create_lagged_features(df, lags=[1, 3, 5]):
    """
    Create lagged features for time series modeling.
    
    Args:
        df (pd.DataFrame): Data with rolling features
        lags (list): List of lag periods
        
    Returns:
        pd.DataFrame: Data with lagged features added
    """
    print(f"\n⏰ CREATING LAGGED FEATURES (lags={lags})")
    print("-" * 35)
    
    df = df.copy()
    
    # Features to lag
    features_to_lag = [
        'zscore_20',
        'pct_change', 
        'vwap_close_spread',
        'hl_ratio',
        'buy_volume_ratio'
    ]
    
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
            print(f"  ✅ {feature}: lags {lags}")
        else:
            print(f"  ⚠️ {feature}: not found, skipping")
    
    return df

def create_target_label(df, forward_periods=5):
    """
    Create binary classification label for mean reversion.
    
    Label Logic:
    1 if future_return and zscore are in opposite directions (mean reversion occurred)
    0 otherwise
    
    Args:
        df (pd.DataFrame): Data with features
        forward_periods (int): Number of periods to look forward for returns
        
    Returns:
        pd.DataFrame: Data with target label added
    """
    print(f"\n🎯 CREATING TARGET LABEL (forward_periods={forward_periods})")
    print("-" * 45)
    
    df = df.copy()
    
    # Calculate forward return
    df['future_close'] = df['close'].shift(-forward_periods)
    df['future_return'] = (df['future_close'] / df['close']) - 1
    
    # Create mean reversion label
    # 1 if zscore and future_return have opposite signs (mean reversion)
    # 0 otherwise
    zscore_col = 'zscore_20'
    if zscore_col in df.columns:
        # Mean reversion occurs when:
        # - zscore > 0 (above mean) and future_return < 0 (price falls)
        # - zscore < 0 (below mean) and future_return > 0 (price rises)
        mean_reversion_condition = (
            ((df[zscore_col] > 0) & (df['future_return'] < 0)) |
            ((df[zscore_col] < 0) & (df['future_return'] > 0))
        )
        
        df['mean_reversion_label'] = mean_reversion_condition.astype(int)
        
        print(f"✅ Target label created:")
        print(f"  • future_return: {forward_periods}-bar forward return")
        print(f"  • mean_reversion_label: 1 if zscore and future_return have opposite signs")
        
        # Label distribution
        label_dist = df['mean_reversion_label'].value_counts()
        total_valid = label_dist.sum()
        
        print(f"\n📊 Label Distribution:")
        print(f"  • Mean Reversion (1): {label_dist.get(1, 0):,} ({label_dist.get(1, 0)/total_valid*100:.1f}%)")
        print(f"  • No Mean Reversion (0): {label_dist.get(0, 0):,} ({label_dist.get(0, 0)/total_valid*100:.1f}%)")
        
    else:
        print(f"❌ {zscore_col} not found - cannot create target label")
        
    return df

def prepare_ml_dataset(df):
    """
    Prepare final dataset for machine learning by selecting features and handling missing values.
    
    Args:
        df (pd.DataFrame): Data with all features and labels
        
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    print("\n🛠️ PREPARING ML DATASET")
    print("-" * 25)
    
    # Define feature columns (exclude target, timestamp, and intermediate columns)
    exclude_cols = [
        'timestamp', 'bar_start_time', 'bar_end_time',
        'open', 'high', 'low', 'close', 'vwap',  # Raw OHLCV
        'volume', 'dollar_volume', 'trade_count',  # Raw volume data
        'first_trade_id', 'last_trade_id',  # Trade IDs
        'buyer_initiated_volume', 'seller_initiated_volume',  # Raw volume components
        'future_close', 'future_return',  # Target calculation intermediates
        'mean_reversion_label',  # Target variable
        'close_rolling_mean_20', 'close_rolling_std_20',  # Intermediate calculations
        'volume_rolling_mean_20',  # Intermediate calculations
        'vol_rolling_20'  # Raw volatility (keep z-score version)
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"📋 Selected Features ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df['mean_reversion_label'].copy()
    
    # Handle missing values
    initial_rows = len(X)
    
    # Drop rows with NaN in target
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Drop rows with too many NaN features (keep rows with at least 80% valid features)
    nan_threshold = 0.8 * len(feature_cols)
    feature_valid_mask = X.count(axis=1) >= nan_threshold
    X = X[feature_valid_mask]
    y = y[feature_valid_mask]
    
    # Forward fill remaining NaN values
    X = X.fillna(method='ffill')
    
    # Drop any remaining rows with NaN
    final_valid_mask = X.notna().all(axis=1)
    X = X[final_valid_mask]
    y = y[final_valid_mask]
    
    final_rows = len(X)
    
    print(f"\n🧹 Data Cleaning Results:")
    print(f"  • Initial rows: {initial_rows:,}")
    print(f"  • Final rows: {final_rows:,}")
    print(f"  • Dropped: {initial_rows - final_rows:,} ({(initial_rows - final_rows)/initial_rows*100:.1f}%)")
    print(f"  • Features: {X.shape[1]}")
    
    # Final label distribution
    label_dist = y.value_counts()
    total = len(y)
    print(f"\n📊 Final Label Distribution:")
    print(f"  • Mean Reversion (1): {label_dist.get(1, 0):,} ({label_dist.get(1, 0)/total*100:.1f}%)")
    print(f"  • No Mean Reversion (0): {label_dist.get(0, 0):,} ({label_dist.get(0, 0)/total*100:.1f}%)")
    
    return X, y

def train_lightgbm_model(X, y, n_splits=5, random_state=42):
    """
    Train LightGBM classifier with TimeSeriesSplit cross-validation.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        n_splits (int): Number of CV splits
        random_state (int): Random seed
        
    Returns:
        tuple: (model, cv_scores, cv_predictions)
    """
    print(f"\n🚀 TRAINING LIGHTGBM MODEL (CV={n_splits})")
    print("-" * 40)
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Model parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': random_state
    }
    
    # Store results
    cv_scores = []
    cv_predictions = []
    fold_models = []
    
    print("Training folds:")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n  📁 Fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"    Train: {len(X_train):,} samples")
        print(f"    Val:   {len(X_val):,} samples")
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predictions
        val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        val_pred = (val_pred_proba > 0.5).astype(int)
        
        # Calculate precision
        precision = precision_score(y_val, val_pred, zero_division=0)
        cv_scores.append(precision)
        cv_predictions.append({
            'fold': fold + 1,
            'y_true': y_val,
            'y_pred': val_pred,
            'y_pred_proba': val_pred_proba,
            'precision': precision
        })
        
        fold_models.append(model)
        
        print(f"    Precision: {precision:.4f}")
    
    # Train final model on all data
    print(f"\n🏁 Training final model on full dataset...")
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # Summary
    mean_precision = np.mean(cv_scores)
    std_precision = np.std(cv_scores)
    
    print(f"\n📊 Cross-Validation Results:")
    print(f"  • Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"  • Individual Folds: {[f'{s:.4f}' for s in cv_scores]}")
    
    return final_model, cv_scores, cv_predictions

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot and analyze feature importance from LightGBM model.
    
    Args:
        model: Trained LightGBM model
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
    """
    print(f"\n📊 FEATURE IMPORTANCE ANALYSIS (Top {top_n})")
    print("-" * 45)
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Display top features
    print("Top features:")
    for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<25} {row['importance']:>8.1f}")
    
    # Create plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)
    
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance (LightGBM)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Feature importance plot saved to {output_dir}/feature_importance.png")
    
    return feature_importance_df

def simulate_trading_strategy(cv_predictions, threshold_long=0.6, threshold_short=0.4):
    """
    Simulate a trading strategy based on model predictions.
    
    Strategy:
    - Go long if p > threshold_long
    - Go short if p < threshold_short  
    - Flat otherwise
    
    Args:
        cv_predictions (list): CV prediction results
        threshold_long (float): Probability threshold for long positions
        threshold_short (float): Probability threshold for short positions
        
    Returns:
        dict: Strategy performance metrics
    """
    print(f"\n📈 TRADING STRATEGY SIMULATION")
    print("-" * 35)
    print(f"Long threshold: {threshold_long}")
    print(f"Short threshold: {threshold_short}")
    
    all_returns = []
    all_positions = []
    
    for fold_result in cv_predictions:
        y_true = fold_result['y_true'].values
        y_pred_proba = fold_result['y_pred_proba']
        
        # Generate positions based on probabilities
        positions = np.zeros_like(y_pred_proba)
        positions[y_pred_proba > threshold_long] = 1   # Long
        positions[y_pred_proba < threshold_short] = -1  # Short
        
        # Calculate returns (simplified - assuming we capture mean reversion)
        # If we predict mean reversion (high prob) and it occurs (y_true=1), we profit
        # This is a simplified return calculation
        returns = positions * (y_true - 0.5) * 2  # Scale to [-2, 2] range
        
        all_returns.extend(returns)
        all_positions.extend(positions)
    
    all_returns = np.array(all_returns)
    all_positions = np.array(all_positions)
    
    # Calculate performance metrics
    cumulative_return = np.sum(all_returns)
    total_trades = np.sum(all_positions != 0)
    
    if total_trades > 0:
        avg_return_per_trade = cumulative_return / total_trades
        
        # Calculate Sharpe ratio (simplified)
        returns_std = np.std(all_returns[all_positions != 0])
        sharpe_ratio = avg_return_per_trade / returns_std if returns_std > 0 else 0
        
        # Calculate max drawdown (simplified)
        cumulative_returns = np.cumsum(all_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Position statistics
        long_positions = np.sum(all_positions == 1)
        short_positions = np.sum(all_positions == -1)
        flat_positions = np.sum(all_positions == 0)
        
        results = {
            'cumulative_return': cumulative_return,
            'total_trades': total_trades,
            'avg_return_per_trade': avg_return_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'flat_positions': flat_positions,
            'position_accuracy': np.mean(all_returns[all_positions != 0] > 0) if total_trades > 0 else 0
        }
        
        print(f"\n📊 Strategy Performance:")
        print(f"  • Total trades: {total_trades:,}")
        print(f"  • Long positions: {long_positions:,}")
        print(f"  • Short positions: {short_positions:,}")
        print(f"  • Flat periods: {flat_positions:,}")
        print(f"  • Cumulative return: {cumulative_return:.4f}")
        print(f"  • Avg return per trade: {avg_return_per_trade:.4f}")
        print(f"  • Sharpe ratio: {sharpe_ratio:.4f}")
        print(f"  • Max drawdown: {max_drawdown:.4f}")
        print(f"  • Position accuracy: {results['position_accuracy']:.2%}")
        
    else:
        print("❌ No trades generated with current thresholds")
        results = {}
    
    return results

def save_results(model, cv_predictions, feature_importance_df, strategy_results):
    """
    Save model and results to files.
    
    Args:
        model: Trained LightGBM model
        cv_predictions (list): CV prediction results
        feature_importance_df (pd.DataFrame): Feature importance results
        strategy_results (dict): Trading strategy results
    """
    print(f"\n💾 SAVING RESULTS")
    print("-" * 20)
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / "btc_mean_reversion_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Save feature importance
    importance_path = output_dir / "feature_importance.csv"
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"✅ Feature importance: {importance_path}")
    
    # Combine CV predictions
    all_predictions = []
    for fold_result in cv_predictions:
        fold_df = pd.DataFrame({
            'fold': fold_result['fold'],
            'y_true': fold_result['y_true'].values,
            'y_pred': fold_result['y_pred'],
            'y_pred_proba': fold_result['y_pred_proba']
        })
        all_predictions.append(fold_df)
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_path = output_dir / "cv_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ CV predictions: {predictions_path}")
    
    # Save strategy results
    if strategy_results:
        strategy_path = output_dir / "strategy_results.csv"
        strategy_df = pd.DataFrame([strategy_results])
        strategy_df.to_csv(strategy_path, index=False)
        print(f"✅ Strategy results: {strategy_path}")
    
    print(f"\n📁 All outputs saved to: {output_dir}/")

def main():
    """
    Main pipeline execution function.
    """
    print("🚀 BTCUSDT DOLLAR BARS MEAN REVERSION ML PIPELINE")
    print("=" * 60)
    print("Testing hypothesis: Dollar bars mean-revert after deviating from local rolling mean")
    print()
    
    try:
        # Step 1: Load data
        df = load_dollar_bars_data(limit_months=6)  # Limit to 6 months for faster execution
        
        # Step 2: Feature engineering
        df = create_basic_features(df)
        df = create_rolling_features(df, window=20)
        df = create_lagged_features(df, lags=[1, 3, 5])
        
        # Step 3: Create target labels
        df = create_target_label(df, forward_periods=5)
        
        # Step 4: Prepare ML dataset
        X, y = prepare_ml_dataset(df)
        
        # Step 5: Train model
        model, cv_scores, cv_predictions = train_lightgbm_model(X, y, n_splits=5)
        
        # Step 6: Analyze results
        feature_importance_df = plot_feature_importance(model, X.columns.tolist())
        strategy_results = simulate_trading_strategy(cv_predictions)
        
        # Step 7: Save results
        save_results(model, cv_predictions, feature_importance_df, strategy_results)
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ PIPELINE ERROR: {e}")
        raise

if __name__ == "__main__":
    main() 