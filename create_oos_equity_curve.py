"""
Out-of-Sample Equity Curve Analysis for BTCUSDT Mean Reversion Strategy

This script creates a proper out-of-sample analysis by:
1. Loading the original dollar bars data with timestamps
2. Using TimeSeriesSplit to ensure proper temporal ordering
3. Creating a realistic equity curve based on actual price movements
4. Calculating proper risk-adjusted metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score

def load_data_with_timestamps():
    """Load dollar bars data and keep timestamp information"""
    print("📊 LOADING DATA WITH TIMESTAMPS FOR OOS ANALYSIS")
    print("=" * 55)
    
    # Load the same data as the original pipeline but keep timestamps
    import glob
    data_dir = "data/BTCUSDT/dollar_bars_5M"
    parquet_files = sorted(glob.glob(f"{data_dir}/*.parquet"))[-6:]  # Last 6 months
    
    print(f"Loading {len(parquet_files)} files...")
    
    dfs = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        if 'bar_start_time' in df.columns:
            df['timestamp'] = df['bar_start_time']
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Loaded {len(combined_df):,} bars")
    print(f"Date range: {combined_df['timestamp'].min().date()} to {combined_df['timestamp'].max().date()}")
    
    return combined_df

def create_features_with_timestamps(df):
    """Create the same features as the original pipeline but preserve timestamps"""
    print("\n🔧 CREATING FEATURES WITH TIMESTAMPS")
    print("-" * 40)
    
    df = df.copy()
    
    # Basic features (same as original pipeline)
    df['pct_change'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['vwap_close_spread'] = df['vwap'] - df['close']
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    
    if 'buy_volume_ratio' not in df.columns:
        df['buy_volume_ratio'] = df['buyer_initiated_volume'] / df['volume']
    
    # Rolling features (20-bar window)
    window = 20
    df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
    df[f'close_rolling_std_{window}'] = df['close'].rolling(window=window).std()
    df[f'zscore_{window}'] = (df['close'] - df[f'close_rolling_mean_{window}']) / df[f'close_rolling_std_{window}']
    
    # Volume and volatility features
    df[f'vol_rolling_{window}'] = df['log_return'].rolling(window=window).std()
    df[f'vol_zscore_{window}'] = (df[f'vol_rolling_{window}'] - df[f'vol_rolling_{window}'].rolling(window=window).mean()) / df[f'vol_rolling_{window}'].rolling(window=window).std()
    df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()
    df[f'volume_zscore_{window}'] = (df['volume'] - df[f'volume_rolling_mean_{window}']) / df['volume'].rolling(window=window).std()
    
    # Lagged features
    lags = [1, 3, 5]
    features_to_lag = [f'zscore_{window}', 'pct_change', 'vwap_close_spread', 'hl_ratio', 'buy_volume_ratio']
    
    for feature in features_to_lag:
        if feature in df.columns:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    # Target label (5-bar forward return)
    df['future_close'] = df['close'].shift(-5)
    df['future_return'] = (df['future_close'] / df['close']) - 1
    
    # Mean reversion label
    zscore_col = f'zscore_{window}'
    if zscore_col in df.columns:
        mean_reversion_condition = (
            ((df[zscore_col] > 0) & (df['future_return'] < 0)) |
            ((df[zscore_col] < 0) & (df['future_return'] > 0))
        )
        df['mean_reversion_label'] = mean_reversion_condition.astype(int)
    
    print("✅ Features created with timestamp preservation")
    
    return df

def prepare_oos_dataset(df):
    """Prepare dataset for proper OOS analysis"""
    print("\n🛠️ PREPARING OOS DATASET")
    print("-" * 30)
    
    # Feature columns (same as original pipeline)
    exclude_cols = [
        'timestamp', 'bar_start_time', 'bar_end_time',
        'open', 'high', 'low', 'close', 'vwap',
        'volume', 'dollar_volume', 'trade_count',
        'first_trade_id', 'last_trade_id',
        'buyer_initiated_volume', 'seller_initiated_volume',
        'future_close', 'future_return',
        'mean_reversion_label',
        'close_rolling_mean_20', 'close_rolling_std_20',
        'volume_rolling_mean_20',
        'vol_rolling_20'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Keep essential columns for analysis
    analysis_df = df[['timestamp', 'close'] + feature_cols + ['mean_reversion_label', 'future_return']].copy()
    
    # Remove rows with NaN values
    analysis_df = analysis_df.dropna()
    
    print(f"✅ OOS dataset prepared: {len(analysis_df):,} samples with {len(feature_cols)} features")
    
    return analysis_df, feature_cols

def walk_forward_oos_analysis(df, feature_cols, n_splits=5):
    """
    Perform walk-forward analysis to get true OOS performance with equity curve
    """
    print(f"\n🚀 WALK-FORWARD OOS ANALYSIS ({n_splits} splits)")
    print("-" * 45)
    
    # Prepare data
    X = df[feature_cols]
    y = df['mean_reversion_label']
    timestamps = df['timestamp']
    close_prices = df['close']
    future_returns = df['future_return']
    
    # TimeSeriesSplit for proper temporal ordering
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Results storage
    oos_predictions = []
    oos_timestamps = []
    oos_close_prices = []
    oos_future_returns = []
    oos_actual_labels = []
    
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
        'random_state': 42
    }
    
    print("Walk-forward validation:")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n  📁 Fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_timestamps = timestamps.iloc[test_idx]
        test_close = close_prices.iloc[test_idx]
        test_future_returns = future_returns.iloc[test_idx]
        
        print(f"    Train period: {timestamps.iloc[train_idx[0]].date()} to {timestamps.iloc[train_idx[-1]].date()}")
        print(f"    Test period:  {test_timestamps.iloc[0].date()} to {test_timestamps.iloc[-1].date()}")
        print(f"    Train samples: {len(X_train):,}")
        print(f"    Test samples:  {len(X_test):,}")
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=100,  # Reduced for faster training
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # OOS predictions
        test_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        
        # Store OOS results
        oos_predictions.extend(test_pred_proba)
        oos_timestamps.extend(test_timestamps)
        oos_close_prices.extend(test_close)
        oos_future_returns.extend(test_future_returns)
        oos_actual_labels.extend(y_test)
        
        # Fold performance
        test_pred = (test_pred_proba > 0.5).astype(int)
        precision = precision_score(y_test, test_pred, zero_division=0)
        print(f"    OOS Precision: {precision:.4f}")
    
    # Create OOS results DataFrame
    oos_df = pd.DataFrame({
        'timestamp': oos_timestamps,
        'close': oos_close_prices,
        'pred_proba': oos_predictions,
        'actual_label': oos_actual_labels,
        'future_return': oos_future_returns
    }).sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n✅ Walk-forward analysis complete")
    print(f"Total OOS samples: {len(oos_df):,}")
    print(f"OOS period: {oos_df['timestamp'].min().date()} to {oos_df['timestamp'].max().date()}")
    
    return oos_df

def create_realistic_trading_strategy(oos_df, threshold_long=0.6, threshold_short=0.4):
    """
    Create realistic trading strategy using actual future returns
    """
    print(f"\n💰 REALISTIC TRADING STRATEGY")
    print("-" * 35)
    print(f"Long threshold: {threshold_long}")
    print(f"Short threshold: {threshold_short}")
    
    df = oos_df.copy()
    
    # Generate trading signals
    df['signal'] = 0
    df.loc[df['pred_proba'] > threshold_long, 'signal'] = 1   # Long signal
    df.loc[df['pred_proba'] < threshold_short, 'signal'] = -1  # Short signal
    
    # Calculate strategy returns using actual future returns
    # If we go long and expect mean reversion UP, we profit from positive future returns
    # If we go short and expect mean reversion DOWN, we profit from negative future returns
    df['strategy_return'] = 0.0
    
    # Long positions: profit from positive future returns (price going up after being below mean)
    long_mask = df['signal'] == 1
    df.loc[long_mask, 'strategy_return'] = df.loc[long_mask, 'future_return']
    
    # Short positions: profit from negative future returns (price going down after being above mean)
    short_mask = df['signal'] == -1
    df.loc[short_mask, 'strategy_return'] = -df.loc[short_mask, 'future_return']
    
    # Calculate cumulative performance
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    
    # Performance metrics
    strategy_returns = df['strategy_return']
    total_trades = (df['signal'] != 0).sum()
    
    if total_trades > 0 and strategy_returns.std() > 0:
        # Annualized metrics (assuming dollar bars represent roughly daily frequency)
        # Note: This is an approximation since dollar bars have variable time intervals
        annualization_factor = np.sqrt(252)  # Approximate
        
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        sharpe_ratio = mean_return / std_return * annualization_factor
        
        # Other metrics
        total_return = df['cumulative_return'].iloc[-1] - 1
        max_drawdown = (df['cumulative_return'] / df['cumulative_return'].expanding().max() - 1).min()
        win_rate = (strategy_returns[df['signal'] != 0] > 0).mean()
        
        # Position statistics
        long_trades = (df['signal'] == 1).sum()
        short_trades = (df['signal'] == -1).sum()
        flat_periods = (df['signal'] == 0).sum()
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'flat_periods': flat_periods,
            'mean_daily_return': mean_return,
            'daily_volatility': std_return
        }
        
        print(f"\n📊 TRUE OOS Strategy Performance:")
        print(f"  • Total Return: {total_return:.2%}")
        print(f"  • Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  • Max Drawdown: {max_drawdown:.2%}")
        print(f"  • Win Rate: {win_rate:.2%}")
        print(f"  • Total Trades: {total_trades:,}")
        print(f"  • Long Trades: {long_trades:,}")
        print(f"  • Short Trades: {short_trades:,}")
        print(f"  • Flat Periods: {flat_periods:,}")
        print(f"  • Daily Volatility: {std_return:.3f}")
        
    else:
        print("❌ No trades generated or zero volatility")
        results = {}
    
    return df, results

def create_equity_curve_plots(strategy_df, results):
    """Create comprehensive equity curve and performance plots"""
    print(f"\n📊 CREATING EQUITY CURVE VISUALIZATION")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('BTCUSDT Mean Reversion Strategy - True OOS Performance', fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    axes[0,0].plot(strategy_df['timestamp'], strategy_df['cumulative_return'], 
                   linewidth=2, color='navy', label='Strategy')
    axes[0,0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even')
    axes[0,0].set_title('Equity Curve (True OOS)', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Cumulative Return')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Add performance text
    if results:
        perf_text = f"Total Return: {results['total_return']:.2%}\nSharpe: {results['sharpe_ratio']:.3f}\nMax DD: {results['max_drawdown']:.2%}"
        axes[0,0].text(0.02, 0.98, perf_text, transform=axes[0,0].transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Daily Returns Distribution
    strategy_returns = strategy_df['strategy_return']
    active_returns = strategy_returns[strategy_df['signal'] != 0]
    
    axes[0,1].hist(active_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0,1].axvline(active_returns.mean(), color='red', linestyle='--', 
                     label=f'Mean: {active_returns.mean():.4f}')
    axes[0,1].set_title('Active Trade Returns Distribution', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Return')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    running_max = strategy_df['cumulative_return'].expanding().max()
    drawdown = (strategy_df['cumulative_return'] / running_max - 1) * 100
    
    axes[0,2].fill_between(strategy_df['timestamp'], drawdown, 0, 
                          alpha=0.7, color='red', label='Drawdown')
    axes[0,2].set_title('Strategy Drawdown', fontsize=12, fontweight='bold')
    axes[0,2].set_ylabel('Drawdown (%)')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Rolling Sharpe Ratio (30-day window)
    window = 30
    rolling_returns = strategy_df['strategy_return'].rolling(window)
    rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
    
    axes[1,0].plot(strategy_df['timestamp'], rolling_sharpe, 
                   alpha=0.8, color='purple', linewidth=1.5)
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1,0].axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Sharpe = 1.0')
    axes[1,0].set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Sharpe Ratio')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Signal Distribution Over Time
    signal_colors = {-1: 'red', 0: 'gray', 1: 'green'}
    for signal_value, color in signal_colors.items():
        signal_mask = strategy_df['signal'] == signal_value
        if signal_mask.any():
            label = {-1: 'Short', 0: 'Flat', 1: 'Long'}[signal_value]
            axes[1,1].scatter(strategy_df.loc[signal_mask, 'timestamp'], 
                            strategy_df.loc[signal_mask, 'pred_proba'],
                            c=color, alpha=0.6, s=1, label=label)
    
    axes[1,1].axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Long threshold')
    axes[1,1].axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Short threshold')
    axes[1,1].set_title('Trading Signals Over Time', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Prediction Probability')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Performance Summary
    axes[1,2].axis('off')
    
    if results:
        summary_text = f"""
📊 TRUE OOS PERFORMANCE SUMMARY

🎯 Returns:
• Total Return: {results['total_return']:.2%}
• Sharpe Ratio: {results['sharpe_ratio']:.3f}
• Max Drawdown: {results['max_drawdown']:.2%}
• Win Rate: {results['win_rate']:.2%}

📈 Trading Activity:
• Total Trades: {results['total_trades']:,}
• Long Trades: {results['long_trades']:,}
• Short Trades: {results['short_trades']:,}
• Flat Periods: {results['flat_periods']:,}

⚠️ IMPORTANT:
This is TRUE out-of-sample performance
using walk-forward validation with 
proper temporal ordering.

Previous Sharpe of 0.42 was NOT
truly out-of-sample.
        """
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("outputs") / "true_oos_equity_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Equity curve saved: {output_path}")
    
    plt.show()
    
    return fig

def main():
    """Run the true OOS analysis"""
    print("🔍 TRUE OUT-OF-SAMPLE EQUITY CURVE ANALYSIS")
    print("=" * 60)
    print("Creating proper walk-forward analysis with realistic returns")
    print()
    
    try:
        # Step 1: Load data with timestamps
        df = load_data_with_timestamps()
        
        # Step 2: Create features while preserving timestamps
        df = create_features_with_timestamps(df)
        
        # Step 3: Prepare OOS dataset
        analysis_df, feature_cols = prepare_oos_dataset(df)
        
        # Step 4: Walk-forward OOS analysis
        oos_df = walk_forward_oos_analysis(analysis_df, feature_cols, n_splits=5)
        
        # Step 5: Create realistic trading strategy
        strategy_df, results = create_realistic_trading_strategy(oos_df)
        
        # Step 6: Create equity curve plots
        create_equity_curve_plots(strategy_df, results)
        
        # Step 7: Save OOS results
        output_dir = Path("outputs")
        oos_path = output_dir / "true_oos_results.csv"
        strategy_df.to_csv(oos_path, index=False)
        print(f"💾 OOS results saved: {oos_path}")
        
        if results:
            oos_summary_path = output_dir / "true_oos_summary.csv"
            pd.DataFrame([results]).to_csv(oos_summary_path, index=False)
            print(f"💾 OOS summary saved: {oos_summary_path}")
        
        print(f"\n🎉 TRUE OOS ANALYSIS COMPLETED!")
        print("="*50)
        
        if results:
            print(f"\n🔍 KEY FINDINGS:")
            print(f"• TRUE OOS Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"• TRUE OOS Total Return: {results['total_return']:.2%}")
            print(f"• TRUE OOS Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"• Total OOS Trades: {results['total_trades']:,}")
            print()
            print("⚠️  IMPORTANT: This is the REAL out-of-sample performance!")
            print("   The previous Sharpe of 0.42 was from mixed CV validation sets.")
        
    except Exception as e:
        print(f"\n❌ OOS ANALYSIS ERROR: {e}")
        raise

if __name__ == "__main__":
    main() 