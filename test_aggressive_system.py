#!/usr/bin/env python3
"""
Test New Aggressive, Parameter-Rich ML System (Optimized for Large Datasets)
Shows actual out-of-sample performance with much more trading activity
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import gc
import psutil
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

import config
from features.engineer import engineer_features
from labels.triple_barrier import create_triple_barrier_labels, align_labels_with_features, create_sequences_for_lstm
from models.cnn_lstm import CNNLSTMModel
from models.random_forest import RandomForestModel

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"💾 Memory usage: {memory_mb:.1f} MB")

def calculate_comprehensive_metrics(returns, positions, strategy_name="Strategy"):
    """Calculate comprehensive trading performance metrics"""
    returns = pd.Series(returns)
    positions = pd.Series(positions)
    
    # Trading frequency
    total_periods = len(positions)
    trading_periods = (positions != 0).sum()
    trade_frequency = trading_periods / total_periods
    
    # Position breakdown
    long_positions = (positions > 0).sum()
    short_positions = (positions < 0).sum()
    flat_positions = (positions == 0).sum()
    
    # Return calculations
    cumulative_return = (1 + returns).prod() - 1
    
    if returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24)  # Assuming hourly data
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Win rate
    trading_returns = returns[positions != 0]
    if len(trading_returns) > 0:
        win_rate = (trading_returns > 0).mean()
        avg_win = trading_returns[trading_returns > 0].mean() if len(trading_returns[trading_returns > 0]) > 0 else 0
        avg_loss = trading_returns[trading_returns < 0].mean() if len(trading_returns[trading_returns < 0]) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # Turnover (how often positions change)
    position_changes = (positions.diff() != 0).sum()
    turnover = position_changes / total_periods
    
    return {
        'strategy_name': strategy_name,
        'total_periods': total_periods,
        'trading_periods': trading_periods,
        'trade_frequency': trade_frequency,
        'long_positions': long_positions,
        'short_positions': short_positions,
        'flat_positions': flat_positions,
        'cumulative_return': cumulative_return,
        'annual_return': (1 + cumulative_return) ** (252 * 24 / total_periods) - 1,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'turnover': turnover,
        'calmar_ratio': (cumulative_return / abs(max_drawdown)) if max_drawdown != 0 else 0
    }

def load_and_sample_data(max_samples=500000):
    """Load and sample data efficiently to avoid memory issues"""
    import glob
    
    # Get ALL dollar bar files
    data_pattern = "data/BTCUSDT/dollar_bars_1M/BTCUSDT-trades-*_dollar_bars.parquet"
    files = sorted(glob.glob(data_pattern))
    
    if not files:
        print("❌ No dollar bar files found!")
        return None
    
    print(f"📊 Found {len(files)} files from {os.path.basename(files[0])} to {os.path.basename(files[-1])}")
    
    # For testing, use recent 12 months for better memory management
    if len(files) > 12:
        files = files[-12:]
        print(f"📊 Limited to most recent 12 months for memory efficiency: {len(files)} files")
    
    # Load data in chunks and sample if too large
    dfs = []
    total_rows = 0
    
    for file in files:
        try:
            df = pd.read_parquet(file)
            total_rows += len(df)
            dfs.append(df)
            print(f"✓ Loaded {file}: {len(df):,} bars (Total: {total_rows:,})")
            
            # Early break if we have enough data
            if total_rows > max_samples * 1.5:  # Leave room for sampling
                print(f"📊 Stopping early - have {total_rows:,} bars (target: {max_samples:,})")
                break
                
        except FileNotFoundError:
            print(f"⚠️ File not found: {file}")
    
    if not dfs:
        print("❌ No data files found!")
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('bar_start_time').reset_index(drop=True)
    
    # Sample data if too large
    if len(df) > max_samples:
        print(f"📊 Sampling {max_samples:,} from {len(df):,} bars for memory efficiency...")
        # Take recent data + random sample to maintain time series properties
        recent_split = int(max_samples * 0.7)  # 70% recent data
        sample_split = max_samples - recent_split  # 30% sampled from rest
        
        # Recent data
        recent_data = df.iloc[-recent_split:].copy()
        
        # Sampled historical data
        historical_data = df.iloc[:-recent_split]
        if len(historical_data) > sample_split:
            sampled_indices = np.sort(np.random.choice(len(historical_data), sample_split, replace=False))
            sampled_data = historical_data.iloc[sampled_indices].copy()
        else:
            sampled_data = historical_data.copy()
        
        # Combine and sort
        df = pd.concat([sampled_data, recent_data], ignore_index=True)
        df = df.sort_values('bar_start_time').reset_index(drop=True)
        print(f"📊 Final dataset: {len(df):,} bars")
    
    print(f"📅 Period: {df['bar_start_time'].min()} to {df['bar_start_time'].max()}")
    print_memory_usage()
    
    return df

def engineer_features_efficiently(df):
    """Engineer features with memory management"""
    print("\n🔧 Engineering features efficiently...")
    print_memory_usage()
    
    # Import feature engineering functions directly
    from features.engineer import calculate_returns, calculate_technical_indicators, calculate_microstructure_features
    
    print("Calculating returns...")
    df = calculate_returns(df)
    gc.collect()  # Force garbage collection
    
    print("Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    gc.collect()
    
    print("Calculating microstructure features...")
    df = calculate_microstructure_features(df)
    gc.collect()
    
    print_memory_usage()
    return df

def create_labels_efficiently(df, chunk_size=100000):
    """Create labels in chunks to manage memory"""
    print(f"\n🏷️ Creating labels efficiently (chunk size: {chunk_size:,})...")
    print_memory_usage()
    
    total_samples = len(df)
    all_labels = []
    
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        print(f"Processing chunk {start_idx:,} to {end_idx:,} ({len(chunk_df):,} bars)")
        
        # Create labels for this chunk
        labels_chunk = create_triple_barrier_labels(chunk_df)
        all_labels.append(labels_chunk)
        
        # Clean up
        del chunk_df
        gc.collect()
        print_memory_usage()
    
    # Combine all labels
    print("Combining label chunks...")
    labels_df = pd.concat(all_labels, ignore_index=False)
    
    # Clean up
    del all_labels
    gc.collect()
    print_memory_usage()
    
    return labels_df

def run_aggressive_backtest():
    """Run comprehensive test of the aggressive system with memory optimization"""
    
    print("🚀 TESTING NEW AGGRESSIVE, PARAMETER-RICH ML SYSTEM (OPTIMIZED)")
    print("=" * 70)
    print(f"🎯 Profit Target: {config.PROFIT_TGT:.1%}")
    print(f"🛑 Stop Target: {config.STOP_TGT:.1%}")
    print(f"⏱️  Timeout: {config.TIMEOUT} bars")
    print(f"📊 Long Threshold: {config.P_THRESH_LONG:.1%}")
    print(f"📊 Short Threshold: {config.P_THRESH_SHORT:.1%}")
    print(f"🔄 Window: {config.WINDOW:,} bars")
    print(f"🔄 Step: {config.STEP:,} bars")
    
    print_memory_usage()
    
    # Load and prepare data efficiently
    print("\n📂 Loading data efficiently...")
    df = load_and_sample_data(max_samples=500000)  # Limit to 500K samples for memory
    
    if df is None:
        return
    
    # Engineer features efficiently
    df = engineer_features_efficiently(df)
    
    # Select feature columns (excluding price/volume basics)
    price_cols = ['bar_start_time', 'bar_end_time', 'open', 'high', 'low', 'close']
    feature_cols = [col for col in df.columns if col not in [
        'bar_start_time', 'bar_end_time', 'open', 'high', 'low', 'close', 'volume', 
        'dollar_volume', 'trade_count', 'vwap', 'buyer_initiated_volume', 
        'seller_initiated_volume', 'buy_volume_ratio', 'hl_ratio'
    ]]
    
    print(f"✓ Available feature columns: {len(feature_cols)}")
    print(f"✓ Sample features: {feature_cols[:10]}")
    
    # Clean data
    df_clean = df.dropna()
    print(f"✓ Clean dataset: {df_clean.shape[1]} total columns, {len(df_clean):,} rows")
    print_memory_usage()
    
    # Create labels efficiently
    labels_df = create_labels_efficiently(df_clean)
    print(f"✓ Labels created: {len(labels_df):,} samples")
    
    # Align features and labels
    print("\n🔗 Aligning features and labels...")
    X_aligned, y_aligned = align_labels_with_features(df_clean[feature_cols], labels_df)
    print(f"✓ Aligned data: {len(X_aligned):,} samples with {len(feature_cols)} features")
    print_memory_usage()
    
    # Clean up large dataframes
    del df, df_clean, labels_df
    gc.collect()
    print_memory_usage()
    
    # Create sequences for LSTM
    print("\n🔄 Creating LSTM sequences...")
    X_sequences, y_sequences, seq_indices = create_sequences_for_lstm(X_aligned, y_aligned, config.LOOKBACK)
    print(f"✓ LSTM sequences: {len(X_sequences):,} sequences")
    print_memory_usage()
    
    # Split data - use 80% for training
    split_point = int(len(X_aligned) * 0.8)
    
    # Training data
    X_train_seq = X_sequences[:split_point]
    y_train_seq = y_sequences[:split_point]
    X_train_flat = X_aligned.iloc[:split_point].values
    y_train_flat = y_aligned.iloc[:split_point]['label'].values
    
    # Test data
    X_test_seq = X_sequences[split_point:]
    y_test_seq = y_sequences[split_point:]
    X_test_flat = X_aligned.iloc[split_point:].values
    y_test_flat = y_aligned.iloc[split_point:]['label'].values
    test_indices = seq_indices[split_point:]
    
    print(f"\n📚 Training: {len(X_train_seq):,} sequences")
    print(f"🧪 Testing: {len(X_test_seq):,} sequences")
    print_memory_usage()
    
    # Train models
    print(f"\n🤖 Training CNN-LSTM on {len(X_train_seq):,} sequences...")
    cnn_lstm = CNNLSTMModel(input_shape=(config.LOOKBACK, X_train_seq.shape[2]))
    cnn_lstm.fit(X_train_seq, y_train_seq, epochs=min(config.EPOCHS, 20), verbose=1)  # Limit epochs for testing
    
    print(f"🌲 Training Random Forest ({config.RF_N_ESTIMATORS} trees, depth {config.RF_MAX_DEPTH}) on {len(X_train_flat):,} samples...")
    rf = RandomForestModel()
    rf.fit(X_train_flat, y_train_flat)
    
    print_memory_usage()
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    cnn_lstm_proba = cnn_lstm.predict_proba(X_test_seq)
    rf_proba = rf.predict_proba(X_test_flat)
    
    # Model accuracy
    cnn_lstm_pred = np.argmax(cnn_lstm_proba, axis=1) - 1  # Convert from {0,1,2} to {-1,0,1}
    rf_pred = rf.predict(X_test_flat)
    
    cnn_lstm_accuracy = accuracy_score(y_test_seq, cnn_lstm_pred)
    rf_accuracy = accuracy_score(y_test_flat, rf_pred)
    
    print(f"📊 CNN-LSTM Accuracy: {cnn_lstm_accuracy:.1%}")
    print(f"📊 Random Forest Accuracy: {rf_accuracy:.1%}")
    
    # Generate trading signals
    print(f"\n📈 Generating trading signals...")
    
    # CNN-LSTM signals
    p_down = cnn_lstm_proba[:, 0]  # Stop loss probability  
    p_flat = cnn_lstm_proba[:, 1]  # Timeout probability
    p_up = cnn_lstm_proba[:, 2]    # Profit probability
    
    cnn_lstm_signals = np.zeros(len(X_test_seq))
    cnn_lstm_signals[p_up > config.P_THRESH_LONG] = 1   # Long
    cnn_lstm_signals[p_down > config.P_THRESH_SHORT] = -1 # Short
    
    # Random Forest signals
    rf_classes = rf.model.classes_
    p_down_rf = rf_proba[:, np.where(rf_classes == -1)[0][0]]
    p_flat_rf = rf_proba[:, np.where(rf_classes == 0)[0][0]]
    p_up_rf = rf_proba[:, np.where(rf_classes == 1)[0][0]]
    
    rf_signals = np.zeros(len(X_test_flat))
    rf_signals[p_up_rf > config.P_THRESH_LONG] = 1   # Long
    rf_signals[p_down_rf > config.P_THRESH_SHORT] = -1 # Short
    
    # Calculate returns with fees and slippage
    print("\n💰 Calculating trading returns...")
    price_data = X_aligned.iloc[split_point:].copy()
    forward_returns = price_data['log_ret'].shift(-1).values  # Next bar return
    
    def calculate_strategy_returns(signals, name):
        strategy_returns = []
        positions = []
        
        for i, signal in enumerate(signals):
            if i >= len(forward_returns) - 1:  # Skip last bar
                break
                
            base_return = forward_returns[i]
            positions.append(signal)
            
            if signal != 0:  # Trade
                # Apply direction
                trade_return = signal * base_return
                
                # Apply costs
                fee_cost = config.FEE_RT_BPS / 10000  # Round-turn fees
                slippage_cost = config.SLIPPAGE_BPS / 10000 * 2  # Both sides
                
                total_cost = fee_cost + slippage_cost
                net_return = trade_return - total_cost
                
                strategy_returns.append(net_return)
            else:
                strategy_returns.append(0.0)  # No position
        
        return np.array(strategy_returns), np.array(positions)
    
    # Calculate returns for both models
    cnn_lstm_returns, cnn_lstm_positions = calculate_strategy_returns(cnn_lstm_signals, "CNN-LSTM")
    rf_returns, rf_positions = calculate_strategy_returns(rf_signals, "Random Forest")
    
    # Performance analysis
    print("\n🏆 PERFORMANCE RESULTS")
    print("=" * 50)
    
    cnn_lstm_metrics = calculate_comprehensive_metrics(cnn_lstm_returns, cnn_lstm_positions, "CNN-LSTM")
    rf_metrics = calculate_comprehensive_metrics(rf_returns, rf_positions, "Random Forest")
    
    # Display results
    for metrics in [cnn_lstm_metrics, rf_metrics]:
        print(f"\n📊 {metrics['strategy_name']} Strategy:")
        print(f"  🎯 Accuracy: {metrics['strategy_name'] == 'CNN-LSTM' and cnn_lstm_accuracy or rf_accuracy:.1%}")
        print(f"  📈 Trade Frequency: {metrics['trade_frequency']:.1%}")
        print(f"  📊 Long Positions: {metrics['long_positions']:,} ({metrics['long_positions']/max(metrics['trading_periods'], 1):.1%})")
        print(f"  📊 Short Positions: {metrics['short_positions']:,} ({metrics['short_positions']/max(metrics['trading_periods'], 1):.1%})")
        print(f"  💰 Cumulative Return: {metrics['cumulative_return']:.2%}")
        print(f"  📊 Annual Return: {metrics['annual_return']:.2%}")
        print(f"  ⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  🎯 Win Rate: {metrics['win_rate']:.1%}")
        print(f"  💵 Avg Win: {metrics['avg_win']:.3%}")
        print(f"  💸 Avg Loss: {metrics['avg_loss']:.3%}")
        print(f"  📊 Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  🔄 Turnover: {metrics['turnover']:.1%}")
        print(f"  📊 Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    # System comparison
    print(f"\n🔥 AGGRESSIVE SYSTEM ANALYSIS")
    print("=" * 40)
    print(f"✅ CNN-LSTM takes {(cnn_lstm_positions != 0).sum():,} trades ({(cnn_lstm_positions != 0).mean():.1%} of time)")
    print(f"✅ Random Forest takes {(rf_positions != 0).sum():,} trades ({(rf_positions != 0).mean():.1%} of time)")
    print(f"🎯 Lower thresholds = Much more trading activity!")
    print(f"📊 50bp profit target = More achievable targets")
    print(f"🔧 {X_aligned.shape[1]} features = Maximum complexity")
    print_memory_usage()
    
    if cnn_lstm_metrics['cumulative_return'] > 0:
        print(f"🏆 CNN-LSTM: PROFITABLE with {cnn_lstm_metrics['cumulative_return']:.2%} return!")
    if rf_metrics['cumulative_return'] > 0:
        print(f"🏆 Random Forest: PROFITABLE with {rf_metrics['cumulative_return']:.2%} return!")
    
    return {
        'cnn_lstm_metrics': cnn_lstm_metrics,
        'rf_metrics': rf_metrics,
        'total_features': X_aligned.shape[1],
        'cnn_lstm_trades': (cnn_lstm_positions != 0).sum(),
        'rf_trades': (rf_positions != 0).sum()
    }

if __name__ == "__main__":
    results = run_aggressive_backtest()
    
    print(f"\n✅ AGGRESSIVE SYSTEM TEST COMPLETE!")
    print(f"🔧 Total features used: {results['total_features']}")
    print(f"📊 CNN-LSTM trades: {results['cnn_lstm_trades']:,}")
    print(f"📊 Random Forest trades: {results['rf_trades']:,}")
    print(f"🚀 System is now much more active and parameter-rich!") 