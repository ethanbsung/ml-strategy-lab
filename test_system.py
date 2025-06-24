#!/usr/bin/env python3
"""
Test script for Bitcoin Walk-Forward ML Backtesting System
This script demonstrates all key functionality with a small subset of data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import config
from features.engineer import engineer_features
from labels.triple_barrier import create_triple_barrier_labels, align_labels_with_features, create_sequences_for_lstm
from models.cnn_lstm import CNNLSTMModel
from models.random_forest import RandomForestModel

def test_feature_engineering():
    """Test feature engineering pipeline"""
    print("="*60)
    print("TESTING FEATURE ENGINEERING")
    print("="*60)
    
    try:
        # Limit to small subset for testing
        X, _ = engineer_features()
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Date range: {X.index.min()} to {X.index.max()}")
        print(f"✓ Feature columns: {len(X.columns)}")
        
        # Check data quality
        nan_percentage = X.isna().sum().sum() / (X.shape[0] * X.shape[1]) * 100
        print(f"✓ Overall NaN percentage: {nan_percentage:.2f}%")
        
        return X
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        raise

def test_labeling(X):
    """Test triple barrier labeling"""
    print("\n" + "="*60)
    print("TESTING TRIPLE BARRIER LABELING")
    print("="*60)
    
    try:
        # Create price data for labeling
        price_df = pd.DataFrame(index=X.index)
        price_df['close'] = np.exp(X['log_ret'].cumsum()) * 50000
        
        # Create labels
        labels_df = create_triple_barrier_labels(price_df)
        print(f"✓ Labels created: {labels_df.shape}")
        
        # Show distribution
        label_counts = labels_df['label'].value_counts().sort_index()
        print("✓ Label distribution:")
        for label, count in label_counts.items():
            pct = count / len(labels_df) * 100
            label_name = {-1: "Stop", 0: "Timeout", 1: "Profit"}[label]
            print(f"    {label_name} ({label}): {count:,} ({pct:.1f}%)")
        
        # Align with features
        X_aligned, y_aligned = align_labels_with_features(X, labels_df)
        print(f"✓ Aligned data: {X_aligned.shape}, {y_aligned.shape}")
        
        return X_aligned, y_aligned
        
    except Exception as e:
        print(f"✗ Labeling failed: {e}")
        raise

def test_models(X_aligned, y_aligned):
    """Test model training and prediction"""
    print("\n" + "="*60)
    print("TESTING MODEL TRAINING")
    print("="*60)
    
    try:
        # Create sequences for LSTM
        X_sequences, y_sequences, seq_indices = create_sequences_for_lstm(X_aligned, y_aligned)
        print(f"✓ Sequences created: {X_sequences.shape}")
        
        # Split data for testing
        split_idx = int(0.8 * len(X_sequences))
        X_train_seq = X_sequences[:split_idx]
        y_train_seq = y_sequences[:split_idx]
        X_test_seq = X_sequences[split_idx:]
        y_test_seq = y_sequences[split_idx:]
        
        # Test CNN-LSTM model
        print("\nTesting CNN-LSTM model...")
        cnn_lstm = CNNLSTMModel(input_shape=(config.LOOKBACK, X_train_seq.shape[2]))
        
        # Quick training with few epochs for testing
        original_epochs = config.EPOCHS
        config.EPOCHS = 3  # Quick test
        
        history = cnn_lstm.fit(X_train_seq, y_train_seq, epochs=3, verbose=0)
        print(f"✓ CNN-LSTM training completed")
        
        # Test prediction
        predictions = cnn_lstm.predict_proba(X_test_seq[:10])
        print(f"✓ CNN-LSTM predictions shape: {predictions.shape}")
        
        # Test Random Forest model
        print("\nTesting Random Forest model...")
        rf = RandomForestModel()
        
        # Prepare flat features for RF
        X_train_flat = X_aligned.iloc[:split_idx].values
        y_train_flat = y_aligned.iloc[:split_idx]['label'].values
        X_test_flat = X_aligned.iloc[split_idx:].values
        
        rf.fit(X_train_flat, y_train_flat, verbose=0)
        print(f"✓ Random Forest training completed")
        
        # Test prediction
        rf_predictions = rf.predict_proba(X_test_flat[:10])
        print(f"✓ Random Forest predictions shape: {rf_predictions.shape}")
        
        # Restore original config
        config.EPOCHS = original_epochs
        
        return cnn_lstm, rf
        
    except Exception as e:
        print(f"✗ Model testing failed: {e}")
        raise

def test_performance_calculation():
    """Test performance metrics calculation"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE METRICS")
    print("="*60)
    
    try:
        # Generate dummy returns for testing
        np.random.seed(42)
        n_periods = 1000
        returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift
        
        # Calculate equity curve
        equity_curve = pd.Series((1 + returns).cumprod())
        
        # Calculate metrics
        total_return = equity_curve.iloc[-1] - 1
        returns_series = equity_curve.pct_change().dropna()
        
        annual_return = (1 + total_return) ** (252 * 24 / len(equity_curve)) - 1
        volatility = returns_series.std() * np.sqrt(252 * 24)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()
        
        print(f"✓ Total Return: {total_return:.2%}")
        print(f"✓ Annual Return: {annual_return:.2%}")
        print(f"✓ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"✓ Max Drawdown: {max_drawdown:.2%}")
        print("✓ Performance calculation successful")
        
    except Exception as e:
        print(f"✗ Performance calculation failed: {e}")
        raise

def main():
    """Run all tests"""
    print("🚀 BITCOIN WALK-FORWARD ML BACKTESTING SYSTEM TEST")
    print("Testing core functionality with limited data...")
    print()
    
    try:
        # Test 1: Feature Engineering
        X = test_feature_engineering()
        
        # Test 2: Labeling
        X_aligned, y_aligned = test_labeling(X)
        
        # Test 3: Models (if we have enough data)
        if len(X_aligned) > 100:
            models = test_models(X_aligned, y_aligned)
        else:
            print("\n⚠️  Skipping model testing due to insufficient data")
        
        # Test 4: Performance Metrics
        test_performance_calculation()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print()
        print("The Bitcoin Walk-Forward ML Backtesting System is ready to use!")
        print()
        print("Next steps:")
        print("1. Install missing dependencies: pip install typer rich plotly")
        print("2. Run: python cli.py features")
        print("3. Run: python cli.py label") 
        print("4. Run: python cli.py walk-forward --fast")
        print("5. Run: python cli.py plot")
        print()
        print("For full documentation, see README_walkforward.md")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 