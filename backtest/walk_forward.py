"""
Walk-Forward Backtesting Engine for Bitcoin ML Strategy
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config
from features.engineer import engineer_features
from labels.triple_barrier import create_triple_barrier_labels, align_labels_with_features, create_sequences_for_lstm
from models.cnn_lstm import CNNLSTMModel
from models.random_forest import RandomForestModel


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine with sliding windows
    """
    
    def __init__(self):
        """Initialize the backtesting engine"""
        self.results = []
        self.equity_curves = {}
        self.models = {}
        self.features = None
        self.labels = None
        self.sequences = None
        
    def prepare_data(self) -> bool:
        """
        Prepare features and labels for backtesting
        
        Returns:
        --------
        bool : Success flag
        """
        try:
            print("Preparing data for walk-forward backtesting...")
            
            # Engineer features
            print("Engineering features...")
            X, _ = engineer_features()
            self.features = X
            
            # Create price data for labeling (reconstruct from log returns)
            print("Creating price data for labeling...")
            price_df = pd.DataFrame(index=X.index)
            price_df['close'] = np.exp(X['log_ret'].cumsum()) * 50000  # Scale to realistic BTC prices
            
            # Create triple barrier labels
            print("Creating triple barrier labels...")
            labels_df = create_triple_barrier_labels(price_df)
            
            # Align features and labels
            print("Aligning features and labels...")
            X_aligned, y_aligned = align_labels_with_features(X, labels_df)
            
            # Create sequences for LSTM
            print("Creating sequences for LSTM...")
            X_sequences, y_sequences, seq_indices = create_sequences_for_lstm(X_aligned, y_aligned)
            
            self.features = X_aligned
            self.labels = y_aligned
            self.sequences = {
                'X': X_sequences,
                'y': y_sequences,
                'indices': seq_indices
            }
            
            print(f"Data preparation completed successfully!")
            print(f"Features shape: {self.features.shape}")
            print(f"Labels shape: {self.labels.shape}")
            print(f"Sequences shape: {X_sequences.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return False
    
    def split_window(self, start_idx: int, end_idx: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split a window into train/val/test sets
        
        Parameters:
        -----------
        start_idx : int
            Start index for the window
        end_idx : int
            End index for the window
        
        Returns:
        --------
        Dict with train/val/test splits for both sequences and flat features
        """
        # Get window data
        window_X_seq = self.sequences['X'][start_idx:end_idx]
        window_y_seq = self.sequences['y'][start_idx:end_idx]
        window_indices = self.sequences['indices'][start_idx:end_idx]
        
        # Get flat features for RF (align with sequence indices)
        window_X_flat = self.features.loc[window_indices].values
        window_y_flat = self.labels.loc[window_indices]['label'].values
        
        # Split ratios
        n_samples = len(window_X_seq)
        train_end = int(n_samples * config.TRAIN_RATIO)
        val_end = int(n_samples * (config.TRAIN_RATIO + config.VAL_RATIO))
        
        # Split sequences for CNN-LSTM
        X_train_seq = window_X_seq[:train_end]
        y_train_seq = window_y_seq[:train_end]
        
        X_val_seq = window_X_seq[train_end:val_end]
        y_val_seq = window_y_seq[train_end:val_end]
        
        X_test_seq = window_X_seq[val_end:]
        y_test_seq = window_y_seq[val_end:]
        
        # Split flat features for RF
        X_train_flat = window_X_flat[:train_end]
        y_train_flat = window_y_flat[:train_end]
        
        X_val_flat = window_X_flat[train_end:val_end]
        y_val_flat = window_y_flat[train_end:val_end]
        
        X_test_flat = window_X_flat[val_end:]
        y_test_flat = window_y_flat[val_end:]
        
        # Test indices for tracking
        test_indices = window_indices[val_end:]
        
        return {
            'sequences': {
                'train': (X_train_seq, y_train_seq),
                'val': (X_val_seq, y_val_seq),
                'test': (X_test_seq, y_test_seq)
            },
            'flat': {
                'train': (X_train_flat, y_train_flat),
                'val': (X_val_flat, y_val_flat),
                'test': (X_test_flat, y_test_flat)
            },
            'test_indices': test_indices
        }
    
    def train_models(self, splits: Dict) -> Dict:
        """
        Train both CNN-LSTM and Random Forest models
        
        Parameters:
        -----------
        splits : dict
            Data splits from split_window
        
        Returns:
        --------
        Dict with trained models
        """
        models = {}
        
        # Train CNN-LSTM
        print("Training CNN-LSTM model...")
        X_train_seq, y_train_seq = splits['sequences']['train']
        X_val_seq, y_val_seq = splits['sequences']['val']
        
        cnn_lstm = CNNLSTMModel(input_shape=(config.LOOKBACK, X_train_seq.shape[2]))
        cnn_lstm.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
                    epochs=config.EPOCHS, verbose=0)
        models['cnn_lstm'] = cnn_lstm
        
        # Train Random Forest
        print("Training Random Forest model...")
        X_train_flat, y_train_flat = splits['flat']['train']
        X_val_flat, y_val_flat = splits['flat']['val']
        
        rf = RandomForestModel()
        rf.fit(X_train_flat, y_train_flat, X_val_flat, y_val_flat, verbose=0)
        models['rf'] = rf
        
        return models
    
    def generate_signals(self, models: Dict, splits: Dict) -> Dict[str, np.ndarray]:
        """
        Generate trading signals from model predictions
        
        Parameters:
        -----------
        models : dict
            Trained models
        splits : dict
            Data splits
        
        Returns:
        --------
        Dict with signals for each model
        """
        signals = {}
        
        # CNN-LSTM signals
        X_test_seq, _ = splits['sequences']['test']
        cnn_lstm_proba = models['cnn_lstm'].predict_proba(X_test_seq)
        
        # Convert probabilities to signals
        # Probabilities are for classes [0, 1, 2] representing [-1, 0, 1]
        p_down = cnn_lstm_proba[:, 0]  # Stop loss probability
        p_flat = cnn_lstm_proba[:, 1]  # Timeout probability  
        p_up = cnn_lstm_proba[:, 2]    # Profit probability
        
        cnn_lstm_signals = np.zeros(len(X_test_seq))
        cnn_lstm_signals[p_up > config.P_THRESH_LONG] = 1   # Long (lower threshold)
        cnn_lstm_signals[p_down > config.P_THRESH_SHORT] = -1 # Short (higher threshold)
        
        signals['cnn_lstm'] = cnn_lstm_signals
        
        # Random Forest signals
        X_test_flat, _ = splits['flat']['test']
        rf_proba = models['rf'].predict_proba(X_test_flat)
        
        # RF might have different class ordering, need to check
        rf_classes = models['rf'].model.classes_
        p_down_rf = rf_proba[:, np.where(rf_classes == -1)[0][0]]
        p_flat_rf = rf_proba[:, np.where(rf_classes == 0)[0][0]]
        p_up_rf = rf_proba[:, np.where(rf_classes == 1)[0][0]]
        
        rf_signals = np.zeros(len(X_test_flat))
        rf_signals[p_up_rf > config.P_THRESH_LONG] = 1   # Long (lower threshold)
        rf_signals[p_down_rf > config.P_THRESH_SHORT] = -1 # Short (higher threshold)
        
        signals['rf'] = rf_signals
        
        return signals
    
    def calculate_returns(self, signals: Dict[str, np.ndarray], 
                         test_indices: pd.Index) -> Dict[str, pd.Series]:
        """
        Calculate strategy returns with fees and slippage
        
        Parameters:
        -----------
        signals : dict
            Trading signals for each model
        test_indices : pd.Index
            Timestamps for the test period
        
        Returns:
        --------
        Dict with return series for each model
        """
        returns = {}
        
        # Get price data for return calculation
        price_data = self.features.loc[test_indices]
        forward_returns = price_data['log_ret'].shift(-1)  # Next bar return
        
        for model_name, model_signals in signals.items():
            strategy_returns = []
            
            for i, signal in enumerate(model_signals):
                if i >= len(forward_returns) - 1:  # Skip last bar
                    break
                    
                base_return = forward_returns.iloc[i]
                
                if signal != 0:  # Only trade when signal is non-zero
                    # Apply direction
                    trade_return = signal * base_return
                    
                    # Apply fees and slippage
                    fee_cost = config.FEE_RT_BPS / 10000  # Round-turn fees
                    slippage_cost = config.SLIPPAGE_BPS / 10000 * 2  # Both entry and exit
                    
                    total_cost = fee_cost + slippage_cost
                    net_return = trade_return - total_cost
                    
                    strategy_returns.append(net_return)
                else:
                    strategy_returns.append(0.0)  # No position
            
            returns[model_name] = pd.Series(strategy_returns, 
                                          index=test_indices[:len(strategy_returns)])
        
        return returns
    
    def run_backtest(self) -> Dict:
        """
        Run the complete walk-forward backtest
        
        Returns:
        --------
        Dict with backtest results
        """
        if not self.prepare_data():
            return {}
        
        print(f"Starting walk-forward backtest...")
        print(f"Window size: {config.WINDOW}, Step size: {config.STEP}")
        
        # Initialize equity curves
        self.equity_curves = {'cnn_lstm': [], 'rf': []}
        all_returns = {'cnn_lstm': [], 'rf': []}
        window_results = []
        
        # Calculate number of windows
        n_sequences = len(self.sequences['X'])
        n_windows = (n_sequences - config.WINDOW) // config.STEP + 1
        
        print(f"Total sequences: {n_sequences:,}")
        print(f"Number of windows: {n_windows}")
        
        # Run walk-forward windows
        for window_idx in tqdm(range(n_windows), desc="Walk-forward windows"):
            start_idx = window_idx * config.STEP
            end_idx = start_idx + config.WINDOW
            
            if end_idx > n_sequences:
                break
            
            print(f"\nWindow {window_idx + 1}/{n_windows}: [{start_idx}:{end_idx}]")
            
            try:
                # Split data
                splits = self.split_window(start_idx, end_idx)
                
                # Train models
                models = self.train_models(splits)
                
                # Generate signals
                signals = self.generate_signals(models, splits)
                
                # Calculate returns
                returns = self.calculate_returns(signals, splits['test_indices'])
                
                # Store results
                for model_name in ['cnn_lstm', 'rf']:
                    if model_name in returns:
                        all_returns[model_name].extend(returns[model_name].tolist())
                
                # Store window metadata
                window_results.append({
                    'window_idx': window_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'test_start': splits['test_indices'][0],
                    'test_end': splits['test_indices'][-1],
                    'n_trades': {model: sum(np.abs(signals[model])) for model in signals.keys()}
                })
                
            except Exception as e:
                print(f"Error in window {window_idx}: {e}")
                continue
        
        # Combine all returns into equity curves
        print("\nCombining results...")
        for model_name in ['cnn_lstm', 'rf']:
            if all_returns[model_name]:
                equity_curve = pd.Series(all_returns[model_name])
                equity_curve = (1 + equity_curve).cumprod()
                self.equity_curves[model_name] = equity_curve
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()
        
        return {
            'equity_curves': self.equity_curves,
            'performance': performance,
            'window_results': window_results,
            'all_returns': all_returns
        }
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics for each strategy"""
        metrics = {}
        
        for model_name, equity_curve in self.equity_curves.items():
            if len(equity_curve) == 0:
                continue
                
            returns = equity_curve.pct_change().dropna()
            
            # Basic metrics
            total_return = equity_curve.iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 * 24 / len(equity_curve)) - 1  # Assuming hourly bars
            
            volatility = returns.std() * np.sqrt(252 * 24)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Drawdown
            cummax = equity_curve.expanding().max()
            drawdown = (equity_curve - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # Hit rate
            hit_rate = (returns > 0).mean()
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            metrics[model_name] = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'hit_rate': hit_rate,
                'profit_factor': profit_factor,
                'n_trades': len(returns[returns != 0])
            }
        
        return metrics
    
    def save_results(self, results: Dict, filepath: str = None):
        """Save backtest results"""
        if filepath is None:
            filepath = config.OUTPUT_PATH / "backtest_results.pkl"
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Run backtest
    backtest = WalkForwardBacktest()
    results = backtest.run_backtest()
    
    # Print performance summary
    if 'performance' in results:
        print("\n" + "="*60)
        print("WALK-FORWARD BACKTEST RESULTS")
        print("="*60)
        
        for model_name, metrics in results['performance'].items():
            print(f"\n{model_name.upper()} Model:")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Annual Return: {metrics['annual_return']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Hit Rate: {metrics['hit_rate']:.2%}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Number of Trades: {metrics['n_trades']:,}")
    
    # Save results
    backtest.save_results(results) 