"""
Configuration file for Bitcoin Walk-Forward ML Backtesting System
"""
import os
from pathlib import Path

# Data paths
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "BTCUSDT" / "dollar_bars_1M"
OUTPUT_PATH = PROJECT_ROOT / "outputs"

# Ensure output directory exists
OUTPUT_PATH.mkdir(exist_ok=True)

# Trading parameters - AGGRESSIVE SETTINGS
FEE_RT_BPS = 4         # round-turn fees, basis-points
SLIPPAGE_BPS = 2       # one-way slippage
PROFIT_TGT = 0.0050    # 50 bps profit target (reduced for more trades)
STOP_TGT = 0.0035      # 35 bps stop loss (reduced for more trades)
TIMEOUT = 15           # bars timeout for triple barrier (reduced)

# Walk-forward parameters - MORE FREQUENT UPDATES
WINDOW = 3_000         # sliding-window length (reduced for more adaptation)
STEP = 500             # step size between windows (smaller steps)
LOOKBACK = 30          # bars fed into the model (increased)
P_THRESH = 0.45        # probability threshold for entry (MUCH LOWER - more trades)
P_THRESH_SHORT = 0.47  # different threshold for shorts (asymmetric)
P_THRESH_LONG = 0.43   # different threshold for longs

# Feature engineering parameters - MORE PARAMETERS
RSI_PERIOD = 14
RSI_PERIOD_SHORT = 7   # Additional short-term RSI
RSI_PERIOD_LONG = 21   # Additional long-term RSI
CCI_PERIOD = 20
CCI_PERIOD_SHORT = 10  # Additional short-term CCI
BOLLINGER_PERIOD = 20  # Bollinger bands
STOCH_PERIOD = 14      # Stochastic oscillator
MACD_FAST = 12         # MACD fast period
MACD_SLOW = 26         # MACD slow period
MACD_SIGNAL = 9        # MACD signal period
ATR_PERIOD = 14        # Average True Range
EMA_PERIODS = [5, 10, 21, 55, 89]  # More EMAs
SMA_PERIODS = [10, 20, 50, 100]    # Simple moving averages
VOLATILITY_PERIOD = 50
VOLATILITY_PERIOD_SHORT = 20       # Short-term volatility
VPIN_PERIOD = 50
VPIN_PERIOD_SHORT = 20             # Short-term VPIN
ZSCORE_LOOKBACK = 100              # Shorter lookback for faster adaptation

# Model parameters - MUCH MORE COMPLEX
CNN_FILTERS_1 = 64     # First conv layer (increased)
CNN_FILTERS_2 = 32     # Second conv layer
CNN_FILTERS_3 = 16     # Third conv layer (new)
CNN_KERNEL_SIZE = 3
CNN_KERNEL_SIZE_2 = 5  # Different kernel size
LSTM_UNITS_1 = 128     # First LSTM layer (much larger)
LSTM_UNITS_2 = 64      # Second LSTM layer (new)
DENSE_UNITS_1 = 128    # Additional dense layers
DENSE_UNITS_2 = 64
LEARNING_RATE = 1e-3   # Higher learning rate
LEARNING_RATE_DECAY = 0.95  # Learning rate decay
BATCH_SIZE = 64        # Smaller batch size for more updates
EPOCHS = 100           # More epochs
VALIDATION_SPLIT = 0.15 # Less validation, more training
DROPOUT_RATE = 0.3     # Lower dropout (less conservative)
DROPOUT_RATE_LSTM = 0.2 # Separate LSTM dropout
L2_REG = 0.0001        # L2 regularization strength

# Random Forest parameters - MORE COMPLEX
RF_N_ESTIMATORS = 500   # Much more trees
RF_MAX_DEPTH = 12       # Deeper trees
RF_MIN_SAMPLES_SPLIT = 2  # More aggressive splits
RF_MIN_SAMPLES_LEAF = 1   # Smaller leaves
RF_MAX_FEATURES = 0.8     # More features per tree
RF_BOOTSTRAP = True
RF_N_JOBS = -1

# Additional ML Models
XGBOOST_N_ESTIMATORS = 300
XGBOOST_MAX_DEPTH = 8
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE = 0.8

# Ensemble parameters
ENSEMBLE_WEIGHTS = {
    'cnn_lstm': 0.4,
    'random_forest': 0.3,
    'xgboost': 0.3
}

# Position sizing parameters
MAX_POSITION_SIZE = 1.0     # Maximum position size
MIN_POSITION_SIZE = 0.2     # Minimum position size
POSITION_SCALE_FACTOR = 2.0 # How much to scale by confidence
USE_KELLY_SIZING = True     # Use Kelly criterion for sizing
VOLATILITY_TARGET = 0.20    # Target portfolio volatility

# Risk management - LESS CONSERVATIVE
MAX_DAILY_TRADES = 50       # Maximum trades per day
MAX_CORRELATION = 0.8       # Maximum correlation between positions
DRAWDOWN_LIMIT = 0.15       # Stop trading if drawdown exceeds
FORCE_FLAT_HOUR = None      # No forced flat periods

# Data splitting
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging
LOG_LEVEL = "INFO" 