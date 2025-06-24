# Bitcoin Walk-Forward ML Backtesting System

A comprehensive machine learning backtesting framework for Bitcoin using $1M dollar bars with walk-forward validation. This system implements López-de-Prado triple barrier labeling and compares CNN-LSTM deep learning models against Random Forest baselines.

## Features

- 🔄 **Walk-Forward Backtesting**: Sliding window validation with realistic out-of-sample testing
- 🤖 **Dual Model Architecture**: CNN-LSTM for sequence learning vs Random Forest baseline
- 📊 **Advanced Features**: Technical indicators, microstructure features, and volatility adjustments
- 🎯 **Triple Barrier Labeling**: Profit target, stop loss, and timeout-based labeling system
- 💰 **Realistic Trading Costs**: Configurable fees and slippage modeling
- 📈 **Rich Visualizations**: Interactive equity curves and performance comparison plots
- ⚡ **Fast Execution**: Optimized with Numba and efficient data structures

## Project Structure

```
.
├── config.py                    # Configuration parameters
├── cli.py                      # Command-line interface
├── features/
│   └── engineer.py            # Feature engineering pipeline
├── labels/
│   └── triple_barrier.py      # Triple barrier labeling system
├── models/
│   ├── cnn_lstm.py           # CNN-LSTM deep learning model
│   └── random_forest.py      # Random Forest baseline
├── backtest/
│   └── walk_forward.py       # Walk-forward backtesting engine
├── data/
│   └── BTCUSDT/
│       └── dollar_bars_1M/   # $1M dollar bars data
├── outputs/                  # Generated results and plots
├── requirements.txt          # Python dependencies
└── README_walkforward.md     # This file
```

## Installation

### 1. Clone and Setup Environment

```bash
git clone <your-repo>
cd ml-strategy-lab

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with TA-Lib, install it separately:

```bash
# On macOS with Homebrew
brew install ta-lib
pip install TA-Lib

# On Ubuntu/Debian
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On Windows
# Download and install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

### 3. Verify Installation

```bash
python cli.py status
```

## Quick Start

### Complete Pipeline (5 minutes)

Run the entire pipeline with fast settings:

```bash
# 1. Create features from dollar bars
python cli.py features

# 2. Generate triple barrier labels  
python cli.py label

# 3. Run walk-forward backtest (fast mode)
python cli.py walk-forward --fast

# 4. Generate performance plots
python cli.py plot
```

### Individual Steps

#### 1. Feature Engineering

```bash
python cli.py features
```

Creates 18 features including:
- **Returns**: Log returns, volatility-adjusted returns
- **Technical Indicators**: Z-scored RSI, CCI, EMA slopes
- **Microstructure**: Dollar imbalance, VPIN, signed volume
- **Momentum**: Multi-timeframe price momentum

#### 2. Triple Barrier Labeling

```bash
python cli.py label
```

Implements López-de-Prado methodology:
- **Profit Target**: 20 bps (configurable)
- **Stop Loss**: 15 bps (configurable)  
- **Timeout**: 10 bars maximum hold time
- **Labels**: {-1: Stop, 0: Timeout, 1: Profit}

#### 3. Model Training (Single Window)

```bash
python cli.py train --window-idx 0
```

Trains both models on the first window:
- **CNN-LSTM**: Conv1D → Conv1D → LSTM → Dense
- **Random Forest**: 200 trees with balanced class weights

#### 4. Walk-Forward Backtest

```bash
# Full backtest (may take 30+ minutes)
python cli.py walk-forward

# Fast mode (5-10 minutes)
python cli.py walk-forward --fast
```

Sliding window parameters:
- **Window Size**: 2,000 bars (1,000 in fast mode)
- **Step Size**: 500 bars (250 in fast mode)
- **Train/Val/Test**: 60%/20%/20% split per window

#### 5. Results and Plotting

```bash
python cli.py plot
```

Generates:
- Interactive equity curves (`equity_curves.html`)
- Performance comparison charts (`performance_comparison.png`)
- Detailed performance metrics table

## Configuration

Edit `config.py` to customize parameters:

```python
# Trading parameters
FEE_RT_BPS = 4         # Round-turn fees (4 bps)
SLIPPAGE_BPS = 2       # One-way slippage (2 bps)
PROFIT_TGT = 0.0020    # Profit target (20 bps)
STOP_TGT = 0.0015      # Stop loss (15 bps)
P_THRESH = 0.55        # Probability threshold

# Walk-forward parameters
WINDOW = 2_000         # Window size
STEP = 500             # Step size
LOOKBACK = 20          # LSTM lookback period

# Model parameters
CNN_FILTERS = 32       # CNN filters
LSTM_UNITS = 64        # LSTM units
EPOCHS = 50            # Training epochs
```

## Model Architecture

### CNN-LSTM Model

```
Input(20, 18) → Conv1D(32, 3) → BatchNorm → Dropout(0.2)
              → Conv1D(32, 3) → BatchNorm → Dropout(0.2)  
              → LSTM(64) → Dense(3, softmax)
```

- **Input**: 20 bars × 18 features
- **Output**: Probability distribution over {Stop, Timeout, Profit}
- **Optimization**: Adam(1e-3) with early stopping

### Random Forest Model

```
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    n_jobs=-1
)
```

- **Input**: Flattened 20×18 = 360 features
- **Output**: Class probabilities
- **Preprocessing**: StandardScaler normalization

## Performance Metrics

The system calculates comprehensive performance statistics:

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative strategy return |
| **Annual Return** | Annualized return (assuming 24/7 trading) |
| **Sharpe Ratio** | Risk-adjusted return measure |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Hit Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Number of Trades** | Total number of executed trades |

## Data Requirements

The system expects dollar bar data in the following format:

```
data/BTCUSDT/dollar_bars_1M/
├── BTCUSDT-trades-2024-01_dollar_bars.parquet
├── BTCUSDT-trades-2024-02_dollar_bars.parquet
└── ...
```

**Required columns:**
- `bar_end_time`: Timestamp
- `close`: Closing price
- `volume`: Volume
- `dollar_volume`: Dollar volume
- `buyer_initiated_volume`: Buy volume
- `seller_initiated_volume`: Sell volume
- `buy_volume_ratio`: Buy ratio
- Additional OHLC and microstructure features

## Advanced Usage

### Custom Feature Engineering

Modify `features/engineer.py` to add new features:

```python
def calculate_custom_features(df):
    # Add your custom features here
    df['custom_feature'] = ...
    return df
```

### Hyperparameter Optimization

Use the training command to test different parameters:

```python
# Modify config.py temporarily
config.EPOCHS = 100
config.LEARNING_RATE = 5e-4

# Train and evaluate
python cli.py train --window-idx 0
```

### Model Persistence

Models are automatically saved during training:

```
outputs/
├── cnn_lstm_window_0.h5      # Keras model
├── rf_window_0.pkl           # Scikit-learn model
└── walk_forward_results.pkl  # Complete results
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `WINDOW` size or use `--fast` mode
2. **TensorFlow Warnings**: Set `TF_CPP_MIN_LOG_LEVEL=2`
3. **TA-Lib Installation**: See installation instructions above
4. **Data Not Found**: Verify dollar bar files exist in correct directory

### Performance Optimization

```bash
# Set environment variables for better performance
export TF_CPP_MIN_LOG_LEVEL=2
export NUMBA_NUM_THREADS=4

# Use fast mode for quick testing
python cli.py walk-forward --fast
```

### Debugging

Enable verbose output:

```python
# In config.py
LOG_LEVEL = "DEBUG"

# Or check system status
python cli.py status
```

## Expected Results

On a typical dataset, you should expect:

- **Feature Engineering**: ~2-3 minutes for full history
- **Label Creation**: ~30 seconds with Numba acceleration
- **Walk-Forward Backtest**: 5-30 minutes depending on settings
- **Model Performance**: Varies by market conditions, typically:
  - Sharpe Ratio: 0.5-2.0
  - Hit Rate: 35-65%
  - Max Drawdown: 5-20%

## Contributing

To extend the system:

1. **New Features**: Add to `features/engineer.py`
2. **New Models**: Create in `models/` directory
3. **New Labels**: Extend `labels/triple_barrier.py`
4. **New Metrics**: Add to `backtest/walk_forward.py`

## License

This project is for educational and research purposes. Please ensure compliance with your local regulations when using financial data and trading algorithms.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Dollar Bar Construction for High-Frequency Data
- Walk-Forward Optimization in Quantitative Finance
- CNN-LSTM Architectures for Time Series Prediction

---

**Happy Backtesting! 🚀**

For questions or issues, please check the troubleshooting section or review the code comments in each module. 