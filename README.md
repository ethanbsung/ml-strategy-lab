# ML Strategy Lab

A machine learning-based trading strategy development project for futures markets.

## Directory Structure

```
ml-strategy-lab/
├── src/                    # Source code and main scripts
│   ├── train_model.py      # Main model training and analysis
│   ├── load_data.py        # Data loading utilities
│   ├── load_features.py    # Feature engineering
│   └── README.md           # Source code documentation
├── data/                   # Market data and databases
│   ├── market.db           # DuckDB database with features
│   ├── *.csv               # Various market data files
│   └── data_*.py           # Data processing scripts
├── outputs/                # Generated plots and analysis results
│   ├── *.png               # Visualization files
│   └── README.md           # Output documentation
├── backtests/              # Backtesting results and scripts
├── models/                 # Trained model artifacts
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run feature engineering:
   ```bash
   python src/load_features.py
   ```

3. Train model and analyze strategy:
   ```bash
   python src/train_model.py
   ```

## Key Features

- **Clean ML Pipeline**: Zero data leakage with proper time-series splitting
- **Feature Engineering**: Technical indicators with leak-free calculations
- **Comprehensive Testing**: Shuffle tests and sanity checks for signal validation
- **Strategy Analysis**: Regression-based trading with performance metrics
- **Visualization**: Detailed plots and analysis outputs

## Output Files

All generated plots and analysis results are saved in the `outputs/` directory. Key visualizations include:
- Feature importance analysis
- Strategy performance charts
- Sanity check results
- Regression analysis plots
