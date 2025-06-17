# Source Code

This directory contains the main Python scripts for the ML trading strategy project:

## Files

- `train_model.py` - Main model training and strategy analysis script
- `load_data.py` - Data loading utilities for market data
- `load_features.py` - Feature engineering and creation utilities

## Usage

Run scripts from the project root directory to ensure proper relative paths:

```bash
# From ml-strategy-lab/ directory
python src/train_model.py
python src/load_features.py
python src/load_data.py
```

All scripts are configured to use relative paths to access the `../data/` and `../outputs/` directories. 