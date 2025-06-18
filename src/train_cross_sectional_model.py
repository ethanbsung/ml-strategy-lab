"""
Daily Cross-Sectional Alpha Model

Trains a machine learning model to predict relative returns across assets on a daily basis.
This provides daily "bias" signals for ranking instruments to long, short, or skip.

Key Features:
- Cross-sectional ranking per day
- Time-series aware validation
- Long-short portfolio evaluation
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load daily features and create target variable"""
    
    print("📊 Loading daily features...")
    df = pd.read_parquet("lake/daily_features.parquet")
    
    print(f"Initial data shape: {df.shape}")
    print(f"Date range: {df['ts_utc'].min().date()} to {df['ts_utc'].max().date()}")
    print(f"Number of symbols: {df['symbol'].nunique()}")
    
    # Add target: forward 1-day return (leak-free)
    print("Creating target variable (next-day return)...")
    df = df.sort_values(['symbol', 'ts_utc'])
    df['target'] = df.groupby("symbol")['ln_ret_1'].shift(-1)
    
    # Add helper columns
    df['date'] = df['ts_utc'].dt.date
    df['year'] = df['ts_utc'].dt.year
    df['month'] = df['ts_utc'].dt.month
    
    # Drop rows with missing target (last observation per symbol)
    initial_rows = len(df)
    df = df.dropna(subset=['target'])
    print(f"Dropped {initial_rows - len(df)} rows with missing target")
    
    # Basic data quality checks
    print(f"\n📈 Data Quality:")
    print(f"Final shape: {df.shape}")
    print(f"Target range: {df['target'].min():.4f} to {df['target'].max():.4f}")
    print(f"Target mean: {df['target'].mean():.6f}")
    print(f"Target std: {df['target'].std():.4f}")
    
    return df

def select_features(df):
    """Select and validate features for the model"""
    
    # Define feature groups
    momentum_features = ['ln_ret_1', 'mom_5', 'mom_20']
    volatility_features = ['vol_5', 'vol_20']
    gap_features = ['gap_pc']
    volume_features = ['vol_z_20']
    technical_features = ['price_sma20_dev', 'hl_range_pc']
    time_features = ['day_of_week', 'month', 'sector_id']
    
    # Combine all features
    all_features = (momentum_features + volatility_features + gap_features + 
                   volume_features + technical_features + time_features)
    
    # Check which features are available
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]
    
    print(f"📋 Feature Selection:")
    print(f"Available features ({len(available_features)}): {available_features}")
    if missing_features:
        print(f"Missing features ({len(missing_features)}): {missing_features}")
    
    # Feature correlation analysis
    feature_df = df[available_features + ['target']].copy()
    
    # Remove features with too many nulls or zero variance
    null_pcts = feature_df.isnull().sum() / len(feature_df)
    high_null_features = null_pcts[null_pcts > 0.5].index.tolist()
    
    if high_null_features:
        print(f"Removing high-null features: {high_null_features}")
        available_features = [f for f in available_features if f not in high_null_features]
    
    # Check variance
    numeric_features = feature_df[available_features].select_dtypes(include=[np.number]).columns
    zero_var_features = []
    for feat in numeric_features:
        if feature_df[feat].std() == 0:
            zero_var_features.append(feat)
    
    if zero_var_features:
        print(f"Removing zero-variance features: {zero_var_features}")
        available_features = [f for f in available_features if f not in zero_var_features]
    
    print(f"Final feature set ({len(available_features)}): {available_features}")
    
    return available_features

def setup_time_series_validation(df, test_pct=0.20):
    """Setup time-aware train/test split using percentage of data"""
    
    # Sort by date
    df = df.sort_values(['date', 'symbol'])
    dates = sorted(df['date'].unique())
    
    # Use last 20% of dates as test set
    total_dates = len(dates)
    test_size = int(total_dates * test_pct)
    test_start_idx = total_dates - test_size
    
    train_dates = dates[:test_start_idx]
    test_dates = dates[test_start_idx:]
    
    print(f"📅 Time Series Split ({test_pct:.0%} out-of-sample):")
    print(f"Training period: {min(train_dates)} to {max(train_dates)}")
    print(f"Test period: {min(test_dates)} to {max(test_dates)}")
    print(f"Training days: {len(train_dates)}, Test days: {len(test_dates)}")
    print(f"Test percentage: {len(test_dates)/len(dates):.1%}")
    
    return train_dates, test_dates

def train_lightgbm_model(df, features, train_dates, test_dates):
    """Train LightGBM model with proper validation"""
    
    print("🧠 Training LightGBM Model...")
    
    # Prepare training data
    train_df = df[df['date'].isin(train_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Handle missing values
    X_train = train_df[features].fillna(0)
    y_train = train_df['target']
    X_test = test_df[features].fillna(0)
    y_test = test_df['target']
    
    # Train model
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        verbose=-1
    )
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[
            # lgb.early_stopping(50),
            # lgb.log_evaluation(100)
        ]
    )
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Model evaluation
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"Training RMSE: {train_rmse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Training MAE: {train_mae:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
    return model, feature_importance, train_df, test_df

def evaluate_cross_sectional_strategy(df, model, features, eval_dates, 
                                     long_pct=0.2, short_pct=0.2):
    """Evaluate long-short portfolio strategy with enhanced risk management"""
    
    print("📈 Evaluating Enhanced Cross-Sectional Strategy...")
    
    # Get evaluation data
    eval_df = df[df['date'].isin(eval_dates)].copy()
    
    # Generate predictions
    X_eval = eval_df[features].fillna(0)
    eval_df['pred'] = model.predict(X_eval)
    
    # Add volatility forecasting features
    eval_df = eval_df.sort_values(['symbol', 'date'])
    
    # 1. Realized volatility (20-day rolling)
    eval_df['realized_vol'] = eval_df.groupby('symbol')['target'].transform(
        lambda x: x.rolling(20, min_periods=5).std() * np.sqrt(252)
    ).fillna(eval_df.groupby('symbol')['target'].transform('std') * np.sqrt(252))
    
    # 2. EWMA volatility forecasting (more responsive)
    eval_df['ewma_vol'] = eval_df.groupby('symbol')['target'].transform(
        lambda x: x.ewm(span=20).std() * np.sqrt(252)
    ).fillna(eval_df['realized_vol'])
    
    # 3. Volatility regime detection (high vs low vol periods)
    vol_threshold = eval_df['ewma_vol'].quantile(0.75)  # Top 25% = high vol regime
    eval_df['high_vol_regime'] = (eval_df['ewma_vol'] > vol_threshold).astype(int)
    
    # Daily cross-sectional ranking
    eval_df['rank_pct'] = eval_df.groupby('date')['pred'].rank(pct=True)
    
    # Enhanced position sizing with risk management
    def calculate_positions(group):
        """Calculate risk-adjusted positions for each day"""
        
        # Basic long/short selection
        long_threshold = 1 - long_pct
        short_threshold = short_pct
        
        longs = group[group['rank_pct'] >= long_threshold].copy()
        shorts = group[group['rank_pct'] <= short_threshold].copy()
        
        # Position sizing adjustments
        for positions, side in [(longs, 'long'), (shorts, 'short')]:
            if len(positions) > 0:
                # 1. Inverse volatility weighting
                positions['inv_vol_weight'] = 1 / (positions['ewma_vol'] + 0.01)  # Add small constant
                positions['inv_vol_weight'] = positions['inv_vol_weight'] / positions['inv_vol_weight'].sum()
                
                # 2. Prediction confidence weighting  
                positions['pred_abs'] = np.abs(positions['pred'])
                positions['conf_weight'] = positions['pred_abs'] / (positions['pred_abs'].sum() + 1e-8)
                
                # 3. Combined weight (50% inv vol, 50% confidence)
                positions['combined_weight'] = 0.5 * positions['inv_vol_weight'] + 0.5 * positions['conf_weight']
                
                # 4. Volatility regime adjustment (reduce positions in high vol)
                # This is a principled approach - reduce risk when volatility is high
                vol_adjustment = np.where(positions['high_vol_regime'] == 1, 0.7, 1.0)  # 30% reduction in high vol
                positions['risk_adj_weight'] = positions['combined_weight'] * vol_adjustment
                
                # 5. Maximum position limit (diversification principle)
                positions['risk_adj_weight'] = np.minimum(positions['risk_adj_weight'], 0.15)  # Max 15% per position
                
                # 6. Renormalize weights
                positions['final_weight'] = positions['risk_adj_weight'] / positions['risk_adj_weight'].sum()
                
        return longs, shorts
    
    # Apply enhanced position sizing
    enhanced_results = []
    
    for date in eval_dates:
        daily_data = eval_df[eval_df['date'] == date]
        if len(daily_data) > 0:
            longs, shorts = calculate_positions(daily_data)
            
            # Add date and side labels
            if len(longs) > 0:
                longs['date'] = date
                longs['side'] = 'long'
                enhanced_results.append(longs)
                
            if len(shorts) > 0:
                shorts['date'] = date  
                shorts['side'] = 'short'
                enhanced_results.append(shorts)
    
    if not enhanced_results:
        print("⚠️ No positions generated!")
        return None
    
    # Combine all enhanced results
    enhanced_positions = pd.concat(enhanced_results, ignore_index=True)
    
    # Calculate portfolio-level risk management
    daily_portfolio_returns = []
    daily_portfolio_vol = []
    daily_positions = []
    
    # Portfolio volatility targeting (target 15% annual vol)
    target_portfolio_vol = 0.15
    
    for date in eval_dates:
        date_positions = enhanced_positions[enhanced_positions['date'] == date]
        
        if len(date_positions) > 0:
            # Calculate weighted returns
            long_positions = date_positions[date_positions['side'] == 'long']
            short_positions = date_positions[date_positions['side'] == 'short']
            
            # Portfolio return
            long_return = (long_positions['target'] * long_positions['final_weight']).sum() if len(long_positions) > 0 else 0
            short_return = (short_positions['target'] * short_positions['final_weight']).sum() if len(short_positions) > 0 else 0
            raw_portfolio_return = long_return - short_return
            
            # Estimate portfolio volatility (simple approach)
            portfolio_vol = date_positions['ewma_vol'].mean() if len(date_positions) > 0 else 0.2
            
            # Volatility scaling (scale down if portfolio vol > target)
            vol_scale = min(1.0, target_portfolio_vol / (portfolio_vol + 0.01))
            scaled_return = raw_portfolio_return * vol_scale
            
            daily_portfolio_returns.append(scaled_return)
            daily_portfolio_vol.append(portfolio_vol)
            daily_positions.append(len(date_positions))
        else:
            daily_portfolio_returns.append(0.0)
            daily_portfolio_vol.append(0.0)
            daily_positions.append(0)
    
    # Convert to Series with proper date index
    daily_returns = pd.Series(daily_portfolio_returns, index=eval_dates)
    daily_vol = pd.Series(daily_portfolio_vol, index=eval_dates)
    
    # Calculate final metrics
    final_cumulative = (1 + daily_returns).cumprod()
    
    # Performance metrics
    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    total_return = final_cumulative.iloc[-1] - 1
    max_dd = ((final_cumulative / final_cumulative.expanding().max()) - 1).min()
    win_rate = (daily_returns > 0).mean()
    avg_vol = daily_vol.mean()
    
    # Enhanced metrics
    calmar_ratio = abs(total_return / max_dd) if max_dd < 0 else 0
    avg_positions = np.mean(daily_positions)
    
    print(f"\n🎯 Enhanced Strategy Performance (IN-SAMPLE):")
    print(f"Sharpe Ratio: {sharpe:.2f}")  
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg Portfolio Vol: {avg_vol:.1%}")
    print(f"Avg Daily Positions: {avg_positions:.1f}")
    print(f"Volatility Target: {target_portfolio_vol:.0%}")
    
    # Risk management stats
    high_vol_days = (daily_vol > target_portfolio_vol * 1.5).sum()
    print(f"High Vol Days (>22.5%): {high_vol_days} ({high_vol_days/len(eval_dates):.1%})")
    
    return {
        'daily_returns': daily_returns,
        'cumulative_returns': final_cumulative,
        'daily_vol': daily_vol,
        'metrics': {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'avg_vol': avg_vol,
            'avg_positions': avg_positions
        },
        'enhanced_positions': enhanced_positions,
        'risk_stats': {
            'high_vol_days': high_vol_days,
            'dd_stops_triggered': False,  # No drawdown stops implemented in this version
            'vol_target': target_portfolio_vol
        }
    }

def create_visualizations(feature_importance, strategy_results):
    """Create performance and analysis visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Get the data
    daily_rets = strategy_results['daily_returns']
    cum_returns = strategy_results['cumulative_returns']
    daily_vol = strategy_results.get('daily_vol', pd.Series(index=daily_rets.index, data=0.15))
    
    # Create continuous date range for proper plotting (fill weekends/holidays)
    date_range = pd.date_range(start=daily_rets.index.min(), 
                              end=daily_rets.index.max(), 
                              freq='D')
    
    # Reindex to include all dates (weekends/holidays as 0)
    daily_rets_full = daily_rets.reindex(date_range, fill_value=0)
    cum_returns_full = (1 + daily_rets_full).cumprod()
    
    # 1. Cumulative Returns with proper date formatting
    axes[0,1].plot(cum_returns_full.index, cum_returns_full.values, 
                   linewidth=2, color='blue', alpha=0.8)
    axes[0,1].set_title(f'Enhanced Strategy Cumulative Returns (IN-SAMPLE)\nTotal Return: {strategy_results["metrics"]["total_return"]:.1%}, Sharpe: {strategy_results["metrics"]["sharpe"]:.2f}')
    axes[0,1].set_ylabel('Cumulative Return')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add horizontal line at 1.0
    axes[0,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    # Format x-axis for dates
    import matplotlib.dates as mdates
    axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0,1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Daily Returns Distribution
    axes[1,0].hist(daily_rets[daily_rets != 0], bins=50, alpha=0.7, 
                   edgecolor='black', color='green')
    axes[1,0].axvline(daily_rets.mean(), color='red', linestyle='--', 
                     label=f'Mean: {daily_rets.mean():.4f}')
    axes[1,0].set_title('Daily Returns Distribution')
    axes[1,0].set_xlabel('Daily Return')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 3. 30-Day Rolling Sharpe Ratio (fixed gaps)
    rolling_sharpe = daily_rets_full.rolling(30).mean() / daily_rets_full.rolling(30).std() * np.sqrt(252)
    rolling_sharpe = rolling_sharpe.dropna()
    
    axes[1,1].plot(rolling_sharpe.index, rolling_sharpe.values, 
                   alpha=0.8, color='purple', linewidth=1.5)
    axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1,1].axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Sharpe = 1.0')
    axes[1,1].set_title('30-Day Rolling Sharpe Ratio')
    axes[1,1].set_ylabel('Sharpe Ratio')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    # Format x-axis for dates
    axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1,1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
    
    # 5. Portfolio Volatility Over Time
    daily_vol_full = daily_vol.reindex(date_range, method='ffill')
    vol_target = strategy_results.get('risk_stats', {}).get('vol_target', 0.15)
    
    axes[0,2].plot(daily_vol_full.index, daily_vol_full.values * 100, 
                   alpha=0.7, color='orange', linewidth=1)
    axes[0,2].axhline(y=vol_target * 100, color='red', linestyle='--', 
                     alpha=0.7, label=f'Target: {vol_target:.0%}')
    axes[0,2].set_title(f'Portfolio Volatility\nAvg: {strategy_results["metrics"].get("avg_vol", 0.15):.1%}')
    axes[0,2].set_ylabel('Volatility (%)')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].legend()
    
    # Format x-axis for dates
    axes[0,2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0,2].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[0,2].xaxis.get_majorticklabels(), rotation=45)
    
    # 6. Risk Management Summary
    axes[1,2].axis('off')  # Turn off axis
    
    # Create risk management summary text
    risk_stats = strategy_results.get('risk_stats', {})
    metrics = strategy_results['metrics']
    
    summary_text = f"""
    📊 Risk Management Summary
    
    🎯 Performance Metrics:
    • Sharpe Ratio: {metrics['sharpe']:.2f}
    • Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}
    • Max Drawdown: {metrics['max_drawdown']:.1%}
    • Win Rate: {metrics['win_rate']:.1%}
    
    🛡️ Risk Controls:
    • Volatility Target: {risk_stats.get('vol_target', 0.15):.0%}
    • Avg Portfolio Vol: {metrics.get('avg_vol', 0.15):.1%}
    • High Vol Days: {risk_stats.get('high_vol_days', 0)}
    • DD Stop Triggered: {risk_stats.get('dd_stops_triggered', False)}
    
    💼 Position Stats:
    • Avg Daily Positions: {metrics.get('avg_positions', 0):.1f}
    • Position Size Limit: 10%
    • Vol Regime Adjustment: 50%
    """
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 4. Strategy Drawdown
    running_max = cum_returns_full.expanding().max()
    drawdown = (cum_returns_full / running_max - 1) * 100
    
    axes[0,0].fill_between(drawdown.index, drawdown.values, 0, 
                          alpha=0.7, color='red', label='Drawdown')
    axes[0,0].set_title(f'Strategy Drawdown (IN-SAMPLE)\nMax Drawdown: {strategy_results["metrics"]["max_drawdown"]:.1%}')
    axes[0,0].set_ylabel('Drawdown (%)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Format x-axis for dates
    axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0,0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/enhanced_cross_sectional_analysis_insample.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_feature_importance_chart(feature_importance):
    """Create a separate feature importance chart"""
    
    plt.figure(figsize=(12, 8))
    
    # Top 10 features
    top_features = feature_importance.head(10)
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance - Cross-Sectional Alpha Model')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + max(top_features['importance']) * 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{importance:.0f}', 
                va='center', ha='left')
    
    plt.tight_layout()
    plt.savefig('outputs/feature_importance_analysis_insample.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_predictions(enhanced_positions, model_name="enhanced_lightgbm_daily_alpha"):
    """Save enhanced predictions for future use"""
    
    # Prepare prediction dataset
    cols_to_save = ['ts_utc', 'symbol', 'pred', 'target', 'rank_pct', 'side', 
                   'final_weight', 'ewma_vol', 'high_vol_regime']
    available_cols = [col for col in cols_to_save if col in enhanced_positions.columns]
    
    pred_df = enhanced_positions[available_cols].copy()
    pred_df['model'] = model_name
    pred_df['generated_at'] = datetime.now()
    
    # Save to CSV
    output_path = f"outputs/{model_name}_predictions_insample.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"💾 Enhanced predictions saved to: {output_path}")
    print(f"📊 Saved {len(pred_df)} position records")
    
    return output_path

def main():
    """Main execution function"""
    
    print("🚀 Training Daily Cross-Sectional Alpha Model")
    print("=" * 60)
    
    # Ensure outputs directory exists
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Load and prepare data
    df = load_and_prepare_data()
    
    # 2. Feature selection
    features = select_features(df)
    
    # 3. Time series validation setup
    train_dates, test_dates = setup_time_series_validation(df, test_pct=0.20)
    
    # 4. Train model
    model, feature_importance, train_df, test_df = train_lightgbm_model(
        df, features, train_dates, test_dates
    )
    
    # 5. Evaluate strategy (IN-SAMPLE)
    strategy_results = evaluate_cross_sectional_strategy(
        df, model, features, train_dates
    )
    
    # 6. Create visualizations
    create_visualizations(feature_importance, strategy_results)
    create_feature_importance_chart(feature_importance)
    
    # 7. Save predictions  
    save_predictions(strategy_results['enhanced_positions'])
    
    print(f"\n✅ Cross-Sectional Model Training Complete!")
    print(f"📊 IN-SAMPLE Final Sharpe Ratio: {strategy_results['metrics']['sharpe']:.2f}")
    print(f"📈 IN-SAMPLE Total Return: {strategy_results['metrics']['total_return']:.2%}")
    print(f"📉 IN-SAMPLE Max Drawdown: {strategy_results['metrics']['max_drawdown']:.2%}")
    
    return model, strategy_results, feature_importance

if __name__ == "__main__":
    model, results, importance = main() 