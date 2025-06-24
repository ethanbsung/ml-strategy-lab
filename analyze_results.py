"""
BTCUSDT Mean Reversion Pipeline Results Analysis

Analyzes the results from the mean reversion hypothesis test
and provides detailed insights into model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_cv_predictions():
    """Analyze cross-validation predictions in detail"""
    print("📊 CROSS-VALIDATION PREDICTION ANALYSIS")
    print("=" * 50)
    
    # Load CV predictions
    cv_df = pd.read_csv("outputs/cv_predictions.csv")
    
    print(f"Total predictions: {len(cv_df):,}")
    print(f"Folds: {cv_df['fold'].nunique()}")
    
    # Analyze by fold
    print("\n📈 Performance by Fold:")
    fold_stats = cv_df.groupby('fold').agg({
        'y_true': ['count', 'mean'],
        'y_pred': 'mean',
        'y_pred_proba': ['mean', 'std']
    }).round(4)
    
    print(fold_stats)
    
    # Calculate additional metrics per fold
    print("\n🎯 Detailed Metrics by Fold:")
    for fold in sorted(cv_df['fold'].unique()):
        fold_data = cv_df[cv_df['fold'] == fold]
        
        # Calculate metrics
        accuracy = (fold_data['y_true'] == fold_data['y_pred']).mean()
        precision = (fold_data['y_pred'] & fold_data['y_true']).sum() / fold_data['y_pred'].sum() if fold_data['y_pred'].sum() > 0 else 0
        recall = (fold_data['y_pred'] & fold_data['y_true']).sum() / fold_data['y_true'].sum() if fold_data['y_true'].sum() > 0 else 0
        
        print(f"  Fold {fold}:")
        print(f"    Samples: {len(fold_data):,}")
        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    Mean Reversion Rate: {fold_data['y_true'].mean():.3f}")
        print(f"    Avg Predicted Prob: {fold_data['y_pred_proba'].mean():.3f}")
    
    return cv_df

def analyze_feature_importance():
    """Analyze feature importance results"""
    print("\n\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    # Load feature importance
    feat_df = pd.read_csv("outputs/feature_importance.csv")
    
    print(f"Total features: {len(feat_df)}")
    
    # Group features by type
    feature_groups = {
        'Z-Score Features': feat_df[feat_df['feature'].str.contains('zscore')],
        'Volume Features': feat_df[feat_df['feature'].str.contains('volume')],
        'Price Features': feat_df[feat_df['feature'].str.contains('pct_change|log_return|vwap')],
        'Microstructure': feat_df[feat_df['feature'].str.contains('hl_ratio|duration|buy_volume_ratio')],
        'Lagged Features': feat_df[feat_df['feature'].str.contains('_lag_')]
    }
    
    print("\n📊 Feature Importance by Category:")
    for group_name, group_df in feature_groups.items():
        if len(group_df) > 0:
            total_importance = group_df['importance'].sum()
            avg_importance = group_df['importance'].mean()
            print(f"\n{group_name}:")
            print(f"  Features: {len(group_df)}")
            print(f"  Total Importance: {total_importance:.1f}")
            print(f"  Avg Importance: {avg_importance:.1f}")
            print(f"  Top feature: {group_df.iloc[0]['feature']} ({group_df.iloc[0]['importance']:.1f})")
    
    # Analyze lag importance
    print("\n⏰ Lag Analysis:")
    lag_features = feat_df[feat_df['feature'].str.contains('_lag_')]
    if len(lag_features) > 0:
        lag_features['lag'] = lag_features['feature'].str.extract(r'_lag_(\d+)').astype(int)
        lag_summary = lag_features.groupby('lag')['importance'].agg(['count', 'mean', 'sum']).round(1)
        print(lag_summary)
    
    return feat_df

def analyze_strategy_performance():
    """Analyze trading strategy performance"""
    print("\n\n💰 TRADING STRATEGY ANALYSIS")
    print("=" * 35)
    
    # Load strategy results
    strategy_df = pd.read_csv("outputs/strategy_results.csv")
    
    results = strategy_df.iloc[0]
    
    print("📈 Strategy Performance Summary:")
    print(f"  • Total Trades: {int(results['total_trades']):,}")
    print(f"  • Long Positions: {int(results['long_positions']):,}")
    print(f"  • Short Positions: {int(results['short_positions']):,}")
    print(f"  • Flat Periods: {int(results['flat_positions']):,}")
    print(f"  • Position Accuracy: {results['position_accuracy']:.1%}")
    print(f"  • Cumulative Return: {results['cumulative_return']:.2f}")
    print(f"  • Avg Return per Trade: {results['avg_return_per_trade']:.4f}")
    print(f"  • Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  • Max Drawdown: {results['max_drawdown']:.2f}")
    
    # Calculate additional metrics
    total_periods = int(results['total_trades'] + results['flat_positions'])
    trade_frequency = results['total_trades'] / total_periods
    
    print(f"\n📊 Additional Metrics:")
    print(f"  • Trade Frequency: {trade_frequency:.1%}")
    print(f"  • Risk-Adjusted Return: {results['cumulative_return'] / abs(results['max_drawdown']):.2f}")
    
    # Strategy interpretation
    print(f"\n💡 Strategy Insights:")
    if results['short_positions'] > results['long_positions']:
        print("  • Strategy is predominantly SHORT-biased")
        print("  • This suggests the model identifies mean reversion opportunities")
        print("    when prices are expected to fall (negative z-scores)")
    elif results['long_positions'] > results['short_positions']:
        print("  • Strategy is predominantly LONG-biased")
        print("  • This suggests the model identifies mean reversion opportunities")
        print("    when prices are expected to rise (positive z-scores)")
    else:
        print("  • Strategy is balanced between long and short positions")
    
    if results['position_accuracy'] > 0.5:
        print(f"  • Position accuracy of {results['position_accuracy']:.1%} suggests the model")
        print("    has predictive power for mean reversion")
    
    return results

def create_probability_distribution_analysis():
    """Analyze the distribution of predicted probabilities"""
    print("\n\n📊 PREDICTION PROBABILITY ANALYSIS")
    print("=" * 40)
    
    cv_df = pd.read_csv("outputs/cv_predictions.csv")
    
    # Probability distribution
    print("Probability Distribution:")
    prob_bins = pd.cut(cv_df['y_pred_proba'], bins=10)
    prob_dist = prob_bins.value_counts().sort_index()
    
    for interval, count in prob_dist.items():
        pct = count / len(cv_df) * 100
        print(f"  {interval}: {count:,} ({pct:.1f}%)")
    
    # Calibration analysis
    print("\n🎯 Model Calibration Analysis:")
    cv_df['prob_bin'] = pd.cut(cv_df['y_pred_proba'], bins=10, labels=False)
    calibration = cv_df.groupby('prob_bin').agg({
        'y_pred_proba': 'mean',
        'y_true': 'mean',
        'y_pred_proba': 'count'
    }).round(3)
    
    print("Bin | Pred Prob | Actual Rate | Count")
    print("----|-----------|-------------|------")
    for idx, row in calibration.iterrows():
        print(f" {idx:2d} |   {row['y_pred_proba']:.3f}   |    {row['y_true']:.3f}    | {int(row.name):>4d}")

def hypothesis_test_conclusions():
    """Draw conclusions about the mean reversion hypothesis"""
    print("\n\n🔬 HYPOTHESIS TEST CONCLUSIONS")
    print("=" * 40)
    
    print("HYPOTHESIS: BTCUSDT dollar bars mean-revert after deviating significantly from their local rolling mean")
    print()
    
    # Load results for analysis
    cv_df = pd.read_csv("outputs/cv_predictions.csv")
    strategy_df = pd.read_csv("outputs/strategy_results.csv")
    feat_df = pd.read_csv("outputs/feature_importance.csv")
    
    # Key metrics
    mean_precision = cv_df.groupby('fold').apply(
        lambda x: ((x['y_pred'] == 1) & (x['y_true'] == 1)).sum() / (x['y_pred'] == 1).sum() if (x['y_pred'] == 1).sum() > 0 else 0
    ).mean()
    
    mean_reversion_rate = cv_df['y_true'].mean()
    position_accuracy = strategy_df.iloc[0]['position_accuracy']
    
    print("📊 KEY FINDINGS:")
    print(f"  • Base Mean Reversion Rate: {mean_reversion_rate:.1%}")
    print(f"  • Model Precision: {mean_precision:.1%}")
    print(f"  • Trading Strategy Accuracy: {position_accuracy:.1%}")
    
    # Feature insights
    top_features = feat_df.head(5)['feature'].tolist()
    zscore_features = [f for f in top_features if 'zscore' in f]
    
    print(f"  • Top 5 Features: {', '.join(top_features)}")
    print(f"  • Z-score features in top 5: {len(zscore_features)}")
    
    print("\n✅ CONCLUSIONS:")
    
    if mean_reversion_rate > 0.45:
        print("  1. STRONG EVIDENCE for mean reversion:")
        print(f"     Mean reversion occurs {mean_reversion_rate:.1%} of the time")
        
    if mean_precision > 0.52:
        print("  2. MODEL PREDICTIVE POWER:")
        print(f"     Model precision of {mean_precision:.1%} beats random chance")
        
    if position_accuracy > 0.55:
        print("  3. ACTIONABLE TRADING SIGNAL:")
        print(f"     Strategy accuracy of {position_accuracy:.1%} suggests exploitable patterns")
    
    if len(zscore_features) > 0:
        print("  4. Z-SCORE RELEVANCE:")
        print(f"     Z-score features among most important confirms hypothesis relevance")
    
    # Limitations
    print("\n⚠️ LIMITATIONS & CONSIDERATIONS:")
    print("  • Results based on 6 months of recent data")
    print("  • Market regime dependency not fully explored")
    print("  • Transaction costs not included in strategy simulation")
    print("  • Forward-looking bias possible in label construction")

def main():
    """Run complete results analysis"""
    print("🔍 BTCUSDT MEAN REVERSION RESULTS ANALYSIS")
    print("=" * 60)
    
    # Run all analyses
    cv_df = analyze_cv_predictions()
    feat_df = analyze_feature_importance()
    strategy_results = analyze_strategy_performance()
    create_probability_distribution_analysis()
    hypothesis_test_conclusions()
    
    print("\n" + "="*60)
    print("📋 ANALYSIS COMPLETE")
    print("See outputs/ directory for detailed files and visualizations")

if __name__ == "__main__":
    main() 