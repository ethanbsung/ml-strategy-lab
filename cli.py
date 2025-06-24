"""
Command Line Interface for Bitcoin Walk-Forward ML Backtesting System
"""
import typer
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import warnings
warnings.filterwarnings('ignore')

import config
from features.engineer import engineer_features
from labels.triple_barrier import create_triple_barrier_labels, align_labels_with_features
from backtest.walk_forward import WalkForwardBacktest

app = typer.Typer(help="Bitcoin Walk-Forward ML Backtesting System")
console = Console()


@app.command()
def features(
    save: bool = typer.Option(True, help="Save features to parquet file")
):
    """Create features from dollar bars data"""
    rprint("[bold blue]Creating features from dollar bars...[/bold blue]")
    
    try:
        X, y = engineer_features()
        
        if save:
            output_path = config.OUTPUT_PATH / "features.parquet"
            X.to_parquet(output_path)
            rprint(f"[green]✓[/green] Features saved to {output_path}")
        
        rprint(f"[green]✓[/green] Feature engineering completed successfully!")
        rprint(f"Shape: {X.shape}")
        rprint(f"Date range: {X.index.min()} to {X.index.max()}")
        rprint(f"Features: {list(X.columns)}")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error creating features: {e}")
        raise typer.Exit(1)


@app.command()
def label(
    features_file: Optional[str] = typer.Option(None, help="Path to features parquet file"),
    save: bool = typer.Option(True, help="Save labels to parquet file")
):
    """Create triple barrier labels"""
    rprint("[bold blue]Creating triple barrier labels...[/bold blue]")
    
    try:
        # Load features
        if features_file:
            X = pd.read_parquet(features_file)
            rprint(f"Loaded features from {features_file}")
        else:
            rprint("Creating features...")
            X, _ = engineer_features()
        
        # Create price data for labeling
        price_df = pd.DataFrame(index=X.index)
        price_df['close'] = np.exp(X['log_ret'].cumsum()) * 50000
        
        # Create labels
        labels_df = create_triple_barrier_labels(price_df)
        
        # Align with features
        X_aligned, y_aligned = align_labels_with_features(X, labels_df)
        
        if save:
            labels_path = config.OUTPUT_PATH / "labels.parquet"
            y_aligned.to_parquet(labels_path)
            rprint(f"[green]✓[/green] Labels saved to {labels_path}")
        
        rprint(f"[green]✓[/green] Triple barrier labeling completed!")
        rprint(f"Labels shape: {y_aligned.shape}")
        
        # Print label distribution
        label_counts = y_aligned['label'].value_counts().sort_index()
        rprint("\nLabel distribution:")
        for label, count in label_counts.items():
            pct = count / len(y_aligned) * 100
            label_name = {-1: "Stop", 0: "Timeout", 1: "Profit"}[label]
            rprint(f"  {label_name} ({label}): {count:,} ({pct:.1f}%)")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error creating labels: {e}")
        raise typer.Exit(1)


@app.command()
def train(
    window_idx: int = typer.Option(0, help="Window index for training (0-based)"),
    save_models: bool = typer.Option(True, help="Save trained models")
):
    """Train models on the first window"""
    rprint(f"[bold blue]Training models on window {window_idx}...[/bold blue]")
    
    try:
        # Initialize backtest engine to prepare data
        backtest = WalkForwardBacktest()
        if not backtest.prepare_data():
            rprint("[red]✗[/red] Failed to prepare data")
            raise typer.Exit(1)
        
        # Get data splits for the specified window
        start_idx = window_idx * config.STEP
        end_idx = start_idx + config.WINDOW
        
        if end_idx > len(backtest.sequences['X']):
            rprint(f"[red]✗[/red] Window {window_idx} exceeds available data")
            raise typer.Exit(1)
        
        rprint(f"Window range: [{start_idx}:{end_idx}]")
        
        # Split data
        splits = backtest.split_window(start_idx, end_idx)
        
        # Train models
        models = backtest.train_models(splits)
        
        # Evaluate models
        rprint("\n[bold]Model Evaluation:[/bold]")
        
        for model_name, model in models.items():
            rprint(f"\n{model_name.upper()} Model:")
            
            if model_name == 'cnn_lstm':
                X_test, y_test = splits['sequences']['test']
                results = model.evaluate(X_test, y_test, verbose=1)
            else:  # Random Forest
                X_test, y_test = splits['flat']['test']
                results = model.evaluate(X_test, y_test, verbose=1)
            
            if save_models:
                model_path = config.OUTPUT_PATH / f"{model_name}_window_{window_idx}"
                if model_name == 'cnn_lstm':
                    model.save_model(str(model_path) + ".h5")
                else:
                    model.save_model(str(model_path) + ".pkl")
                rprint(f"[green]✓[/green] {model_name} model saved")
        
        rprint(f"[green]✓[/green] Training completed successfully!")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error in training: {e}")
        raise typer.Exit(1)


@app.command()
def walk_forward(
    fast: bool = typer.Option(False, help="Run with reduced parameters for faster execution"),
    save_results: bool = typer.Option(True, help="Save results to pickle file")
):
    """Run full walk-forward backtest"""
    rprint("[bold blue]Starting walk-forward backtest...[/bold blue]")
    
    if fast:
        rprint("[yellow]⚡[/yellow] Fast mode: Using reduced parameters")
        # Temporarily modify config for faster execution
        original_window = config.WINDOW
        original_step = config.STEP
        original_epochs = config.EPOCHS
        
        config.WINDOW = 1000
        config.STEP = 250
        config.EPOCHS = 10
    
    try:
        # Run backtest
        backtest = WalkForwardBacktest()
        results = backtest.run_backtest()
        
        if save_results:
            results_path = config.OUTPUT_PATH / "walk_forward_results.pkl"
            backtest.save_results(results, results_path)
            rprint(f"[green]✓[/green] Results saved to {results_path}")
        
        # Print summary
        if 'performance' in results:
            print_performance_summary(results['performance'])
        
        rprint(f"[green]✓[/green] Walk-forward backtest completed!")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error in walk-forward backtest: {e}")
        raise typer.Exit(1)
    
    finally:
        if fast:
            # Restore original config
            config.WINDOW = original_window
            config.STEP = original_step
            config.EPOCHS = original_epochs


@app.command()
def plot(
    results_file: Optional[str] = typer.Option(None, help="Path to results pickle file"),
    save_plots: bool = typer.Option(True, help="Save plots to files"),
    show_plots: bool = typer.Option(False, help="Display plots in browser")
):
    """Generate equity curves and performance plots"""
    rprint("[bold blue]Generating plots...[/bold blue]")
    
    try:
        # Load results
        if results_file:
            results_path = Path(results_file)
        else:
            results_path = config.OUTPUT_PATH / "walk_forward_results.pkl"
        
        if not results_path.exists():
            rprint(f"[red]✗[/red] Results file not found: {results_path}")
            rprint("Run 'python cli.py walk-forward' first to generate results")
            raise typer.Exit(1)
        
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        # Create plots
        create_equity_curve_plot(results, save_plots, show_plots)
        create_performance_comparison_plot(results, save_plots, show_plots)
        
        if 'performance' in results:
            print_performance_summary(results['performance'])
        
        rprint(f"[green]✓[/green] Plots generated successfully!")
        
    except Exception as e:
        rprint(f"[red]✗[/red] Error generating plots: {e}")
        raise typer.Exit(1)


def print_performance_summary(performance: dict):
    """Print formatted performance summary"""
    rprint("\n[bold]WALK-FORWARD BACKTEST RESULTS[/bold]")
    rprint("=" * 60)
    
    # Create comparison table
    table = Table(title="Performance Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("CNN-LSTM", style="magenta")
    table.add_column("Random Forest", style="green")
    
    metrics_display = {
        'total_return': ('Total Return', '.2%'),
        'annual_return': ('Annual Return', '.2%'),
        'sharpe_ratio': ('Sharpe Ratio', '.2f'),
        'max_drawdown': ('Max Drawdown', '.2%'),
        'hit_rate': ('Hit Rate', '.2%'),
        'profit_factor': ('Profit Factor', '.2f'),
        'n_trades': ('Number of Trades', ':,')
    }
    
    for metric_key, (metric_name, fmt) in metrics_display.items():
        cnn_val = performance.get('cnn_lstm', {}).get(metric_key, 0)
        rf_val = performance.get('rf', {}).get(metric_key, 0)
        
        if fmt == ':,':
            cnn_str = f"{cnn_val:,}"
            rf_str = f"{rf_val:,}"
        else:
            cnn_str = f"{cnn_val:{fmt}}"
            rf_str = f"{rf_val:{fmt}}"
        
        table.add_row(metric_name, cnn_str, rf_str)
    
    console.print(table)


def create_equity_curve_plot(results: dict, save: bool = True, show: bool = False):
    """Create equity curve plot"""
    if 'equity_curves' not in results:
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative Returns', 'Drawdown'),
        vertical_spacing=0.1
    )
    
    colors = {'cnn_lstm': '#1f77b4', 'rf': '#ff7f0e'}
    
    for model_name, equity_curve in results['equity_curves'].items():
        if len(equity_curve) == 0:
            continue
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=list(range(len(equity_curve))),
                y=equity_curve,
                name=f'{model_name.upper()}',
                line=dict(color=colors.get(model_name, '#000000'))
            ),
            row=1, col=1
        )
        
        # Drawdown
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(drawdown))),
                y=drawdown,
                name=f'{model_name.upper()} DD',
                line=dict(color=colors.get(model_name, '#000000')),
                fill='tonexty' if model_name == 'cnn_lstm' else None
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title="Walk-Forward Backtest Results",
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    
    if save:
        plot_path = config.OUTPUT_PATH / "equity_curves.html"
        fig.write_html(plot_path)
        rprint(f"[green]✓[/green] Equity curve plot saved to {plot_path}")
    
    if show:
        fig.show()


def create_performance_comparison_plot(results: dict, save: bool = True, show: bool = False):
    """Create performance comparison bar plot"""
    if 'performance' not in results:
        return
    
    # Create matplotlib subplot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Performance Comparison: CNN-LSTM vs Random Forest', fontsize=16)
    
    performance = results['performance']
    models = list(performance.keys())
    
    metrics = [
        ('total_return', 'Total Return (%)', 100),
        ('annual_return', 'Annual Return (%)', 100),
        ('sharpe_ratio', 'Sharpe Ratio', 1),
        ('max_drawdown', 'Max Drawdown (%)', 100),
        ('hit_rate', 'Hit Rate (%)', 100),
        ('profit_factor', 'Profit Factor', 1)
    ]
    
    for idx, (metric, title, scale) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = [performance[model].get(metric, 0) * scale for model in models]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = ax.bar(models, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(title.split('(')[0].strip())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save:
        plot_path = config.OUTPUT_PATH / "performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        rprint(f"[green]✓[/green] Performance comparison plot saved to {plot_path}")
    
    if show:
        plt.show()
    
    plt.close()


@app.command()
def status():
    """Show system status and configuration"""
    rprint("[bold blue]Bitcoin Walk-Forward ML Backtesting System Status[/bold blue]\n")
    
    # Configuration
    config_table = Table(title="Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_items = [
        ("Data Path", str(config.DATA_PATH)),
        ("Output Path", str(config.OUTPUT_PATH)),
        ("Window Size", f"{config.WINDOW:,}"),
        ("Step Size", f"{config.STEP:,}"),
        ("Lookback", f"{config.LOOKBACK}"),
        ("Profit Target", f"{config.PROFIT_TGT:.4f}"),
        ("Stop Target", f"{config.STOP_TGT:.4f}"),
        ("Probability Threshold", f"{config.P_THRESH:.2f}"),
        ("Random Seed", f"{config.RANDOM_SEED}")
    ]
    
    for param, value in config_items:
        config_table.add_row(param, value)
    
    console.print(config_table)
    
    # File status
    rprint("\n[bold]File Status:[/bold]")
    files_to_check = [
        ("Features", config.OUTPUT_PATH / "features.parquet"),
        ("Labels", config.OUTPUT_PATH / "labels.parquet"),
        ("Results", config.OUTPUT_PATH / "walk_forward_results.pkl"),
        ("Equity Plot", config.OUTPUT_PATH / "equity_curves.html"),
        ("Performance Plot", config.OUTPUT_PATH / "performance_comparison.png")
    ]
    
    for name, filepath in files_to_check:
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            rprint(f"  [green]✓[/green] {name}: {filepath} ({size:.1f} MB)")
        else:
            rprint(f"  [red]✗[/red] {name}: Not found")
    
    # Data availability
    rprint(f"\n[bold]Data Status:[/bold]")
    try:
        import glob
        parquet_files = glob.glob(str(config.DATA_PATH / "*.parquet"))
        if parquet_files:
            rprint(f"  [green]✓[/green] Found {len(parquet_files)} dollar bar files")
            # Get date range
            first_file = sorted(parquet_files)[0]
            last_file = sorted(parquet_files)[-1]
            rprint(f"  Date range: {Path(first_file).stem} to {Path(last_file).stem}")
        else:
            rprint(f"  [red]✗[/red] No dollar bar files found in {config.DATA_PATH}")
    except Exception as e:
        rprint(f"  [red]✗[/red] Error checking data: {e}")


if __name__ == "__main__":
    app() 