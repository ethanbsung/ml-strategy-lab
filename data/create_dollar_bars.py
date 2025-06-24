"""
Dollar Bar Generation for BTCUSDT Trade Data

This module converts tick-by-tick trade data into dollar bars - bars that close when
a certain dollar volume threshold is reached. Dollar bars are superior to time bars
for financial analysis as they adapt to market activity.

Features:
- Memory-efficient chunked processing
- Configurable dollar thresholds  
- OHLCV + additional statistics
- Preserves all metadata (timestamps, trade counts, etc.)
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

class DollarBarGenerator:
    """
    Generates dollar bars from tick data with memory-efficient processing
    """
    
    def __init__(self, symbol='BTCUSDT', dollar_threshold=1000000, output_dir=None):
        """
        Initialize dollar bar generator
        
        Args:
            symbol: The symbol to process (e.g., 'BTCUSDT', 'ETHUSDT')
            dollar_threshold: Dollar volume to trigger a new bar (default: $1M)
            output_dir: Directory to save dollar bar parquet files. If None, it's auto-generated.
        """
        self.symbol = symbol
        self.dollar_threshold = dollar_threshold
        if output_dir is None:
            self.output_dir = f'data/{self.symbol}/dollar_bars'
        else:
            self.output_dir = output_dir
            
        self.current_bar = None
        self.cumulative_dollar_volume = 0
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize the first bar
        self.reset_bar()
        
        print(f"🔧 {self.symbol} Dollar Bar Generator initialized")
        print(f"   Threshold: ${dollar_threshold:,} per bar")
        print(f"   Output: {self.output_dir}")
    
    def reset_bar(self):
        """Reset current bar accumulator"""
        self.current_bar = {
            'bar_start_time': None,
            'bar_end_time': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0,  # Total quantity traded
            'dollar_volume': 0,  # Total dollar volume
            'trade_count': 0,
            'vwap': 0,  # Volume weighted average price
            'first_trade_id': None,
            'last_trade_id': None,
            'buyer_initiated_volume': 0,
            'seller_initiated_volume': 0
        }
        self.cumulative_dollar_volume = 0
    
    def add_trade_to_bar(self, trade):
        """
        Add a single trade to the current bar
        
        Args:
            trade: Single row from trade data (pandas Series)
        """
        if self.current_bar['bar_start_time'] is None:
            # Initialize new bar
            self.current_bar['bar_start_time'] = trade['timestamp']
            self.current_bar['open'] = trade['price']
            self.current_bar['high'] = trade['price']
            self.current_bar['low'] = trade['price']
            self.current_bar['first_trade_id'] = trade['trade_id']
        
        # Update OHLC
        self.current_bar['bar_end_time'] = trade['timestamp']
        self.current_bar['close'] = trade['price']
        self.current_bar['high'] = max(self.current_bar['high'], trade['price'])
        self.current_bar['low'] = min(self.current_bar['low'], trade['price'])
        
        # Update volumes
        trade_dollar_volume = trade['price'] * trade['qty']
        self.current_bar['volume'] += trade['qty']
        self.current_bar['dollar_volume'] += trade_dollar_volume
        self.current_bar['trade_count'] += 1
        self.current_bar['last_trade_id'] = trade['trade_id']
        
        # Track buyer vs seller initiated volume
        if trade['is_buyer_maker']:
            self.current_bar['seller_initiated_volume'] += trade['qty']  # Market sell order
        else:
            self.current_bar['buyer_initiated_volume'] += trade['qty']   # Market buy order
        
        # Update cumulative dollar volume for threshold check
        self.cumulative_dollar_volume += trade_dollar_volume
    
    def is_bar_complete(self):
        """Check if current bar has reached the dollar threshold"""
        return self.cumulative_dollar_volume >= self.dollar_threshold
    
    def finalize_bar(self):
        """
        Complete the current bar and calculate final statistics
        
        Returns:
            dict: Completed bar data
        """
        if self.current_bar['bar_start_time'] is None:
            return None
        
        # Calculate VWAP
        if self.current_bar['volume'] > 0:
            self.current_bar['vwap'] = self.current_bar['dollar_volume'] / self.current_bar['volume']
        else:
            self.current_bar['vwap'] = self.current_bar['close']
        
        # Calculate order flow imbalance
        total_volume = self.current_bar['volume']
        if total_volume > 0:
            self.current_bar['buy_volume_ratio'] = self.current_bar['buyer_initiated_volume'] / total_volume
        else:
            self.current_bar['buy_volume_ratio'] = 0.5
        
        # Calculate bar duration in seconds
        duration = (self.current_bar['bar_end_time'] - self.current_bar['bar_start_time']).total_seconds()
        self.current_bar['duration_seconds'] = duration
        
        # Calculate volatility within bar (high-low range)
        if self.current_bar['close'] > 0:
            self.current_bar['hl_ratio'] = (self.current_bar['high'] - self.current_bar['low']) / self.current_bar['close']
        else:
            self.current_bar['hl_ratio'] = 0
        
        completed_bar = self.current_bar.copy()
        return completed_bar
    
    def process_trades_chunk(self, trades_df):
        """
        Process a chunk of trades and return completed bars
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            list: List of completed dollar bars
        """
        completed_bars = []
        
        for _, trade in trades_df.iterrows():
            self.add_trade_to_bar(trade)
            
            if self.is_bar_complete():
                # Complete current bar
                completed_bar = self.finalize_bar()
                if completed_bar:
                    completed_bars.append(completed_bar)
                
                # Start new bar
                self.reset_bar()
        
        return completed_bars
    
    def process_monthly_file(self, parquet_file):
        """
        Process a single monthly parquet file
        
        Args:
            parquet_file: Path to parquet file
            
        Returns:
            pd.DataFrame: Dollar bars for this month
        """
        print(f"Processing {os.path.basename(parquet_file)}...")
        
        # Read the file
        df = pd.read_parquet(parquet_file)
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Process in smaller chunks to manage memory
        chunk_size = 1000000  # 1M trades per chunk
        all_bars = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            bars = self.process_trades_chunk(chunk)
            all_bars.extend(bars)
            
            if i % (chunk_size * 5) == 0:  # Progress every 5M trades
                print(f"  Processed {i:,} / {len(df):,} trades...")
        
        # Convert to DataFrame
        if all_bars:
            bars_df = pd.DataFrame(all_bars)
            return bars_df
        else:
            return pd.DataFrame()
    
    def generate_dollar_bars(self, start_year=None, end_year=None):
        """
        Generate dollar bars for all available data
        
        Args:
            start_year: Start year (default: all available)
            end_year: End year (default: all available)
        """
        # Get all parquet files for the specified symbol
        input_dir = f'data/{self.symbol}/parquet'
        parquet_files = sorted(glob.glob(f'{input_dir}/*.parquet'))

        if not parquet_files:
            print(f"⚠️  No parquet files found for {self.symbol} in '{input_dir}'")
            print(f"   Please ensure your raw data is in the correct directory.")
            return 0
        
        # Filter by year if specified
        if start_year or end_year:
            filtered_files = []
            for f in parquet_files:
                filename = os.path.basename(f)
                year = int(filename.split('-')[2])
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue
                filtered_files.append(f)
            parquet_files = filtered_files
        
        print(f"🚀 Generating dollar bars for {len(parquet_files)} files...")
        print(f"   Threshold: ${self.dollar_threshold:,} per bar")
        
        total_bars_created = 0
        
        # Process each monthly file
        for file_path in tqdm(parquet_files, desc="Processing months"):
            filename = os.path.basename(file_path).replace('.parquet', '')
            
            # Generate bars for this month
            monthly_bars = self.process_monthly_file(file_path)
            
            if len(monthly_bars) > 0:
                # Save monthly dollar bars
                output_file = f"{self.output_dir}/{filename}_dollar_bars.parquet"
                monthly_bars.to_parquet(output_file, index=False)
                
                total_bars_created += len(monthly_bars)
                print(f"  ✅ {filename}: {len(monthly_bars):,} dollar bars saved")
            else:
                print(f"  ⚠️  {filename}: No complete bars generated")
        
        print(f"\\n🎉 Dollar bar generation complete!")
        print(f"   Total bars created: {total_bars_created:,}")
        print(f"   Average trades per bar: {4_847_615_740 / total_bars_created:.0f}")
        print(f"   Files saved to: {self.output_dir}")
        
        return total_bars_created


def main():
    """Example usage"""

    # --- Configuration ---
    # Change the symbol to 'BTCUSDT', 'ETHUSDT', etc. as needed
    target_symbol = 'ETHUSDT' 
    
    # Set the desired dollar volume threshold for each bar
    dollar_threshold = 500_000  # e.g., $500k bars
    # -------------------

    print(f"🚀 {target_symbol} DOLLAR BAR GENERATION")
    print("=" * 50)
    
    # Auto-generate a descriptive output directory name
    output_dir_name = f'dollar_bars_{dollar_threshold / 1_000_000:.0f}M'
    output_dir = f'data/{target_symbol}/{output_dir_name}'

    # Initialize the generator with our configuration
    generator = DollarBarGenerator(
        symbol=target_symbol,
        dollar_threshold=dollar_threshold,
        output_dir=output_dir
    )
    
    # Generate bars for all data starting from a specific year
    # You can remove the start_year argument to process all available files
    generator.generate_dollar_bars(start_year=2017)


if __name__ == "__main__":
    main() 