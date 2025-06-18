"""
Load raw bars data into DuckDB (one-time setup)
Handles both daily and 5-minute data with robust CSV parsing and error handling
"""

import duckdb
import glob
import os
import pandas as pd
from pathlib import Path

def setup_database():
    """Initialize DuckDB with proper schemas for bars data"""
    
    DB_PATH = "lake/market.db"
    
    # Ensure lake directory exists
    Path("lake").mkdir(exist_ok=True)
    
    con = duckdb.connect(DB_PATH)
    
    # Create tables with proper schemas
    daily_schema = """
    CREATE TABLE IF NOT EXISTS bars_daily (
        ts_utc TIMESTAMP,
        symbol VARCHAR,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        open_int BIGINT
    );
    """
    
    fivemin_schema = """
    CREATE TABLE IF NOT EXISTS bars_5m (
        ts_utc TIMESTAMP,
        symbol VARCHAR,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        open_int BIGINT
    );
    """
    
    con.execute("DROP TABLE IF EXISTS bars_daily;")
    con.execute("DROP TABLE IF EXISTS bars_5m;")
    con.execute(daily_schema)
    con.execute(fivemin_schema)
    
    return con

def load_csv_robust(file_path):
    """Robust CSV loading with error handling"""
    
    try:
        # Try standard CSV reading first
        df = pd.read_csv(file_path)
        return df
    except Exception as e1:
        try:
            # Try with different encoding
            df = pd.read_csv(file_path, encoding='latin-1')
            return df
        except Exception as e2:
            try:
                # Try with error handling
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                return df
            except Exception as e3:
                print(f"  Failed to load {file_path}: {e3}")
                return None

def clean_numeric_column(series):
    """Clean numeric columns by removing non-numeric values"""
    # Convert to string, remove non-numeric characters, then to float
    cleaned = pd.to_numeric(series.astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    return cleaned

def load_daily_data(con):
    """Load all daily CSV files into bars_daily table with robust error handling"""
    
    print("Loading daily data...")
    daily_files = glob.glob("data/*_daily_data.csv")
    
    total_loaded = 0
    successful = 0
    
    for i, file_path in enumerate(daily_files):
        # Extract symbol from filename
        base_name = os.path.basename(file_path)
        symbol = base_name.replace("_daily_data.csv", "").upper()
        
        print(f"Processing {symbol} ({i+1}/{len(daily_files)})...")
        
        # Load CSV with robust error handling
        df = load_csv_robust(file_path)
        
        if df is None:
            continue
            
        try:
            # Check if required columns exist
            if 'Time' not in df.columns:
                print(f"  Skipping {symbol}: No 'Time' column found")
                continue
                
            # Determine price column name (could be 'Last', 'Close', etc.)
            price_col = None
            for col in ['Last', 'Close', 'close']:
                if col in df.columns:
                    price_col = col
                    break
                    
            if price_col is None:
                print(f"  Skipping {symbol}: No price column found")
                continue
            
            # Clean and prepare data
            df = df.copy()
            df = df.dropna(subset=['Time', price_col])
            
            # Clean numeric columns
            df['Open'] = clean_numeric_column(df.get('Open', 0))
            df['High'] = clean_numeric_column(df.get('High', 0)) 
            df['Low'] = clean_numeric_column(df.get('Low', 0))
            df[price_col] = clean_numeric_column(df[price_col])
            df['Volume'] = pd.to_numeric(df.get('Volume', 0), errors='coerce').fillna(0)
            
            # Handle Open Interest column (different possible names)
            oi_col = None
            for col in ['Open Int', 'OpenInt', 'OI']:
                if col in df.columns:
                    oi_col = col
                    break
            
            if oi_col:
                df['OpenInt'] = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)
            else:
                df['OpenInt'] = 0
            
            # Try to parse dates
            try:
                df['ts_utc'] = pd.to_datetime(df['Time'], errors='coerce')
            except:
                print(f"  Skipping {symbol}: Cannot parse dates")
                continue
            
            # Filter valid rows
            df = df.dropna(subset=['ts_utc', price_col])
            df = df[df[price_col] > 0]  # Remove zero/negative prices
            
            if len(df) == 0:
                print(f"  Skipping {symbol}: No valid data after cleaning")
                continue
            
            # Prepare final dataset
            final_df = pd.DataFrame({
                'ts_utc': df['ts_utc'],
                'symbol': symbol,
                'open': df['Open'],
                'high': df['High'], 
                'low': df['Low'],
                'close': df[price_col],
                'volume': df['Volume'].astype('int64'),
                'open_int': df['OpenInt'].astype('int64')
            })
            
            # Insert into database
            con.execute("INSERT INTO bars_daily SELECT * FROM final_df")
            
            total_loaded += len(final_df)
            successful += 1
            print(f"  ✅ Loaded {len(final_df)} bars for {symbol}")
            
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
            continue
    
    print(f"✅ Daily data complete: {successful}/{len(daily_files)} files, {total_loaded:,} total bars")

def load_5min_data(con):
    """Load 5-minute CSV files into bars_5m table with robust error handling"""
    
    print("Loading 5-minute data...")
    fivemin_files = glob.glob("data/*_5m_*.csv")
    
    if len(fivemin_files) == 0:
        print("No 5-minute files found")
        return
    
    total_loaded = 0
    
    for file_path in fivemin_files:
        # Extract symbol from filename
        base_name = os.path.basename(file_path)
        symbol = base_name.split("_")[0].upper()
        
        print(f"Processing 5m data for {symbol}...")
        
        # Load CSV with robust error handling
        df = load_csv_robust(file_path)
        
        if df is None:
            continue
            
        try:
            # Check columns
            if 'Time' not in df.columns:
                print(f"  Skipping {symbol}: No 'Time' column")
                continue
            
            # Find price column
            price_col = None
            for col in ['Last', 'Close', 'close']:
                if col in df.columns:
                    price_col = col
                    break
                    
            if price_col is None:
                print(f"  Skipping {symbol}: No price column")
                continue
            
            # Clean data
            df = df.copy()
            df = df.dropna(subset=['Time', price_col])
            
            # Clean numeric columns  
            df['Open'] = clean_numeric_column(df.get('Open', 0))
            df['High'] = clean_numeric_column(df.get('High', 0))
            df['Low'] = clean_numeric_column(df.get('Low', 0))
            df[price_col] = clean_numeric_column(df[price_col])
            df['Volume'] = pd.to_numeric(df.get('Volume', 0), errors='coerce').fillna(0)
            
            # Parse timestamps
            try:
                df['ts_utc'] = pd.to_datetime(df['Time'], errors='coerce')
            except:
                print(f"  Skipping {symbol}: Cannot parse timestamps")
                continue
            
            # Filter valid data
            df = df.dropna(subset=['ts_utc', price_col])
            df = df[df[price_col] > 0]
            
            if len(df) == 0:
                print(f"  Skipping {symbol}: No valid 5m data")
                continue
            
            # Prepare final dataset
            final_df = pd.DataFrame({
                'ts_utc': df['ts_utc'],
                'symbol': symbol,
                'open': df['Open'],
                'high': df['High'],
                'low': df['Low'], 
                'close': df[price_col],
                'volume': df['Volume'].astype('int64'),
                'open_int': 0  # 5m data typically doesn't have OI
            })
            
            # Insert into database
            con.execute("INSERT INTO bars_5m SELECT * FROM final_df")
            
            total_loaded += len(final_df)
            print(f"  ✅ Loaded {len(final_df)} 5m bars for {symbol}")
            
        except Exception as e:
            print(f"  ❌ Error processing 5m {symbol}: {e}")
            continue
    
    print(f"✅ 5-minute data complete: {total_loaded:,} total bars")

def validate_data(con):
    """Basic validation of loaded data"""
    
    print("\n📊 Data Validation:")
    
    # Daily data stats
    try:
        daily_stats = con.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT symbol) as unique_symbols,
                MIN(ts_utc) as earliest_date,
                MAX(ts_utc) as latest_date,
                SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_closes
            FROM bars_daily
        """).fetchone()
        
        if daily_stats[0] > 0:
            print(f"Daily bars: {daily_stats[0]:,} rows, {daily_stats[1]} symbols")
            print(f"Date range: {daily_stats[2]} to {daily_stats[3]}")
            print(f"Null closes: {daily_stats[4]}")
            
            # Show sample symbols
            symbols = con.execute("SELECT DISTINCT symbol FROM bars_daily LIMIT 10").fetchall()
            print(f"Sample symbols: {', '.join([s[0] for s in symbols])}")
        else:
            print("❌ No daily data loaded")
    except Exception as e:
        print(f"Error validating daily data: {e}")
    
    # 5-minute data stats
    try:
        fivemin_count = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
        if fivemin_count > 0:
            fivemin_stats = con.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(ts_utc) as earliest_datetime,
                    MAX(ts_utc) as latest_datetime
                FROM bars_5m
            """).fetchone()
            
            print(f"5-min bars: {fivemin_count:,} rows, {fivemin_stats[0]} symbols")
            print(f"Datetime range: {fivemin_stats[1]} to {fivemin_stats[2]}")
        else:
            print("No 5-minute data loaded")
    except Exception as e:
        print(f"Error validating 5m data: {e}")

def main():
    """Main execution function"""
    
    print("🚀 Setting up leak-free Data Lake...")
    
    con = setup_database()
    
    # Load raw data
    load_daily_data(con)
    load_5min_data(con)
    
    # Validate results
    validate_data(con)
    
    con.close()
    print("\n✅ Raw data loading complete!")
    print("Next: Run build_features.py to create feature views")

if __name__ == "__main__":
    main() 