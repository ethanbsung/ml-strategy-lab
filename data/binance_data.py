import os
import requests
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from tqdm import tqdm

def download_and_convert_binance_monthly(symbol='BTCUSDT', start_year=2017, end_year=None, base_dir='data'):
    base_url = 'https://data.binance.vision/data/spot/monthly/trades'
    if end_year is None:
        end_year = datetime.utcnow().year

    # Organize into data/<symbol>/
    root_dir = os.path.join(base_dir, symbol)
    zip_dir = os.path.join(root_dir, 'zips')
    csv_dir = os.path.join(root_dir, 'csv')
    parquet_dir = os.path.join(root_dir, 'parquet')

    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(parquet_dir, exist_ok=True)

    # Loop over years and months
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            date_str = f"{year}-{month:02d}"
            filename = f"{symbol}-trades-{date_str}"
            url = f"{base_url}/{symbol}/{filename}.zip"
            zip_path = os.path.join(zip_dir, filename + '.zip')
            csv_path = os.path.join(csv_dir, filename + '.csv')
            parquet_path = os.path.join(parquet_dir, filename + '.parquet')

            # Skip if already converted
            if os.path.exists(parquet_path):
                print(f"✅ Already converted: {filename}")
                continue

            try:
                # Download
                r = requests.get(url, stream=True, timeout=10)
                if r.status_code != 200:
                    print(f"❌ File not found: {url}")
                    continue

                with open(zip_path, 'wb') as f:
                    f.write(r.content)

                # Unzip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(csv_dir)

                # Convert to Parquet
                df = pd.read_csv(csv_path,
                                 names=['trade_id', 'price', 'qty', 'quote_qty', 'timestamp', 'is_buyer_maker', 'is_best_match'],
                                 dtype={'trade_id': 'int64', 'price': 'float64', 'qty': 'float64',
                                        'quote_qty': 'float64', 'timestamp': 'int64',
                                        'is_buyer_maker': 'bool', 'is_best_match': 'bool'})

                # 🔧 FIX: Convert timestamps from microseconds to datetime
                # Binance timestamps are in microseconds, so divide by 1,000,000 to get seconds
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'] / 1000000, unit='s')
                except Exception as ts_error:
                    print(f"⚠️ Timestamp conversion error for {filename}: {ts_error}")
                    # Fallback: try treating as milliseconds (for older data)
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        print(f"✅ Fallback to milliseconds worked for {filename}")
                    except Exception as fallback_error:
                        print(f"❌ Both timestamp conversions failed for {filename}: {fallback_error}")
                        continue
                
                table = pa.Table.from_pandas(df)
                pq.write_table(table, parquet_path, compression='zstd')

                print(f"✅ Done: {filename}")

                # Optional cleanup
                os.remove(zip_path)
                os.remove(csv_path)

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

# 🔧 Run it
download_and_convert_binance_monthly(
    symbol='ETHUSDT',
    start_year=2017,
    base_dir='data'
)