import duckdb
import pandas as pd
import os

# Setup
DATA_FOLDER = "../data"
CSV_FILE = os.path.join(DATA_FOLDER, "es_30m_data.csv")  # Update if needed
DB_FILE = "../data/market.db"
SYMBOL = "ES"

# Step 1: Load CSV with selected columns and rename
df = pd.read_csv(
    CSV_FILE,
    usecols=["Time", "Symbol", "Open", "High", "Low", "Last", "Volume"],
    parse_dates=["Time"]
)

# Standardize column names
df = df.rename(columns={
    "Time": "ts_utc",
    "Last": "close",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Volume": "volume"
})

# Overwrite symbol with standardized name (e.g., 'ES') instead of contract codes
df["symbol"] = SYMBOL

# Drop the original Symbol column since we have our standardized symbol column
df = df.drop(columns=["Symbol"])

# Drop rows with missing values (e.g., early contract data)
df = df.dropna(subset=["ts_utc", "open", "high", "low", "close", "volume"])

# Ensure columns are in the correct order for the database table
df = df[["ts_utc", "symbol", "open", "high", "low", "close", "volume"]]

# Step 2: Connect to DuckDB
con = duckdb.connect(DB_FILE)

# Step 3: Create the table if not already there
con.execute("""
CREATE TABLE IF NOT EXISTS bars_30m (
    ts_utc  TIMESTAMP,
    symbol  VARCHAR,
    open    DOUBLE,
    high    DOUBLE,
    low     DOUBLE,
    close   DOUBLE,
    volume  DOUBLE
)
""")

# Step 4: Insert cleaned data
con.execute("INSERT INTO bars_30m SELECT * FROM df")

# Step 5: Verify success
print(con.execute("SELECT COUNT(*), MIN(ts_utc), MAX(ts_utc) FROM bars_30m").fetchall())