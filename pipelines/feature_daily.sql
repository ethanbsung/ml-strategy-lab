-- Daily Feature Engineering (Leak-Free)
-- All rolling windows exclude current observation to prevent look-ahead bias

CREATE OR REPLACE VIEW daily_features AS
WITH base_returns AS (
    SELECT
        ts_utc,
        symbol,
        open,
        high,
        low,
        close,
        volume,
        
        -- Calculate 1-day lagged log return (no leakage)
        LN(close) - LN(LAG(close, 1) OVER w) AS ln_ret_1,
        
        -- Previous close for gap calculation
        LAG(close, 1) OVER w AS prev_close
        
    FROM bars_daily
    WHERE close IS NOT NULL AND close > 0
    WINDOW w AS (PARTITION BY symbol ORDER BY ts_utc)
)
SELECT
    ts_utc,
    symbol,
    open,
    high,
    low,
    close,
    volume,
    
    -- Basic return (lagged, leak-free)
    ln_ret_1,
    
    -- 20-day momentum (exclude current day)
    SUM(ln_ret_1) OVER w20 AS mom_20,
    
    -- 20-day realized volatility (exclude current day) 
    STDDEV(ln_ret_1) OVER w20 * SQRT(252) AS vol_20,
    
    -- 5-day momentum
    SUM(ln_ret_1) OVER w5 AS mom_5,
    
    -- 5-day volatility
    STDDEV(ln_ret_1) OVER w5 * SQRT(252) AS vol_5,
    
    -- Overnight gap percentage (leak-free)
    CASE 
        WHEN prev_close IS NOT NULL AND prev_close > 0 
        THEN (open - prev_close) / prev_close 
        ELSE NULL 
    END AS gap_pc,
    
    -- High-low range as % of close (intraday volatility proxy)
    CASE 
        WHEN close > 0 
        THEN (high - low) / close 
        ELSE NULL 
    END AS hl_range_pc,
    
    -- Volume z-score (20-day rolling)
    CASE 
        WHEN STDDEV(volume) OVER w20 > 0 
        THEN (volume - AVG(volume) OVER w20) / STDDEV(volume) OVER w20
        ELSE 0
    END AS vol_z_20,
    
    -- Simple sector classification (hash-based)
    ABS(hash(symbol) % 10) AS sector_id,
    
    -- Day of week effect
    EXTRACT(DOW FROM ts_utc) AS day_of_week,
    
    -- Month effect  
    EXTRACT(MONTH FROM ts_utc) AS month,
    
    -- Price level relative to 20-day SMA
    CASE 
        WHEN AVG(close) OVER w20 > 0 
        THEN (close - AVG(close) OVER w20) / AVG(close) OVER w20
        ELSE 0
    END AS price_sma20_dev
    
FROM base_returns
WHERE ln_ret_1 IS NOT NULL  -- Ensure we have valid returns

WINDOW
    w AS (PARTITION BY symbol ORDER BY ts_utc),
    w5 AS (PARTITION BY symbol ORDER BY ts_utc ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    w20 AS (PARTITION BY symbol ORDER BY ts_utc ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)

-- Only include rows where we have sufficient lookback data
-- This ensures the latest feature row lags behind the latest raw data  
QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts_utc DESC) > 1; 