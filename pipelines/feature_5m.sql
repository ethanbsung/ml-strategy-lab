-- 5-Minute Feature Engineering (Leak-Free)  
-- All rolling windows exclude current observation to prevent look-ahead bias
-- Focus on intraday patterns, volatility regimes, and microstructure

CREATE OR REPLACE VIEW fivemin_features AS
WITH base_features AS (
    SELECT
        ts_utc,
        symbol,
        open,
        high,
        low,
        close,
        volume,
        
        -- 1-period lagged log return (5-min, leak-free)
        LN(close) - LN(LAG(close, 1) OVER w) AS ln_ret_1,
        
        -- Previous close for calculations
        LAG(close, 1) OVER w AS prev_close,
        
        -- Time-of-day features
        EXTRACT(HOUR FROM ts_utc) AS hour,
        EXTRACT(MINUTE FROM ts_utc) AS minute,
        EXTRACT(DOW FROM ts_utc) AS day_of_week
        
    FROM bars_5m
    WHERE close IS NOT NULL AND close > 0
    WINDOW w AS (PARTITION BY symbol ORDER BY ts_utc)
),
enriched_features AS (
    SELECT
        *,
        
        -- Intraday session indicators
        CASE 
            WHEN hour >= 9 AND hour < 16 THEN 1  -- Regular trading hours (approx)
            ELSE 0 
        END AS is_regular_hours,
        
        CASE 
            WHEN hour >= 9 AND hour < 10 THEN 1  -- Market open hour
            ELSE 0 
        END AS is_open_hour,
        
        CASE 
            WHEN hour >= 15 AND hour < 16 THEN 1  -- Market close hour
            ELSE 0 
        END AS is_close_hour,
        
        -- Minute-of-hour buckets for finer time effects
        CASE 
            WHEN minute < 15 THEN 0
            WHEN minute < 30 THEN 1
            WHEN minute < 45 THEN 2
            ELSE 3
        END AS minute_bucket
        
    FROM base_features
    WHERE ln_ret_1 IS NOT NULL
)
SELECT
    ts_utc,
    symbol,
    open,
    high,
    low, 
    close,
    volume,
    hour,
    minute,
    day_of_week,
    is_regular_hours,
    is_open_hour,
    is_close_hour,
    minute_bucket,
    
    -- Basic return (lagged, leak-free)
    ln_ret_1,
    
    -- Short-term momentum (1-hour = 12 5-min bars, exclude current)
    SUM(ln_ret_1) OVER w12 AS mom_1h,
    
    -- 1-hour simple moving average (exclude current)
    AVG(close) OVER w12 AS sma_1h,
    
    -- Distance from 1-hour SMA (momentum/mean reversion signal)
    CASE 
        WHEN AVG(close) OVER w12 > 0 
        THEN (close - AVG(close) OVER w12) / AVG(close) OVER w12
        ELSE NULL 
    END AS dist_sma1h,
    
    -- 6-hour volatility regime (72 bars, exclude current)
    STDDEV(ln_ret_1) OVER w72 * SQRT(72) AS vol_6h,
    
    -- 1-hour volatility (12 bars, exclude current) 
    STDDEV(ln_ret_1) OVER w12 * SQRT(12) AS vol_1h,
    
    -- Volume z-score (6-hour rolling, exclude current)
    CASE 
        WHEN STDDEV(volume) OVER w72 > 0 
        THEN (volume - AVG(volume) OVER w72) / STDDEV(volume) OVER w72
        ELSE 0
    END AS vol_z,
    
    -- High-low range as % of close (bar-level microstructure)
    CASE 
        WHEN close > 0 
        THEN (high - low) / close 
        ELSE NULL 
    END AS hl_range_pc,
    
    -- Gap from previous close (5-min gap)
    CASE 
        WHEN prev_close IS NOT NULL AND prev_close > 0 
        THEN (open - prev_close) / prev_close 
        ELSE NULL 
    END AS gap_5m_pc,
    
    -- Volume-weighted return proxy (volume * return)
    ln_ret_1 * COALESCE(volume, 0) AS vol_weighted_ret,
    
    -- Relative volume (current vs 6h average)
    CASE 
        WHEN AVG(volume) OVER w72 > 0 
        THEN volume / AVG(volume) OVER w72
        ELSE 1
    END AS rel_volume_6h,
    
    -- Price velocity (rate of price change over 1h)
    CASE 
        WHEN COUNT(*) OVER w12 >= 12 
        THEN SUM(ln_ret_1) OVER w12 / 12.0
        ELSE NULL 
    END AS price_velocity_1h,
    
    -- Price acceleration (change in velocity over 1h)
    CASE 
        WHEN COUNT(*) OVER w12 >= 12 
        THEN (SUM(ln_ret_1) OVER w12 / 12.0) - (SUM(ln_ret_1) OVER w12_lag / 12.0)
        ELSE NULL 
    END AS price_acceleration_1h
    
FROM enriched_features

WINDOW
    w AS (PARTITION BY symbol ORDER BY ts_utc),
    w12 AS (PARTITION BY symbol ORDER BY ts_utc ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING),
    w72 AS (PARTITION BY symbol ORDER BY ts_utc ROWS BETWEEN 72 PRECEDING AND 1 PRECEDING),
    w12_lag AS (PARTITION BY symbol ORDER BY ts_utc ROWS BETWEEN 24 PRECEDING AND 13 PRECEDING)

-- Only include rows where we have sufficient lookback data
-- This ensures the latest feature row lags behind the latest raw data
QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts_utc DESC) > 1; 