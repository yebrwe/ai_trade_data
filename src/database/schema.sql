CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL NOT NULL,
    high DECIMAL NOT NULL,
    low DECIMAL NOT NULL,
    close DECIMAL NOT NULL,
    volume DECIMAL NOT NULL,
    ma_5 DECIMAL,
    ma_10 DECIMAL,
    ma_20 DECIMAL,
    ma_60 DECIMAL,
    ma_120 DECIMAL,
    ma_200 DECIMAL,
    ema_5 DECIMAL,
    ema_10 DECIMAL,
    ema_20 DECIMAL,
    ema_60 DECIMAL,
    ema_120 DECIMAL,
    ema_200 DECIMAL,
    bb_upper DECIMAL,
    bb_middle DECIMAL,
    bb_lower DECIMAL,
    rsi DECIMAL,
    macd DECIMAL,
    macd_signal DECIMAL,
    macd_hist DECIMAL,
    stoch_k DECIMAL,
    stoch_d DECIMAL,
    atr DECIMAL,
    obv DECIMAL,
    dmi_plus DECIMAL,
    dmi_minus DECIMAL,
    adx DECIMAL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (timestamp, symbol, timeframe)
);

CREATE INDEX idx_market_data_timestamp ON market_data (timestamp);
CREATE INDEX idx_market_data_symbol ON market_data (symbol);
CREATE INDEX idx_market_data_timeframe ON market_data (timeframe);
CREATE INDEX idx_market_data_symbol_timeframe ON market_data (symbol, timeframe); 