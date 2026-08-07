import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Contract specifications database representing institutional futures configurations
CONTRACTS = {
    'ES=F': {
        'name': 'S&P 500 E-mini',
        'multiplier': 50.0,
        'tick_size': 0.25,
        'margin_rate': 0.10,
        'maintenance_margin_rate': 0.08,
        'slippage': 0.25,          # Expressed in absolute price units (1 tick)
        'commission': 2.50,        # USD per contract (one-way)
        'default_capital': 100000.0,
        'sector': 'Equity Index'
    },
    'CL=F': {
        'name': 'Crude Oil (WTI)',
        'multiplier': 1000.0,
        'tick_size': 0.01,
        'margin_rate': 0.12,
        'maintenance_margin_rate': 0.09,
        'slippage': 0.02,
        'commission': 3.00,
        'default_capital': 50000.0,
        'sector': 'Energy'
    },
    'GC=F': {
        'name': 'Gold',
        'multiplier': 100.0,
        'tick_size': 0.10,
        'margin_rate': 0.08,
        'maintenance_margin_rate': 0.06,
        'slippage': 0.10,
        'commission': 2.50,
        'default_capital': 80000.0,
        'sector': 'Metals'
    },
    'ZN=F': {
        'name': '10-Year US Treasury Note',
        'multiplier': 1000.0,
        'tick_size': 0.0156,       # ~1/64th
        'margin_rate': 0.03,
        'maintenance_margin_rate': 0.02,
        'slippage': 0.0156,
        'commission': 2.00,
        'default_capital': 100000.0,
        'sector': 'Fixed Income'
    }
}

def fetch_real_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical futures continuous contract price data from Yahoo Finance.
    Cleans, interpolates, and formats to standard schema.
    """
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {symbol} from Yahoo Finance.")
        
        # In newer versions of yfinance, MultiIndex is returned even for single ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        # Clean column names to standard lower_snake_case
        df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Set date index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Forward fill any missing rows and drop NaNs
        df = df.ffill().dropna()
        
        # Ensure we have all necessary columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # If some column is missing, estimate or raise
                if col == 'open':
                    df['open'] = df['close']
                elif col == 'high':
                    df['high'] = df['close']
                elif col == 'low':
                    df['low'] = df['close']
                elif col == 'volume':
                    df['volume'] = 0
        
        return df[required_cols]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}. Falling back to synthetic.")
        raise e

def generate_synthetic_data(symbol: str, start_date: str, end_date: str, 
                            trend_strength: float = 0.05, 
                            volatility_level: float = 0.20,
                            regime_switching: bool = True) -> pd.DataFrame:
    """
    Generates high-fidelity synthetic price series for stress testing strategies.
    Models volatility clustering (GARCH-lite), regime switches, and jumps.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start, end, freq='B') # Business days
    n_days = len(dates)
    
    # Get contract specifications or use defaults
    spec = CONTRACTS.get(symbol, {
        'default_capital': 100000.0,
        'tick_size': 0.01
    })
    
    # Seed based on symbol string hash for reproducibility
    seed = abs(hash(symbol)) % (2**32)
    np.random.seed(seed)
    
    # Initialize base parameters
    base_price = {
        'ES=F': 4500.0,
        'CL=F': 75.0,
        'GC=F': 1900.0,
        'ZN=F': 110.0
    }.get(symbol, 100.0)
    
    prices = np.zeros(n_days)
    prices[0] = base_price
    
    # Volatility process (GARCH-like: variance depends on previous return)
    vols = np.zeros(n_days)
    vols[0] = volatility_level / np.sqrt(252) # daily vol
    omega = 0.05 * (vols[0]**2)
    alpha = 0.10
    beta = 0.85
    
    # Drift/Trend processes for regimes
    current_regime = 1 # 1: normal, 2: trend-up, 3: trend-down, 4: mean-reverting
    regime_timer = 0
    
    # Generate prices day by day
    for i in range(1, n_days):
        # Regime switching logic
        if regime_switching and (regime_timer <= 0 or np.random.rand() < 0.02):
            current_regime = np.random.choice([1, 2, 3, 4], p=[0.4, 0.25, 0.25, 0.1])
            regime_timer = np.random.randint(15, 60) # duration of regime
        
        regime_timer -= 1
        
        # Drift determination
        if current_regime == 2:   # Strong Trend Up
            drift = (trend_strength + 0.15) / 252
        elif current_regime == 3: # Strong Trend Down
            drift = (-trend_strength - 0.15) / 252
        elif current_regime == 4: # Mean-reverting regime
            drift = 0.1 * (base_price - prices[i-1]) / base_price / 252
        else:                     # Normal drift
            drift = trend_strength / 252
            
        # Update daily volatility
        prev_ret = (prices[i-1] - prices[max(0, i-2)]) / prices[max(0, i-2)] if i > 1 else 0
        var = omega + alpha * (prev_ret**2) + beta * (vols[i-1]**2)
        vols[i] = np.sqrt(max(var, 1e-6))
        
        # Daily return logic with jump probability
        jump = 0
        if np.random.rand() < 0.015: # 1.5% daily jump chance
            jump = np.random.normal(0, vols[i] * 3) # large jump
            
        daily_return = drift + np.random.normal(0, vols[i]) + jump
        prices[i] = prices[i-1] * np.exp(daily_return)
        
    # Round prices to the tick size
    tick = spec['tick_size']
    prices = np.round(prices / tick) * tick
    
    # Create DataFrame with high, low, open, close
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    
    # Add intraday range
    noise_hl = np.random.exponential(scale=0.008, size=n_days)
    df['high'] = df['close'] * (1 + noise_hl)
    df['low'] = df['close'] * (1 - noise_hl)
    
    # Noise for open
    noise_op = np.random.normal(0, vols, size=n_days)
    df['open'] = df['close'] * (1 + noise_op * 0.3)
    
    # Adjust boundaries so High >= Max(Open, Close) and Low <= Min(Open, Close)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # Round all to tick size
    for col in ['open', 'high', 'low', 'close']:
        df[col] = np.round(df[col] / tick) * tick
        
    # Generate realistic volume
    df['volume'] = np.random.negative_binomial(20, 0.0001, size=n_days)
    
    df.index.name = 'date'
    return df

def get_futures_data(symbol: str, start_date: str, end_date: str, 
                     use_synthetic: bool = False,
                     trend_strength: float = 0.05, 
                     volatility_level: float = 0.20,
                     regime_switching: bool = True) -> pd.DataFrame:
    """
    Primary interface for data retrieval. Fetches real data with instant fallback
    to synthetic path generation in case of API failure or offline mode.
    """
    if use_synthetic:
        return generate_synthetic_data(symbol, start_date, end_date, 
                                       trend_strength, volatility_level, regime_switching)
    try:
        return fetch_real_data(symbol, start_date, end_date)
    except Exception:
        # Graceful fallback to synthetic data
        print(f"Fallback initiated: generating synthetic data for {symbol} due to download failure.")
        return generate_synthetic_data(symbol, start_date, end_date, 
                                       trend_strength, volatility_level, regime_switching)
