import pandas as pd
import numpy as np
from core.strategies.base import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """
    Time-Series Momentum (TSMOM) Strategy.
    Generates signals based on historical asset returns (lookback window),
    scaled by volatility, and optionally filtered by trend strength (ADX)
    to prevent whip-saws in range-bound markets.
    """
    def __init__(self, lookback_window: int = 50, adx_threshold: float = 20.0, use_adx_filter: bool = True):
        super().__init__(name="Time-Series Momentum")
        self.lookback_window = lookback_window
        self.adx_threshold = adx_threshold
        self.use_adx_filter = use_adx_filter

    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Vectorized ADX (Average Directional Index) calculator.
        Represents quantitative-grade trend strength filtration.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Shift values
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        # True Range (TR)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        
        # Directional Movement (+DM, -DM)
        up_move = high - prev_high
        down_move = prev_low - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_dm_series = pd.Series(plus_dm, index=df.index)
        minus_dm_series = pd.Series(minus_dm, index=df.index)
        
        plus_di = 100 * (plus_dm_series.ewm(alpha=1/window, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm_series.ewm(alpha=1/window, adjust=False).mean() / atr)
        
        # Directional Index (DX) and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        dx = dx.fillna(0)
        
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        return adx

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates signal based on direction of return over the lookback window.
        If use_adx_filter is True, filters out signals where trend strength is weak.
        """
        close = df['close']
        
        # Calculate trailing momentum (return over the lookback window)
        mom_return = close.pct_change(periods=self.lookback_window)
        
        # Base momentum signal: +1 if positive, -1 if negative
        raw_signals = np.sign(mom_return).fillna(0.0)
        
        # If ADX filter is enabled, suppress signals in low-trend environments
        if self.use_adx_filter:
            adx = self._calculate_adx(df, window=14)
            signals = np.where(adx >= self.adx_threshold, raw_signals, 0.0)
            signals = pd.Series(signals, index=df.index)
        else:
            signals = raw_signals
            
        # Clean warmup periods
        warmup = max(self.lookback_window, 30)
        signals.iloc[:warmup] = 0.0
        
        return signals
