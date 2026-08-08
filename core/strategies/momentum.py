import pandas as pd
import numpy as np
from core.strategies.base import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback_window: int = 50, adx_threshold: float = 20.0, use_adx_filter: bool = True):
        super().__init__(name="Time-Series Momentum")
        self.lookback_window = lookback_window
        self.adx_threshold = adx_threshold
        self.use_adx_filter = use_adx_filter

    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        
        up_move = high - prev_high
        down_move = prev_low - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_dm_series = pd.Series(plus_dm, index=df.index)
        minus_dm_series = pd.Series(minus_dm, index=df.index)
        
        plus_di = 100 * (plus_dm_series.ewm(alpha=1/window, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm_series.ewm(alpha=1/window, adjust=False).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        dx = dx.fillna(0)
        
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        return adx

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        
        mom_return = close.pct_change(periods=self.lookback_window)
        
        raw_signals = np.sign(mom_return).fillna(0.0)
        
        if self.use_adx_filter:
            adx = self._calculate_adx(df, window=14)
            signals = np.where(adx >= self.adx_threshold, raw_signals, 0.0)
            signals = pd.Series(signals, index=df.index)
        else:
            signals = raw_signals
            
        warmup = max(self.lookback_window, 30)
        signals.iloc[:warmup] = 0.0
        
        return signals
