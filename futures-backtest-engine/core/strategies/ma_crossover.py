import pandas as pd
from core.strategies.base import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    Generates long signal when fast moving average is above slow moving average,
    and short signal when it is below.
    """
    def __init__(self, fast_window: int = 10, slow_window: int = 50, ma_type: str = 'EMA'):
        super().__init__(name="Moving Average Crossover")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.ma_type = ma_type.upper()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals.
        1.0 for Long (fast > slow)
        -1.0 for Short (fast < slow)
        """
        close = df['close']
        
        if self.ma_type == 'EMA':
            fast_ma = close.ewm(span=self.fast_window, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow_window, adjust=False).mean()
        else: # Default to SMA
            fast_ma = close.rolling(window=self.fast_window).mean()
            slow_ma = close.rolling(window=self.slow_window).mean()
            
        # Create signals: 1 for long, -1 for short
        # We fill NaNs with 0 until the slow MA warm-up period is finished
        signals = pd.Series(0.0, index=df.index)
        
        # Determine signals
        signals[fast_ma > slow_ma] = 1.0
        signals[fast_ma < slow_ma] = -1.0
        
        # Zero out the initial warm-up period to avoid premature trades on unaligned data
        warmup_period = max(self.fast_window, self.slow_window)
        signals.iloc[:warmup_period] = 0.0
        
        return signals
