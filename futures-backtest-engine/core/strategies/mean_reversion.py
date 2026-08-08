import pandas as pd
import numpy as np
from core.strategies.base import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, rolling_window: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.0):
        super().__init__(name="Mean Reversion")
        self.rolling_window = rolling_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        
        rolling_mean = close.rolling(window=self.rolling_window).mean()
        rolling_std = close.rolling(window=self.rolling_window).std()
        
        z_score = (close - rolling_mean) / (rolling_std + 1e-8)
        
        signals = np.zeros(len(df))
        current_position = 0.0
        
        z_values = z_score.values
        
        for i in range(len(df)):
            if i < self.rolling_window:
                continue
                
            z = z_values[i]
            
            if current_position == 0.0:
                if z < -self.entry_threshold:
                    current_position = 1.0
                elif z > self.entry_threshold:
                    current_position = -1.0
            elif current_position == 1.0:
                if z >= -self.exit_threshold:
                    current_position = 0.0
            elif current_position == -1.0:
                if z <= self.exit_threshold:
                    current_position = 0.0
                    
            signals[i] = current_position
            
        return pd.Series(signals, index=df.index)
