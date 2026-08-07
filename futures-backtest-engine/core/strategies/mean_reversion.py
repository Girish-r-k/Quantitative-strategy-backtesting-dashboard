import pandas as pd
import numpy as np
from core.strategies.base import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy based on rolling Z-scores.
    Long entry: Z-score falls below -entry_threshold.
    Short entry: Z-score rises above entry_threshold.
    Exit conditions:
      - Long exit when Z-score returns to or exceeds -exit_threshold.
      - Short exit when Z-score returns to or falls below exit_threshold.
    """
    def __init__(self, rolling_window: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.0):
        super().__init__(name="Mean Reversion")
        self.rolling_window = rolling_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates mean reversion signals using state-tracking iteration.
        """
        close = df['close']
        
        # Calculate rolling statistics
        rolling_mean = close.rolling(window=self.rolling_window).mean()
        rolling_std = close.rolling(window=self.rolling_window).std()
        
        # Z-score calculation (epsilon added to avoid division by zero)
        z_score = (close - rolling_mean) / (rolling_std + 1e-8)
        
        # Iteratively calculate signals due to hysteretic path dependency
        signals = np.zeros(len(df))
        current_position = 0.0 # 0.0: Flat, 1.0: Long, -1.0: Short
        
        # Convert to numpy for execution performance
        z_values = z_score.values
        
        for i in range(len(df)):
            if i < self.rolling_window:
                continue
                
            z = z_values[i]
            
            if current_position == 0.0:
                # Flat state: Look for entries
                if z < -self.entry_threshold:
                    current_position = 1.0 # Buy signal
                elif z > self.entry_threshold:
                    current_position = -1.0 # Sell/Short signal
            elif current_position == 1.0:
                # Long state: Look for exit
                if z >= -self.exit_threshold:
                    current_position = 0.0 # Exit long
            elif current_position == -1.0:
                # Short state: Look for exit
                if z <= self.exit_threshold:
                    current_position = 0.0 # Exit short
                    
            signals[i] = current_position
            
        return pd.Series(signals, index=df.index)
