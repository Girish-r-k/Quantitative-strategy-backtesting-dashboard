import pandas as pd
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Ensures a consistent interface for signal generation.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates position signals from historical market data.
        Returns a Pandas Series of floats containing signals:
         - 1.0 : Long position
         - -1.0: Short position
         - 0.0 : Flat/Cash
        """
        pass
