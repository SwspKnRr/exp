"""
Market Regime Filter
Determines if strategy should be active based on market regime.
Rules:
- SPY > MA200 → strategy active
- SPY ≤ MA200 → no positions
"""

import pandas as pd
import numpy as np
from typing import Union


class RegimeFilter:
    """Market regime filter based on price vs moving average."""
    
    def __init__(
        self,
        benchmark: str = 'SPY',
        ma_period: int = 200,
        use_simple_ma: bool = True
    ):
        """
        Initialize regime filter.
        
        Parameters
        ----------
        benchmark : str
            Benchmark ticker (default 'SPY')
        ma_period : int
            Moving average period (default 200)
        use_simple_ma : bool
            Use simple MA (True) or exponential MA (False)
        """
        self.benchmark = benchmark
        self.ma_period = ma_period
        self.use_simple_ma = use_simple_ma
    
    def calculate_regime(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate regime signal.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data with benchmark column
        
        Returns
        -------
        pd.Series
            Boolean series: True if uptrend, False if downtrend
        """
        
        if self.benchmark not in prices.columns:
            raise ValueError(f"{self.benchmark} not found in price data")
        
        benchmark_price = prices[self.benchmark]
        
        # Calculate moving average
        if self.use_simple_ma:
            ma = benchmark_price.rolling(window=self.ma_period).mean()
        else:
            ma = benchmark_price.ewm(span=self.ma_period, adjust=False).mean()
        
        # Create regime signal
        regime = benchmark_price > ma
        regime.name = 'regime_signal'
        
        return regime
    
    def get_active_dates(self, prices: pd.DataFrame) -> np.ndarray:
        """
        Get dates when strategy should be active.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        np.ndarray
            Boolean array of active dates
        """
        
        regime = self.calculate_regime(prices)
        return regime.values
    
    def filter_signal(
        self,
        signals: Union[pd.DataFrame, pd.Series],
        prices: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Apply regime filter to trading signals.
        
        Parameters
        ----------
        signals : pd.DataFrame or pd.Series
            Trading signals
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        pd.DataFrame or pd.Series
            Filtered signals (set to 0/False during downtrends)
        """
        
        regime = self.calculate_regime(prices)
        
        # Align regime with signals
        regime_aligned = regime.reindex(signals.index)
        
        # Apply filter
        filtered = signals.copy()
        filtered[~regime_aligned] = 0
        
        return filtered


def is_market_uptrend(
    prices: pd.DataFrame,
    benchmark: str = 'SPY',
    ma_period: int = 200
) -> pd.Series:
    """
    Convenience function to check if market is in uptrend.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    benchmark : str
        Benchmark ticker
    ma_period : int
        MA period
    
    Returns
    -------
    pd.Series
        Boolean series indicating uptrend
    """
    
    regime_filter = RegimeFilter(benchmark=benchmark, ma_period=ma_period)
    return regime_filter.calculate_regime(prices)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    from loader import download_price_data, ALL_TICKERS
    
    prices = download_price_data(ALL_TICKERS, start_date='2015-01-01')
    
    regime_filter = RegimeFilter(benchmark='SPY', ma_period=200)
    regime = regime_filter.calculate_regime(prices)
    
    print(f"Regime signal shape: {regime.shape}")
    print(f"Uptrend ratio: {regime.sum() / len(regime):.2%}")
    print(f"\nLast 10 regime signals:")
    print(regime.tail(10))
