"""
Dual Momentum Strategy
Combines absolute and relative momentum:
- Absolute momentum: ETF return > 0
- Relative momentum: ETF return > SPY return
Final filter: ETF return > 0 AND ETF return > SPY return
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class DualMomentum:
    """Dual momentum signal generator."""
    
    def __init__(
        self,
        momentum_period: int = 20,
        benchmark: str = 'SPY',
        require_both: bool = True
    ):
        """
        Initialize dual momentum.
        
        Parameters
        ----------
        momentum_period : int
            Period for momentum calculation (default 20 days)
        benchmark : str
            Benchmark for relative momentum (default 'SPY')
        require_both : bool
            Require both absolute and relative momentum (default True)
        """
        self.momentum_period = momentum_period
        self.benchmark = benchmark
        self.require_both = require_both
    
    def calculate_absolute_momentum(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate absolute momentum (price change > 0).
        
        Parameters
        ----------
        returns : pd.DataFrame
            Log returns data
        
        Returns
        -------
        pd.DataFrame
            Boolean signals (True if return > 0)
        """
        
        absolute_momentum = returns > 0
        absolute_momentum.columns = [f'{col}_abs_mom' for col in returns.columns]
        
        return absolute_momentum
    
    def calculate_relative_momentum(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate relative momentum (vs benchmark).
        
        Parameters
        ----------
        returns : pd.DataFrame
            Log returns for each ETF
        benchmark_returns : pd.Series
            Benchmark (SPY) log returns
        
        Returns
        -------
        pd.DataFrame
            Boolean signals (True if ETF return > benchmark return)
        """
        
        # Broadcast benchmark returns to all columns
        relative_momentum = returns.gt(benchmark_returns, axis=0)
        relative_momentum.columns = [f'{col}_rel_mom' for col in returns.columns]
        
        return relative_momentum
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        include_benchmark: bool = False
    ) -> pd.DataFrame:
        """
        Generate dual momentum signals.
        Uses LOG RETURNS for consistency with ranking system.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data with all tickers
        include_benchmark : bool
            Whether to include benchmark in signals
        
        Returns
        -------
        pd.DataFrame
            Momentum signals (binary: 1 if momentum, 0 if not)
        
        Notes
        -----
        Signals are based on:
        - Absolute momentum: log_return_period > 0
        - Relative momentum: log_return_period > benchmark_log_return_period
        Both conditions must be TRUE (AND logic)
        """
        
        if self.benchmark not in prices.columns:
            raise ValueError(f"{self.benchmark} not found in price data")
        
        # Calculate LOG returns (consistent with ranking.py)
        returns = np.log(prices / prices.shift(self.momentum_period))
        benchmark_returns = returns[self.benchmark]
        
        # Absolute momentum
        abs_mom = self.calculate_absolute_momentum(returns)
        
        # Relative momentum (vs benchmark)
        rel_mom = self.calculate_relative_momentum(returns, benchmark_returns)
        
        # Combine signals
        if self.require_both:
            # Both absolute AND relative momentum required
            signals = pd.DataFrame(index=prices.index)
            
            for col in prices.columns:
                if col == self.benchmark and not include_benchmark:
                    continue
                
                abs_col = f'{col}_abs_mom'
                rel_col = f'{col}_rel_mom'
                
                signals[col] = (abs_mom[abs_col] & rel_mom[rel_col]).astype(int)
        else:
            # Either absolute OR relative momentum
            signals = pd.DataFrame(index=prices.index)
            
            for col in prices.columns:
                if col == self.benchmark and not include_benchmark:
                    continue
                
                abs_col = f'{col}_abs_mom'
                rel_col = f'{col}_rel_mom'
                
                signals[col] = (abs_mom[abs_col] | rel_mom[rel_col]).astype(int)
        
        return signals
    
    def get_momentum_strengths(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get momentum strength (actual return values).
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        pd.DataFrame
            Returns for momentum period
        """
        
        returns = np.log(prices / prices.shift(self.momentum_period))
        
        if self.benchmark in prices.columns:
            benchmark_returns = returns[self.benchmark]
            for col in returns.columns:
                if col != self.benchmark:
                    returns[col] = returns[col] - benchmark_returns
        
        return returns


def create_dual_momentum_filter(
    prices: pd.DataFrame,
    momentum_period: int = 20,
    benchmark: str = 'SPY'
) -> pd.DataFrame:
    """
    Convenience function to generate dual momentum signals.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    momentum_period : int
        Momentum period
    benchmark : str
        Benchmark ticker
    
    Returns
    -------
    pd.DataFrame
        Dual momentum signals
    """
    
    dm = DualMomentum(
        momentum_period=momentum_period,
        benchmark=benchmark,
        require_both=True
    )
    
    return dm.generate_signals(prices, include_benchmark=False)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    from loader import download_price_data, ALL_TICKERS
    
    prices = download_price_data(ALL_TICKERS, start_date='2015-01-01')
    
    dm = DualMomentum(momentum_period=20, benchmark='SPY', require_both=True)
    signals = dm.generate_signals(prices)
    
    print(f"Dual momentum signals shape: {signals.shape}")
    print(f"\nSignal summary (number of positive signals per ETF):")
    print((signals == 1).sum())
    print(f"\nLast 10 signals:")
    print(signals.tail(10))
