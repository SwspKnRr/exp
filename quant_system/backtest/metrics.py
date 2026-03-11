"""
Performance Metrics Calculation
Calculate key backtest performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class PerformanceMetrics:
    """Calculate backtest performance metrics."""
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        days_per_year: int = 252
    ):
        """
        Initialize metrics calculator.
        
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate (default 2%)
        days_per_year : int
            Trading days per year (default 252)
        """
        
        self.risk_free_rate = risk_free_rate
        self.days_per_year = days_per_year
    
    def calculate_cagr(
        self,
        initial_value: float,
        final_value: float,
        years: float
    ) -> float:
        """
        Calculate Compound Annual Growth Rate (CAGR).
        
        Parameters
        ----------
        initial_value : float
            Starting value
        final_value : float
            Ending value
        years : float
            Number of years
        
        Returns
        -------
        float
            CAGR
        """
        
        if years <= 0:
            return 0
        
        cagr = (final_value / initial_value) ** (1 / years) - 1
        return cagr
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
        risk_free_rate : float, optional
            Annual risk-free rate
        
        Returns
        -------
        float
            Sharpe Ratio
        """
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Daily risk-free rate (using exact compound formula)
        daily_rf = (1 + risk_free_rate) ** (1 / self.days_per_year) - 1
        
        # Excess returns
        excess_returns = returns - daily_rf
        
        # Annualized Sharpe
        if excess_returns.std() == 0:
            return 0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.days_per_year)
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None,
        target_return: float = 0
    ) -> float:
        """
        Calculate Sortino Ratio (penalizes downside volatility only).
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
        risk_free_rate : float, optional
            Annual risk-free rate
        target_return : float
            Target return threshold (default 0)
        
        Returns
        -------
        float
            Sortino Ratio
        
        Notes
        -----
        Sortino = (mean_excess_return) / downside_deviation
        where downside_deviation = sqrt(mean((max(target - return, 0))^2))
        """
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Daily risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1 / self.days_per_year) - 1
        excess_returns = returns - daily_rf
        
        # Downside deviation: only penalize returns below target
        downside_returns = np.minimum(excess_returns - target_return, 0)
        downside_deviation = np.sqrt((downside_returns ** 2).mean())
        
        if downside_deviation == 0:
            return 0
        
        sortino = (excess_returns.mean() / downside_deviation) * np.sqrt(self.days_per_year)
        return sortino
    
    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown and period.
        
        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio value over time (index=date, values=portfolio value)
        
        Returns
        -------
        Tuple[float, pd.Timestamp, pd.Timestamp]
            Max drawdown %, peak date, trough date
        """
        
        # Calculate running max
        running_max = equity_curve.expanding().max()
        
        # Drawdown from peak
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Peak date (running max just before trough)
        peak_value = running_max[max_dd_date]
        peak_dates = equity_curve[equity_curve == peak_value].index
        
        if len(peak_dates) > 0:
            peak_date = peak_dates[0]
        else:
            peak_date = None
        
        return max_dd, peak_date, max_dd_date
    
    def calculate_volatility(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate annualized volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
        
        Returns
        -------
        float
            Annualized volatility
        """
        
        annual_vol = returns.std() * np.sqrt(self.days_per_year)
        return annual_vol
    
    def calculate_win_rate(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate win rate (% of positive return days).
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
        
        Returns
        -------
        float
            Win rate
        """
        
        if len(returns) == 0:
            return 0
        
        winning_days = (returns > 0).sum()
        win_rate = winning_days / len(returns)
        return win_rate
    
    def calculate_profit_factor(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns
        
        Returns
        -------
        float
            Profit factor
        """
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        
        if negative_returns == 0:
            return float('inf') if positive_returns > 0 else 0
        
        profit_factor = positive_returns / negative_returns
        return profit_factor
    
    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        initial_capital: float,
        final_capital: float = None
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio value over time
        initial_capital : float
            Starting capital
        final_capital : float, optional
            Final capital. If None, uses last value from equity_curve
        
        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values
        """
        
        if final_capital is None:
            final_capital = equity_curve.iloc[-1]
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Time period
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # Calculate basic metrics
        total_profit = final_capital - initial_capital
        max_drawdown = self.calculate_max_drawdown(equity_curve)[0]
        profit_factor = self.calculate_profit_factor(returns)
        
        # Recovery Factor = Total Profit / Max Drawdown
        # If max drawdown is positive or zero, recovery factor is infinite or undefined
        if max_drawdown < 0:
            recovery_factor = abs(total_profit / (initial_capital * max_drawdown))
        else:
            recovery_factor = 0  # No drawdown, so recovery factor is undefined
        
        # Calculate metrics
        metrics = {
            'Initial Capital': initial_capital,
            'Final Capital': final_capital,
            'Total Return': (final_capital - initial_capital) / initial_capital,
            'CAGR': self.calculate_cagr(initial_capital, final_capital, years),
            'Volatility': self.calculate_volatility(returns),
            'Sharpe Ratio': self.calculate_sharpe_ratio(returns),
            'Sortino Ratio': self.calculate_sortino_ratio(returns),
            'Max Drawdown': max_drawdown,
            'Win Rate': self.calculate_win_rate(returns),
            'Profit Factor': profit_factor,
            'Recovery Factor': recovery_factor,
            'Average Daily Return': returns.mean(),
            'Std Dev Daily Return': returns.std()
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Pretty print metrics.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics dictionary
        """
        
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        
        # Capital metrics
        print(f"\nCapital:")
        print(f"  Initial Capital:    ${metrics['Initial Capital']:>15,.2f}")
        print(f"  Final Capital:      ${metrics['Final Capital']:>15,.2f}")
        
        # Return metrics
        print(f"\nReturns:")
        print(f"  Total Return:       {metrics['Total Return']:>15.2%}")
        print(f"  CAGR:               {metrics['CAGR']:>15.2%}")
        
        # Risk metrics
        print(f"\nRisk:")
        print(f"  Volatility:         {metrics['Volatility']:>15.2%}")
        print(f"  Max Drawdown:       {metrics['Max Drawdown']:>15.2%}")
        
        # Risk-adjusted returns
        print(f"\nRisk-Adjusted Returns:")
        print(f"  Sharpe Ratio:       {metrics['Sharpe Ratio']:>15.2f}")
        print(f"  Sortino Ratio:      {metrics['Sortino Ratio']:>15.2f}")
        
        # Trading metrics
        print(f"\nTrading Metrics:")
        print(f"  Win Rate:           {metrics['Win Rate']:>15.2%}")
        print(f"  Profit Factor:      {metrics['Profit Factor']:>15.2f}")
        print(f"  Recovery Factor:    {metrics['Recovery Factor']:>15.2f}")
        
        # Daily metrics
        print(f"\nDaily Statistics:")
        print(f"  Avg Daily Return:   {metrics['Average Daily Return']:>15.2%}")
        print(f"  Daily Volatility:   {metrics['Std Dev Daily Return']:>15.2%}")
        
        print("="*50)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    from loader import download_price_data, ALL_TICKERS
    
    # Load data
    prices = download_price_data(ALL_TICKERS, start_date='2015-01-01', end_date='2020-12-31')
    
    # Create dummy equity curve (for testing)
    dates = prices.index
    equity_values = pd.Series(
        100000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, len(dates)))),
        index=dates
    )
    
    # Calculate metrics
    calculator = PerformanceMetrics()
    metrics = calculator.calculate_all_metrics(equity_values, 100000)
    
    # Print results
    calculator.print_metrics(metrics)
