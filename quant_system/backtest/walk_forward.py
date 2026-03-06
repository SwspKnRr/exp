"""
Walk-Forward Analysis
Rolling window training and testing for out-of-sample backtesting.
- Train window: 5 years
- Test window: 1 year
- Retraining: Rolling forward every 1 year
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from datetime import timedelta
from .engine import BacktestEngine


class WalkForwardAnalyzer:
    """Walk-forward analysis for robust backtesting."""
    
    def __init__(
        self,
        train_years: int = 5,
        test_years: int = 1,
        initial_capital: float = 100000,
        transaction_cost: float = 0.0005
    ):
        """
        Initialize walk-forward analyzer.
        
        Parameters
        ----------
        train_years : int
            Training window in years (default 5)
        test_years : int
            Testing window in years (default 1)
        initial_capital : float
            Starting capital for each period
        transaction_cost : float
            Transaction cost per trade
        """
        
        self.train_years = train_years
        self.test_years = test_years
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.periods = []
        self.results = []
    
    def create_walk_forward_windows(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[Dict]:
        """
        Create walk-forward windows.
        
        Parameters
        ----------
        start_date : pd.Timestamp
            Start date for full backtest period
        end_date : pd.Timestamp
            End date for full backtest period
        
        Returns
        -------
        List[Dict]
            List of periods with train_start, train_end, test_start, test_end
        """
        
        windows = []
        current_year = start_date.year
        
        while True:
            train_start = pd.Timestamp(year=current_year, month=1, day=1)
            train_end = pd.Timestamp(year=current_year + self.train_years - 1, month=12, day=31)
            
            test_start = train_end + timedelta(days=1)
            test_end = pd.Timestamp(year=current_year + self.train_years + self.test_years - 1, month=12, day=31)
            
            # Adjust dates to available data
            if train_start < start_date:
                train_start = start_date
            
            if test_end > end_date:
                test_end = end_date
            
            # Check if we have enough data
            if train_start >= test_start or test_start > end_date:
                break
            
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'period': current_year
            })
            
            current_year += self.test_years
        
        return windows
    
    def run_walk_forward(
        self,
        prices: pd.DataFrame,
        selection_func: Callable,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run walk-forward backtest.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        selection_func : Callable
            Function to select portfolio based on training data
        start_date : pd.Timestamp
            Start date
        end_date : pd.Timestamp
            End date
        verbose : bool
            Print progress
        
        Returns
        -------
        Dict
            Results with equity curve and trades
        """
        
        if start_date is None:
            start_date = prices.index[0]
        if end_date is None:
            end_date = prices.index[-1]
        
        # Create windows
        windows = self.create_walk_forward_windows(start_date, end_date)
        
        if verbose:
            print(f"\nRunning Walk-Forward Analysis")
            print(f"Total periods: {len(windows)}")
            print(f"Train window: {self.train_years} years | Test window: {self.test_years} year")
            print()
        
        # Results container
        combined_results = {
            'equity_curve': pd.DataFrame(),
            'trades': [],
            'periods': []
        }
        
        all_equity_curves = []
        
        # Run each period
        for i, window in enumerate(windows):
            if verbose:
                print(f"Period {i+1}/{len(windows)}: {window['test_start'].date()} → {window['test_end'].date()}")
            
            # Get training data
            train_mask = (prices.index >= window['train_start']) & (prices.index <= window['train_end'])
            train_prices = prices[train_mask]
            
            # Select portfolio using training data
            last_train_date = window['train_end']
            
            # Find last Monday in training data
            while last_train_date.weekday() != 0:  # 0 = Monday
                last_train_date = last_train_date - timedelta(days=1)
            
            if last_train_date in train_prices.index:
                portfolio_selections = selection_func(train_prices, None, last_train_date)
            else:
                portfolio_selections = selection_func(train_prices, None, train_prices.index[-1])
            
            # Get test data
            test_mask = (prices.index >= window['test_start']) & (prices.index <= window['test_end'])
            test_prices = prices[test_mask]
            
            # Run backtest on test period
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                transaction_cost=self.transaction_cost
            )
            
            results = engine.run_backtest(
                test_prices,
                portfolio_selections,
                start_date=window['test_start'],
                end_date=window['test_end']
            )
            
            # Store results
            results['equity_curve']['window'] = i
            all_equity_curves.append(results['equity_curve'])
            
            combined_results['trades'].extend(results['trades'])
            
            # Calculate additional metrics from trades
            winning_trades = 0
            total_trades = len(results['trades'])
            
            for trade in results['trades']:
                if 'pnl' in trade and trade['pnl'] > 0:
                    winning_trades += 1
            
            win_rate = float(winning_trades) / float(total_trades) if total_trades > 0 else 0.0
            
            # Calculate max drawdown for this period
            eq_curve = results['equity_curve']['portfolio_value']
            if len(eq_curve) > 0:
                running_max = eq_curve.expanding().max()
                drawdown = (eq_curve - running_max) / running_max
                max_drawdown = float(drawdown.min())
            else:
                max_drawdown = 0.0
            
            combined_results['periods'].append({
                'period': window['period'],
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'final_value': float(results['final_value']),
                'total_return': float(results['total_return']),
                'num_trades': float(total_trades),
                'winning_trades': float(winning_trades),
                'win_rate': float(win_rate),
                'max_drawdown': float(max_drawdown),
                'period_start': window['test_start'],
                'period_end': window['test_end']
            })
            
            if verbose:
                print(f"  Final Value: ${results['final_value']:,.2f} | Return: {results['total_return']:.2%} | Trades: {results['total_trades']}")
        
        # Combine equity curves
        if all_equity_curves:
            combined_results['equity_curve'] = pd.concat(all_equity_curves)
        
        self.results = combined_results
        self.periods = combined_results['periods']
        
        return combined_results
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all periods.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        
        if not self.periods:
            return pd.DataFrame()
        
        return pd.DataFrame(self.periods)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    from loader import download_price_data, ALL_TICKERS
    
    # Load data
    prices = download_price_data(ALL_TICKERS, start_date='2010-01-01')
    
    # Define simple selection function
    def simple_selection(train_prices, train_features, date):
        return ['SPY', 'QQQ', 'VTI']
    
    # Run analysis
    analyzer = WalkForwardAnalyzer(
        train_years=5,
        test_years=1,
        initial_capital=100000,
        transaction_cost=0.0005
    )
    
    results = analyzer.run_walk_forward(
        prices,
        simple_selection,
        start_date=pd.Timestamp('2015-01-01'),
        end_date=pd.Timestamp('2026-01-01'),
        verbose=True
    )
    
    print("\nWalk-Forward Results:")
    summary = analyzer.get_summary_statistics()
    print(summary)
