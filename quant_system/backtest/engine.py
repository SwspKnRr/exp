"""
Backtest Engine
Simulates portfolio rebalancing with transaction costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class BacktestEngine:
    """Backtesting engine for portfolio simulation."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.0005,
        rebalance_day: str = 'Monday',
        max_positions: int = 3
    ):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        initial_capital : float
            Starting portfolio value
        transaction_cost : float
            Transaction cost per trade (default 0.05%)
        rebalance_day : str
            Day of week for rebalancing (default 'Monday')
        max_positions : int
            Maximum number of positions
        """
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_day = rebalance_day
        self.max_positions = max_positions
        
        self.portfolio_values = []
        self.positions = {}
        self.trades = []
        self.daily_returns = []
    
    def get_rebalance_dates(
        self,
        dates: pd.DatetimeIndex,
        rebalance_day: str = None
    ) -> List[pd.Timestamp]:
        """
        Get dates when portfolio should be rebalanced.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            Trading dates
        rebalance_day : str, optional
            Day of week. If None, uses self.rebalance_day
        
        Returns
        -------
        List[pd.Timestamp]
            Rebalance dates
        """
        
        if rebalance_day is None:
            rebalance_day = self.rebalance_day
        
        # Get day numbers (0=Monday, 6=Sunday)
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        target_day = day_map.get(rebalance_day, 0)
        
        # Find all dates matching the day
        rebalance_dates = [d for d in dates if d.dayofweek == target_day]
        
        return rebalance_dates
    
    def execute_rebalance(
        self,
        date: pd.Timestamp,
        current_prices: pd.Series,
        target_positions: Dict[str, float],
        current_positions: Dict[str, float],
        cash: float,
        portfolio_value: float
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """
        Execute portfolio rebalancing.
        
        Parameters
        ----------
        date : pd.Timestamp
            Current date
        current_prices : pd.Series
            Current prices for all assets
        target_positions : Dict[str, float]
            Target position weights
        current_positions : Dict[str, float]
            Current position counts
        cash : float
            Available cash
        portfolio_value : float
            Current portfolio value
        
        Returns
        -------
        Tuple[Dict[str, float], float, List[Dict]]
            Updated positions, cash, and list of trades executed
        """
        
        trades_executed = []
        new_positions = current_positions.copy()
        new_cash = cash
        
        # First: Sell positions we want to reduce/eliminate
        for ticker in current_positions.keys():
            if ticker not in current_prices.index:
                continue
            
            target_weight = target_positions.get(ticker, 0)
            current_shares = current_positions[ticker]
            
            if current_shares > 0:
                current_value = current_shares * current_prices[ticker]
                target_value = target_weight * portfolio_value
                
                if target_value < current_value * 0.99:  # Need to sell
                    shares_to_sell = current_shares - (target_value / current_prices[ticker])
                    
                    if shares_to_sell > 0:
                        # Execution price with bid-ask spread cost
                        execution_price = current_prices[ticker] * (1 - self.transaction_cost)
                        sale_value = shares_to_sell * execution_price
                        
                        new_cash += sale_value
                        new_positions[ticker] -= shares_to_sell
                        
                        trades_executed.append({
                            'date': date,
                            'ticker': ticker,
                            'side': 'SELL',
                            'shares': shares_to_sell,
                            'price': current_prices[ticker],
                            'execution_price': execution_price,
                            'transaction_cost': shares_to_sell * current_prices[ticker] * self.transaction_cost
                        })
        
        # Second: Buy positions we want to add
        for ticker, target_weight in target_positions.items():
            if ticker not in current_prices.index:
                continue
            
            target_value = target_weight * portfolio_value
            current_shares = new_positions.get(ticker, 0)
            current_value = current_shares * current_prices[ticker]
            
            shares_needed = (target_value - current_value) / current_prices[ticker]
            
            if shares_needed > 0:
                # Execution price with bid-ask spread cost
                execution_price = current_prices[ticker] * (1 + self.transaction_cost)
                buy_cost = shares_needed * execution_price
                
                if buy_cost <= new_cash:
                    new_cash -= buy_cost
                    new_positions[ticker] = current_shares + shares_needed
                    
                    trades_executed.append({
                        'date': date,
                        'ticker': ticker,
                        'side': 'BUY',
                        'shares': shares_needed,
                        'price': current_prices[ticker],
                        'execution_price': execution_price,
                        'transaction_cost': shares_needed * current_prices[ticker] * self.transaction_cost
                    })
        
        # Remove zero positions
        new_positions = {k: v for k, v in new_positions.items() if v > 0}
        
        return new_positions, new_cash, trades_executed
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        portfolio_selections: Dict[pd.Timestamp, List[str]],
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None
    ) -> Dict:
        """
        Run backtest with dynamic portfolio rebalancing.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data for all tickers
        portfolio_selections : Dict[pd.Timestamp, List[str]]
            Selected tickers for each rebalance date
        start_date : pd.Timestamp, optional
            Start date for backtest
        end_date : pd.Timestamp, optional
            End date for backtest
        
        Returns
        -------
        Dict
            Backtest results
        """
        
        if start_date is None:
            start_date = prices.index[0]
        if end_date is None:
            end_date = prices.index[-1]
        
        # Filter price data
        mask = (prices.index >= start_date) & (prices.index <= end_date)
        prices_backtest = prices[mask].copy()
        
        # Initialize
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}  # {ticker: shares}
        
        equity_curve = []
        all_trades = []
        
        # Get rebalance dates
        rebalance_dates = self.get_rebalance_dates(prices_backtest.index)
        
        # Main backtest loop
        for date in prices_backtest.index:
            day_prices = prices_backtest.loc[date]
            
            # Calculate portfolio value
            position_value = 0
            for ticker, shares in positions.items():
                if ticker in day_prices.index:
                    position_value += shares * day_prices[ticker]
            
            portfolio_value = cash + position_value
            
            # Check if rebalancing day
            if date in rebalance_dates and date in portfolio_selections:
                # Get target portfolio
                selected_tickers = portfolio_selections[date]
                
                if len(selected_tickers) > 0:
                    # Equal weight allocation
                    target_weight = 1.0 / len(selected_tickers)
                    target_positions = {
                        ticker: target_weight for ticker in selected_tickers
                    }
                else:
                    target_positions = {}
                
                # Execute rebalance
                positions, cash, trades = self.execute_rebalance(
                    date, day_prices, target_positions, positions, cash, portfolio_value
                )
                
                all_trades.extend(trades)
            
            # Recalculate portfolio value after potential rebalance
            position_value = 0
            for ticker, shares in positions.items():
                if ticker in day_prices.index:
                    position_value += shares * day_prices[ticker]
            
            portfolio_value = cash + position_value
            
            equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'num_positions': len(positions)
            })
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
        
        results = {
            'equity_curve': equity_df,
            'total_trades': len(all_trades),
            'trades': all_trades,
            'final_value': portfolio_value,
            'total_return': (portfolio_value - self.initial_capital) / self.initial_capital,
            'positions': positions
        }
        
        return results


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    from loader import download_price_data, ALL_TICKERS
    
    # Load data
    prices = download_price_data(ALL_TICKERS, start_date='2015-01-01')
    
    # Create simple portfolio selections (for testing)
    portfolio_selections = {}
    for date in prices.index:
        if date.dayofweek == 0:  # Monday
            portfolio_selections[date] = ['SPY', 'QQQ', 'VTI']
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000, transaction_cost=0.0005)
    results = engine.run_backtest(
        prices, portfolio_selections,
        start_date='2015-01-01', end_date='2020-12-31'
    )
    
    print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
