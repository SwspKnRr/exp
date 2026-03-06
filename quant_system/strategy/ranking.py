"""
ETF Ranking System
Ranks ETFs by momentum score and selects top performers.

Score formula:
score = 0.4 * return_20d + 0.3 * return_60d + 0.3 * return_120d

Selection: Top 3 ETFs by score
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class ETFRanker:
    """ETF ranking system based on multi-period momentum."""
    
    def __init__(
        self,
        weights: Dict[int, float] = None,
        max_positions: int = 3,
        min_positions: int = 1
    ):
        """
        Initialize ETF ranker.
        
        Parameters
        ----------
        weights : Dict[int, float]
            Weights for different periods {period: weight}
            Default: {20: 0.4, 60: 0.3, 120: 0.3}
        max_positions : int
            Maximum number of positions to select (default 3)
        min_positions : int
            Minimum number of positions to select (default 1)
        """
        
        if weights is None:
            weights = {20: 0.4, 60: 0.3, 120: 0.3}
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        
        self.max_positions = max_positions
        self.min_positions = min_positions
    
    def calculate_momentum_score(
        self,
        prices: pd.DataFrame,
        exclude_tickers: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate momentum scores for all ETFs.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        exclude_tickers : List[str], optional
            Tickers to exclude from ranking (e.g., benchmarks)
        
        Returns
        -------
        pd.DataFrame
            Momentum scores (index=date, columns=tickers)
        """
        
        if exclude_tickers is None:
            exclude_tickers = []
        
        scores = pd.DataFrame(index=prices.index)
        
        for col in prices.columns:
            if col in exclude_tickers:
                continue
            
            score = pd.Series(0.0, index=prices.index)
            
            for period, weight in self.weights.items():
                # Calculate return for this period
                returns = np.log(prices[col] / prices[col].shift(period))
                score += weight * returns
            
            scores[col] = score
        
        return scores
    
    def rank_etfs(
        self,
        prices: pd.DataFrame,
        date: pd.Timestamp = None,
        exclude_tickers: List[str] = None
    ) -> pd.DataFrame:
        """
        Rank ETFs on a specific date.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        date : pd.Timestamp, optional
            Date for ranking. If None, uses latest date.
        exclude_tickers : List[str], optional
            Tickers to exclude from ranking
        
        Returns
        -------
        pd.DataFrame
            Ranking DataFrame with columns: rank, ticker, score
        """
        
        if date is None:
            date = prices.index[-1]
        
        if date not in prices.index:
            raise ValueError(f"Date {date} not in price index")
        
        scores = self.calculate_momentum_score(prices, exclude_tickers)
        
        # Get scores for the date
        date_scores = scores.loc[date].dropna().sort_values(ascending=False)
        
        # Create ranking dataframe
        ranking = pd.DataFrame({
            'ticker': date_scores.index,
            'score': date_scores.values,
            'rank': range(1, len(date_scores) + 1)
        })
        
        return ranking
    
    def select_portfolio(
        self,
        prices: pd.DataFrame,
        date: pd.Timestamp = None,
        exclude_tickers: List[str] = None,
        min_score: float = None
    ) -> List[str]:
        """
        Select top ETFs for portfolio.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        date : pd.Timestamp, optional
            Date for selection. If None, uses latest date.
        exclude_tickers : List[str], optional
            Tickers to exclude
        min_score : float, optional
            Minimum score threshold. If set, can select fewer than max_positions.
        
        Returns
        -------
        List[str]
            List of selected tickers
        """
        
        ranking = self.rank_etfs(prices, date, exclude_tickers)
        
        # Apply max positions constraint
        selected = ranking.head(self.max_positions)
        
        # Apply min score constraint if provided
        if min_score is not None:
            selected = selected[selected['score'] >= min_score]
        
        # Ensure minimum positions
        if len(selected) < self.min_positions:
            selected = ranking.head(self.min_positions)
        
        return selected['ticker'].tolist()
    
    def get_portfolio_weights(
        self,
        prices: pd.DataFrame,
        date: pd.Timestamp = None,
        exclude_tickers: List[str] = None,
        weight_method: str = 'equal'
    ) -> Dict[str, float]:
        """
        Get portfolio weights for selected ETFs.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        date : pd.Timestamp, optional
            Date for selection
        exclude_tickers : List[str], optional
            Tickers to exclude
        weight_method : str
            'equal', 'momentum_score', or 'inverse_rank'
        
        Returns
        -------
        Dict[str, float]
            Dictionary of ticker: weight
        """
        
        ranking = self.rank_etfs(prices, date, exclude_tickers)
        selected = ranking.head(self.max_positions)
        
        if len(selected) == 0:
            return {}
        
        if weight_method == 'equal':
            weight = 1.0 / len(selected)
            return {ticker: weight for ticker in selected['ticker']}
        
        elif weight_method == 'momentum_score':
            # Weight by score magnitude
            total_score = selected['score'].sum()
            return {
                row['ticker']: row['score'] / total_score
                for _, row in selected.iterrows()
            }
        
        elif weight_method == 'inverse_rank':
            # Higher rank (lower number) gets higher weight
            max_rank = len(selected)
            weights = (max_rank - selected['rank'] + 1).values
            total_weight = weights.sum()
            return {
                ticker: w / total_weight
                for ticker, w in zip(selected['ticker'], weights)
            }
        
        else:
            raise ValueError(f"Unknown weight method: {weight_method}")


def get_top_momentum_etfs(
    prices: pd.DataFrame,
    date: pd.Timestamp = None,
    num_etfs: int = 3,
    exclude_tickers: List[str] = None
) -> List[str]:
    """
    Convenience function to select top momentum ETFs.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    date : pd.Timestamp, optional
        Date for selection
    num_etfs : int
        Number of ETFs to select
    exclude_tickers : List[str], optional
        Tickers to exclude
    
    Returns
    -------
    List[str]
        List of selected tickers
    """
    
    ranker = ETFRanker(max_positions=num_etfs)
    return ranker.select_portfolio(prices, date, exclude_tickers)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    from loader import download_price_data, ALL_TICKERS
    
    prices = download_price_data(ALL_TICKERS, start_date='2015-01-01')
    
    ranker = ETFRanker(max_positions=3)
    
    # Get latest ranking
    ranking = ranker.rank_etfs(prices)
    print("Latest ETF Ranking:")
    print(ranking.head(10))
    
    # Select portfolio
    selected = ranker.select_portfolio(prices)
    print(f"\nSelected ETFs: {selected}")
    
    # Get weights
    weights = ranker.get_portfolio_weights(prices, weight_method='equal')
    print(f"\nPortfolio Weights:")
    for ticker, weight in weights.items():
        print(f"  {ticker}: {weight:.2%}")
