"""
Momentum Feature Engineering
Calculate momentum-based technical indicators and features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


def calculate_returns(prices: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    Calculate returns over different periods.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    periods : List[int]
        List of periods (default [5, 10, 20, 50])
    
    Returns
    -------
    pd.DataFrame
        Returns for each period and ticker
    """
    
    if periods is None:
        periods = [5, 10, 20, 50]
    
    returns = pd.DataFrame(index=prices.index)
    
    for period in periods:
        returns[f'return_{period}d'] = prices.pct_change(period)
    
    return returns


def calculate_moving_average_ratios(
    prices: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50
) -> pd.DataFrame:
    """
    Calculate price to moving average ratios.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    fast_period : int
        Fast MA period (default 20)
    slow_period : int
        Slow MA period (default 50)
    
    Returns
    -------
    pd.DataFrame
        MA ratios for each ticker
    """
    
    ratios = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        ma_fast = prices[ticker].rolling(fast_period).mean()
        ma_slow = prices[ticker].rolling(slow_period).mean()
        
        # Price to MA ratios
        ratios[f'{ticker}_price_to_ma_fast'] = prices[ticker] / ma_fast
        ratios[f'{ticker}_price_to_ma_slow'] = prices[ticker] / ma_slow
        
        # MA crossover
        ratios[f'{ticker}_ma_crossover'] = (ma_fast > ma_slow).astype(int)
    
    return ratios


def calculate_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    period : int
        RSI period (default 14)
    
    Returns
    -------
    pd.DataFrame
        RSI for each ticker
    """
    
    rsi = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        delta = prices[ticker].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
    
    return rsi


def create_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive momentum features.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    
    Returns
    -------
    pd.DataFrame
        All momentum features
    """
    
    features = pd.DataFrame(index=prices.index)
    
    # Returns
    returns_df = calculate_returns(prices, periods=[5, 10, 20, 50, 100, 200])
    features = pd.concat([features, returns_df], axis=1)
    
    # Moving average ratios
    ma_ratios = calculate_moving_average_ratios(prices, fast_period=20, slow_period=50)
    features = pd.concat([features, ma_ratios], axis=1)
    
    # RSI
    rsi_df = calculate_rsi(prices, period=14)
    features = pd.concat([features, rsi_df], axis=1)
    
    # Momentum score (composite)
    for ticker in prices.columns:
        momentum_score = 0
        
        # Trend component
        if f'{ticker}_ma_crossover' in features.columns:
            momentum_score += features[f'{ticker}_ma_crossover'] * 0.3
        
        # Return component
        if f'{ticker}_return_20d' in features.columns:
            momentum_score += features[f'{ticker}_return_20d'].rank() / len(features) * 0.4
        
        # RSI component
        if f'{ticker}_rsi' in features.columns:
            momentum_score += (features[f'{ticker}_rsi'] / 100) * 0.3
        
        features[f'{ticker}_momentum_score'] = momentum_score
    
    return features.fillna(0)
