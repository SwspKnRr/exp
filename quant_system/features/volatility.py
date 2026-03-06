"""
Volatility Feature Engineering
Calculate volatility-based technical indicators and features.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_volatility(prices: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculate volatility (standard deviation of returns).
    
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    period : int
        Lookback period (default 20)
    
    Returns
    -------
    pd.DataFrame
        Volatility for each ticker
    """
    
    returns = prices.pct_change()
    volatility = returns.rolling(period).std() * np.sqrt(252)
    
    volatility.columns = [f'{col}_volatility' for col in volatility.columns]
    
    return volatility


def calculate_atr(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range.
    
    Parameters
    ----------
    prices : pd.DataFrame
        OHLC data or just close prices
    period : int
        ATR period (default 14)
    
    Returns
    -------
    pd.DataFrame
        ATR for each ticker
    """
    
    atr = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        # Using close prices only (approximation)
        # True range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr = prices[ticker].diff().abs()
        atr[f'{ticker}_atr'] = tr.rolling(period).mean()
    
    return atr


def calculate_atr_ratio(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate ATR as percentage of price (volatility ratio).
    
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    period : int
        ATR period (default 14)
    
    Returns
    -------
    pd.DataFrame
        ATR ratio for each ticker
    """
    
    atr = calculate_atr(prices, period)
    ratio = pd.DataFrame(index=prices.index)
    
    for ticker in prices.columns:
        if f'{ticker}_atr' in atr.columns:
            ratio[f'{ticker}_atr_ratio'] = atr[f'{ticker}_atr'] / prices[ticker]
    
    return ratio


def create_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive volatility features.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    
    Returns
    -------
    pd.DataFrame
        All volatility features
    """
    
    features = pd.DataFrame(index=prices.index)
    
    # Volatility
    vol_df = calculate_volatility(prices, period=20)
    features = pd.concat([features, vol_df], axis=1)
    
    # ATR
    atr_df = calculate_atr(prices, period=14)
    features = pd.concat([features, atr_df], axis=1)
    
    # ATR Ratio
    atr_ratio_df = calculate_atr_ratio(prices, period=14)
    features = pd.concat([features, atr_ratio_df], axis=1)
    
    # Bollinger Bands
    for ticker in prices.columns:
        sma = prices[ticker].rolling(20).mean()
        std = prices[ticker].rolling(20).std()
        
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        
        features[f'{ticker}_bb_upper'] = bb_upper
        features[f'{ticker}_bb_lower'] = bb_lower
        features[f'{ticker}_bb_width'] = bb_upper - bb_lower
        features[f'{ticker}_bb_position'] = (prices[ticker] - bb_lower) / (bb_upper - bb_lower)
    
    return features.fillna(0)
