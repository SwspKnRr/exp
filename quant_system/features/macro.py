"""
Macro Feature Engineering
Calculates macro-level indicators and market regime signals.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_vix_features(vix: pd.Series) -> pd.DataFrame:
    """
    Create VIX-based macro features.
    
    Parameters
    ----------
    vix : pd.Series
        VIX index values
    
    Returns
    -------
    pd.DataFrame
        VIX features
    """
    
    features = pd.DataFrame(index=vix.index)
    
    # VIX level
    features['VIX'] = vix
    
    # VIX change (daily percent change)
    features['VIX_change'] = vix.pct_change()
    
    # VIX MA20 (rolling 20-day average)
    features['VIX_MA20'] = vix.rolling(window=20).mean()
    
    # VIX volatility (rolling volatility of VIX)
    features['VIX_volatility'] = vix.pct_change().rolling(window=20).std()
    
    return features


def calculate_spy_gld_ratio(spy: pd.Series, gld: pd.Series) -> pd.Series:
    """
    Calculate SPY to GLD ratio (risk-on/risk-off indicator).
    
    Parameters
    ----------
    spy : pd.Series
        SPY price series
    gld : pd.Series
        GLD price series
    
    Returns
    -------
    pd.Series
        SPY/GLD ratio
    """
    
    ratio = spy / gld
    ratio.name = 'SPY_GLD_ratio'
    
    return ratio


def calculate_bond_stock_ratio(tlt: pd.Series, spy: pd.Series) -> pd.Series:
    """
    Calculate TLT to SPY ratio (bond/stock ratio).
    
    Parameters
    ----------
    tlt : pd.Series
        TLT (bond ETF) price series
    spy : pd.Series
        SPY price series
    
    Returns
    -------
    pd.Series
        TLT/SPY ratio
    """
    
    ratio = tlt / spy
    ratio.name = 'Bond_Stock_ratio'
    
    return ratio


def create_macro_features(
    prices: pd.DataFrame,
    vix: pd.Series = None
) -> pd.DataFrame:
    """
    Create all macro-level features.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with tickers as columns
    vix : pd.Series, optional
        VIX data. If None, tried to be computed from prices
    
    Returns
    -------
    pd.DataFrame
        All macro features
    """
    
    features = pd.DataFrame(index=prices.index)
    
    # VIX features
    if vix is not None:
        vix_aligned = vix.reindex(prices.index).fillna(method='ffill')
        vix_features = calculate_vix_features(vix_aligned)
        features = features.join(vix_features)
    
    # SPY/GLD ratio
    if 'SPY' in prices.columns and 'GLD' in prices.columns:
        spy_gld = calculate_spy_gld_ratio(prices['SPY'], prices['GLD'])
        features['SPY_GLD_ratio'] = spy_gld
        
        # SPY/GLD momentum
        features['SPY_GLD_change'] = spy_gld.pct_change()
    
    # Bond/Stock ratio
    if 'TLT' in prices.columns and 'SPY' in prices.columns:
        bond_stock = calculate_bond_stock_ratio(prices['TLT'], prices['SPY'])
        features['Bond_Stock_ratio'] = bond_stock
    
    return features


def calculate_regime_filter(
    prices: pd.DataFrame,
    benchmark: str = 'SPY',
    ma_period: int = 200
) -> pd.Series:
    """
    Calculate market regime filter based on price vs MA200.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    benchmark : str
        Benchmark ticker (default 'SPY')
    ma_period : int
        Moving average period for regime (default 200)
    
    Returns
    -------
    pd.Series
        Boolean series: True if market is in uptrend (price > MA), False otherwise
    """
    
    if benchmark not in prices.columns:
        raise ValueError(f"{benchmark} not found in price data")
    
    benchmark_price = prices[benchmark]
    ma = benchmark_price.rolling(window=ma_period).mean()
    
    # True if price > MA (uptrend), False if price <= MA (downtrend)
    regime = benchmark_price > ma
    regime.name = f'{benchmark}_regime'
    
    return regime


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    from loader import download_price_data, download_vix_data, ALL_TICKERS
    
    prices = download_price_data(ALL_TICKERS, start_date='2015-01-01')
    vix = download_vix_data(start_date='2015-01-01')
    
    # Create macro features
    macro_features = create_macro_features(prices, vix)
    print(macro_features.head())
    print(f"\nMacro features shape: {macro_features.shape}")
    print(f"Features: {list(macro_features.columns)}")
    
    # Calculate regime filter
    regime = calculate_regime_filter(prices)
    print(f"\nRegime filter shape: {regime.shape}")
    print(f"Regime uptrend ratio: {regime.sum() / len(regime):.2%}")
