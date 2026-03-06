"""
Data Loader - Download and manage price data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# All tickers
ALL_TICKERS = [
    'SPY', 'QQQ', 'VTI',  # Broad market
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',  # Sectors
    'SMH', 'LIT', 'ARKK',  # Thematic
    'ITA', 'ITB',  # Industrials
    'GLD', 'TLT'  # Commodities/Bonds
]


def download_price_data(
    tickers: List[str] = None,
    start_date: str = '2010-01-01',
    end_date: str = None
) -> pd.DataFrame:
    """
    Download price data from Yahoo Finance.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD), defaults to today
    
    Returns
    -------
    pd.DataFrame
        Adjusted close prices with tickers as columns
    """
    
    if tickers is None:
        tickers = ALL_TICKERS
    
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=True,
        ignore_tz=True
    )
    
    # Extract closing prices
    # yfinance returns MultiIndex columns: (Price Type, Ticker)
    if len(tickers) == 1:
        # Single ticker: data.columns is simple Index, not MultiIndex
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                prices = pd.DataFrame({tickers[0]: data['Close']})
            else:
                prices = pd.DataFrame({tickers[0]: data.iloc[:, 0]})
        else:
            # data is a Series (single ticker)
            prices = pd.DataFrame({tickers[0]: data})
    else:
        # Multiple tickers: extract Close prices from MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            # Access by price type: 'Close' is at level 0
            prices = data.xs('Close', level=0, axis=1)
        else:
            # Fallback for older yfinance versions
            prices = data['Close'] if 'Close' in data.columns else data
    
    # Ensure prices is a DataFrame
    if isinstance(prices, pd.Series):
        prices = pd.DataFrame(prices)
    
    # Filter to only include tickers that were successfully downloaded
    prices = prices[[ticker for ticker in tickers if ticker in prices.columns]]
    
    # Remove rows with NaN values
    prices = prices.dropna()
    
    print(f"\nDownloaded data shape: {prices.shape}")
    if len(prices) > 0:
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Successfully loaded tickers: {list(prices.columns)}")
    
    return prices


def download_vix_data(
    start_date: str = '2010-01-01',
    end_date: str = None
) -> pd.Series:
    """
    Download VIX (volatility index) data.
    
    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns
    -------
    pd.Series
        VIX values
    """
    
    print("Downloading VIX data...")
    
    vix = yf.download(
        '^VIX',
        start=start_date,
        end=end_date,
        progress=False,
        ignore_tz=True
    )
    
    vix_series = vix['Adj Close']
    vix_series.name = 'VIX'
    
    print(f"VIX data shape: {vix_series.shape}")
    
    return vix_series


def create_macro_signals(
    prices: pd.DataFrame,
    vix: pd.Series
) -> pd.DataFrame:
    """
    Create macro signals from price and VIX data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    vix : pd.Series
        VIX data
    
    Returns
    -------
    pd.DataFrame
        Macro signals
    """
    
    # Align data
    aligned_prices = prices.loc[prices.index.intersection(vix.index)]
    aligned_vix = vix.loc[prices.index.intersection(vix.index)]
    
    macro_signals = pd.DataFrame(index=aligned_prices.index)
    
    # VIX signals
    macro_signals['VIX'] = aligned_vix
    macro_signals['VIX_MA20'] = aligned_vix.rolling(20).mean()
    macro_signals['VIX_Above_MA'] = (aligned_vix > macro_signals['VIX_MA20']).astype(int)
    
    return macro_signals


def align_all_data(
    prices: pd.DataFrame,
    vix: pd.Series,
    macro_signals: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align all data to common date range.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    vix : pd.Series
        VIX data
    macro_signals : pd.DataFrame
        Macro signals (optional)
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Aligned prices and macro signals
    """
    
    # Find common dates
    common_index = prices.index.intersection(vix.index)
    
    if macro_signals is not None:
        common_index = common_index.intersection(macro_signals.index)
    
    aligned_prices = prices.loc[common_index]
    
    if macro_signals is not None:
        aligned_macro = macro_signals.loc[common_index]
    else:
        aligned_macro = create_macro_signals(aligned_prices, vix.loc[common_index])
    
    return aligned_prices, aligned_macro
