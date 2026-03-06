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
    print(f"Tickers: {tickers}")
    
    try:
        # Download data
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,  # Disable progress bar for cleaner logs
            ignore_tz=True
        )
    except Exception as e:
        print(f"⚠️  Error downloading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    # Check if data is empty
    if data is None:
        print("⚠️  yfinance returned None")
        return pd.DataFrame()
    
    if isinstance(data, pd.DataFrame) and data.empty:
        print("⚠️  yfinance returned empty DataFrame")
        return pd.DataFrame()
    
    if isinstance(data, pd.Series) and len(data) == 0:
        print("⚠️  yfinance returned empty Series")
        return pd.DataFrame()
    
    print(f"Raw data type: {type(data)}, shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
    
    # Extract closing prices
    # yfinance returns MultiIndex columns: (Price Type, Ticker)
    prices = None
    
    if len(tickers) == 1:
        # Single ticker case
        ticker = tickers[0]
        print(f"Processing single ticker: {ticker}")
        
        if isinstance(data, pd.Series):
            # data is already a Series (single ticker, single price type)
            print(f"  → Data is a Series: {len(data)} rows")
            prices = data.to_frame(name=ticker)
        elif isinstance(data, pd.DataFrame):
            # data is a DataFrame
            print(f"  → Data is DataFrame with columns: {list(data.columns)}")
            if 'Close' in data.columns:
                close_col = data['Close']
                if isinstance(close_col, pd.Series):
                    print(f"    → Using 'Close' column: {len(close_col)} rows")
                    prices = close_col.to_frame(name=ticker)
                else:
                    # Single scalar - need to handle this case
                    print(f"⚠️  'Close' column is not a Series: {type(close_col)}")
            else:
                # Try to use the first column
                first_col = data.iloc[:, 0]
                print(f"    → Using first column: {first_col.name}, {len(first_col)} rows")
                if isinstance(first_col, pd.Series) and len(first_col) > 0:
                    prices = first_col.to_frame(name=ticker)
                else:
                    print(f"⚠️  Could not extract valid price data for {ticker}")
    else:
        # Multiple tickers case
        print(f"Processing multiple tickers: {len(tickers)}")
        print(f"  → Data columns type: {type(data.columns)}")
        
        if isinstance(data.columns, pd.MultiIndex):
            # Access by price type: 'Close' is at level 0
            print(f"  → MultiIndex detected, levels: {data.columns.names}")
            try:
                prices = data.xs('Close', level=0, axis=1)
                print(f"    → Extracted 'Close' prices: {prices.shape}")
            except KeyError:
                # 'Close' not in MultiIndex, try first level
                print(f"⚠️  'Close' not in MultiIndex, using first level")
                prices = data
        elif 'Close' in data.columns:
            # Simple columns, Close is available
            print(f"  → Simple columns with 'Close'")
            prices = data[['Close']]
            prices.columns = [col[1] if isinstance(col, tuple) else col for col in prices.columns]
        else:
            # Fallback: use all data
            print(f"⚠️  No 'Close' column found, using all data")
            prices = data
    
    # Check if we got any prices
    if prices is None:
        print("⚠️  prices is None after processing")
        return pd.DataFrame()
    
    # Ensure prices is a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    print(f"Prices shape before filtering: {prices.shape}")
    print(f"Prices columns: {list(prices.columns)}")
    
    # Filter to only include tickers that were successfully downloaded
    if len(prices.columns) > 0:
        available_cols = [col for col in tickers if col in prices.columns]
        print(f"Available tickers: {available_cols}")
        if len(available_cols) > 0:
            prices = prices[available_cols]
        else:
            # No requested tickers found, just keep what we have
            print(f"⚠️  No requested tickers found, keeping all columns: {list(prices.columns)}")
    
    # Remove rows with NaN values
    print(f"Rows before dropna: {len(prices)}")
    initial_rows = len(prices)
    
    if initial_rows == 0:
        print("⚠️  CRITICAL: prices DataFrame is empty!")
        return prices
    
    # Try strict approach first: remove any row with NaN
    prices_strict = prices.dropna(how='any')
    
    # If we lose too much data (more than 50%), try a looser approach
    if len(prices_strict) < len(prices) * 0.5:
        print(f"⚠️  Strict dropna would remove {len(prices) - len(prices_strict)} rows ({(len(prices) - len(prices_strict))/len(prices)*100:.1f}%)")
        print(f"    Using looser approach: keep rows with 80%+ data coverage...")
        prices = prices.dropna(thresh=len(prices.columns) * 0.8)
    else:
        prices = prices_strict
    
    print(f"Rows after dropna: {len(prices)}")
    
    if initial_rows > 0 and len(prices) == 0:
        print(f"⚠️  CRITICAL: All {initial_rows} rows were removed!")
        print(f"    The first few rows had these values:")
        print(prices_strict.head(3))
    
    if len(prices) > 0:
        print(f"\n✓ Downloaded data shape: {prices.shape}")
        print(f"✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"✓ Successfully loaded tickers: {list(prices.columns)}")
    else:
        print("⚠️  No valid data after filtering NaN values")
    
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
    
    # Extract close prices flexibly
    if isinstance(vix, pd.DataFrame):
        # Try different column names
        if 'Adj Close' in vix.columns:
            vix_series = vix['Adj Close']
        elif 'Close' in vix.columns:
            vix_series = vix['Close']
        elif isinstance(vix.columns, pd.MultiIndex):
            # Handle MultiIndex columns
            try:
                vix_series = vix.xs('Adj Close', level=1, axis=1).iloc[:, 0]
            except (KeyError, IndexError):
                try:
                    vix_series = vix.xs('Close', level=1, axis=1).iloc[:, 0]
                except (KeyError, IndexError):
                    # Fallback to first column
                    vix_series = vix.iloc[:, 0]
        else:
            # Not MultiIndex, use first column
            vix_series = vix.iloc[:, 0]
    else:
        # Single series returned
        vix_series = vix
    
    # Ensure it's a Series
    if isinstance(vix_series, pd.DataFrame):
        vix_series = vix_series.iloc[:, 0]
    
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
