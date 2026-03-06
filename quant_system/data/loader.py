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
    
    print(f"\n{'='*60}")
    print(f"📥 데이터 다운로드 시작")
    print(f"{'='*60}")
    print(f"기간: {start_date} ~ {end_date}")
    print(f"티커: {tickers}")
    print(f"{'='*60}\n")
    
    try:
        # Download data
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            ignore_tz=True
        )
        print(f"✓ yfinance 다운로드 성공")
    except Exception as e:
        print(f"❌ yfinance 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    # Check if data is empty
    print(f"📊 다운로드된 데이터 타입: {type(data)}")
    print(f"📊 데이터 형태: {data.shape if hasattr(data, 'shape') else 'N/A'}")
    
    if data is None:
        print("❌ yfinance가 None을 반환했습니다")
        return pd.DataFrame()
    
    if isinstance(data, pd.DataFrame) and data.empty:
        print("❌ yfinance가 빈 DataFrame을 반환했습니다")
        return pd.DataFrame()
    
    if isinstance(data, pd.Series) and len(data) == 0:
        print("❌ yfinance가 빈 Series를 반환했습니다")
        return pd.DataFrame()
    
    # Extract closing prices
    prices = None
    
    if len(tickers) == 1:
        # Single ticker case
        ticker = tickers[0]
        print(f"\n📍 단일 티커 처리: {ticker}")
        
        if isinstance(data, pd.Series):
            print(f"  → Series 형태: {len(data)} 행")
            prices = data.to_frame(name=ticker)
        elif isinstance(data, pd.DataFrame):
            print(f"  → DataFrame 형태")
            print(f"     컬럼: {list(data.columns)}")
            
            if 'Close' in data.columns:
                close_col = data['Close']
                print(f"     'Close' 컬럼 사용: {len(close_col)} 행")
                if isinstance(close_col, pd.Series):
                    prices = close_col.to_frame(name=ticker)
                else:
                    print(f"     ⚠️  'Close'이 Series가 아님: {type(close_col)}")
            else:
                first_col = data.iloc[:, 0]
                print(f"     첫 번째 컬럼 사용: {data.columns[0]}, {len(first_col)} 행")
                if isinstance(first_col, pd.Series) and len(first_col) > 0:
                    prices = first_col.to_frame(name=ticker)
                else:
                    print(f"     ⚠️  첫 컬럼 추출 실패")
    else:
        # Multiple tickers case
        print(f"\n📍 다중 티커 처리: {len(tickers)}개")
        
        if isinstance(data.columns, pd.MultiIndex):
            print(f"  → MultiIndex 컬럼")
            try:
                prices = data.xs('Close', level=0, axis=1)
                print(f"     'Close' 추출 성공: {prices.shape}")
            except KeyError:
                print(f"     ⚠️  'Close' 레벨 없음")
                prices = data
        elif 'Close' in data.columns:
            print(f"  → Simple 컬럼 with 'Close'")
            prices = data[['Close']]
        else:
            print(f"  → 'Close' 없음, 모든 데이터 사용")
            prices = data
    
    # Check if we got any prices
    if prices is None:
        print("\n❌ prices가 None입니다")
        return pd.DataFrame()
    
    # Ensure prices is a DataFrame
    if isinstance(prices, pd.Series):
        print(f"\nSeries를 DataFrame으로 변환")
        prices = prices.to_frame()
    
    print(f"\n📊 필터링 전:")
    print(f"   형태: {prices.shape}")
    print(f"   컬럼: {list(prices.columns)}")
    print(f"   인덱스 범위: {prices.index[0]} ~ {prices.index[-1]}")
    
    # Filter to only include tickers that were successfully downloaded
    if len(prices.columns) > 0:
        available_cols = [col for col in tickers if col in prices.columns]
        print(f"   요청 티커: {tickers}")
        print(f"   수신 티커: {available_cols}")
        if len(available_cols) > 0:
            prices = prices[available_cols]
        else:
            print(f"   ⚠️  요청 티커가 없습니다. 모든 컬럼 유지: {list(prices.columns)}")
    
    # Remove rows with NaN values
    initial_len = len(prices)
    prices_strict = prices.dropna(how='any')
    
    print(f"\n🧹 NaN 제거:")
    print(f"   제거 전: {initial_len} 행")
    print(f"   제거 후: {len(prices_strict)} 행")
    
    if len(prices_strict) < len(prices) * 0.5:
        print(f"   ⚠️  50% 이상 손실! 유연한 방식 사용...")
        prices = prices.dropna(thresh=len(prices.columns) * 0.8)
        print(f"   완화 후: {len(prices)} 행 (80%+ 커버리지)")
    else:
        prices = prices_strict
    
    # Final result
    print(f"\n✅ 최종 결과:")
    if len(prices) > 0:
        print(f"   데이터: {prices.shape[0]} 행 × {prices.shape[1]} 컬럼")
        print(f"   기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")
        print(f"   티커: {list(prices.columns)}")
    else:
        print(f"   ❌ 데이터 없음!")
    
    print(f"{'='*60}\n")
    
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
