"""
Day Trading Strategy Module
Intraday trading signals using short-term technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class DayTradingSignals:
    """Generate intraday trading signals using RSI, MACD, Bollinger Bands."""
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0
    ):
        """
        Initialize day trading strategy.
        
        Parameters
        ----------
        rsi_period : int
            RSI lookback period
        rsi_overbought : float
            RSI overbought threshold
        rsi_oversold : float
            RSI oversold threshold
        macd_fast : int
            Fast EMA period for MACD
        macd_slow : int
            Slow EMA period for MACD
        macd_signal : int
            Signal line period for MACD
        bb_period : int
            Bollinger Bands period
        bb_std : float
            Bollinger Bands standard deviation
        """
        
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Parameters
        ----------
        prices : pd.Series
            Price data
        period : int, optional
            RSI period
        
        Returns
        -------
        pd.Series
            RSI values (0-100)
        """
        
        if period is None:
            period = self.rsi_period
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Parameters
        ----------
        prices : pd.Series
            Price data
        
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            MACD line, Signal line, Histogram
        """
        
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = None,
        std: float = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Parameters
        ----------
        prices : pd.Series
            Price data
        period : int, optional
            BB period
        std : float, optional
            Standard deviation multiplier
        
        Returns
        -------
        Tuple[pd.Series, pd.Series, pd.Series]
            Upper band, Middle band (SMA), Lower band
        """
        
        if period is None:
            period = self.bb_period
        if std is None:
            std = self.bb_std
        
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    def generate_signals(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate buy/sell signals for each ticker.
        
        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV data or close prices
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Signals for each ticker (1=buy, -1=sell, 0=hold)
        """
        
        signals = {}
        
        # Ensure prices is a DataFrame
        if isinstance(prices, pd.Series):
            close_prices = pd.DataFrame({prices.name or 'price': prices})
        elif not isinstance(prices, pd.DataFrame):
            close_prices = pd.DataFrame(prices)
        else:
            close_prices = prices
        
        # Get column names (tickers)
        tickers = close_prices.columns.tolist()
        
        for ticker in tickers:
            price = close_prices[ticker]
            
            # Skip if insufficient data
            if len(price) < max(self.bb_period, self.macd_slow):
                continue
            
            # Calculate indicators
            rsi = self.calculate_rsi(price)
            macd, signal, hist = self.calculate_macd(price)
            upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(price)
            
            # Generate signals
            signal_series = pd.Series(0, index=price.index)
            
            # RSI-based signals
            rsi_buy = rsi < self.rsi_oversold  # Oversold = buy
            rsi_sell = rsi > self.rsi_overbought  # Overbought = sell
            
            # MACD-based signals
            macd_buy = (macd > signal) & (macd.shift(1) <= signal.shift(1))  # Bullish crossover
            macd_sell = (macd < signal) & (macd.shift(1) >= signal.shift(1))  # Bearish crossover
            
            # Bollinger Bands-based signals
            bb_buy = price < lower_bb  # Price breaks lower band
            bb_sell = price > upper_bb  # Price breaks upper band
            
            # Combine signals (majority rules)
            buy_signals = (rsi_buy.astype(int) + macd_buy.astype(int) + bb_buy.astype(int)) >= 2
            sell_signals = (rsi_sell.astype(int) + macd_sell.astype(int) + bb_sell.astype(int)) >= 2
            
            signal_series[buy_signals] = 1
            signal_series[sell_signals] = -1
            
            signals[ticker] = pd.DataFrame({
                'signal': signal_series,
                'rsi': rsi,
                'macd': macd,
                'signal_line': signal,
                'histogram': hist,
                'upper_bb': upper_bb,
                'middle_bb': middle_bb,
                'lower_bb': lower_bb,
                'price': price
            })
        
        return signals
    
    def calculate_backtest_metrics(self, prices: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate reliability metrics based on historical signals and price movements.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        Dict[str, Dict]
            Metrics including win rate, avg returns, signal frequency
        """
        
        signals_dict = self.generate_signals(prices)
        metrics = {}
        
        for ticker, indicators in signals_dict.items():
            price = indicators['price']
            signal = indicators['signal']
            
            # Get signal points (buy=1, sell=-1)
            buy_signals = signal == 1
            sell_signals = signal == -1
            
            buy_dates = signal[buy_signals].index.tolist()
            sell_dates = signal[sell_signals].index.tolist()
            
            # Calculate returns from buy signals
            returns = []
            win_count = 0
            loss_count = 0
            
            for buy_date in buy_dates:
                buy_price = price.loc[buy_date]
                
                # Find next sell signal or use next prices
                future_dates = price.index[price.index > buy_date]
                if len(future_dates) == 0:
                    continue
                
                # Calculate 5-day returns as benchmark
                if len(future_dates) >= 5:
                    sell_price = price.loc[future_dates[4]]
                else:
                    sell_price = price.iloc[-1]
                
                ret = (sell_price - buy_price) / buy_price
                returns.append(ret)
                
                if ret > 0:
                    win_count += 1
                else:
                    loss_count += 1
            
            total_signals = len(buy_signals[buy_signals])
            win_rate = (win_count / (win_count + loss_count)) * 100 if (win_count + loss_count) > 0 else 0
            avg_return = np.mean(returns) * 100 if returns else 0
            
            metrics[ticker] = {
                'total_buy_signals': len(buy_dates),
                'total_sell_signals': len(sell_dates),
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'avg_return_pct': avg_return,
                'returns': returns
            }
        
        return metrics
    
    def get_signal_history(self, prices: pd.DataFrame, ticker: str, lookback_days: int = 100) -> pd.DataFrame:
        """
        Get historical buy/hold/sell signals.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        ticker : str
            Ticker symbol
        lookback_days : int
            How many recent days to include
        
        Returns
        -------
        pd.DataFrame
            History with signals and prices
        """
        
        signals_dict = self.generate_signals(prices)
        
        if ticker not in signals_dict:
            return pd.DataFrame()
        
        indicators = signals_dict[ticker]
        
        # Get last N days
        if len(indicators) > lookback_days:
            indicators = indicators.tail(lookback_days)
        
        # Create result dataframe
        result = pd.DataFrame({
            '날짜': indicators.index,
            '가격': indicators['price'].values,
            '신호': indicators['signal'].values,
            'RSI': indicators['rsi'].values.round(2),
            'MACD': indicators['macd'].values.round(4),
            'Signal': indicators['signal_line'].values.round(4)
        })
        
        # Replace signal values with labels
        result['신호_이름'] = result['신호'].map({
            1: '🟢 매수',
            -1: '🔴 매도',
            0: '⚪ 관망'
        })
        
        # Calculate signal strength
        result['강도'] = result.apply(
            lambda row: self._calculate_signal_strength_row(row['RSI'], row['MACD']),
            axis=1
        )
        
        return result.reset_index(drop=True)
    
    def _calculate_signal_strength_row(self, rsi: float, macd: float) -> str:
        """Determine signal strength based on RSI and MACD."""
        
        if np.isnan(rsi) or np.isnan(macd):
            return '약함'
        
        # Strong when RSI is extreme and MACD is large
        rsi_extreme = np.abs(rsi - 50) > 20
        macd_strong = np.abs(macd) > 0.1
        
        if rsi_extreme and macd_strong:
            return '강함'
        elif rsi_extreme or macd_strong:
            return '보통'
        else:
            return '약함'
    
    def get_signal_strength(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate signal strength (0-1) for each ticker.
        Combines RSI extreme levels with MACD momentum.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        pd.DataFrame
            Signal strength for each ticker
        """
        
        signals_dict = self.generate_signals(prices)
        strength = {}
        
        for ticker, indicators in signals_dict.items():
            rsi = indicators['rsi']
            histogram = indicators['histogram']
            
            # RSI strength (normalized)
            rsi_strength = np.abs(rsi - 50) / 50  # 0-1 scale
            
            # MACD strength (normalized)
            macd_strength = histogram.rolling(20).std()
            macd_strength = (macd_strength - macd_strength.min()) / (macd_strength.max() - macd_strength.min() + 1e-8)
            
            # Combined strength
            combined = (rsi_strength * 0.5) + (macd_strength * 0.5)
            strength[ticker] = combined
        
        return pd.DataFrame(strength)
