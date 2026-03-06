"""
ML Signal Filter
Applies ML model predictions to trading signals.
Accept trades if probability > 0.6, reject otherwise.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .model import MomentumClassifier


class MLSignalFilter:
    """Apply ML model to filter trading signals."""
    
    def __init__(
        self,
        probability_threshold: float = 0.6,
        model: MomentumClassifier = None
    ):
        """
        Initialize ML signal filter.
        
        Parameters
        ----------
        probability_threshold : float
            Probability threshold for accepting trade (default 0.6)
        model : MomentumClassifier, optional
            Pre-trained model. If None, creates new one.
        """
        
        self.threshold = probability_threshold
        self.model = model if model is not None else MomentumClassifier()
    
    def train_model(
        self,
        momentum_features: pd.DataFrame,
        volatility_features: pd.DataFrame,
        macro_features: pd.DataFrame,
        prices: pd.DataFrame,
        ticker: str,
        prediction_horizon: int = 5,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train ML model for a specific ticker.
        
        Parameters
        ----------
        momentum_features : pd.DataFrame
            Momentum features for ticker
        volatility_features : pd.DataFrame
            Volatility features for ticker
        macro_features : pd.DataFrame
            Macro features
        prices : pd.DataFrame
            Price data for ticker
        ticker : str
            Ticker symbol
        prediction_horizon : int
            Days ahead to predict
        validation_split : float
            Fraction of data for validation
        
        Returns
        -------
        Dict[str, float]
            Training metrics
        """
        
        # Prepare features
        X, feature_names = self.model.prepare_features(
            momentum_features, volatility_features, macro_features
        )
        
        # Create target
        y = self.model.create_target(prices[ticker], horizon=prediction_horizon)
        
        # Fit model
        metrics = self.model.fit(X, y, validation_split=validation_split)
        
        return metrics
    
    def filter_signals(
        self,
        signals: pd.DataFrame,
        momentum_features: pd.DataFrame,
        volatility_features: pd.DataFrame,
        macro_features: pd.DataFrame,
        return_probabilities: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply ML filter to trading signals.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Binary trading signals (rows=dates, cols=tickers)
        momentum_features : pd.DataFrame
            Momentum features
        volatility_features : pd.DataFrame
            Volatility features
        macro_features : pd.DataFrame
            Macro features
        return_probabilities : bool
            Whether to return probability scores
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Filtered signals and (optionally) probabilities
        """
        
        if not self.model.is_fitted:
            raise ValueError("Model not fitted. Call train_model() first.")
        
        # Prepare features
        X, feature_names = self.model.prepare_features(
            momentum_features, volatility_features, macro_features
        )
        
        # Get predictions
        probabilities = self.model.predict_proba(X)
        
        # Create probability dataframe
        proba_df = pd.DataFrame(
            index=X.index,
            columns=signals.columns,
            data=probabilities
        )
        
        # Align with signals
        proba_aligned = proba_df.reindex(signals.index, fill_value=0.5)
        
        # Filter signals: accept only if proba > threshold
        filtered_signals = signals.copy()
        filtered_signals[proba_aligned <= self.threshold] = 0
        
        if return_probabilities:
            return filtered_signals, proba_aligned
        else:
            return filtered_signals
    
    def filter_portfolio_signals(
        self,
        signals: pd.DataFrame,
        momentum_features: pd.DataFrame,
        volatility_features: pd.DataFrame,
        macro_features: pd.DataFrame,
        min_probability: float = None
    ) -> pd.DataFrame:
        """
        Filter portfolio-level signals.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Portfolio signals (rows=dates, cols=tickers)
        momentum_features : pd.DataFrame
            Momentum features
        volatility_features : pd.DataFrame
            Volatility features
        macro_features : pd.DataFrame
            Macro features
        min_probability : float, optional
            Minimum probability. If None, uses self.threshold
        
        Returns
        -------
        pd.DataFrame
            Filtered signals
        """
        
        if min_probability is None:
            min_probability = self.threshold
        
        filtered_signals, probabilities = self.filter_signals(
            signals,
            momentum_features,
            volatility_features,
            macro_features,
            return_probabilities=True
        )
        
        # For portfolio signals, set to 0 if probability too low
        filtered_signals[probabilities < min_probability] = 0
        
        return filtered_signals
    
    def get_signal_confidence(
        self,
        signals: pd.DataFrame,
        probabilities: pd.DataFrame
    ) -> pd.Series:
        """
        Get average confidence of active signals.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Trading signals
        probabilities : pd.DataFrame
            Probability scores
        
        Returns
        -------
        pd.Series
            Average probability for active signals per date
        """
        
        # Where signals are active (1), get average probability
        confidence = pd.Series(index=signals.index, dtype=float)
        
        for date in signals.index:
            active_signals = signals.loc[date] == 1
            if active_signals.sum() > 0:
                confidence[date] = probabilities.loc[date, active_signals].mean()
            else:
                confidence[date] = 0
        
        return confidence


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    sys.path.insert(0, '../features')
    sys.path.insert(0, '../strategy')
    
    from loader import download_price_data, download_vix_data, ALL_TICKERS
    from momentum import create_momentum_features
    from volatility import create_volatility_features
    from macro import create_macro_features
    from dual_momentum import DualMomentum
    
    # Load data
    prices = download_price_data(ALL_TICKERS, start_date='2015-01-01')
    vix = download_vix_data(start_date='2015-01-01')
    
    # Create features (example for SPY)
    momentum_features = create_momentum_features(prices[['SPY']])
    volatility_features = create_volatility_features(prices[['SPY']])
    macro_features = create_macro_features(prices[['SPY', 'GLD']], vix)
    
    # Create signals
    dm = DualMomentum(momentum_period=20, benchmark='SPY')
    signals = dm.generate_signals(prices[['SPY']])
    
    # Initialize filter
    ml_filter = MLSignalFilter(probability_threshold=0.6)
    
    # Train model
    metrics = ml_filter.train_model(
        momentum_features, volatility_features, macro_features,
        prices, 'SPY', prediction_horizon=5
    )
    
    print("\nML Model Training Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Filter signals
    filtered_signals = ml_filter.filter_signals(
        signals, momentum_features, volatility_features, macro_features
    )
    
    print(f"\nOriginal signals sum: {signals.sum().sum()}")
    print(f"Filtered signals sum: {filtered_signals.sum().sum()}")
    print(f"Acceptance ratio: {filtered_signals.sum().sum() / signals.sum().sum():.2%}")
