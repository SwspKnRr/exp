"""
ML Model Module
Trains RandomForestClassifier to predict future positive returns.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class MomentumClassifier:
    """Random Forest classifier for predicting positive returns."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 20,
        random_state: int = 42,
        prediction_horizon: int = 5
    ):
        """
        Initialize momentum classifier.
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in forest
        max_depth : int
            Max tree depth
        min_samples_split : int
            Min samples to split node
        random_state : int
            Random seed
        prediction_horizon : int
            Days ahead to predict (for target creation)
        """
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.prediction_horizon = prediction_horizon
        self.importance = None
    
    def create_target(
        self,
        prices: pd.Series,
        horizon: int = None
    ) -> pd.Series:
        """
        Create binary target (1 if future return > 0, else 0).
        
        Parameters
        ----------
        prices : pd.Series
            Price data
        horizon : int, optional
            Days ahead. If None, uses self.prediction_horizon
        
        Returns
        -------
        pd.Series
            Binary target (index offset by horizon)
        """
        
        if horizon is None:
            horizon = self.prediction_horizon
        
        # Calculate forward returns
        forward_returns = np.log(prices / prices.shift(-horizon))
        
        # Create binary target
        target = (forward_returns > 0).astype(int)
        
        return target
    
    def prepare_features(
        self,
        momentum_features: pd.DataFrame,
        volatility_features: pd.DataFrame,
        macro_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Combine all features for model.
        
        Parameters
        ----------
        momentum_features : pd.DataFrame
            Momentum-based features
        volatility_features : pd.DataFrame
            Volatility-based features
        macro_features : pd.DataFrame
            Macro indicators
        
        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            Combined features and feature names
        """
        
        combined = pd.DataFrame(index=momentum_features.index)
        
        # Add momentum features
        for col in momentum_features.columns:
            combined[col] = momentum_features[col]
        
        # Add volatility features
        for col in volatility_features.columns:
            combined[col] = volatility_features[col]
        
        # Add macro features
        for col in macro_features.columns:
            combined[col] = macro_features[col]
        
        # Remove rows with NaN
        combined = combined.dropna()
        
        feature_names = combined.columns.tolist()
        
        return combined, feature_names
    
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Fit model on training data.
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        target : pd.Series
            Binary target
        validation_split : float
            Fraction of data for validation
        
        Returns
        -------
        Dict[str, float]
            Training and validation metrics
        """
        
        # Align features and target
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        # Remove rows with NaN target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            print(f"Warning: Only {len(X)} samples available for training")
        
        # Split into train/val
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Remove NaN values
        train_valid = X_train.notna().all(axis=1)
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        
        val_valid = X_val.notna().all(axis=1)
        X_val = X_val[val_valid]
        y_val = y_val[val_valid]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        self.feature_names = X_train.columns.tolist()
        
        # Calculate metrics
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        # Feature importance
        self.importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        metrics = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
        
        return metrics
    
    def predict_proba(
        self,
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict probability of positive return.
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        np.ndarray
            Probability of class 1 (positive return)
        """
        
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Ensure feature order matches training
        features_ordered = features[self.feature_names].copy()
        
        # Scale features
        features_scaled = self.scaler.transform(features_ordered)
        
        # Predict probabilities
        proba = self.model.predict_proba(features_scaled)
        
        return proba[:, 1]  # Probability of class 1
    
    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """
        Get feature importance ranking.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
        
        Returns
        -------
        pd.Series
            Feature importance scores
        """
        
        if self.importance is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.importance.head(top_n)


from typing import Tuple, Dict, List


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../data')
    sys.path.insert(0, '../features')
    
    from loader import download_price_data, download_vix_data, ALL_TICKERS
    from momentum import create_momentum_features
    from volatility import create_volatility_features
    from macro import create_macro_features
    
    # Load data
    prices = download_price_data(ALL_TICKERS, start_date='2015-01-01')
    vix = download_vix_data(start_date='2015-01-01')
    
    # Create features (for SPY only)
    momentum_features = create_momentum_features(prices[['SPY']])
    volatility_features = create_volatility_features(prices[['SPY']])
    macro_features = create_macro_features(prices[['SPY', 'GLD']], vix)
    
    # Initialize and train classifier
    clf = MomentumClassifier(prediction_horizon=5)
    
    # Prepare features
    X, feature_names = clf.prepare_features(
        momentum_features, volatility_features, macro_features
    )
    
    # Create target
    y = clf.create_target(prices['SPY'], horizon=5)
    
    # Fit model
    metrics = clf.fit(X, y, validation_split=0.2)
    print("\nModel Training Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    print(clf.get_feature_importance(top_n=10))
