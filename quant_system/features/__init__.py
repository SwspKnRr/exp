"""Feature engineering modules for momentum, volatility, and macro indicators."""
from .momentum import (
    calculate_returns,
    calculate_moving_average_ratios,
    calculate_rsi,
    create_momentum_features
)
from .volatility import (
    calculate_atr,
    calculate_volatility,
    calculate_atr_ratio,
    create_volatility_features
)
from .macro import (
    calculate_vix_features,
    calculate_spy_gld_ratio,
    calculate_bond_stock_ratio,
    create_macro_features,
    calculate_regime_filter
)

__all__ = [
    'calculate_returns',
    'calculate_moving_average_ratios',
    'calculate_rsi',
    'create_momentum_features',
    'calculate_atr',
    'calculate_volatility',
    'calculate_atr_ratio',
    'create_volatility_features',
    'calculate_vix_features',
    'calculate_spy_gld_ratio',
    'calculate_bond_stock_ratio',
    'create_macro_features',
    'calculate_regime_filter'
]
