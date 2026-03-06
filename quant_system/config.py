"""
Strategy Configuration & Parameters
All key strategy parameters in one place for easy reference and modification.
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Date Range for Backtesting
BACKTEST_START_DATE = '2010-01-01'
BACKTEST_END_DATE = None  # None = today

# Market Data
PRICE_DATA_INTERVAL = 'daily'  # Only daily supported
PRICE_COLUMN = 'Adj Close'

# ============================================================================
# ETF UNIVERSE (25 Total)
# ============================================================================

ETF_UNIVERSE = {
    'Broad Market': {
        'SPY': 'S&P 500 ETF Trust',
        'QQQ': 'Invesco QQQ ETF',
        'VTI': 'Vanguard Total Stock Market ETF'
    },
    'Sectors': {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Health Care',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate'
    },
    'Thematic': {
        'SMH': 'Semiconductors',
        'LIT': 'Lithium/Battery Tech',
        'ARKK': 'Innovation (Growth)',
        'ITA': 'Aerospace/Defense',
        'ITB': 'Homebuilders'
    },
    'Bonds & Commodities': {
        'GLD': 'Gold ETF',
        'TLT': '20+ Year Treasury Bonds'
    }
}

BENCHMARK_TICKER = 'SPY'
EXCLUDE_FROM_SELECTION = ['SPY']  # Don't select benchmark in portfolio

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Momentum Features
MOMENTUM_PERIODS = [5, 10, 20, 60, 120]  # Days
MA_PERIODS = [20, 60]  # Moving average periods
RSI_PERIOD = 14

# Volatility Features
VOLATILITY_PERIODS = [20, 60]  # Days
ATR_PERIOD = 14

# Macro Features
VIX_TICKER = '^VIX'
MACRO_LOOKBACK = 20  # Days for macro calculations

# ============================================================================
# MARKET REGIME FILTER
# ============================================================================

REGIME_BENCHMARK = 'SPY'
REGIME_MA_PERIOD = 200  # 200-day moving average
REGIME_USE_SIMPLE_MA = True  # True=SMA, False=EMA

# Rule: SPY > MA200 → UPTREND (active)
#       SPY ≤ MA200 → DOWNTREND (no positions)

# ============================================================================
# DUAL MOMENTUM STRATEGY
# ============================================================================

# Momentum Calculation
MOMENTUM_PERIOD = 20  # Days

# Conditions (both must be true):
# 1. Absolute: ETF_return_20d > 0
# 2. Relative: ETF_return_20d > SPY_return_20d

REQUIRE_BOTH_CONDITIONS = True

# ============================================================================
# ETF RANKING SYSTEM
# ============================================================================

# Momentum Score Formula
# score = w1*return_20d + w2*return_60d + w3*return_120d

RANKING_WEIGHTS = {
    20: 0.4,   # 40% weight to 20-day return
    60: 0.3,   # 30% weight to 60-day return
    120: 0.3   # 30% weight to 120-day return
}

MAX_POSITIONS = 3
MIN_POSITIONS = 1

# Ranking methods: 'equal', 'momentum_score', 'inverse_rank'
RANKING_METHOD = 'equal'

# ============================================================================
# ML SIGNAL FILTER
# ============================================================================

# RandomForestClassifier Parameters
ML_MODEL_TYPE = 'RandomForestClassifier'
ML_N_ESTIMATORS = 100
ML_MAX_DEPTH = 10
ML_MIN_SAMPLES_SPLIT = 20
ML_RANDOM_STATE = 42

# Prediction Target
ML_PREDICTION_HORIZON = 5  # Days ahead

# Signal Acceptance
ML_PROBABILITY_THRESHOLD = 0.60  # Accept signal if P(positive return) > 60%

# Training Data Split
ML_VALIDATION_SPLIT = 0.20  # 20% for validation

# ============================================================================
# PORTFOLIO PARAMETERS
# ============================================================================

# Initial Capital
INITIAL_CAPITAL = 100000

# Allocation
ALLOCATION_METHOD = 'equal'  # 'equal' = 1/N allocation, 'momentum_score', 'inverse_rank'
WEIGHT_PER_POSITION = 1.0 / MAX_POSITIONS  # Equal weight: 33.3% each

# Rebalancing
REBALANCE_DAY = 'Monday'  # Monday, Tuesday, ... Sunday
REBALANCE_FREQUENCY = 'weekly'

# Transaction Costs
TRANSACTION_COST = 0.0005  # 0.05% per trade
MARKET_IMPACT = 0.0  # Additional market impact cost (optional)

# ============================================================================
# BACKTEST ENGINE PARAMETERS
# ============================================================================

# Slippage & Execution
USE_CLOSE_PRICE = True  # Use close price for entry/exit
SLIPPAGE = 0.0  # Additional slippage (as %)

# Portfolio Tracking
TRACK_DAILY_CHANGES = True
TRACK_POSITIONS = True

# ============================================================================
# WALK-FORWARD ANALYSIS PARAMETERS
# ============================================================================

WALK_FORWARD_TRAIN_YEARS = 5  # Training window
WALK_FORWARD_TEST_YEARS = 1   # Testing window
WALK_FORWARD_STEP = 1         # Years to move forward each iteration

# Example:
# Train: 2010-2014, Test: 2015
# Train: 2011-2015, Test: 2016
# ... continues until end of data

# ============================================================================
# PERFORMANCE METRICS PARAMETERS
# ============================================================================

# Risk-Free Rate (for Sharpe/Sortino)
RISK_FREE_RATE = 0.02  # 2% annual

# Trading Days per Year
TRADING_DAYS_PER_YEAR = 252

# Minimum Holding Period Analysis
HOLDING_PERIODS = [1, 5, 20, 60]  # Days

# ============================================================================
# REPORTING & VISUALIZATION
# ============================================================================

# Dashboard Settings
STREAMLIT_PORT = 8501
STREAMLIT_THEME = 'light'  # 'light' or 'dark'

# Chart Colors
COLOR_UPTREND = 'green'
COLOR_DOWNTREND = 'red'
COLOR_EQUITY = 'blue'
COLOR_DRAWDOWN = 'orange'

# Metrics Display Precision
DISPLAY_DECIMALS = 2
DISPLAY_PERCENT_DECIMALS = 2

# ============================================================================
# ADVANCED PARAMETERS
# ============================================================================

# Missing Data Handling
FILL_METHOD = 'forward'  # 'forward', 'back', 'interpolate'
MAX_MISSING_DAYS = 5

# Outlier Detection
REMOVE_OUTLIERS = False
OUTLIER_THRESHOLD = 5  # Standard deviations

# Correlation Matrix
MIN_CORRELATION_THRESHOLD = -1.0  # Include all
MAX_CORRELATION_THRESHOLD = 1.0

# ============================================================================
# OPTIMIZATION PARAMETERS (for parameter search)
# ============================================================================

# Momentum Period variations to test
OPTIMIZE_MOMENTUM_PERIODS = [10, 15, 20, 25, 30]

# Regime MA periods to test
OPTIMIZE_REGIME_MA_PERIODS = [150, 200, 250]

# Max positions to test
OPTIMIZE_MAX_POSITIONS = [1, 2, 3, 4, 5]

# ML thresholds to test
OPTIMIZE_ML_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

VERBOSE = True  # Print detailed progress
DEBUG_MODE = False
LOG_TRADES = True
LOG_REBALANCES = True

# ============================================================================
# CONFIGURATION NOTES
#
# 1. REGIME FILTER: Controls when strategy is active
#    - Only enter positions when SPY > 200-day MA
#    - This filters out weak market conditions
#
# 2. DUAL MOMENTUM: Two-stage filtering
#    - Absolute: Returns must be positive
#    - Relative: Must outperform SPY (benchmark)
#    - Both must be true → robust signal
#
# 3. RANKING: Multi-period weighted score
#    - 20-day return (40%)
#    - 60-day return (30%)
#    - 120-day return (30%)
#    - Balances short & medium term trends
#
# 4. ML FILTER: Predict positive returns using features
#    - Reduces false signals
#    - Uses historical data to learn patterns
#    - Requires sufficient training data
#
# 5. WALK-FORWARD: Robust backtesting technique
#    - Trains on 5 years, tests on 1 year
#    - Moves forward 1 year at a time
#    - Prevents look-ahead bias
#    - More realistic out-of-sample testing
#
# 6. TRANSACTION COSTS: Critical for realistic results
#    - 0.05% = typical cost for liquid ETFs
#    - Weekly rebalancing = ~4 trades/position/month
#    - Impacts strategy profitability significantly
#
# ============================================================================

# Example custom configuration for more aggressive strategy

class AggressiveConfig:
    """Aggressive version - higher momentum thresholds."""
    MOMENTUM_PERIOD = 10  # Shorter lookback
    MAX_POSITIONS = 5  # More positions
    ML_PROBABILITY_THRESHOLD = 0.55  # Lower threshold


class ConservativeConfig:
    """Conservative version - lower risk."""
    MOMENTUM_PERIOD = 60  # Longer lookback (trend following)
    MAX_POSITIONS = 2  # Fewer positions
    ML_PROBABILITY_THRESHOLD = 0.70  # Higher threshold
    WALK_FORWARD_TRAIN_YEARS = 7  # More training data


if __name__ == '__main__':
    print("Strategy Configuration Loaded")
    print(f"Total ETFs in universe: {sum(len(v) for v in ETF_UNIVERSE.values())}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Max Positions: {MAX_POSITIONS}")
    print(f"Transaction Cost: {TRANSACTION_COST:.2%}")
    print(f"Walk-Forward: Train {WALK_FORWARD_TRAIN_YEARS}y, Test {WALK_FORWARD_TEST_YEARS}y")
