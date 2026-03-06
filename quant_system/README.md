# Quantitative Trading System - Sector Rotation Strategy

A production-grade quantitative trading system implementing a sophisticated sector rotation strategy with dual momentum, market regime filtering, and machine learning signal enhancement for US ETFs.

## 📋 Overview

This system combines multiple advanced techniques to identify and trade the best-performing ETF sectors:

- **Dual Momentum**: Absolute momentum (return > 0) + Relative momentum (vs benchmark)
- **Market Regime Filter**: Only trade when SPY > 200-day MA
- **ML Signal Filter**: RandomForest classifier to predict positive returns
- **Walk-Forward Analysis**: Robust out-of-sample backtesting (5-year train, 1-year test)
- **Portfolio Rules**: Max 3 positions, equal weight, weekly rebalancing

## 🏗️ Project Structure

```
quant_system/
├── data/
│   ├── loader.py              # Download and preprocess price data
│   └── __init__.py
├── features/
│   ├── momentum.py            # Momentum features: returns, RSI, MAs
│   ├── volatility.py          # Volatility features: ATR, volatility
│   ├── macro.py               # Macro features: VIX, SPY/GLD ratio
│   └── __init__.py
├── strategy/
│   ├── regime_filter.py       # Market regime filter (SPY > MA200)
│   ├── dual_momentum.py       # Dual momentum signal generation
│   ├── ranking.py             # ETF ranking and portfolio selection
│   └── __init__.py
├── ml/
│   ├── model.py               # RandomForestClassifier for signal prediction
│   ├── filter.py              # ML-based signal filtering
│   └── __init__.py
├── backtest/
│   ├── engine.py              # Portfolio backtesting engine
│   ├── walk_forward.py        # Walk-forward analysis framework
│   ├── metrics.py             # Performance metrics calculation
│   └── __init__.py
├── dashboard/
│   ├── streamlit_app.py       # Interactive Streamlit dashboard
│   └── __init__.py
├── main.py                    # Complete working example
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🎯 Strategy Logic

### 1. Data Universe (25 ETFs)

**Broad Market** (3): SPY, QQQ, VTI

**Sectors** (10): XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE

**Thematic** (5): SMH, LIT, ARKK, ITA, ITB

**Bonds & Commodities** (2): GLD, TLT

### 2. Feature Engineering

#### Momentum Features
- 5d, 10d, 20d, 60d, 120d returns
- Price-to-MA20 and Price-to-MA60 ratios
- 14-day RSI

#### Volatility Features
- 20-day and 60-day rolling volatility
- ATR (Average True Range)
- ATR-to-Price ratio

#### Macro Features
- VIX level and change
- SPY-to-GLD ratio (risk-on/risk-off)
- Bond-to-Stock ratio

### 3. Market Regime Filter

```
SPY price > 200-day Moving Average → Strategy ACTIVE
SPY price ≤ 200-day Moving Average → NO POSITIONS
```

### 4. Dual Momentum Screening

**Absolute Momentum**: ETF return (20d) > 0

**Relative Momentum**: ETF return (20d) > SPY return (20d)

**Final Signal**: Both conditions must be TRUE

### 5. Ranking System

Momentum score per ETF:
```
score = 0.4 × return_20d + 0.3 × return_60d + 0.3 × return_120d
```

Select **top 3 ETFs** by score.

### 6. ML Signal Filter

**Model**: RandomForestClassifier

**Features**: All momentum + volatility + macro features

**Target**: Binary classification of 5-day forward return

**Rule**: Accept signal only if P(positive return) > 0.60

### 7. Portfolio Rules

- **Max Positions**: 3
- **Allocation**: Equal weight (33.3% each)
- **Rebalancing**: Weekly (Monday)
- **Transaction Cost**: 0.05% per trade

## 🔄 Walk-Forward Backtest

Robust out-of-sample testing using rolling windows:

```
Train: 2010-2014 | Test: 2015
Train: 2011-2015 | Test: 2016
Train: 2012-2016 | Test: 2017
...
Train: 2018-2022 | Test: 2023
```

Each period independnently trained and tested to avoid look-ahead bias.

## 📊 Performance Metrics

The system calculates:

- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted returns (vs SPY, rf=2%)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation
- **Win Rate**: % of profitable days
- **Profit Factor**: Gross profit / Gross loss

## 🚀 Getting Started

### Installation

```bash
# Clone repository
cd quant_system

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Backtest

```bash
python main.py
```

This will:
1. Download 15 years of ETF price data
2. Calculate all features (momentum, volatility, macro)
3. Analyze strategy components (regime, momentum, ranking)
4. Train ML signal filter
5. Run walk-forward backtest (2015-2023)
6. Print comprehensive performance metrics

### Launch Interactive Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Access dashboard at `http://localhost:8501`

Features:
- ETF ranking visualizations
- Market regime filter display
- Portfolio equity curve
- Performance metrics
- Feature statistics
- Walk-forward results

## 📈 Configuration

Key parameters can be adjusted in respective modules:

### BacktestEngine (backtest/engine.py)
```python
engine = BacktestEngine(
    initial_capital=100000,        # Starting capital
    transaction_cost=0.0005,       # 0.05% per trade
    rebalance_day='Monday',        # Rebalancing day
    max_positions=3                # Max portfolio size
)
```

### WalkForwardAnalyzer (backtest/walk_forward.py)
```python
analyzer = WalkForwardAnalyzer(
    train_years=5,                 # Training window
    test_years=1,                  # Testing window
    initial_capital=100000,
    transaction_cost=0.0005
)
```

### ETFRanker (strategy/ranking.py)
```python
ranker = ETFRanker(
    weights={20: 0.4, 60: 0.3, 120: 0.3},  # Period weights
    max_positions=3,
    min_positions=1
)
```

### MLSignalFilter (ml/filter.py)
```python
ml_filter = MLSignalFilter(
    probability_threshold=0.6      # Signal acceptance threshold
)
```

## 🔧 Advanced Usage

### Custom Feature Engineering

Add new features in `features/` modules:

```python
from features.momentum import create_momentum_features

# Create custom features
def create_custom_features(prices):
    return pd.DataFrame({
        'custom_1': ...,
        'custom_2': ...,
    })
```

### Custom Portfolio Selection

Implement custom selection function:

```python
def custom_selection_func(train_prices, train_features, date):
    # Your logic here
    return ['SPY', 'QQQ', 'VTI']

# Use in backtest
analyzer.run_walk_forward(prices, custom_selection_func)
```

### Parameter Optimization

Backtest different parameters:

```python
for train_years in [3, 5, 7]:
    for max_positions in [2, 3, 4]:
        analyzer = WalkForwardAnalyzer(
            train_years=train_years,
            max_positions=max_positions
        )
        # Run backtest...
```

## 📋 Key Features

✅ **Production-Quality Code**
- Modular design with clean separation of concerns
- Comprehensive docstrings and comments
- Type hints for better code clarity
- Error handling and validation

✅ **Robust Backtesting**
- Walk-forward analysis to avoid look-ahead bias
- Daily price updates with weekly rebalancing
- Transaction cost simulation
- Realistic portfolio tracking

✅ **Advanced Strategy**
- Multiple signal filters (regime, momentum, ML)
- Flexible ranking system
- Dynamic portfolio allocation
- Out-of-sample validation

✅ **Easy to Extend**
- Modular feature engineering
- Pluggable ML models
- Customizable selection functions
- Dashboard for visualization

## 📚 Module Reference

### data/loader.py
```python
download_price_data(tickers, start_date, end_date)  # Get OHLC data
download_vix_data(start_date, end_date)             # Get VIX index
create_macro_signals(prices, vix)                    # Create macro features
```

### features/
```python
create_momentum_features(prices)     # Momentum indicators
create_volatility_features(prices)   # Volatility indicators
create_macro_features(prices, vix)   # Macro indicators
```

### strategy/
```python
RegimeFilter()                       # Market regime detection
DualMomentum()                       # Dual momentum signals
ETFRanker()                          # ETF ranking and selection
```

### ml/
```python
MomentumClassifier()                 # RandomForest model
MLSignalFilter()                     # Signal filtering
```

### backtest/
```python
BacktestEngine.run_backtest()        # Simulation engine
WalkForwardAnalyzer.run_walk_forward() # W-F analysis
PerformanceMetrics.calculate_all_metrics() # Metrics
```

## 📊 Example Output

```
======================================================================
 STEP 5: PERFORMANCE METRICS
======================================================================

==================================================
PERFORMANCE METRICS
==================================================

Capital:
  Initial Capital:            100,000.00
  Final Capital:              156,234.50

Returns:
  Total Return:                   56.23%
  CAGR:                           7.45%

Risk:
  Volatility:                      9.23%
  Max Drawdown:                  -18.42%

Risk-Adjusted Returns:
  Sharpe Ratio:                   0.85
  Sortino Ratio:                  1.12

Trading Metrics:
  Win Rate:                       52.34%
  Profit Factor:                   1.45

Daily Statistics:
  Avg Daily Return:               0.013%
  Daily Volatility:               0.58%
==================================================
```

## ⚠️ Disclaimer

This is an educational system for learning quantitative finance concepts. Past performance does not guarantee future results. Always:

- Thoroughly backtest strategies before live trading
- Use proper risk management
- Start with small position sizes
- Monitor performance continuously
- Adjust parameters based on market conditions

## 🔍 Troubleshooting

### No data downloaded
- Check internet connection
- Verify ticker symbols are correct
- Check yfinance API status

### ML model won't train
- Ensure sufficient historical data (>1000 days recommended)
- Check for NaN values in features
- Verify target variable has variation

### Dashboard won't load
- Run: `pip install --upgrade streamlit`
- Clear Streamlit cache: `streamlit cache clear`
- Check port 8501 is available

## 📝 License

MIT License - Feel free to use and modify for research and educational purposes.

## 📧 Contact

For questions and suggestions, refer to the code documentation and comments.

---

**Built with**: Python 3.8+ | Pandas | NumPy | scikit-learn | Streamlit | yfinance

**Latest Update**: 2026

Happy Quantitative Trading! 📈
