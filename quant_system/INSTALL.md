# Installation & First Run Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ~2GB disk space (for data cache)
- Internet connection (for data download)

## Step-by-Step Installation

### 1. Navigate to Project Directory

```bash
cd z:\codexprojects\quant_system
```

### 2. Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **yfinance**: Stock data download
- **scikit-learn**: Machine learning
- **plotly**: Interactive charts
- **streamlit**: Dashboard framework
- And supporting libraries

Installation typically takes 2-5 minutes.

### 4. Verify Installation

```bash
python -c "import pandas, numpy, yfinance, sklearn, streamlit; print('✓ All dependencies installed!')"
```

## First Run - Complete Backtest

### Run the Main Script

```bash
python main.py
```

**What it does:**
1. Downloads 15 years of ETF data from Yahoo Finance (~2-3 minutes)
2. Calculates technical features (momentum, volatility, macro)
3. Analyzes strategy components (regime filter, momentum, ranking)
4. Trains ML classification model
5. Runs walk-forward backtest (5-year train / 1-year test rolling windows)
6. Calculates performance metrics
7. Prints detailed results

**Expected Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  QUANTITATIVE TRADING SYSTEM - SECTOR ROTATION STRATEGY          ║
║  Dual Momentum + Market Regime Filter + ML Signal Filter         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

======================================================================
 STEP 1: DOWNLOAD DATA & PREPARE FEATURES
======================================================================

[Downloaded data...]
✓ Created momentum, volatility, and macro features

======================================================================
 STEP 2: ANALYZE STRATEGY COMPONENTS
======================================================================

[Market regime, momentum signals, ETF ranking...]

======================================================================
 STEP 3: TRAIN ML SIGNAL FILTER
======================================================================

[Training RandomForestClassifier...]

======================================================================
 STEP 4: RUN WALK-FORWARD BACKTEST
======================================================================

Period 1: Train 2010-2014, Test 2015
  Final Value: $105,234.50 | Return: 5.23% | Trades: 12
...

======================================================================
 STEP 5: PERFORMANCE METRICS
======================================================================

CAGR:               7.45%
Sharpe Ratio:       0.85
Max Drawdown:      -18.42%
Win Rate:           52.34%
...

✓ Backtest completed successfully in 234.5 seconds
```

## Second Step - Interactive Dashboard

### Launch Streamlit App

```bash
streamlit run dashboard/streamlit_app.py
```

**Browser opens automatically to:** http://localhost:8501

**Features:**
- 📊 ETF ranking charts
- 📈 Market regime visualization
- 💹 Portfolio equity curve
- 📉 Performance metrics
- 📊 Feature statistics
- 🔄 Walk-forward results

### Exit Dashboard

Press `Ctrl+C` in the terminal

## Common First-Run Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'yfinance'"

**Solution:**
```bash
pip install --upgrade yfinance
```

### Issue: "No data downloaded" or "Empty DataFrame"

**Causes & Solutions:**
1. **No internet connection** → Check your connection
2. **Ticker not found** → Verify ticker symbols in config
3. **yfinance rate limited** → Wait a few minutes and retry
4. **Firewall blocking** → Check network settings

### Issue: "ML Model won't train"

**Solutions:**
1. Need at least 1000+ days of data
2. Check for NaN values in features
3. Ensure sufficient market variation in data

### Issue: "Streamlit won't load"

**Solutions:**
```bash
# Update Streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear

# Try different port
streamlit run dashboard/streamlit_app.py --server.port 8502
```

### Issue: "Slow data download"

**Solutions:**
1. Reduce date range in code
2. Download fewer tickers
3. Network might be slow - try later

## Running Custom Backtests

### Modify Time Period

Edit `main.py`:
```python
prices, vix, ... = download_and_prepare_data(
    start_date='2015-01-01',  # Change start date
    end_date='2023-12-31'      # Change end date
)
```

### Change Universe

Edit `data/loader.py`:
```python
ETF_UNIVERSE = {
    'Broad Market': ['SPY', 'QQQ'],  # Remove VTI
    # ... customize
}
```

### Adjust Strategy Parameters

Edit `main.py` in backtest section:
```python
# Use 2 positions instead of 3
ranker = ETFRanker(max_positions=2)

# Change momentum period
dm = DualMomentum(momentum_period=30)

# Lower ML threshold
ml_filter = MLSignalFilter(probability_threshold=0.55)
```

## Understanding the Output

### Key Metrics Explained

| Metric | What it means | Good value |
|--------|---------------|-----------|
| CAGR | Annual return | 5-10%+ |
| Sharpe Ratio | Risk-adjusted return | 0.5-1.0+ |
| Max Drawdown | Worst loss | -10% to -20% |
| Win Rate | % profitable days | 50-55%+ |
| Profit Factor | Gains/Losses ratio | 1.5-2.0+ |

### Interpreting Results

**If returns are too low:**
- Increase max positions (take more risk)
- Reduce ML threshold (accept more trades)
- Use shorter momentum period

**If drawdown is too high:**
- Use market regime filter more strictly
- Reduce max positions
- Increase ML threshold

**If strategy has few trades:**
- Reduce ML probability threshold
- Change momentum period
- Remove regime filter

## Next Steps

### 1. Explore the Code
```bash
# Key files to understand
main.py          # Complete strategy implementation
strategy/        # Signal generation
backtest/        # Backtesting engine
features/        # Feature calculation
```

### 2. Modify Parameters
- Edit `main.py` or `config.py`
- Run `python main.py` again
- Compare results

### 3. Enhance Strategy
- Add new features in `features/`
- Try different ML models in `ml/`
- Implement custom selection logic

### 4. Visualize Results
- Use Streamlit dashboard
- Export results to CSV
- Create custom visualizations

## File Structure Reminder

```
quant_system/
├── main.py                      ← START HERE
├── requirements.txt             ← Install dependencies
├── config.py                    ← All configuration
├── README.md                    ← Full documentation
├── QUICKSTART.md                ← Quick reference
├── INSTALL.md                   ← This file
│
├── data/
│   └── loader.py               ← Download price data
│
├── features/
│   ├── momentum.py             ← Momentum indicators
│   ├── volatility.py           ← Volatility indicators
│   └── macro.py                ← Macro indicators
│
├── strategy/
│   ├── regime_filter.py        ← Market regime
│   ├── dual_momentum.py        ← Momentum signals
│   └── ranking.py              ← ETF ranking
│
├── ml/
│   ├── model.py                ← ML classifier
│   └── filter.py               ← Signal filtering
│
├── backtest/
│   ├── engine.py               ← Backtest engine
│   ├── walk_forward.py         ← Walk-forward analysis
│   └── metrics.py              ← Performance metrics
│
└── dashboard/
    └── streamlit_app.py        ← Visual dashboard
```

## Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete backtest
python main.py

# Launch interactive dashboard
streamlit run dashboard/streamlit_app.py

# Check Python version
python --version

# List installed packages
pip list

# Update all packages
pip install --upgrade -r requirements.txt

# Deactivate virtual environment
deactivate
```

## Getting Help

1. **Check code comments** - All functions have docstrings
2. **Read README.md** - Comprehensive documentation
3. **Review example code** - `main.py` shows everything
4. **Check error messages** - Python errors are usually clear

## Expected Time Breakdown

| Task | Duration |
|------|----------|
| Installation | 2-5 min |
| First run (data download) | 3-5 min |
| Feature calculation | 1-2 min |
| ML training | 1-2 min |
| Walk-forward backtest | 2-5 min |
| **Total first run** | **8-20 min** |

Subsequent runs are faster (data cached).

## System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| Python | 3.8 | 3.9+ |
| RAM | 2GB | 4GB+ |
| Disk Space | 500MB | 2GB |
| Internet | Required | 10+ Mbps |
| OS | Windows/Mac/Linux | Any |

## Success Indicators

✓ Complete installation without errors
✓ Download 15 years of ETF data successfully
✓ Calculate features in <2 minutes
✓ ML model trains with >50% accuracy
✓ Walk-forward backtest completes
✓ Metrics print without errors
✓ Streamlit dashboard launches

---

Ready to start? Run: `python main.py`

Happy Trading! 📈
