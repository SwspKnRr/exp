# Quantitative Trading System - Quick Start Guide

## 🚀 5-Minute Quick Start

### 1. Install

```bash
cd quant_system
pip install -r requirements.txt
```

### 2. Run Backtest

```bash
python main.py
```

You'll see:
- ✅ Data downloads (15 years of ETF prices)
- ✅ Feature calculations (momentum, volatility, macro)
- ✅ Walk-forward backtest (2015-2023)
- ✅ Performance metrics

### 3. Launch Dashboard (optional)

```bash
streamlit run dashboard/streamlit_app.py
```

Open browser to: http://localhost:8501

---

## 📦 What You Get

### Data (25 ETFs)
- **Broad Market**: SPY, QQQ, VTI
- **Sectors**: XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE
- **Thematic**: SMH, LIT, ARKK, ITA, ITB
- **Bonds/Commodities**: GLD, TLT

### Features Calculated
- **Momentum**: 5d/10d/20d/60d/120d returns, RSI, MAs
- **Volatility**: 20d/60d volatility, ATR
- **Macro**: VIX, SPY/GLD ratio, Bond/Stock ratio

### Strategy Rules
1. **Regime Filter**: Only trade when SPY > 200-day MA
2. **Dual Momentum**: Both absolute (return > 0) AND relative (return > SPY)
3. **Ranking**: Score = 0.4×return_20d + 0.3×return_60d + 0.3×return_120d
4. **Selection**: Top 3 ETFs by momentum score
5. **ML Filter**: RandomForest predicts P(positive return) - accept if > 0.60
6. **Portfolio**: 3 positions, equal weight, weekly rebalance

### Performance Metrics
- CAGR (annual return)
- Sharpe Ratio (risk-adjusted)
- Sortino Ratio (downside risk-adjusted)
- Max Drawdown (worst peak-to-trough)
- Volatility (risk level)
- Win Rate (% profitable days)
- Profit Factor (gains/losses ratio)

---

## 🔧 Key Files to Know

| File | Purpose |
|------|---------|
| `main.py` | ⭐ Run this for complete backtest |
| `data/loader.py` | Download price data from Yahoo Finance |
| `features/*.py` | Calculate technical indicators |
| `strategy/*.py` | Generate trading signals |
| `ml/*.py` | Train ML classifier |
| `backtest/*.py` | Run backtest simulations |
| `dashboard/streamlit_app.py` | Interactive visualization |

---

## ⚙️ Customize Strategy

### Change Universe

Edit `data/loader.py`:
```python
ETF_UNIVERSE = {
    'Broad Market': ['SPY', 'QQQ', 'VTI'],
    # Add or remove tickers here
}
```

### Change Strategy Parameters

Edit `main.py` selection function:
```python
# Reduce to 2 positions
ranker = ETFRanker(max_positions=2)

# Change ranking weights
ranker = ETFRanker(weights={10: 0.5, 60: 0.3, 120: 0.2})

# Change ML threshold
ml_filter = MLSignalFilter(probability_threshold=0.55)
```

### Change Backtest Settings

Edit `main.py` call to `run_walk_forward`:
```python
analyzer = WalkForwardAnalyzer(
    train_years=3,      # Reduce training window
    test_years=2,       # Longer test period
    initial_capital=50000  # Smaller starting capital
)
```

---

## 📊 Expected Results (as of 2023)

Based on sector rotation strategy from 2015-2023:

| Metric | Expected Range |
|--------|-----------------|
| CAGR | 5-12% |
| Sharpe Ratio | 0.6-1.0 |
| Max Drawdown | -15% to -25% |
| Win Rate | 48-54% |

*Results vary based on market conditions and parameter selection*

---

## 🎓 Learning Path

1. **Understand Strategy** → Read `README.md`
2. **Run Example** → `python main.py`
3. **Review Results** → Check console output
4. **Explore Code** → Look at `main.py` and modules
5. **Modify Parameters** → Change and re-run
6. **Visualize** → `streamlit run dashboard/streamlit_app.py`
7. **Optimize** → Run backtests with different parameters

---

## ⚡ Common Tasks

### See current ETF rankings
```python
from strategy.ranking import ETFRanker
from data.loader import download_price_data, ALL_TICKERS

prices = download_price_data(ALL_TICKERS)
ranker = ETFRanker(max_positions=3)
ranking = ranker.rank_etfs(prices)
print(ranking)
```

### Check market regime
```python
from features.macro import calculate_regime_filter
from data.loader import download_price_data, ALL_TICKERS

prices = download_price_data(ALL_TICKERS)
regime = calculate_regime_filter(prices)
print(f"Uptrend: {regime.iloc[-1]}")
```

### Train custom ML model
```python
from ml.filter import MLSignalFilter

ml_filter = MLSignalFilter(probability_threshold=0.6)
metrics = ml_filter.train_model(
    momentum_features, volatility_features, 
    macro_features, prices, 'SPY'
)
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Data download fails | Check internet, verify tickers |
| ML model won't train | Need > 1000 days data |
| Dashboard won't start | Run: `pip install --upgrade streamlit` |
| Slow backtest | Reduce date range in `run_walk_forward()` |

---

## 📚 Next Steps

- ✅ Run `main.py` to see complete system
- ✅ Launch dashboard with `streamlit run dashboard/streamlit_app.py`
- ✅ Modify strategy parameters in `main.py`
- ✅ Implement your own selection logic
- ✅ Add additional features in `features/`
- ✅ Optimize using different ML models

---

## 💡 Pro Tips

1. **Start Simple**: Begin with basic momentum ranking before adding ML
2. **Test Often**: Small parameter changes can have big impacts
3. **Use Walk-Forward**: Always test out-of-sample to avoid overfitting
4. **Monitor Risk**: Check max drawdown, not just returns
5. **Track Trades**: Monitor transaction costs impact on performance

---

**Ready to get started?** → Run `python main.py`

**Questions?** → Check code comments and docstrings

**Want to visualize?** → Run `streamlit run dashboard/streamlit_app.py`

Happy Trading! 📈
