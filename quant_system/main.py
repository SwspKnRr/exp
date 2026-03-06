"""
Quantitative Trading System - Main Entry Point (Streamlit UI)
Complete working example demonstrating:
1. Data download
2. Feature engineering
3. Walk-forward backtest
4. Performance metrics calculation
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import sys

warnings.filterwarnings('ignore')

# Import all modules
from data.loader import (
    download_price_data,
    download_vix_data,
    create_macro_signals,
    align_all_data,
    ALL_TICKERS
)

from features.momentum import create_momentum_features
from features.volatility import create_volatility_features
from features.macro import create_macro_features, calculate_regime_filter

from strategy.regime_filter import RegimeFilter
from strategy.dual_momentum import DualMomentum
from strategy.ranking import ETFRanker
from strategy.day_trading import DayTradingSignals

from ml.model import MomentumClassifier
from ml.filter import MLSignalFilter

from backtest.engine import BacktestEngine
from backtest.walk_forward import WalkForwardAnalyzer
from backtest.metrics import PerformanceMetrics

# Streamlit page config
st.set_page_config(
    page_title="Quant Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐싱 설정
@st.cache_data(ttl=3600, show_spinner=False)
def get_price_data(start_date='2010-01-01', end_date=None):
    """캐시된 가격 데이터 반환"""
    return download_price_data(ALL_TICKERS, start_date=start_date, end_date=end_date)

@st.cache_data(ttl=3600, show_spinner=False)
def get_vix_data(start_date='2010-01-01', end_date=None):
    """캐시된 VIX 데이터 반환"""
    return download_vix_data(start_date=start_date, end_date=end_date)


@st.cache_data(ttl=3600)
def download_and_prepare_data(start_date='2010-01-01', end_date=None):
    """
    Step 1: Download price data and prepare features.
    
    Returns:
        prices, vix, momentum_feat, volatility_feat, macro_feat
    """
    
    progress_bar = st.progress(0)
    status_container = st.empty()
    
    # Download prices
    status_container.info("📥 Downloading price data...")
    prices = download_price_data(ALL_TICKERS, start_date=start_date, end_date=end_date)
    progress_bar.progress(20)
    
    # Check if price data is valid
    if prices is None or prices.empty:
        status_container.error("❌ Failed to download price data")
        st.error("Could not retrieve price data from Yahoo Finance")
        st.info("Possible reasons:")
        st.write("1. Network connectivity issue")
        st.write("2. Yahoo Finance API temporary unavailable")
        st.write("3. Data not available for selected date range")
        st.write("\nPlease try again or check your internet connection.")
        return None, None, None, None, None
    
    st.write(f"✓ Loaded price data: {prices.shape[0]} dates × {prices.shape[1]} tickers")
    
    # Download VIX
    status_container.info("📥 Downloading VIX data...")
    try:
        vix = download_vix_data(start_date=start_date, end_date=end_date)
        if vix is None or len(vix) == 0:
            st.warning("⚠️  VIX data unavailable, continuing without it")
            vix = pd.Series(dtype=float)
    except Exception as e:
        st.warning(f"⚠️  Could not download VIX: {e}")
        vix = pd.Series(dtype=float)
    progress_bar.progress(40)
    
    # Create momentum features
    status_container.info("⚙️  Creating momentum features...")
    try:
        momentum_features = create_momentum_features(prices)
        st.write(f"✓ Created {len(momentum_features.columns)} momentum features")
    except Exception as e:
        st.error(f"❌ Error creating momentum features: {e}")
        return None, None, None, None, None
    progress_bar.progress(60)
    
    # Create volatility features
    status_container.info("⚙️  Creating volatility features...")
    try:
        volatility_features = create_volatility_features(prices)
        st.write(f"✓ Created {len(volatility_features.columns)} volatility features")
    except Exception as e:
        st.error(f"❌ Error creating volatility features: {e}")
        return None, None, None, None, None
    progress_bar.progress(80)
    
    # Create macro features
    status_container.info("⚙️  Creating macro features...")
    try:
        macro_features = create_macro_features(prices, vix)
        st.write(f"✓ Created {len(macro_features.columns)} macro features")
    except Exception as e:
        st.error(f"❌ Error creating macro features: {e}")
        return None, None, None, None, None
    progress_bar.progress(100)
    
    status_container.success("✓ Data preparation complete!")
    
    return prices, vix, momentum_features, volatility_features, macro_features


def analyze_strategy_components(prices, momentum_features, volatility_features, macro_features):
    """
    Step 2: Analyze strategy components.
    
    Shows:
    - Market regime filter
    - Dual momentum signals
    - ETF ranking
    """
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market Regime Filter
        st.subheader("📊 Market Regime Filter")
        regime_filter = RegimeFilter(benchmark='SPY', ma_period=200)
        regime = regime_filter.calculate_regime(prices)
        uptrend_ratio = regime.sum() / len(regime)
        
        st.metric("Uptrend Ratio (SPY > MA200)", f"{uptrend_ratio:.1%}")
        st.caption(f"Last 5 days: {regime.tail(5).values}")
    
    with col2:
        # Dual Momentum
        st.subheader("⚡ Dual Momentum Analysis")
        dm = DualMomentum(momentum_period=20, benchmark='SPY')
        signals = dm.generate_signals(prices)
        signal_ratio = (signals == 1).sum().sum() / (signals.shape[0] * signals.shape[1])
        
        st.metric("Signal Generation Rate", f"{signal_ratio:.1%}")
        st.caption(f"Active signals today: {(signals.iloc[-1] == 1).sum()}")
    
    # ETF Ranking
    st.subheader("🏆 ETF Ranking")
    ranker = ETFRanker(max_positions=3)
    latest_ranking = ranker.rank_etfs(prices, exclude_tickers=['SPY'])
    selected = ranker.select_portfolio(prices, exclude_tickers=['SPY'])
    weights = ranker.get_portfolio_weights(prices, exclude_tickers=['SPY'], weight_method='equal')
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.write("**Top 10 ETFs by Momentum Score:**")
        ranking_display = latest_ranking.head(10)[['rank', 'ticker', 'score']].copy()
        ranking_display['score'] = ranking_display['score'].round(4)
        st.dataframe(ranking_display, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Selected Portfolio (3 positions):**")
        portfolio_data = []
        for ticker in selected:
            weight = weights.get(ticker, 0)
            portfolio_data.append({'Ticker': ticker, 'Weight': f'{weight:.1%}'})
        st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True, hide_index=True)
    
    return regime, signals, selected, weights


def train_ml_filter(momentum_features, volatility_features, macro_features, prices):
    """
    Step 3: Train ML signal filter.
    """
    
    st.subheader("🤖 ML Signal Filter Training")
    
    ticker = 'SPY'
    
    # Select features for SPY
    spy_momentum = momentum_features[[col for col in momentum_features.columns if not col.endswith('RSI')]]
    spy_volatility = volatility_features
    spy_macro = macro_features
    
    # Initialize and train classifier
    ml_filter = MLSignalFilter(probability_threshold=0.6)
    
    try:
        with st.spinner("Training RandomForest model..."):
            metrics = ml_filter.train_model(
                spy_momentum,
                spy_volatility,
                spy_macro,
                prices,
                ticker,
                prediction_horizon=5,
                validation_split=0.2
            )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train Accuracy", f"{metrics['train_accuracy']:.1%}")
        with col2:
            st.metric("Val Accuracy", f"{metrics['val_accuracy']:.1%}")
        with col3:
            st.metric("Train Samples", f"{metrics['train_samples']:,}")
        with col4:
            st.metric("Val Samples", f"{metrics['val_samples']:,}")
        
        # Feature importance
        st.write("**Top 10 Important Features:**")
        importance = ml_filter.model.get_feature_importance(top_n=10)
        importance_df = pd.DataFrame([
            {'Feature': feature, 'Importance': score}
            for feature, score in importance.items()
        ])
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', height=400)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        return ml_filter
        
    except Exception as e:
        st.warning(f"⚠️  Could not train ML model: {e}")
        st.info("Continuing analysis without ML filter...")
        return None


def run_walk_forward_backtest(prices, start_date='2015-01-01', end_date=None):
    """
    Step 4: Run walk-forward backtest with enhanced analysis.
    """
    
    st.subheader("📈 Walk-Forward Backtest")
    
    if end_date is None:
        end_date = prices.index[-1].strftime('%Y-%m-%d')
    
    # Define portfolio selection function
    def select_portfolio_for_backtest(train_prices, train_features, date):
        """Simple portfolio selection based on momentum ranking."""
        ranker = ETFRanker(max_positions=3)
        selected = ranker.select_portfolio(
            train_prices, 
            date=date, 
            exclude_tickers=['SPY']
        )
        return selected
    
    # Run walk-forward analysis
    st.info("Train window: 5 years | Test window: 1 year | Rebalance: Weekly | Transaction cost: 0.05%")
    
    with st.spinner("Executing Walk-Forward Analysis..."):
        analyzer = WalkForwardAnalyzer(
            train_years=5,
            test_years=1,
            initial_capital=100000,
            transaction_cost=0.0005
        )
        
        results = analyzer.run_walk_forward(
            prices,
            select_portfolio_for_backtest,
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            verbose=False
        )
    
    st.success("✓ Backtest execution complete!")
    
    # Enhanced analysis
    st.divider()
    st.subheader("📊 Period-by-Period Analysis")
    
    summary = analyzer.get_summary_statistics()
    
    if len(summary) > 0:
        # Detailed period table
        period_display = summary[[
            'test_start', 'test_end', 'total_return', 'num_trades', 
            'winning_trades', 'win_rate', 'max_drawdown'
        ]].copy()
        
        period_display.columns = [
            'Period Start', 'Period End', 'Return', 'Trades', 
            'Wins', 'Win Rate', 'Max DD'
        ]
        
        period_display['Return'] = period_display['Return'].apply(lambda x: f"{x:+.2%}")
        period_display['Win Rate'] = period_display['Win Rate'].apply(lambda x: f"{x:.1%}")
        period_display['Max DD'] = period_display['Max DD'].apply(lambda x: f"{x:.2%}")
        
        st.write("**All Backtest Periods:**")
        st.dataframe(period_display, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Period comparison charts
        st.subheader("📈 Period Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Return by period
            fig_return = go.Figure()
            fig_return.add_trace(go.Bar(
                x=[f"Period {i+1}" for i in range(len(summary))],
                y=summary['total_return'],
                marker=dict(
                    color=summary['total_return'],
                    colorscale='RdYlGn',
                    showscale=True
                ),
                text=[f"{r:.2%}" for r in summary['total_return']],
                textposition='outside'
            ))
            fig_return.update_layout(
                title="Return by Period",
                xaxis_title="Period",
                yaxis_title="Return",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_return, use_container_width=True)
        
        with col2:
            # Win rate by period
            try:
                # Convert win_rate to numeric if needed
                win_rate_data = pd.to_numeric(summary['win_rate'], errors='coerce')
                
                fig_win = go.Figure()
                fig_win.add_trace(go.Bar(
                    x=[f"Period {i+1}" for i in range(len(summary))],
                    y=win_rate_data,
                    marker=dict(color='#1f77b4'),
                    text=[f"{r:.1%}" if not pd.isna(r) else "N/A" for r in win_rate_data],
                    textposition='outside'
                ))
                fig_win.update_layout(
                    title="Win Rate by Period",
                    xaxis_title="Period",
                    yaxis_title="Win Rate",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_win, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display win rate chart: {e}")
        
        st.divider()
        
        # Best and worst periods
        st.subheader("🏆 Best vs 📉 Worst Periods")
        
        best_period_idx = summary['total_return'].idxmax()
        worst_period_idx = summary['total_return'].idxmin()
        
        best_period = summary.loc[best_period_idx]
        worst_period = summary.loc[worst_period_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **🏆 BEST PERIOD: Period {best_period_idx + 1}**
            
            Date: {best_period['period_start'].strftime('%Y-%m-%d')} → {best_period['period_end'].strftime('%Y-%m-%d')}
            
            Return: **{best_period['total_return']:+.2%}**
            
            Trades: {int(best_period['num_trades'])} (W: {int(best_period['winning_trades'])})
            
            Win Rate: {best_period['win_rate']:.1%}
            
            Max Drawdown: {best_period['max_drawdown']:.2%}
            """)
        
        with col2:
            st.error(f"""
            **📉 WORST PERIOD: Period {worst_period_idx + 1}**
            
            Date: {worst_period['period_start'].strftime('%Y-%m-%d')} → {worst_period['period_end'].strftime('%Y-%m-%d')}
            
            Return: **{worst_period['total_return']:+.2%}**
            
            Trades: {int(worst_period['num_trades'])} (W: {int(worst_period['winning_trades'])})
            
            Win Rate: {worst_period['win_rate']:.1%}
            
            Max Drawdown: {worst_period['max_drawdown']:.2%}
            """)
        
        st.divider()
        
        # Aggregate statistics
        st.subheader("📊 Aggregate Statistics")
        
        total_return = float(summary['total_return'].sum())
        avg_return = float(summary['total_return'].mean())
        median_return = float(summary['total_return'].median())
        best_return = float(summary['total_return'].max())
        worst_return = float(summary['total_return'].min())
        
        total_trades = int(summary['num_trades'].sum())
        total_wins = int(summary['winning_trades'].sum())
        aggregate_win_rate = float(total_wins) / float(total_trades) if total_trades > 0 else 0.0
        
        avg_dd = float(summary['max_drawdown'].mean())
        max_dd = float(summary['max_drawdown'].min())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return (All Periods)", f"{total_return:+.2%}")
        with col2:
            st.metric("Average Period Return", f"{avg_return:+.2%}")
        with col3:
            st.metric("Median Period Return", f"{median_return:+.2%}")
        with col4:
            st.metric("Return StdDev", f"{summary['total_return'].std():.2%}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", f"{total_trades}")
        with col2:
            st.metric("Total Wins", f"{total_wins}")
        with col3:
            st.metric("Aggregate Win Rate", f"{aggregate_win_rate:.1%}")
        with col4:
            st.metric("Trades per Period", f"{total_trades/len(summary):.1f}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Max Drawdown", f"{avg_dd:.2%}")
        with col2:
            st.metric("Worst Period DD", f"{max_dd:.2%}")
        with col3:
            st.metric("Number of Periods", f"{len(summary)}")
        
        st.divider()
        
        # Trade distribution
        st.subheader("📊 Trade Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_trades = go.Figure()
            fig_trades.add_trace(go.Bar(
                x=[f"Period {i+1}" for i in range(len(summary))],
                y=summary['num_trades'],
                marker=dict(color='#2ca02c'),
                text=[f"{t:.0f}" for t in summary['num_trades']],
                textposition='outside'
            ))
            fig_trades.update_layout(
                title="Number of Trades by Period",
                xaxis_title="Period",
                yaxis_title="Number of Trades",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_trades, use_container_width=True)
        
        with col2:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Bar(
                x=[f"Period {i+1}" for i in range(len(summary))],
                y=summary['max_drawdown'],
                marker=dict(color='#d62728'),
                text=[f"{dd:.2%}" for dd in summary['max_drawdown']],
                textposition='outside'
            ))
            fig_dd.update_layout(
                title="Max Drawdown by Period",
                xaxis_title="Period",
                yaxis_title="Max Drawdown",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_dd, use_container_width=True)
    
    return analyzer, results


def calculate_performance_metrics(equity_curve, initial_capital):
    """
    Step 5: Calculate and display performance metrics with detailed analysis.
    """
    
    st.subheader("📊 Performance Metrics & Analysis")
    
    calculator = PerformanceMetrics(risk_free_rate=0.02)
    metrics = calculator.calculate_all_metrics(
        equity_curve['portfolio_value'],
        initial_capital
    )
    
    # Display key metrics in grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Capital", f"${metrics['Final Capital']:,.0f}")
    with col2:
        st.metric("Total Return", f"{metrics['Total Return']:.2%}")
    with col3:
        st.metric("CAGR", f"{metrics['CAGR']:.2%}")
    with col4:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
    with col2:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
    with col3:
        st.metric("Volatility", f"{metrics['Volatility']:.2%}")
    with col4:
        st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
    with col2:
        st.metric("Recovery Factor", f"{metrics['Recovery Factor']:.2f}")
    
    st.divider()
    
    # Equity curve with annotations
    st.subheader("📈 Portfolio Value Over Time")
    
    # Calculate drawdown for visualization
    running_max = equity_curve['portfolio_value'].expanding().max()
    drawdown = (equity_curve['portfolio_value'] - running_max) / running_max
    
    fig = go.Figure()
    
    # Portfolio value
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    
    # Mark max drawdown point
    max_dd_idx = drawdown.idxmin()
    max_dd_value = equity_curve.loc[max_dd_idx, 'portfolio_value']
    
    fig.add_vline(
        x=max_dd_idx,
        line_dash="dash",
        line_color="red",
        annotation_text="Max Drawdown",
        annotation_position="top right"
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Drawdown analysis
    st.subheader("📉 Drawdown Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=equity_curve.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#d62728')
        ))
        fig_dd.update_layout(
            title="Drawdown Over Time (%)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    
    with col2:
        # Monthly returns heatmap
        monthly_returns = equity_curve['portfolio_value'].resample('M').last().pct_change()
        
        st.metric("Average Monthly Return", f"{monthly_returns.mean():+.2%}")
        st.metric("Monthly Return Std Dev", f"{monthly_returns.std():.2%}")
        st.metric("Best Month", f"{monthly_returns.max():+.2%}")
        st.metric("Worst Month", f"{monthly_returns.min():+.2%}")
    
    st.divider()
    
    # Return distribution
    st.subheader("📊 Return Distribution")
    
    daily_returns = equity_curve['portfolio_value'].pct_change().dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=daily_returns * 100,
            nbinsx=50,
            name='Daily Returns',
            marker=dict(color='#2ca02c')
        ))
        fig_hist.update_layout(
            title="Daily Returns Distribution (%)",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.write("**Return Summary Statistics:**")
        return_stats = pd.DataFrame({
            'Metric': [
                'Mean Daily Return',
                'Median Daily Return',
                'Std Dev',
                'Skewness',
                'Kurtosis',
                'Best Day',
                'Worst Day'
            ],
            'Value': [
                f"{daily_returns.mean():+.3%}",
                f"{daily_returns.median():+.3%}",
                f"{daily_returns.std():.3%}",
                f"{daily_returns.skew():.3f}",
                f"{daily_returns.kurtosis():.3f}",
                f"{daily_returns.max():+.2%}",
                f"{daily_returns.min():+.2%}"
            ]
        })
        st.dataframe(return_stats, use_container_width=True, hide_index=True)
    
    return metrics


def generate_summary_report(analyzer, results, metrics):
    """
    Step 6: Generate comprehensive summary report.
    """
    
    st.subheader("📋 Final Summary Report")
    
    # Get summary statistics
    summary = analyzer.get_summary_statistics()
    
    if len(summary) > 0:
        st.write("**Walk-Forward Period Summary:**")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Key metrics summary
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Number of Periods", f"{len(summary)}")
        with col2:
            st.metric("Avg Period Return", f"{summary['total_return'].mean():.2%}")
        with col3:
            st.metric("Best Period Return", f"{summary['total_return'].max():.2%}")
        with col4:
            st.metric("Worst Period Return", f"{summary['total_return'].min():.2%}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", f"{summary['num_trades'].sum():.0f}")
        with col2:
            st.metric("Avg Trades/Period", f"{summary['num_trades'].mean():.1f}")
        with col3:
            st.metric("Aggregate Win Rate", f"{(summary['winning_trades'].sum() / summary['num_trades'].sum()):.1%}")
        with col4:
            st.metric("Avg Max Drawdown", f"{summary['max_drawdown'].mean():.2%}")
        
        st.divider()
        
        # Risk-adjusted returns
        st.subheader("⚖️ Risk-Adjusted Performance")
        
        col1, col2, col3 = st.columns(3)
        
        total_return = float(summary['total_return'].sum())
        avg_return = float(summary['total_return'].mean())
        return_std = float(summary['total_return'].std())
        
        with col1:
            if return_std > 0:
                sharpe_periods = avg_return / return_std * np.sqrt(252.0 / float(len(summary)))
                st.metric("Period Sharpe Ratio", f"{sharpe_periods:.2f}")
        
        with col2:
            st.metric("Return Volatility", f"{return_std:.2%}")
        
        with col3:
            max_dd = float(summary['max_drawdown'].min())
            if max_dd < 0:
                recovery_factor = total_return / abs(max_dd)
                st.metric("Recovery Factor", f"{recovery_factor:.2f}")
        
        st.divider()
        
        # Period consistency
        st.subheader("📊 Strategy Consistency")
        
        winning_periods = int((summary['total_return'] > 0).sum())
        losing_periods = int((summary['total_return'] < 0).sum())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Profitable Periods", f"{winning_periods}/{len(summary)}")
        
        with col2:
            st.metric("Losing Periods", f"{losing_periods}/{len(summary)}")
        
        with col3:
            period_win_rate = float(winning_periods) / float(len(summary)) if len(summary) > 0 else 0.0
            st.metric("Period Win Rate", f"{period_win_rate:.1%}")
        
        st.divider()
        
        # Overall verdict
        st.subheader("✅ Strategy Assessment")
        
        if period_win_rate >= 0.7:
            verdict = "🟢 EXCELLENT"
            color = "success"
            description = "Very consistent profitability across periods"
        elif period_win_rate >= 0.5:
            verdict = "🟡 GOOD"
            color = "info"
            description = "Profitable in most periods with acceptable drawdown"
        else:
            verdict = "🔴 NEEDS IMPROVEMENT"
            color = "warning"
            description = "Strategy struggles in certain market conditions"
        
        if color == "success":
            st.success(f"""
            **{verdict} - {description}**
            
            - Period Win Rate: {period_win_rate:.1%}
            - Average Return: {avg_return:+.2%}
            - Return Std Dev: {return_std:.2%}
            - Max Drawdown: {max_dd:.2%}
            """)
        elif color == "info":
            st.info(f"""
            **{verdict} - {description}**
            
            - Period Win Rate: {period_win_rate:.1%}
            - Average Return: {avg_return:+.2%}
            - Return Std Dev: {return_std:.2%}
            - Max Drawdown: {max_dd:.2%}
            """)
        else:
            st.warning(f"""
            **{verdict} - {description}**
            
            - Period Win Rate: {period_win_rate:.1%}
            - Average Return: {avg_return:+.2%}
            - Return Std Dev: {return_std:.2%}
            - Max Drawdown: {max_dd:.2%}
            """)
        
        st.divider()
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if avg_return > 0.05:
            st.success("✅ Strategy has positive average return")
        else:
            st.warning("⚠️ Average return may be too low")
        
        if return_std < 0.15:
            st.success("✅ Strategy volatility is reasonable")
        else:
            st.warning("⚠️ Strategy volatility is high, consider risk management")
        
        if max_dd > -0.3:
            st.success("✅ Maximum drawdown is acceptable")
        else:
            st.error("❌ Maximum drawdown is excessive, needs risk limits")
        
        if period_win_rate >= 0.5:
            st.success("✅ Strategy is profitable in majority of periods")
        else:
            st.warning("⚠️ Strategy is unprofitable in many periods")
        
        st.divider()
        
        # Next steps
        st.subheader("🚀 Next Steps")
        
        st.markdown("""
        1. **Review Strategy Logic** - Analyze why certain periods underperform
        2. **Optimize Parameters** - Adjust momentum periods, rebalance frequency
        3. **Add Risk Controls** - Implement stop-loss, position sizing
        4. **Paper Trade** - Test with real market data using paper trading
        5. **Live Trading** - Consider starting with small positions
        """)



def show_what_to_do():
    """
    Show current top 3 ETFs to buy today with detailed analysis.
    Also compare custom ticker with the 20 default tickers.
    """
    st.subheader("💰 Current Recommendation")
    
    # Sidebar: Custom ticker comparison
    st.sidebar.subheader("🔍 Ticker Comparison")
    custom_ticker = st.sidebar.text_input(
        "Compare with our 20 ETFs",
        value="",
        placeholder="예: AAPL, MSFT, TSLA",
        help="20개 ETF와 비교하고 싶은 티커를 입력하세요"
    ).upper().strip()
    
    try:
        with st.spinner("Analyzing today's market..."):
            # Determine tickers to download
            download_tickers = ALL_TICKERS.copy()
            if custom_ticker and custom_ticker not in download_tickers:
                download_tickers.append(custom_ticker)
            
            # Download today's data
            prices = download_price_data(download_tickers, start_date='2010-01-01', end_date=None)
            
            # Calculate ranking
            ranker = ETFRanker(max_positions=3)
            latest_ranking = ranker.rank_etfs(prices, exclude_tickers=['SPY'])
            selected = ranker.select_portfolio(prices, exclude_tickers=['SPY'])
            weights = ranker.get_portfolio_weights(prices, exclude_tickers=['SPY'], weight_method='equal')
            
            # Market regime
            regime_filter = RegimeFilter(benchmark='SPY', ma_period=200)
            regime = regime_filter.calculate_regime(prices)
            current_regime = regime.iloc[-1]
        
        # Display custom ticker comparison if provided
        if custom_ticker:
            if custom_ticker in prices.columns:
                st.info(f"🔍 **{custom_ticker} 비교 분석**")
                
                # Get ranking info for custom ticker
                custom_rank_row = latest_ranking[latest_ranking['ticker'] == custom_ticker]
                
                if len(custom_rank_row) > 0:
                    custom_rank = int(custom_rank_row['rank'].iloc[0])
                    custom_score = custom_rank_row['score'].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Ranking position
                    with col1:
                        st.metric(
                            "전체 순위",
                            f"#{custom_rank}위",
                            delta=f"20개 중" if custom_rank <= 20 else "랭킹 외"
                        )
                    
                    # Score comparison with top 1
                    with col2:
                        top_score = latest_ranking.iloc[0]['score']
                        score_diff = custom_score - top_score
                        st.metric(
                            "점수",
                            f"{custom_score:.4f}",
                            delta=f"1위 대비 {score_diff:+.4f}"
                        )
                    
                    # Price and returns
                    with col3:
                        current_price = prices[custom_ticker].iloc[-1]
                        change_5d = prices[custom_ticker].pct_change(5).iloc[-1]
                        st.metric(
                            f"{custom_ticker} 가격",
                            f"${current_price:.2f}",
                            delta=f"5일: {change_5d:+.2%}"
                        )
                    
                    # Signal
                    with col4:
                        if custom_rank <= 3:
                            signal_text = "🟢 매수"
                        elif custom_rank <= 7:
                            signal_text = "🟡 관망"
                        else:
                            signal_text = "🔴 회피"
                        
                        st.metric(
                            "추천",
                            signal_text
                        )
                    
                    st.divider()
                    
                    # Detailed comparison with top 5
                    st.markdown("**상위 5개 및 비교 대상 티커**")
                    
                    top_5 = latest_ranking.head(5)
                    comparison_tickers = list(top_5['ticker']) + [custom_ticker]
                    comparison_tickers = list(dict.fromkeys(comparison_tickers))  # Remove duplicates
                    
                    comparison_data = []
                    for ticker in comparison_tickers:
                        if ticker not in prices.columns:
                            continue
                        
                        rank_row = latest_ranking[latest_ranking['ticker'] == ticker]
                        if len(rank_row) == 0:
                            continue
                        
                        rank = int(rank_row['rank'].iloc[0])
                        score = rank_row['score'].iloc[0]
                        
                        current_price = prices[ticker].iloc[-1]
                        change_1d = prices[ticker].pct_change(1).iloc[-1]
                        change_5d = prices[ticker].pct_change(5).iloc[-1]
                        change_20d = prices[ticker].pct_change(20).iloc[-1]
                        
                        comparison_data.append({
                            '순위': rank,
                            '티커': '⭐ ' + ticker if ticker == custom_ticker else ticker,
                            '점수': f"{score:.4f}",
                            '가격': f"${current_price:.2f}",
                            '1일': f"{change_1d:+.2%}",
                            '5일': f"{change_5d:+.2%}",
                            '20일': f"{change_20d:+.2%}"
                        })
                    
                    df_comp = pd.DataFrame(comparison_data)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # Comparison chart
                    st.markdown("**성과 비교 차트**")
                    
                    chart_tickers = comparison_tickers[:6]  # Top 5 + custom
                    chart_data = []
                    
                    for ticker in chart_tickers:
                        if ticker not in prices.columns:
                            continue
                        
                        rank_row = latest_ranking[latest_ranking['ticker'] == ticker]
                        score = rank_row['score'].iloc[0] if len(rank_row) > 0 else 0
                        ret_5d = prices[ticker].pct_change(5).iloc[-1] * 100
                        
                        chart_data.append({
                            'Ticker': ticker,
                            'Score': score,
                            'Return (5D)': ret_5d,
                            'Is Custom': ticker == custom_ticker
                        })
                    
                    df_chart = pd.DataFrame(chart_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Score comparison
                        fig_score = px.bar(
                            df_chart.sort_values('Score', ascending=True),
                            x='Score',
                            y='Ticker',
                            orientation='h',
                            color='Is Custom',
                            color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'},
                            title='점수 비교'
                        )
                        st.plotly_chart(fig_score, use_container_width=True)
                    
                    with col2:
                        # Return comparison
                        fig_ret = px.bar(
                            df_chart.sort_values('Return (5D)', ascending=True),
                            x='Return (5D)',
                            y='Ticker',
                            orientation='h',
                            color='Is Custom',
                            color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'},
                            title='5일 수익률 비교'
                        )
                        st.plotly_chart(fig_ret, use_container_width=True)
                else:
                    st.warning(f"⚠️ {custom_ticker} 순위 정보를 찾을 수 없습니다.")
            else:
                st.error(f"❌ {custom_ticker} 데이터를 다운로드할 수 없습니다. 올바른 티커를 확인해주세요.")
        
        st.divider()
        
        # Display top 3 ETFs with signals
        st.success("✅ Today's Top 3 ETFs (Updated Daily)")
        
        col1, col2, col3 = st.columns(3)
        
        top_3 = latest_ranking.head(3)
        for idx, (col, (_, row)) in enumerate(zip([col1, col2, col3], top_3.iterrows())):
            with col:
                ticker = row['ticker']
                score = row['score']
                
                # Get price change
                recent_return = prices[ticker].pct_change(5).iloc[-1]
                
                st.metric(
                    f"#{int(row['rank'])} **{ticker}**",
                    f"Score: {score:.4f}",
                    delta=f"5D Return: {recent_return:+.2%}"
                )
                
                # Color-coded button
                st.success("🟢 **STRONG BUY**", icon="✅")
                weight = weights.get(ticker, 0)
                st.caption(f"Weight: {weight:.1%} | ${100000 * weight:,.0f}")
        
        st.divider()
        
        # Detailed ticker analysis table
        st.subheader("🔍 Detailed Ticker Analysis (All 20 ETFs)")
        
        # Build detailed data
        ticker_data = []
        for _, row in latest_ranking.iterrows():
            ticker = row['ticker']
            score = row['score']
            rank = int(row['rank'])
            
            # Price data
            current_price = prices[ticker].iloc[-1]
            prev_price = prices[ticker].iloc[-2]
            daily_change = (current_price - prev_price) / prev_price
            
            # Various timeframes
            change_5d = prices[ticker].pct_change(5).iloc[-1]
            change_20d = prices[ticker].pct_change(20).iloc[-1]
            change_50d = prices[ticker].pct_change(50).iloc[-1]
            
            # Trend analysis
            ma20 = prices[ticker].rolling(20).mean().iloc[-1]
            trend = "📈 UP" if current_price > ma20 else "📉 DOWN"
            
            # Recommendation signal
            if rank <= 3:
                signal = "🟢 STRONG BUY"
                color = "positive"
            elif rank <= 7:
                signal = "🟡 HOLD"
                color = "neutral"
            else:
                signal = "🔴 STRONG SELL"
                color = "negative"
            
            ticker_data.append({
                'Rank': rank,
                'Ticker': ticker,
                'Score': f"{score:.4f}",
                'Price': f"${current_price:.2f}",
                'Today': f"{daily_change:+.2%}",
                '5D': f"{change_5d:+.2%}",
                '20D': f"{change_20d:+.2%}",
                'Trend': trend,
                'Signal': signal,
                'Action': 'Buy' if rank <= 3 else ('Hold' if rank <= 7 else 'Sell')
            })
        
        # Display table
        df_display = pd.DataFrame(ticker_data)
        
        # Color coding function
        def color_signal(val):
            if '🟢' in str(val):
                return 'background-color: #90EE90'
            elif '🟡' in str(val):
                return 'background-color: #FFFFE0'
            else:
                return 'background-color: #FFB6C6'
        
        styled_df = df_display.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Portfolio allocation
        st.subheader("📊 Portfolio Allocation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Table
            allocation_data = []
            for ticker in selected:
                weight = weights.get(ticker, 0)
                allocation_data.append({
                    'Ticker': ticker,
                    'Weight': f'{weight:.1%}',
                    'Amount ($100K)': f'${100000 * weight:,.0f}'
                })
            
            st.dataframe(
                pd.DataFrame(allocation_data),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Pie chart
            fig = px.pie(
                values=[weights.get(ticker, 0) for ticker in selected],
                names=selected,
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Action items with detailed recommendation
        st.subheader("🎯 Specific Action Items")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        for col, ticker in zip([action_col1, action_col2, action_col3], selected):
            with col:
                rank = int(latest_ranking[latest_ranking['ticker'] == ticker]['rank'].values[0])
                score = latest_ranking[latest_ranking['ticker'] == ticker]['score'].values[0]
                recent_return = prices[ticker].pct_change(5).iloc[-1]
                current_price = prices[ticker].iloc[-1]
                
                st.success(f"""
                **Position: {selected.index(ticker) + 1}**
                
                Ticker: **{ticker}**
                
                Rank: #{rank}
                
                Score: {score:.4f}
                
                Current Price: ${current_price:.2f}
                
                5D Return: {recent_return:+.2%}
                
                **→ BUY ${100000/3:,.0f}**
                """)
        
        st.divider()
        
        # Market regime with recommendation
        st.subheader("📈 Market Regime & Recommendation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uptrend_ratio = regime.sum() / len(regime)
            st.metric(
                "Market Status",
                "📈 UPTREND" if current_regime else "📉 DOWNTREND",
                delta=f"SPY {'>' if current_regime else '<'} MA200"
            )
            st.metric(
                "Uptrend Ratio (All Time)",
                f"{uptrend_ratio:.1%}"
            )
        
        with col2:
            if current_regime:
                st.success("""
                ✅ **Market is in UPTREND**
                
                Current positions are appropriate.
                
                Proceed with momentum strategy.
                
                Review and rebalance **next Monday**.
                """)
            else:
                st.warning("""
                ⚠️ **Market is in DOWNTREND**
                
                Consider reducing exposure.
                
                The system will adjust automatically.
                
                Review and rebalance **next Monday**.
                """)
        
        st.divider()
        
        # Legend
        st.subheader("📖 Signal Legend")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("🟢 **STRONG BUY** - Top 3 ETFs, high momentum")
        
        with col2:
            st.warning("🟡 **HOLD** - Middle tier, neutral signal")
        
        with col3:
            st.error("🔴 **STRONG SELL** - Weak momentum, avoid")
        
        st.divider()
        
        # Timing recommendation
        st.subheader("⏰ Rebalancing Schedule")
        
        # Find next rebalance date (Monday)
        today = pd.Timestamp.today()
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        next_rebalance = today + pd.Timedelta(days=days_until_monday)
        
        st.info(f"""
        **Next Rebalancing Date: {next_rebalance.strftime('%A, %B %d, %Y')}**
        
        Rebalance Time: 09:30 AM (Market Open)
        
        Current Positions: Hold until Monday
        
        Action: Execute trades on next Monday morning
        """)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.error(traceback.format_exc())


def show_day_trading_analysis():
    """
    Analyze intraday trading signals using RSI, MACD, Bollinger Bands.
    """
    
    st.header("⚡ Day Trading Analysis")
    st.markdown("""
    Real-time intraday trading signals using short-term technical indicators.
    """)
    
    st.divider()
    
    # Sidebar configuration for day trading
    st.sidebar.subheader("🕐 Day Trading Settings")
    
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        selected_ticker = st.sidebar.text_input(
            "Enter Ticker",
            value="SPY",
            placeholder="예: SPY, QQQ, GLD",
            help="원하는 ETF 티커 기호를 입력하세요"
        ).upper().strip()
        
        if not selected_ticker:
            st.error("❌ 티커를 입력해주세요 (예: SPY, QQQ)")
            return
    
    with col2:
        lookback_days = st.sidebar.slider(
            "기간 (일)",
            min_value=30,
            max_value=365,
            value=90,
            step=10
        )
    
    st.info(f"📊 분석 대상: **{selected_ticker}**")
    
    try:
        # Download recent data
        st.info("📥 데이터 다운로드 중...")
        
        # Use longer lookback period to ensure we get data
        start_date = (pd.Timestamp.today() - pd.Timedelta(days=max(lookback_days, 90))).strftime('%Y-%m-%d')
        st.write(f"📅 기간: {start_date} ~ 오늘")
        
        prices = download_price_data([selected_ticker], start_date=start_date)
        
        # Debug info
        st.info("**다운로드 상태:**")
        if prices is not None and not prices.empty:
            st.write(f"✓ 데이터 형태: {prices.shape[0]} 행 × {prices.shape[1]} 컬럼")
            st.write(f"✓ 컬럼명: {list(prices.columns)}")
            st.write(f"✓ 기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")
        else:
            st.write(f"✗ 데이터 로드 실패")
        
        if prices is None:
            st.error(f"❌ 데이터 다운로드 실패: {selected_ticker}")
            st.warning("**해결책:**")
            st.write("1. 올바른 티커 확인 (예: SPY, AAPL, BTC-USD)")
            st.write("2. 인터넷 연결 확인")  
            st.write("3. 기간 연장 (90일 이상)")
            return
        
        if prices.empty:
            st.error(f"❌ 수신한 데이터가 비어있음: {selected_ticker}")
            st.warning("**해결책:**")
            st.write("1. 티커 기호 정확성 확인")
            st.write("2. 다른 기간으로 시도 (예: 1년)")
            st.write("3. 다른 티커로 테스트 (예: SPY, GLD)")
            return
        
        if len(prices) < 20:
            st.warning(f"⚠️  데이터 부족: {len(prices)}개 (권장: 30개)")
            st.info(f"기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")
            if len(prices) < 10:
                st.error("❌ 최소한 10개 이상의 데이터 필요")
                return
        
        # Get the price series
        if isinstance(prices, pd.DataFrame):
            if selected_ticker in prices.columns:
                price_series = prices[selected_ticker]
            elif len(prices.columns) > 0:
                # Use first column
                col_name = prices.columns[0]
                price_series = prices[col_name]
                st.info(f"📍 '{selected_ticker}' 대신 '{col_name}' 사용")
            else:
                st.error(f"❌ DataFrame에 컬럼 없음")
                return
        else:
            price_series = prices
        
        # Verify price series
        if price_series is None or len(price_series) == 0:
            st.error(f"❌ 가격 데이터 비어있음")
            return
        
        # Check for all NaN
        if price_series.isna().all():
            st.error(f"❌ 모든 가격이 NaN입니다")
            return
        
        # Remove NaN for analysis
        price_series_clean = price_series.dropna()
        
        if len(price_series_clean) == 0:
            st.error(f"❌ NaN 제거 후 데이터 없음")
            return
        
        st.success(f"✓ 데이터 로드 완료!")
        st.write(f"📊 유효 데이터: {len(price_series_clean)}개")
        st.write(f"📅 기간: {price_series_clean.index[0].date()} ~ {price_series_clean.index[-1].date()}")
        st.write(f"💰 범위: ${price_series_clean.min():.2f} ~ ${price_series_clean.max():.2f}")
        
        st.divider()
        
        # Check if we have enough data for indicators
        if len(price_series_clean) < 26:
            st.error("❌ 지표 계산 불가 (최소 26개 필요)")
            st.write(f"현재: {len(price_series_clean)}개")
            return
        
        # Use clean series for analysis
        price_series = price_series_clean
        
        # Initialize day trading strategy
        dt_strategy = DayTradingSignals()
        
        # Calculate indicators
        rsi = dt_strategy.calculate_rsi(price_series)
        macd, signal_line, histogram = dt_strategy.calculate_macd(price_series)
        upper_bb, middle_bb, lower_bb = dt_strategy.calculate_bollinger_bands(price_series)
        
        # Generate signals
        signals_dict = dt_strategy.generate_signals(pd.DataFrame({selected_ticker: price_series}))
        
        if not signals_dict or selected_ticker not in signals_dict:
            st.error(f"❌ 신호 생성 실패")
            return
        
        indicators = signals_dict[selected_ticker]
        
        # Display current metrics
        st.subheader("📊 현재 지표")
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = price_series.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        with col1:
            st.metric("현재가", f"${current_price:.2f}", delta=f"{price_series.pct_change().iloc[-1]:+.2%}")
        
        with col2:
            color_rsi = "🔴 과매수" if current_rsi > 70 else "🟢 과매도" if current_rsi < 30 else "🟡 중립"
            st.metric("RSI (14)", f"{current_rsi:.1f}", delta=color_rsi)
        
        with col3:
            st.metric("MACD", f"{current_macd:.4f}", delta=f"{current_macd - current_signal:+.4f}")
        
        with col4:
            st.metric("신호선", f"{current_signal:.4f}")
        
        st.divider()
        
        # Signal generation
        st.subheader("🎯 거래 신호")
        
        latest_signal = indicators['signal'].iloc[-1]
        
        if latest_signal == 1:
            st.success("### 🟢 매수 신호")
            st.markdown(f"""
            **현재 상태:**
            - RSI: {current_rsi:.1f} (과매도 < 30)
            - 현재가: ${current_price:.2f}
            - MACD: 상승 모멘텀 감지됨
            
            **거래액션:** 매수 포지션 진입
            **손절매:** ${current_price * 0.99:.2f} (1% 하락)
            **익절매:** ${current_price * 1.02:.2f} (2% 상승)
            """)
        elif latest_signal == -1:
            st.error("### 🔴 매도 신호")
            st.markdown(f"""
            **현재 상태:**
            - RSI: {current_rsi:.1f} (과매수 > 70)
            - 현재가: ${current_price:.2f}
            - MACD: 하강 모멘텀 감지됨
            
            **거래액션:** 매도 포지션 진입
            **손절매:** ${current_price * 1.01:.2f} (1% 상승)
            **익절매:** ${current_price * 0.98:.2f} (2% 하락)
            """)
        else:
            st.warning("### 🟡 관망 신호")
            st.markdown(f"""
            **현재 상태:**
            - RSI: {current_rsi:.1f} (중립 30-70)
            - 현재가: ${current_price:.2f}
            - MACD: 혼합 신호
            
            **거래액션:** 명확한 신호 대기
            **다음 지지선:** ${lower_bb.iloc[-1]:.2f}
            **다음 저항선:** ${upper_bb.iloc[-1]:.2f}
            """)
        
        st.divider()
        
        # Detailed indicator charts
        st.subheader("📈 상세 차트 분석")
        
        tab1, tab2, tab3, tab4 = st.tabs(["가격 & 볼린저밴드", "RSI", "MACD", "신호 이력"])
        
        with tab1:
            # Price and Bollinger Bands
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=price_series.index, y=price_series.values, mode='lines', name='가격', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=upper_bb.index, y=upper_bb.values, mode='lines', name='상단밴드', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=middle_bb.index, y=middle_bb.values, mode='lines', name='중간밴드', line=dict(color='blue', dash='dash')))
            fig.add_trace(go.Scatter(
                x=lower_bb.index.tolist() + upper_bb.index.tolist()[::-1],
                y=lower_bb.values.tolist() + upper_bb.values.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,200,0.1)',
                name='밴드폭'
            ))
            
            fig.update_layout(title=f"{selected_ticker} 가격 & 볼린저밴드", height=400, xaxis_title="날짜", yaxis_title="가격")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # RSI
            fig = go.Figure()
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수 (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="과매도 (30)")
            fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values, mode='lines', name='RSI', line=dict(color='purple')))
            
            fig.update_layout(title=f"{selected_ticker} RSI (14)", height=400, xaxis_title="날짜", yaxis_title="RSI", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # MACD
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=macd.index, y=macd.values, mode='lines', name='MACD', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line.values, mode='lines', name='신호선', line=dict(color='red')))
            fig.add_trace(go.Bar(x=histogram.index, y=histogram.values, name='히스토그램', marker=dict(color='gray')))
            
            fig.update_layout(title=f"{selected_ticker} MACD", height=400, xaxis_title="날짜", yaxis_title="MACD")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Signal history and reliability metrics
            st.subheader("📊 신호 이력 & 신뢰성")
            
            # Get signal history
            signal_history = dt_strategy.get_signal_history(
                pd.DataFrame({selected_ticker: price_series}),
                selected_ticker,
                lookback_days=min(100, len(price_series))
            )
            
            # Display signal history
            st.markdown("**최근 신호 이력 (최근 20개)**")
            
            if len(signal_history) > 0:
                display_history = signal_history[['날짜', '가격', '신호_이름', 'RSI', '강도']].tail(20).copy()
                display_history['날짜'] = display_history['날짜'].dt.strftime('%Y-%m-%d')
                display_history['가격'] = display_history['가격'].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(display_history, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Calculate and display reliability metrics
            st.markdown("**신뢰성 지표**")
            
            metrics = dt_strategy.calculate_backtest_metrics(pd.DataFrame({selected_ticker: price_series}))
            
            if selected_ticker in metrics:
                metric_data = metrics[selected_ticker]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "총 매수 신호",
                        f"{metric_data['total_buy_signals']}개",
                        help="지난 기간 동안 발생한 매수 신호 수"
                    )
                
                with col2:
                    st.metric(
                        "총 매도 신호",
                        f"{metric_data['total_sell_signals']}개",
                        help="지난 기간 동안 발생한 매도 신호 수"
                    )
                
                with col3:
                    win_rate = metric_data['win_rate']
                    color_delta = f"+{win_rate:.1f}%" if win_rate > 50 else f"{win_rate:.1f}%"
                    st.metric(
                        "승률",
                        f"{win_rate:.1f}%",
                        delta=color_delta,
                        help="매수 후 5일 수익을 얻은 비율"
                    )
                
                with col4:
                    avg_ret = metric_data['avg_return_pct']
                    color_delta = f"+{avg_ret:.2f}%" if avg_ret > 0 else f"{avg_ret:.2f}%"
                    st.metric(
                        "평균 수익률",
                        f"{avg_ret:.2f}%",
                        delta=color_delta,
                        help="매수 후 5일 평균 수익률"
                    )
                
                st.divider()
                
                # Detailed reliability analysis
                st.markdown("**신뢰성 분석**")
                
                if metric_data['total_buy_signals'] > 0:
                    win_count = metric_data['win_count']
                    loss_count = metric_data['loss_count']
                    
                    # Risk/Reward analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"""
                        **신호 분포:**
                        - 수익 신호: {win_count}개 ({win_count/metric_data['total_buy_signals']*100:.1f}%)
                        - 손실 신호: {loss_count}개 ({loss_count/metric_data['total_buy_signals']*100:.1f}%)
                        - 총합: {metric_data['total_buy_signals']}개
                        """)
                    
                    with col2:
                        returns = metric_data['returns']
                        if returns:
                            max_ret = max(returns) * 100
                            min_ret = min(returns) * 100
                            st.info(f"""
                            **수익률 범위:**
                            - 최대 수익: +{max_ret:.2f}%
                            - 최대 손실: {min_ret:.2f}%
                            - 중위값: {np.median(returns)*100:.2f}%
                            """)
                else:
                    st.warning("📊 신호가 충분하지 않아 신뢰성 지표를 계산할 수 없습니다.")
            
            st.divider()
            
            # Historical signal chart
            st.markdown("**신호 발생 추이**")
            
            if len(signal_history) > 0:
                # Create a chart showing signals over time
                signal_chart_data = signal_history[['날짜', '가격', '신호']].copy()
                signal_chart_data = signal_chart_data.set_index('날짜')
                
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=signal_chart_data.index,
                    y=signal_chart_data['가격'].values,
                    mode='lines',
                    name='가격',
                    line=dict(color='black', width=2)
                ))
                
                # Add buy signals (green dots)
                buy_signals = signal_chart_data[signal_chart_data['신호'] == 1]
                if len(buy_signals) > 0:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['가격'].values,
                        mode='markers',
                        name='매수',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ))
                
                # Add sell signals (red dots)
                sell_signals = signal_chart_data[signal_chart_data['신호'] == -1]
                if len(sell_signals) > 0:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['가격'].values,
                        mode='markers',
                        name='매도',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ))
                
                fig.update_layout(
                    title=f"{selected_ticker} - 신호 이력 (과거 {len(signal_history)}일)",
                    xaxis_title="날짜",
                    yaxis_title="가격",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Risk Management
        st.subheader("⚠️ 위험관리 규칙")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("""
            **포지션 크기:**
            - 거래당 최대 2%
            - 손절매: 1%
            - 익절매: 2%
            """)
        
        with col2:
            st.info("""
            **일일 거래 규칙:**
            - 일일 최대 3회 거래
            - 첫 거래: 오전 10:00-11:00
            - 마지막 거래: 오후 3:00 이전
            """)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.error(traceback.format_exc())


def main():
    """
    Main execution flow with Streamlit UI.
    """
    
    # Header
    st.title("📈 Quantitative Trading System")
    st.markdown("""
    **Dual Momentum + Market Regime Filter + ML Signal Filter**
    
    Comprehensive backtesting platform for sector rotation strategy
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.divider()
    
    # Data parameters
    st.sidebar.subheader("Data Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        data_start = st.sidebar.date_input("Data Start Date", value=pd.Timestamp('2010-01-01'))
    with col2:
        data_end = st.sidebar.date_input("Data End Date", value=pd.Timestamp.today())
    
    # Backtest parameters
    st.sidebar.subheader("Backtest Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        backtest_start = st.sidebar.date_input("Backtest Start Date", value=pd.Timestamp('2015-01-01'))
    with col2:
        backtest_end = st.sidebar.date_input("Backtest End Date", value=pd.Timestamp.today())
    
    # Display information
    st.sidebar.divider()
    st.sidebar.subheader("📊 System Information")
    st.sidebar.info(f"""
    **Tickers:** {len(ALL_TICKERS)}
    
    **Data Span:** 2010-2026
    
    **Strategy:** Sector Rotation
    
    **Rebalance:** Weekly
    
    **Initial Capital:** $100,000
    """)
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["📌 What to Do?", "🚀 Full Pipeline", "⚡ Day Trading"])
    
    with tab1:
        show_what_to_do()
    
    with tab3:
        show_day_trading_analysis()
    
    with tab2:
        run_button = st.button(
            "🚀 RUN COMPLETE PIPELINE",
            use_container_width=True,
            key="run_button"
        )
        
        if not run_button:
            # Initial state
            st.info("👈 Click 'RUN COMPLETE PIPELINE' button above to start full analysis")
            st.divider()
            
            st.subheader("🎯 System Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tickers", len(ALL_TICKERS))
            with col2:
                st.metric("Strategy", "Sector Rotation")
            with col3:
                st.metric("Rebalance Frequency", "Weekly")
            
            st.divider()
            st.subheader("📚 About This System")
            st.markdown("""
            This quantitative trading system performs comprehensive backtesting of a sector rotation strategy using:
            
            1. **Data Download** - Download price data from Yahoo Finance
            2. **Feature Engineering** - Create momentum, volatility, and macro features
            3. **Strategy Analysis** - Analyze regime filters, momentum signals, and rankings
            4. **ML Filtering** - Train machine learning models to improve signals
            5. **Walk-Forward Backtest** - Perform robust out-of-sample testing
            6. **Performance Analysis** - Calculate comprehensive performance metrics
            """)
            
            return
        
        # Execution phase
        st.header("Running Complete Pipeline...")
        st.divider()
        
        try:
            # Step 1: Download and prepare data
            st.header("1️⃣ Data Download & Feature Engineering")
            with st.spinner("Preparing data..."):
                result = download_and_prepare_data(
                    start_date=data_start.strftime('%Y-%m-%d'),
                    end_date=data_end.strftime('%Y-%m-%d')
                )
                
                # Check if result is None (error occurred)
                if result[0] is None:
                    st.error("❌ Failed to prepare data. Cannot continue with pipeline.")
                    return
                
                prices, vix, momentum_features, volatility_features, macro_features = result
            
            # Verify all data is valid
            if prices is None or len(prices) == 0:
                st.error("❌ Price data is empty")
                return
            
            st.write(f"✓ Downloaded **{len(prices)}** trading days")
            st.write(f"✓ Downloaded **{len(prices.columns)}** tickers")
            st.write(f"✓ Created **{len(momentum_features.columns)}** momentum features")
            st.write(f"✓ Created **{len(volatility_features.columns)}** volatility features")
            st.write(f"✓ Created **{len(macro_features.columns)}** macro features")
            
            st.divider()
            
            # Step 2: Analyze strategy components
            st.header("2️⃣ Strategy Analysis")
            regime, signals, selected, weights = \
                analyze_strategy_components(prices, momentum_features, volatility_features, macro_features)
            
            st.divider()
            
            # Step 3: Train ML filter
            st.header("3️⃣ Machine Learning Signal Filter")
            ml_filter = train_ml_filter(momentum_features, volatility_features, macro_features, prices)
            
            st.divider()
            
            # Step 4: Run walk-forward backtest
            st.header("4️⃣ Walk-Forward Backtest")
            analyzer, results = run_walk_forward_backtest(
                prices,
                start_date=backtest_start.strftime('%Y-%m-%d'),
                end_date=backtest_end.strftime('%Y-%m-%d')
            )
            
            st.divider()
            
            # Step 5: Calculate performance metrics
            st.header("5️⃣ Performance Analysis")
            equity_curve = results['equity_curve']
            metrics = calculate_performance_metrics(equity_curve, initial_capital=100000)
            
            st.divider()
            
            # Step 6: Generate summary report
            st.header("6️⃣ Summary Report")
            generate_summary_report(analyzer, results, metrics)
            
            st.divider()
            
            # Completion message
            st.success("✅ Pipeline execution complete!")
            
            # Export results
            st.subheader("💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download CSV Reports"):
                    # Create CSV from equity curve
                    csv_data = equity_curve.to_csv()
                    st.download_button(
                        label="Download Equity Curve",
                        data=csv_data,
                        file_name="equity_curve.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 View Raw Data"):
                    st.subheader("Equity Curve Data")
                    st.dataframe(equity_curve, use_container_width=True)
            
            return {
                'prices': prices,
                'features': {
                    'momentum': momentum_features,
                    'volatility': volatility_features,
                    'macro': macro_features
                },
                'backtest_results': results,
                'metrics': metrics,
                'analyzer': analyzer
            }
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.error(traceback.format_exc())


if __name__ == '__main__':
    main()
