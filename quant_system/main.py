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
    
    # Download VIX
    status_container.info("📥 Downloading VIX data...")
    vix = download_vix_data(start_date=start_date, end_date=end_date)
    progress_bar.progress(40)
    
    # Create momentum features
    status_container.info("⚙️  Creating momentum features...")
    momentum_features = create_momentum_features(prices)
    progress_bar.progress(60)
    
    # Create volatility features
    status_container.info("⚙️  Creating volatility features...")
    volatility_features = create_volatility_features(prices)
    progress_bar.progress(80)
    
    # Create macro features
    status_container.info("⚙️  Creating macro features...")
    macro_features = create_macro_features(prices, vix)
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
    """
    st.subheader("💰 Current Recommendation")
    
    try:
        with st.spinner("Analyzing today's market..."):
            # Download today's data
            prices = download_price_data(ALL_TICKERS, start_date='2010-01-01', end_date=None)
            
            # Calculate ranking
            ranker = ETFRanker(max_positions=3)
            latest_ranking = ranker.rank_etfs(prices, exclude_tickers=['SPY'])
            selected = ranker.select_portfolio(prices, exclude_tickers=['SPY'])
            weights = ranker.get_portfolio_weights(prices, exclude_tickers=['SPY'], weight_method='equal')
            
            # Market regime
            regime_filter = RegimeFilter(benchmark='SPY', ma_period=200)
            regime = regime_filter.calculate_regime(prices)
            current_regime = regime.iloc[-1]
        
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
    tab1, tab2 = st.tabs(["📌 What to Do?", "🚀 Full Pipeline"])
    
    with tab1:
        show_what_to_do()
    
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
                prices, vix, momentum_features, volatility_features, macro_features = \
                    download_and_prepare_data(
                        start_date=data_start.strftime('%Y-%m-%d'),
                        end_date=data_end.strftime('%Y-%m-%d')
                    )
            
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
