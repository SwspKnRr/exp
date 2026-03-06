"""
Streamlit Dashboard for Quantitative Trading System
Display backtest results, portfolio allocation, and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys

# Add parent directory to path
sys.path.insert(0, '..')

from data.loader import download_price_data, download_vix_data, create_macro_signals, ALL_TICKERS
from features.momentum import create_momentum_features
from features.volatility import create_volatility_features
from features.macro import create_macro_features, calculate_regime_filter
from strategy.regime_filter import RegimeFilter
from strategy.dual_momentum import DualMomentum
from strategy.ranking import ETFRanker
from backtest.walk_forward import WalkForwardAnalyzer
from backtest.metrics import PerformanceMetrics


def load_data(start_date='2015-01-01', end_date=None):
    """Load price data and features."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    prices = download_price_data(ALL_TICKERS, start_date=start_date, end_date=end_date, progress=False)
    vix = download_vix_data(start_date=start_date, end_date=end_date)
    
    return prices, vix


def create_features(prices, vix):
    """Create feature sets."""
    momentum_features = create_momentum_features(prices)
    volatility_features = create_volatility_features(prices)
    macro_features = create_macro_features(prices, vix)
    
    return momentum_features, volatility_features, macro_features


def plot_equity_curve(equity_curve):
    """Plot equity curve."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_rankings(prices, date=None):
    """Plot ETF rankings."""
    if date is None:
        date = prices.index[-1]
    
    ranker = ETFRanker(max_positions=10)
    ranking = ranker.rank_etfs(prices, date=date, exclude_tickers=['SPY'])
    
    fig = px.bar(
        ranking.head(10),
        x='score',
        y='ticker',
        orientation='h',
        title='Top 10 ETF Momentum Scores',
        labels={'score': 'Momentum Score', 'ticker': 'ETF'}
    )
    
    fig.update_layout(height=400)
    
    return fig, ranking.head(10)


def plot_regime_filter(prices):
    """Plot market regime over time."""
    regime = calculate_regime_filter(prices, benchmark='SPY', ma_period=200)
    
    fig = go.Figure()
    
    # Add price
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices['SPY'],
        mode='lines',
        name='SPY Price',
        yaxis='y1',
        line=dict(color='blue')
    ))
    
    # Add MA200
    ma200 = prices['SPY'].rolling(window=200).mean()
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=ma200,
        mode='lines',
        name='MA200',
        yaxis='y1',
        line=dict(color='red', dash='dash')
    ))
    
    # Add regime bands
    fig.add_trace(go.Scatter(
        x=regime[regime].index,
        y=prices.loc[regime[regime].index, 'SPY'],
        mode='markers',
        name='Uptrend',
        marker=dict(color='green', size=4, opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=regime[~regime].index,
        y=prices.loc[regime[~regime].index, 'SPY'],
        mode='markers',
        name='Downtrend',
        marker=dict(color='red', size=4, opacity=0.5)
    ))
    
    fig.update_layout(
        title='Market Regime Filter (SPY > MA200)',
        xaxis_title='Date',
        yaxis_title='SPY Price ($)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def main():
    """Main Streamlit app."""
    
    st.set_page_config(
        page_title='Quantitative Trading System',
        page_icon='📈',
        layout='wide'
    )
    
    st.title('📈 Quantitative Trading System - Sector Rotation Strategy')
    
    # Sidebar controls
    st.sidebar.header('Configuration')
    
    start_date = st.sidebar.date_input(
        'Start Date',
        value=datetime(2015, 1, 1)
    )
    
    end_date = st.sidebar.date_input(
        'End Date',
        value=datetime.now()
    )
    
    show_rankings = st.sidebar.checkbox('Show ETF Rankings', value=True)
    show_regime = st.sidebar.checkbox('Show Market Regime', value=True)
    run_backtest = st.sidebar.checkbox('Run Walk-Forward Backtest', value=False)
    
    # Main content
    st.header('📊 Dashboard')
    
    # Load data
    with st.spinner('Loading data...'):
        prices, vix = load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        momentum_features, volatility_features, macro_features = create_features(prices, vix)
    
    st.success('Data loaded successfully!')
    
    # Display key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('Current SPY Price', f"${prices['SPY'].iloc[-1]:.2f}")
    
    with col2:
        spy_return = (prices['SPY'].iloc[-1] / prices['SPY'].iloc[0] - 1) * 100
        st.metric('SPY Return', f"{spy_return:.2f}%")
    
    with col3:
        st.metric('Current VIX', f"{vix.iloc[-1]:.2f}")
    
    with col4:
        st.metric('Data Points', f"{len(prices):,}")
    
    st.markdown('---')
    
    # ETF Rankings
    if show_rankings:
        st.subheader('🏆 ETF Rankings (Latest)')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ranking_df = plot_rankings(prices)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(ranking_df, use_container_width=True)
    
    # Market Regime
    if show_regime:
        st.subheader('📈 Market Regime Filter')
        
        regime = calculate_regime_filter(prices)
        uptrend_ratio = regime.sum() / len(regime)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = plot_regime_filter(prices)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric('Uptrend Ratio', f"{uptrend_ratio:.2%}")
            st.metric('Current Status', 'UPTREND' if regime.iloc[-1] else 'DOWNTREND')
    
    # Walk-Forward Backtest
    if run_backtest:
        st.subheader('🔄 Walk-Forward Backtest')
        
        with st.spinner('Running walk-forward analysis... This may take several minutes'):
            # Define portfolio selection function
            def select_portfolio(train_prices, train_features, date):
                ranker = ETFRanker(max_positions=3)
                return ranker.select_portfolio(train_prices, date, exclude_tickers=['SPY'])
            
            # Run walk-forward analysis
            analyzer = WalkForwardAnalyzer(
                train_years=5,
                test_years=1,
                initial_capital=100000
            )
            
            results = analyzer.run_walk_forward(
                prices, select_portfolio,
                start_date=pd.Timestamp(start_date),
                end_date=pd.Timestamp(end_date),
                verbose=False
            )
            
            # Display results
            if 'equity_curve' in results and len(results['equity_curve']) > 0:
                fig = plot_equity_curve(results['equity_curve'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics
                summary = analyzer.get_summary_statistics()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric('Avg Period Return', f"{summary['total_return'].mean():.2%}")
                
                with col2:
                    st.metric('Total Periods', len(summary))
                
                with col3:
                    st.metric('Total Trades', summary['num_trades'].sum())
                
                with col4:
                    st.metric('Best Period', f"{summary['total_return'].max():.2%}")
                
                # Period details
                st.dataframe(summary, use_container_width=True)
    
    # Feature Statistics
    st.subheader('📉 Feature Statistics')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write('**Momentum Features**')
        st.write(momentum_features[['return_20d', 'return_60d', 'return_120d']].describe())
    
    with col2:
        st.write('**Volatility Features**')
        st.write(volatility_features[['volatility_20', 'volatility_60']].describe())
    
    with col3:
        st.write('**Macro Features**')
        st.write(macro_features[['VIX', 'SPY_GLD_ratio']].describe())
    
    # Footer
    st.markdown('---')
    st.markdown('''
    **Quantitative Trading System - Sector Rotation with Dual Momentum**
    
    - Strategy: Dual momentum + Market regime filter + ML signal filter
    - Rebalancing: Weekly (Monday)
    - Risk Management: 3 max positions, equal weight
    - Backtesting: Walk-forward analysis (5 years train, 1 year test)
    
    Built with Python | Streamlit | Pandas | scikit-learn
    ''')


if __name__ == '__main__':
    main()
