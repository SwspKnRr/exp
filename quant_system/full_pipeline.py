"""
Full Pipeline Streamlit Dashboard
Complete interactive UI for the quantitative trading system.
- Run full pipeline from data download to backtest
- Adjust parameters and visualize results
- Multi-language support (EN/KO)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, '..')

from data.loader import download_price_data, download_vix_data, ALL_TICKERS
from features.momentum import create_momentum_features
from features.volatility import create_volatility_features
from features.macro import create_macro_features, calculate_regime_filter
from strategy.regime_filter import RegimeFilter
from strategy.dual_momentum import DualMomentum
from strategy.ranking import ETFRanker
from ml.filter import MLSignalFilter
from backtest.walk_forward import WalkForwardAnalyzer
from backtest.metrics import PerformanceMetrics

# Translations
TRANSLATIONS = {
    'en': {
        'title': '📈 Quantitative Trading System - Complete Pipeline',
        'subtitle': 'Interactive Dashboard for Strategy Development & Backtesting',
        'language': 'Language',
        'sidebar_config': 'Configuration',
        'run_pipeline': '▶️ Run Complete Pipeline',
        'stop_pipeline': '⏹️ Stop',
        
        # Step 1
        'step1': 'Step 1: Data Download & Features',
        'start_date': 'Start Date',
        'end_date': 'End Date',
        'select_tickers': 'Select Tickers',
        'data_preview': 'Data Preview',
        'features_summary': 'Features Summary',
        'downloaded_days': 'Trading Days Downloaded',
        'tickers_loaded': 'Tickers Loaded',
        
        # Step 2
        'step2': 'Step 2: Strategy Analysis',
        'market_regime': 'Market Regime Filter',
        'uptrend_ratio': 'Uptrend Ratio (SPY > MA200)',
        'momentum_analysis': 'Dual Momentum Analysis',
        'signal_rate': 'Signal Generation Rate',
        'active_signals': 'Active Signals Today',
        'etf_ranking': 'ETF Ranking',
        'top_etfs': 'Top 10 ETFs by Momentum',
        'selected_portfolio': 'Selected Portfolio (Top 3)',
        
        # Step 3
        'step3': 'Step 3: ML Signal Filter',
        'ml_training': 'RandomForest Training',
        'train_accuracy': 'Train Accuracy',
        'val_accuracy': 'Validation Accuracy',
        'train_samples': 'Training Samples',
        'val_samples': 'Validation Samples',
        'feature_importance': 'Top 10 Important Features',
        'training': 'Training in progress...',
        'training_complete': 'Training completed!',
        
        # Step 4
        'step4': 'Step 4: Walk-Forward Backtest',
        'train_years': 'Train Window (years)',
        'test_years': 'Test Window (years)',
        'initial_capital': 'Initial Capital ($)',
        'transaction_cost': 'Transaction Cost (%)',
        'backtest_progress': 'Backtest Progress',
        'running_backtest': 'Running walk-forward backtest...',
        'backtest_complete': 'Backtest completed!',
        
        # Step 5 & 6
        'step5': 'Step 5: Performance Metrics',
        'step6': 'Step 6: Results Summary',
        'final_results': 'Final Results',
        'total_return': 'Total Return',
        'cagr': 'CAGR',
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'max_drawdown': 'Max Drawdown',
        'volatility': 'Volatility',
        'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor',
        'final_value': 'Final Capital',
        'total_trades': 'Total Trades',
        'period_summary': 'Walk-Forward Period Summary',
        'avg_return': 'Average Period Return',
        'best_return': 'Best Period Return',
        'worst_return': 'Worst Period Return',
        'trades_per_period': 'Average Trades per Period',
        
        # Plots
        'equity_curve': 'Portfolio Equity Curve',
        'drawdown_chart': 'Portfolio Drawdown',
        'returns_distribution': 'Daily Returns Distribution',
        'period_returns': 'Returns by Walk-Forward Period',
        
        # Messages
        'loading': 'Loading...',
        'waiting': 'Waiting for input...',
        'no_data': 'No data available',
        'error': 'Error',
        'success': 'Success',
    },
    'ko': {
        'title': '📈 정량화 거래 시스템 - 전체 파이프라인',
        'subtitle': '전략 개발 및 백테스트를 위한 대화형 대시보드',
        'language': '언어',
        'sidebar_config': '설정',
        'run_pipeline': '▶️ 전체 파이프라인 실행',
        'stop_pipeline': '⏹️ 중지',
        
        # Step 1
        'step1': '단계 1: 데이터 다운로드 및 특성 생성',
        'start_date': '시작 날짜',
        'end_date': '종료 날짜',
        'select_tickers': '티커 선택',
        'data_preview': '데이터 미리보기',
        'features_summary': '특성 요약',
        'downloaded_days': '다운로드된 거래일',
        'tickers_loaded': '로드된 티커',
        
        # Step 2
        'step2': '단계 2: 전략 분석',
        'market_regime': '시장 체제 필터',
        'uptrend_ratio': '상승 추세 비율 (SPY > MA200)',
        'momentum_analysis': '듀얼 모멘텀 분석',
        'signal_rate': '신호 생성률',
        'active_signals': '오늘의 활성 신호',
        'etf_ranking': 'ETF 순위',
        'top_etfs': '상위 10개 ETF (모멘텀)',
        'selected_portfolio': '선택된 포트폴리오 (상위 3개)',
        
        # Step 3
        'step3': '단계 3: ML 신호 필터',
        'ml_training': 'RandomForest 학습',
        'train_accuracy': '학습 정확도',
        'val_accuracy': '검증 정확도',
        'train_samples': '학습 샘플',
        'val_samples': '검증 샘플',
        'feature_importance': '상위 10개 중요 특성',
        'training': '학습 중...',
        'training_complete': '학습 완료!',
        
        # Step 4
        'step4': '단계 4: 워크-포워드 백테스트',
        'train_years': '학습 윈도우 (년)',
        'test_years': '테스트 윈도우 (년)',
        'initial_capital': '초기 자본금 ($)',
        'transaction_cost': '거래 비용 (%)',
        'backtest_progress': '백테스트 진행률',
        'running_backtest': '워크-포워드 백테스트 실행 중...',
        'backtest_complete': '백테스트 완료!',
        
        # Step 5 & 6
        'step5': '단계 5: 성과 메트릭',
        'step6': '단계 6: 결과 요약',
        'final_results': '최종 결과',
        'total_return': '총 수익률',
        'cagr': 'CAGR',
        'sharpe_ratio': 'Sharpe 비율',
        'sortino_ratio': 'Sortino 비율',
        'max_drawdown': '최대 낙폭',
        'volatility': '변동성',
        'win_rate': '승률',
        'profit_factor': '이익 계수',
        'final_value': '최종 자본금',
        'total_trades': '총 거래',
        'period_summary': '워크-포워드 기간 요약',
        'avg_return': '평균 기간 수익률',
        'best_return': '최고 기간 수익률',
        'worst_return': '최악 기간 수익률',
        'trades_per_period': '기간당 평균 거래',
        
        # Plots
        'equity_curve': '포트폴리오 자산 곡선',
        'drawdown_chart': '포트폴리오 낙폭',
        'returns_distribution': '일일 수익률 분포',
        'period_returns': '워크-포워드 기간별 수익률',
        
        # Messages
        'loading': '로드 중...',
        'waiting': '입력 대기 중...',
        'no_data': '사용 가능한 데이터 없음',
        'error': '오류',
        'success': '성공',
    }
}

def t(key):
    """Get translated text."""
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS[lang].get(key, key)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'pipeline_data' not in st.session_state:
    st.session_state.pipeline_data = {}

# Page config
st.set_page_config(
    page_title='Quantitative Trading System',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Configuration
st.sidebar.markdown(f"## {t('sidebar_config')}")

# Language selector
language = st.sidebar.radio(
    t('language'),
    options=['en', 'ko'],
    format_func=lambda x: '🇬🇧 English' if x == 'en' else '🇰🇷 한국어',
    key='language'
)

st.sidebar.markdown("---")

# Pipeline parameters
st.sidebar.subheader("📊 Pipeline Parameters")

# Data parameters
with st.sidebar.expander("📅 Data Settings", expanded=True):
    start_date = st.date_input(
        t('start_date'),
        value=pd.Timestamp('2015-01-01')
    )
    end_date = st.date_input(
        t('end_date'),
        value=pd.Timestamp.now()
    )

# Strategy parameters
with st.sidebar.expander("⚙️ Strategy Settings", expanded=False):
    train_years = st.slider(t('train_years'), 3, 10, 5)
    test_years = st.slider(t('test_years'), 1, 3, 1)
    initial_capital = st.number_input(
        t('initial_capital'),
        min_value=10000,
        max_value=1000000,
        value=100000,
        step=10000
    )
    transaction_cost = st.slider(
        t('transaction_cost'),
        0.0, 1.0, 0.05
    ) / 100

# Main title
st.markdown(f"# {t('title')}")
st.markdown(f"### {t('subtitle')}")

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t('step1'), t('step2'), t('step3'), 
    t('step4'), t('step5'), t('step6')
])

# Run Pipeline Button
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run_button = st.button('▶️ ' + t('run_pipeline'), use_container_width=True, key='run_btn')

# ============================================================================
# TAB 1: DATA DOWNLOAD & FEATURES
# ============================================================================
with tab1:
    st.header(t('step1'))
    
    if run_button or st.session_state.get('execute_step1', False):
        st.session_state.execute_step1 = True
        
        with st.spinner(t('loading')):
            try:
                # Download data
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Downloading price data...")
                progress_bar.progress(10)
                
                prices = download_price_data(
                    ALL_TICKERS,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    progress=False
                )
                
                progress_bar.progress(30)
                status_text.text("Downloading VIX data...")
                
                vix = download_vix_data(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                progress_bar.progress(50)
                status_text.text("Creating momentum features...")
                
                momentum_features = create_momentum_features(prices)
                
                progress_bar.progress(65)
                status_text.text("Creating volatility features...")
                
                volatility_features = create_volatility_features(prices)
                
                progress_bar.progress(80)
                status_text.text("Creating macro features...")
                
                macro_features = create_macro_features(prices, vix)
                
                progress_bar.progress(100)
                status_text.text("✓ Data preparation complete!")
                
                # Store in session
                st.session_state.pipeline_data['prices'] = prices
                st.session_state.pipeline_data['vix'] = vix
                st.session_state.pipeline_data['momentum_features'] = momentum_features
                st.session_state.pipeline_data['volatility_features'] = volatility_features
                st.session_state.pipeline_data['macro_features'] = macro_features
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(t('downloaded_days'), f"{len(prices):,}")
                with col2:
                    st.metric(t('tickers_loaded'), len(prices.columns))
                with col3:
                    st.metric("Momentum Features", len(momentum_features.columns))
                with col4:
                    st.metric("Volatility Features", len(volatility_features.columns))
                
                # Data preview
                st.subheader(t('data_preview'))
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Price Data (Last 5 days)**")
                    st.dataframe(prices.tail(), use_container_width=True)
                
                with col2:
                    st.write("**VIX Data (Last 5 days)**")
                    st.dataframe(vix.tail(), use_container_width=True)
                
                st.markdown('<div class="success-box">✓ Data loaded successfully!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">✗ Error: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.info(t('waiting'))

# ============================================================================
# TAB 2: STRATEGY ANALYSIS
# ============================================================================
with tab2:
    st.header(t('step2'))
    
    if 'prices' in st.session_state.pipeline_data:
        prices = st.session_state.pipeline_data['prices']
        momentum_features = st.session_state.pipeline_data['momentum_features']
        volatility_features = st.session_state.pipeline_data['volatility_features']
        macro_features = st.session_state.pipeline_data['macro_features']
        
        # Market Regime
        st.subheader(t('market_regime'))
        with st.spinner("Analyzing market regime..."):
            regime = calculate_regime_filter(prices, benchmark='SPY', ma_period=200)
            uptrend_ratio = regime.sum() / len(regime)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(t('uptrend_ratio'), f"{uptrend_ratio:.2%}")
            with col2:
                st.metric("Market Status", "📈 Uptrend" if regime.iloc[-1] else "📉 Downtrend")
            
            # Regime chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prices.index, y=prices['SPY'],
                mode='lines', name='SPY Price', line=dict(color='blue')
            ))
            ma200 = prices['SPY'].rolling(200).mean()
            fig.add_trace(go.Scatter(
                x=prices.index, y=ma200,
                mode='lines', name='MA200', line=dict(color='red', dash='dash')
            ))
            fig.update_layout(height=400, title='SPY Price vs MA200')
            st.plotly_chart(fig, use_container_width=True)
        
        # Dual Momentum
        st.subheader(t('momentum_analysis'))
        with st.spinner("Analyzing momentum signals..."):
            dm = DualMomentum(momentum_period=20, benchmark='SPY')
            signals = dm.generate_signals(prices)
            signal_ratio = (signals == 1).sum().sum() / (signals.shape[0] * signals.shape[1])
            active_signals = (signals.iloc[-1] == 1).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t('signal_rate'), f"{signal_ratio:.2%}")
            with col2:
                st.metric(t('active_signals'), int(active_signals))
            with col3:
                top_signals = signals.iloc[-1][signals.iloc[-1] == 1].head(3).index.tolist()
                st.write(f"**Top Signals:** {', '.join(top_signals)}")
        
        # ETF Ranking
        st.subheader(t('etf_ranking'))
        with st.spinner("Ranking ETFs..."):
            ranker = ETFRanker(max_positions=3)
            ranking = ranker.rank_etfs(prices, exclude_tickers=['SPY'])
            selected = ranker.select_portfolio(prices, exclude_tickers=['SPY'])
            
            # Chart
            charcol1, charcol2 = st.columns([1, 2])
            
            with charcol1:
                st.write("**Selected Portfolio**")
                for i, ticker in enumerate(selected, 1):
                    st.write(f"{i}. {ticker} (33.33%)")
            
            with charcol2:
                fig = px.bar(
                    ranking.head(10),
                    x='score', y='ticker',
                    orientation='h',
                    title=t('top_etfs'),
                    labels={'score': 'Momentum Score', 'ticker': 'ETF'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.session_state.pipeline_data['ranking'] = ranking
    else:
        st.info(t('waiting'))

# ============================================================================
# TAB 3: ML SIGNAL FILTER
# ============================================================================
with tab3:
    st.header(t('step3'))
    
    if 'prices' in st.session_state.pipeline_data:
        prices = st.session_state.pipeline_data['prices']
        momentum_features = st.session_state.pipeline_data['momentum_features']
        volatility_features = st.session_state.pipeline_data['volatility_features']
        macro_features = st.session_state.pipeline_data['macro_features']
        
        with st.spinner(t('training')):
            try:
                ml_filter = MLSignalFilter(probability_threshold=0.6)
                
                # Train on SPY
                spy_momentum = momentum_features[[col for col in momentum_features.columns if not col.endswith('RSI')]]
                
                metrics = ml_filter.train_model(
                    spy_momentum, volatility_features, macro_features,
                    prices, 'SPY', prediction_horizon=5, validation_split=0.2
                )
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(t('train_accuracy'), f"{metrics['train_accuracy']:.2%}")
                with col2:
                    st.metric(t('val_accuracy'), f"{metrics['val_accuracy']:.2%}")
                with col3:
                    st.metric(t('train_samples'), int(metrics['train_samples']))
                with col4:
                    st.metric(t('val_samples'), int(metrics['val_samples']))
                
                # Feature importance
                st.subheader(t('feature_importance'))
                importance = ml_filter.model.get_feature_importance(top_n=10)
                importance_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in importance.items()
                ])
                
                fig = px.bar(
                    importance_df,
                    x='Importance', y='Feature',
                    orientation='h',
                    title='Feature Importance Scores'
                )
                fig.update_yaxes(autorange='reversed')
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.pipeline_data['ml_filter'] = ml_filter
                st.markdown('<div class="success-box">✓ ML training completed!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">✗ Error: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.info(t('waiting'))

# ============================================================================
# TAB 4: WALK-FORWARD BACKTEST
# ============================================================================
with tab4:
    st.header(t('step4'))
    
    if 'prices' in st.session_state.pipeline_data:
        prices = st.session_state.pipeline_data['prices']
        
        with st.spinner(t('running_backtest')):
            try:
                def select_portfolio(train_prices, train_features, date):
                    ranker = ETFRanker(max_positions=3)
                    return ranker.select_portfolio(train_prices, date=date, exclude_tickers=['SPY'])
                
                analyzer = WalkForwardAnalyzer(
                    train_years=train_years,
                    test_years=test_years,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost
                )
                
                st.write(f"Train window: {train_years} years | Test window: {test_years} year(s) | Rebalance: Weekly")
                
                progress_bar = st.progress(0)
                results = analyzer.run_walk_forward(
                    prices, select_portfolio,
                    start_date=pd.Timestamp(start_date),
                    end_date=pd.Timestamp(end_date),
                    verbose=False
                )
                progress_bar.progress(100)
                
                st.session_state.pipeline_data['analyzer'] = analyzer
                st.session_state.pipeline_data['backtest_results'] = results
                
                # Summary statistics
                summary = analyzer.get_summary_statistics()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("# Periods", len(summary))
                with col2:
                    st.metric(t('total_trades'), f"{summary['num_trades'].sum():.0f}")
                with col3:
                    st.metric(t('avg_return'), f"{summary['total_return'].mean():.2%}")
                with col4:
                    st.metric(t('best_return'), f"{summary['total_return'].max():.2%}")
                
                st.subheader(t('period_summary'))
                st.dataframe(summary, use_container_width=True)
                
                st.markdown('<div class="success-box">✓ Backtest completed!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">✗ Error: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.info(t('waiting'))

# ============================================================================
# TAB 5: PERFORMANCE METRICS
# ============================================================================
with tab5:
    st.header(t('step5'))
    
    if 'backtest_results' in st.session_state.pipeline_data:
        results = st.session_state.pipeline_data['backtest_results']
        
        if len(results['equity_curve']) > 0:
            eq_curve = results['equity_curve']
            
            # Calculate metrics
            calculator = PerformanceMetrics(risk_free_rate=0.02)
            metrics = calculator.calculate_all_metrics(
                eq_curve['portfolio_value'],
                initial_capital
            )
            
            st.session_state.pipeline_data['metrics'] = metrics
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(t('total_return'), f"{metrics['Total Return']:.2%}")
            with col2:
                st.metric(t('cagr'), f"{metrics['CAGR']:.2%}")
            with col3:
                st.metric(t('sharpe_ratio'), f"{metrics['Sharpe Ratio']:.2f}")
            with col4:
                st.metric(t('max_drawdown'), f"{metrics['Max Drawdown']:.2%}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(t('volatility'), f"{metrics['Volatility']:.2%}")
            with col2:
                st.metric(t('sortino_ratio'), f"{metrics['Sortino Ratio']:.2f}")
            with col3:
                st.metric(t('win_rate'), f"{metrics['Win Rate']:.2%}")
            with col4:
                st.metric(t('profit_factor'), f"{metrics['Profit Factor']:.2f}")
            
            # Equity curve chart
            st.subheader(t('equity_curve'))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_curve.index, y=eq_curve['portfolio_value'],
                mode='lines', name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=t('equity_curve'),
                xaxis_title='Date', yaxis_title='Portfolio Value ($)',
                hovermode='x unified', height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily returns distribution
            daily_ret = eq_curve['portfolio_value'].pct_change().dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=daily_ret * 100, nbinsx=50))
            fig.update_layout(
                title=t('returns_distribution'),
                xaxis_title='Daily Return (%)', height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(t('no_data'))
    else:
        st.info(t('waiting'))

# ============================================================================
# TAB 6: SUMMARY REPORT
# ============================================================================
with tab6:
    st.header(t('step6'))
    
    if 'metrics' in st.session_state.pipeline_data:
        metrics = st.session_state.pipeline_data['metrics']
        results = st.session_state.pipeline_data['backtest_results']
        analyzer = st.session_state.pipeline_data['analyzer']
        
        st.subheader(t('final_results'))
        
        # Summary table
        summary_data = {
            t('initial_capital'): f"${initial_capital:,.2f}",
            t('final_value'): f"${metrics['Final Capital']:,.2f}",
            t('total_return'): f"{metrics['Total Return']:.2%}",
            t('cagr'): f"{metrics['CAGR']:.2%}",
            t('sharpe_ratio'): f"{metrics['Sharpe Ratio']:.2f}",
            t('sortino_ratio'): f"{metrics['Sortino Ratio']:.2f}",
            t('max_drawdown'): f"{metrics['Max Drawdown']:.2%}",
            t('volatility'): f"{metrics['Volatility']:.2%}",
            t('win_rate'): f"{metrics['Win Rate']:.2%}",
            t('profit_factor'): f"{metrics['Profit Factor']:.2f}",
            t('total_trades'): f"{len(results['trades']):.0f}",
        }
        
        summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Period returns chart
        st.subheader(t('period_returns'))
        period_summary = analyzer.get_summary_statistics()
        
        fig = px.bar(
            period_summary,
            x='period', y='total_return',
            labels={'period': 'Period', 'total_return': 'Return'},
            title=t('period_returns'),
            color='total_return',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Aggregate statistics
        st.subheader("📊 Aggregate Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("# Walk-Forward Periods", len(period_summary))
        with col2:
            st.metric(t('avg_return'), f"{period_summary['total_return'].mean():.2%}")
        with col3:
            st.metric(t('best_return'), f"{period_summary['total_return'].max():.2%}")
        with col4:
            st.metric(t('worst_return'), f"{period_summary['total_return'].min():.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t('total_trades'), f"{period_summary['num_trades'].sum():.0f}")
        with col2:
            st.metric(t('trades_per_period'), f"{period_summary['num_trades'].mean():.1f}")
        
        # Download report
        st.divider()
        if st.button("📥 Download Summary Report (CSV)"):
            csv = period_summary.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info(t('waiting'))

# Footer
st.divider()
st.markdown("""
---
**💡 Tips:**
- Configure parameters in the sidebar and click **▶️ Run Complete Pipeline** to execute
- Use tabs to explore different stages of the analysis
- See the equity curve, rankings, and metrics in real-time
- Download results for further analysis
""")
