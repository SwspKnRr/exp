"""
Enhanced Dashboard Analytics - Daily/Weekly Returns Analysis
Streamlit widget functions for backtest result visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from backtest_analysis import (
    calculate_daily_returns,
    calculate_weekly_returns,
    get_cumulative_returns,
    get_returns_statistics,
    plot_cumulative_returns,
    plot_daily_returns,
    plot_weekly_returns,
    plot_returns_heatmap,
    plot_returns_distribution
)


def show_daily_weekly_analysis(analyzer, results,initial_capital):
    """
    Display comprehensive daily and weekly returns analysis.
    Extracts equity curve from analyzer and visualizes returns.
    """
    
    st.divider()
    st.header("💹 상세 수익률 분석 (Daily/Weekly Returns)")
    
    try:
        # Get equity curve from analyzer results
        # The structure depends on what run_walk_forward returns
        if hasattr(analyzer, 'equity_curve') and analyzer.equity_curve is not None:
            equity_curve = analyzer.equity_curve
        elif isinstance(results, dict) and 'equity_curve' in results:
            equity_curve = results['equity_curve']
        else:
            # Try to reconstruct from summary statistics
            st.warning("⚠️ Equity curve data not available for detailed analysis")
            return
        
        if equity_curve is None or len(equity_curve) == 0:
            st.warning("⚠️ No equity curve data available")
            return
        
        # Ensure equity_curve is a Series with DatetimeIndex
        if isinstance(equity_curve, dict):
            equity_curve = pd.Series(equity_curve)
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.to_datetime(equity_curve.index)
        
        equity_curve = equity_curve.sort_index()
        
        # Calculate returns
        daily_returns = calculate_daily_returns(equity_curve)
        weekly_returns = calculate_weekly_returns(equity_curve)
        cumulative_returns = get_cumulative_returns(equity_curve)
        
        # Get statistics
        stats = get_returns_statistics(daily_returns, weekly_returns)
        
        st.subheader("📊 수익률 요약 통계")
        
        # Daily returns statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "일일 평균 수익률",
                f"{stats['daily_mean']:+.3%}",
                delta=f"표준편차: {stats['daily_std']:.3%}"
            )
        
        with col2:
            st.metric(
                "일일 최대 수익률",
                f"{stats['daily_max']:+.3%}",
                delta=f"최소: {stats['daily_min']:+.3%}"
            )
        
        with col3:
            st.metric(
                "일일 승률",
                f"{stats['daily_win_rate']:.1%}",
                delta=f"양수: {stats['num_positive_days']}일"
            )
        
        with col4:
            st.metric(
                "수익성 일수",
                f"{stats['num_positive_days']}일",
                delta=f"손실: {stats['num_negative_days']}일"
            )
        
        # Weekly returns statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "주간 평균 수익률",
                f"{stats['weekly_mean']:+.2%}",
                delta=f"표준편차: {stats['weekly_std']:.2%}"
            )
        
        with col2:
            st.metric(
                "주간 최대 수익률",
                f"{stats['weekly_max']:+.2%}",
                delta=f"최소: {stats['weekly_min']:+.2%}"
            )
        
        with col3:
            st.metric(
                "주간 승률",
                f"{stats['weekly_win_rate']:.1%}",
                delta=f"양수: {stats['num_positive_weeks']}주"
            )
        
        with col4:
            st.metric(
                "수익성 주수",
                f"{stats['num_positive_weeks']}주",
                delta=f"손실: {stats['num_negative_weeks']}주"
            )
        
        st.divider()
        
        # Cumulative Returns
        st.subheader("📈 누적 수익률 추이")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_cumulative = plot_cumulative_returns(
                equity_curve,
                title="포트폴리오 누적 수익률",
                show_benchmark=False
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        with col2:
            # Cumulative return summary
            final_return = cumulative_returns.iloc[-1]
            best_return = cumulative_returns.max()
            worst_return = cumulative_returns.min()
            
            st.metric("최종 수익률", f"{final_return:+.2%}")
            st.metric("최고 수익률", f"{best_return:+.2%}", delta=f"누적")
            st.metric("최저 수익률", f"{worst_return:+.2%}", delta=f"누적")
            
            # Additional info
            st.caption(f"시작: {equity_curve.index[0].strftime('%Y-%m-%d')}")
            st.caption(f"종료: {equity_curve.index[-1].strftime('%Y-%m-%d')}")
            st.caption(f"기간: {len(equity_curve)}일")
        
        st.divider()
        
        # Daily Returns Visualization
        st.subheader("📊 일일 수익률 분석")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_daily = plot_daily_returns(daily_returns)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            # Daily returns distribution summary
            positive_days = (daily_returns > 0).sum()
            negative_days = (daily_returns < 0).sum()
            neutral_days = (daily_returns == 0).sum()
            
            st.metric("양수 매매일", f"{positive_days}일")
            st.metric("음수 매매일", f"{negative_days}일")
            st.metric("중립 매매일", f"{neutral_days}일")
            
            daily_avg = daily_returns.mean()
            daily_std = daily_returns.std()
            
            st.metric("평균", f"{daily_avg:+.3%}")
            st.metric("표준편차", f"{daily_std:.3%}")
        
        st.divider()
        
        # Weekly Returns Visualization
        st.subheader("📈 주간 수익률 분석")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_weekly = plot_weekly_returns(weekly_returns)
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        with col2:
            # Weekly returns distribution summary
            pos_weeks = (weekly_returns > 0).sum()
            neg_weeks = (weekly_returns < 0).sum()
            neutral_weeks = (weekly_returns == 0).sum()
            
            st.metric("양수 주간", f"{pos_weeks}주")
            st.metric("음수 주간", f"{neg_weeks}주")
            st.metric("중립 주간", f"{neutral_weeks}주")
            
            weekly_avg = weekly_returns.mean()
            weekly_std = weekly_returns.std()
            
            st.metric("평균", f"{weekly_avg:+.2%}")
            st.metric("표준편차", f"{weekly_std:.2%}")
        
        st.divider()
        
        # Returns Distribution
        st.subheader("📊 수익률 분포 비교")
        
        fig_distribution = plot_returns_distribution(daily_returns, weekly_returns)
        st.plotly_chart(fig_distribution, use_container_width=True)
        
        st.divider()
        
        # Heatmap: Weekly Returns by Week/Year
        st.subheader("🔥 주간 수익률 히트맵")
        
        try:
            fig_heatmap_week = plot_returns_heatmap(daily_returns, category='week')
            st.plotly_chart(fig_heatmap_week, use_container_width=True)
            st.caption("연도별 주차 수익률 히트맵 - 녹색(양수), 빨강색(음수)")
        except Exception as e:
            st.warning(f"⚠️ 주간 히트맵 생성 불가: {e}")
        
        st.divider()
        
        # Heatmap: Monthly Returns by Month/Year
        st.subheader("🔥 월간 수익률 히트맵")
        
        try:
            fig_heatmap_month = plot_returns_heatmap(daily_returns, category='month')
            st.plotly_chart(fig_heatmap_month, use_container_width=True)
            st.caption("연도별 월별 수익률 히트맵 - 녹색(양수), 빨강색(음수)")
        except Exception as e:
            st.warning(f"⚠️ 월간 히트맵 생성 불가: {e}")
        
        st.divider()
        
        # Detailed Daily Returns Table
        st.subheader("📋 일일 수익률 상세 (최근 20일)")
        
        daily_df = pd.DataFrame({
            'Date': daily_returns.index,
            'Portfolio Value': equity_curve.values,
            'Daily Return': daily_returns.values * 100,
            'Cumulative Return': cumulative_returns.values * 100
        })
        
        daily_df_display = daily_df.tail(20).copy()
        daily_df_display['Date'] = daily_df_display['Date'].dt.strftime('%Y-%m-%d')
        daily_df_display['Portfolio Value'] = daily_df_display['Portfolio Value'].apply(lambda x: f"${x:,.0f}")
        daily_df_display['Daily Return'] = daily_df_display['Daily Return'].apply(lambda x: f"{x:+.3f}%")
        daily_df_display['Cumulative Return'] = daily_df_display['Cumulative Return'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(daily_df_display, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Detailed Weekly Returns Table
        st.subheader("📋 주간 수익률 상세")
        
        weekly_df = pd.DataFrame({
            'Week Ending': weekly_returns.index,
            'Weekly Return': weekly_returns.values * 100,
            'Cumulative Return': [cumulative_returns.loc[d] * 100 if d in cumulative_returns.index else np.nan for d in weekly_returns.index]
        })
        
        weekly_df_display = weekly_df.copy()
        weekly_df_display['Week Ending'] = weekly_df_display['Week Ending'].dt.strftime('%Y-%m-%d')
        weekly_df_display['Weekly Return'] = weekly_df_display['Weekly Return'].apply(lambda x: f"{x:+.2f}%")
        weekly_df_display['Cumulative Return'] = weekly_df_display['Cumulative Return'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A")
        
        st.dataframe(weekly_df_display, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"❌ 분석 중 오류 발생: {e}")
        st.info("Import traceback to debug this issue")
