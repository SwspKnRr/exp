"""
Enhanced Backtest Analysis Module
Extracts and visualizes daily/weekly returns from backtest results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Tuple


def calculate_daily_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate daily returns from equity curve.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio values by date (index=date, values=portfolio_value)
    
    Returns
    -------
    pd.Series
        Daily returns (index=date, values=daily_return_pct)
    """
    daily_returns = equity_curve.pct_change()
    daily_returns.name = 'daily_return'
    return daily_returns


def calculate_weekly_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate weekly returns from equity curve (Monday to Sunday).
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio values by date
    
    Returns
    -------
    pd.Series
        Weekly returns with week ending dates as index
    """
    # Resample to week end (Friday)
    weekly_equity = equity_curve.resample('W-FRI').last()
    weekly_returns = weekly_equity.pct_change()
    weekly_returns.name = 'weekly_return'
    return weekly_returns


def calculate_monthly_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate monthly returns from equity curve.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio values by date
    
    Returns
    -------
    pd.Series
        Monthly returns with month end dates as index
    """
    monthly_equity = equity_curve.resample('M').last()
    monthly_returns = monthly_equity.pct_change()
    monthly_returns.name = 'monthly_return'
    return monthly_returns


def get_cumulative_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from start date.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio values by date
    
    Returns
    -------
    pd.Series
        Cumulative return % (index=date, values=cumulative_return_pct)
    """
    initial_value = equity_curve.iloc[0]
    cumulative = (equity_curve - initial_value) / initial_value
    cumulative.name = 'cumulative_return'
    return cumulative


def create_daily_returns_dataframe(
    equity_curve: pd.Series,
    daily_returns: pd.Series = None
) -> pd.DataFrame:
    """
    Create detailed daily returns dataframe.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio values by date
    daily_returns : pd.Series, optional
        Pre-calculated daily returns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with date, portfolio_value, daily_return, cumulative_return
    """
    if daily_returns is None:
        daily_returns = calculate_daily_returns(equity_curve)
    
    cumulative = get_cumulative_returns(equity_curve)
    
    df = pd.DataFrame({
        'date': equity_curve.index,
        'portfolio_value': equity_curve.values,
        'daily_return': daily_returns.fillna(0).values,
        'cumulative_return': cumulative.fillna(0).values
    })
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    return df


def create_return_heatmap_data(
    daily_returns: pd.Series,
    category: str = 'week'
) -> pd.DataFrame:
    """
    Create heatmap data for returns by week/month.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns
    category : str
        'week' for week heatmap, 'month' for month/year heatmap
    
    Returns
    -------
    pd.DataFrame
        Pivot table for heatmap visualization
    """
    df = pd.DataFrame({
        'date': daily_returns.index,
        'return': daily_returns.values
    })
    df['date'] = pd.to_datetime(df['date'])
    
    if category == 'week':
        df['year'] = df['date'].dt.isocalendar().year
        df['week'] = df['date'].dt.isocalendar().week
        heatmap = df.pivot_table(
            values='return',
            index='week',
            columns='year',
            aggfunc='sum'
        )
    else:  # month
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        heatmap = df.pivot_table(
            values='return',
            index='month',
            columns='year',
            aggfunc='sum'
        )
    
    return heatmap


def get_returns_statistics(
    daily_returns: pd.Series,
    weekly_returns: pd.Series,
    annual_risk_free_rate: float = 0.02
) -> Dict:
    """
    Calculate comprehensive return statistics.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns
    weekly_returns : pd.Series
        Weekly returns
    annual_risk_free_rate : float
        Annual risk-free rate (default 2%)
    
    Returns
    -------
    Dict
        Statistics dictionary
    """
    daily_returns_clean = daily_returns.dropna()
    weekly_returns_clean = weekly_returns.dropna()
    
    stats = {
        # Daily stats
        'daily_mean': daily_returns_clean.mean(),
        'daily_std': daily_returns_clean.std(),
        'daily_min': daily_returns_clean.min(),
        'daily_max': daily_returns_clean.max(),
        'daily_win_rate': (daily_returns_clean > 0).sum() / len(daily_returns_clean),
        
        # Weekly stats
        'weekly_mean': weekly_returns_clean.mean(),
        'weekly_std': weekly_returns_clean.std(),
        'weekly_min': weekly_returns_clean.min(),
        'weekly_max': weekly_returns_clean.max(),
        'weekly_win_rate': (weekly_returns_clean > 0).sum() / len(weekly_returns_clean),
        
        # Distribution
        'num_positive_days': (daily_returns_clean > 0).sum(),
        'num_negative_days': (daily_returns_clean < 0).sum(),
        'num_positive_weeks': (weekly_returns_clean > 0).sum(),
        'num_negative_weeks': (weekly_returns_clean < 0).sum(),
    }
    
    return stats


def plot_cumulative_returns(
    equity_curve: pd.Series,
    title: str = "Portfolio Cumulative Returns",
    show_benchmark: bool = False,
    benchmark_prices: pd.Series = None
) -> go.Figure:
    """
    Plot cumulative returns from equity curve.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio values by date
    title : str
        Plot title
    show_benchmark : bool
        Whether to show benchmark comparison
    benchmark_prices : pd.Series, optional
        Benchmark price series
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    cumulative_returns = get_cumulative_returns(equity_curve)
    
    fig = go.Figure()
    
    # Portfolio returns
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=cumulative_returns.values * 100,
        mode='lines',
        name='Strategy',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Benchmark (if provided)
    if show_benchmark and benchmark_prices is not None:
        # Align dates
        common_dates = equity_curve.index.intersection(benchmark_prices.index)
        eq_aligned = equity_curve.loc[common_dates]
        bench_aligned = benchmark_prices.loc[common_dates]
        
        if len(common_dates) > 0:
            bench_cumulative = (bench_aligned - bench_aligned.iloc[0]) / bench_aligned.iloc[0]
            
            fig.add_trace(go.Scatter(
                x=common_dates,
                y=bench_cumulative.values * 100,
                mode='lines',
                name='Benchmark (SPY)',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_daily_returns(daily_returns: pd.Series) -> go.Figure:
    """
    Plot daily returns as bar chart.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Color bars by positive/negative
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in daily_returns.values]
    
    fig.add_trace(go.Bar(
        x=daily_returns.index,
        y=daily_returns.values * 100,
        marker=dict(color=colors),
        name='Daily Return',
        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.3f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Daily Returns (%)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_weekly_returns(weekly_returns: pd.Series) -> go.Figure:
    """
    Plot weekly returns as bar chart.
    
    Parameters
    ----------
    weekly_returns : pd.Series
        Weekly returns
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Color bars
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in weekly_returns.values]
    
    fig.add_trace(go.Bar(
        x=weekly_returns.index,
        y=weekly_returns.values * 100,
        marker=dict(color=colors),
        name='Weekly Return',
        hovertemplate='Week of %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Weekly Returns (%)",
        xaxis_title="Week Ending Date",
        yaxis_title="Return (%)",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_returns_heatmap(
    daily_returns: pd.Series,
    category: str = 'week'
) -> go.Figure:
    """
    Plot heatmap of returns by week or month/year.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns series
    category : str
        'week' for year weeks heatmap, 'month' for month/year heatmap
    
    Returns
    -------
    go.Figure
        Plotly heatmap figure
    """
    heatmap_data = create_return_heatmap_data(daily_returns, category)
    
    if category == 'week':
        title = "Weekly Returns Heatmap (Year by Week)"
        xaxis_title = "Year"
        yaxis_title = "Week of Year"
    else:
        title = "Monthly Returns Heatmap (Year by Month)"
        xaxis_title = "Year"
        yaxis_title = "Month"
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values * 100,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(heatmap_data.values * 100, 1),
        texttemplate='%{text:.1f}%',
        textfont={"size": 9},
        hovertemplate='%{y} - %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_returns_distribution(
    daily_returns: pd.Series,
    weekly_returns: pd.Series
) -> go.Figure:
    """
    Plot distribution of daily and weekly returns.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns
    weekly_returns : pd.Series
        Weekly returns
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Daily returns histogram
    fig.add_trace(go.Histogram(
        x=daily_returns.dropna().values * 100,
        name='Daily Returns',
        nbinsx=50,
        opacity=0.6,
        marker=dict(color='#1f77b4')
    ))
    
    # Weekly returns histogram
    fig.add_trace(go.Histogram(
        x=weekly_returns.dropna().values * 100,
        name='Weekly Returns',
        nbinsx=30,
        opacity=0.6,
        marker=dict(color='#ff7f0e')
    ))
    
    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400,
        template='plotly_white'
    )
    
    return fig
