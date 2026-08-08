import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import scipy.stats as stats

COLOR_STRATEGY = '#1f77b4'
COLOR_BENCHMARK = '#7f7f7f'
COLOR_DRAWDOWN = '#d62728'
COLOR_LONG = '#2ca02c'
COLOR_SHORT = '#ff7f0e'

def _apply_dark_theme(fig):
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def plot_equity_curves(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['benchmark_equity'],
        name='Benchmark (Buy & Hold)',
        line=dict(color=COLOR_BENCHMARK, width=1.5, dash='dot'),
        hovertemplate='Benchmark: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['equity'],
        name='Strategy Portfolio',
        line=dict(color=COLOR_STRATEGY, width=2.5),
        hovertemplate='Strategy: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Portfolio Value Curve (USD)", font=dict(size=16, weight="bold")),
        hovermode="x unified"
    )
    return _apply_dark_theme(fig)

def plot_drawdowns(df: pd.DataFrame) -> go.Figure:
    cum_max = df['equity'].cummax()
    drawdowns = (df['equity'] - cum_max) / cum_max * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=drawdowns,
        name='Drawdown',
        fill='tozeroy',
        line=dict(color=COLOR_DRAWDOWN, width=1.5),
        fillcolor='rgba(239, 68, 68, 0.15)',
        hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Rolling Drawdown Profile (%)", font=dict(size=16, weight="bold")),
        yaxis=dict(ticksuffix="%")
    )
    return _apply_dark_theme(fig)

def plot_signal_overlay(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, 
                        row_heights=[0.75, 0.25])
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        name='Underlying Price',
        line=dict(color='#94a3b8', width=1.5),
        hovertemplate='Close: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    signals = df['position']
    shifted_signals = signals.shift(1).fillna(0.0)
    
    buys = df[(signals == 1.0) & (shifted_signals != 1.0)]
    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys['close'],
        mode='markers',
        name='Enter Long / Exit Short',
        marker=dict(symbol='triangle-up', size=11, color=COLOR_LONG, line=dict(width=1, color='#ffffff')),
        hovertemplate='Buy Price: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    sells = df[(signals == -1.0) & (shifted_signals != -1.0)]
    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells['close'],
        mode='markers',
        name='Enter Short / Exit Long',
        marker=dict(symbol='triangle-down', size=11, color=COLOR_SHORT, line=dict(width=1, color='#ffffff')),
        hovertemplate='Sell Price: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    flats = df[(signals == 0.0) & (shifted_signals != 0.0)]
    fig.add_trace(go.Scatter(
        x=flats.index,
        y=flats['close'],
        mode='markers',
        name='Flatten Position',
        marker=dict(symbol='x', size=8, color='#ffffff', line=dict(width=1)),
        hovertemplate='Exit Price: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['position'],
        name='Position state',
        line=dict(color=COLOR_STRATEGY, width=1.5, shape='hv'),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.05)',
        hovertemplate='Position: %{y}<extra></extra>'
    ), row=2, col=1)
    
    fig.update_layout(
        title=dict(text="Trade Entry / Exit Diagnostics", font=dict(size=16, weight="bold")),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Position Space", row=2, col=1, tickvals=[-1, 0, 1])
    
    return _apply_dark_theme(fig)

def plot_return_distribution(df: pd.DataFrame) -> go.Figure:
    returns = df['strategy_return'] * 100
    
    counts, bins = np.histogram(returns, bins=50)
    bins = 0.5 * (bins[:-1] + bins[1:])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bins,
        y=counts,
        name='Returns Frequency',
        marker_color='rgba(31, 119, 180, 0.4)',
        marker_line=dict(color=COLOR_STRATEGY, width=1),
        hovertemplate='Daily Return Bin: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))
    
    mu, sigma = returns.mean(), returns.std()
    if sigma > 0:
        x_fit = np.linspace(returns.min(), returns.max(), 200)
        y_fit = stats.norm.pdf(x_fit, mu, sigma)
        y_fit = y_fit * len(returns) * (bins[1] - bins[0])
        
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            name='Gaussian Fit',
            line=dict(color='#d62728', width=2),
            hovertemplate='Prob density: %{y:.2f}<extra></extra>'
        ))
        
    fig.update_layout(
        title=dict(text="Daily Return Density vs Normal Distribution", font=dict(size=16, weight="bold")),
        xaxis=dict(ticksuffix="%"),
        yaxis=dict(title_text="Occurrences")
    )
    return _apply_dark_theme(fig)

def plot_monthly_heatmap(df: pd.DataFrame) -> plt.Figure:
    monthly_ret = df['strategy_return'].groupby([df.index.year, df.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    ).unstack() * 100
    
    monthly_ret.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_ret.index.name = 'Year'
    
    yearly_ret = df['strategy_return'].groupby(df.index.year).apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    monthly_ret['YTD'] = yearly_ret
    
    fig, ax = plt.subplots(figsize=(10, max(3, len(monthly_ret) * 0.4)))
    
    sns.heatmap(
        monthly_ret,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0.0,
        cbar=True,
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title("Monthly Return Heatmap (%)", fontsize=12, pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_rolling_metrics(df: pd.DataFrame, window: int = 60) -> go.Figure:
    returns = df['strategy_return']
    
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    
    rolling_mean = returns.rolling(window).mean()
    rolling_sharpe = (rolling_mean / returns.rolling(window).std()) * np.sqrt(252)
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(f"Rolling {window}-Day Sharpe Ratio", f"Rolling {window}-Day Realized Volatility (%)"))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=rolling_sharpe,
        name='Rolling Sharpe',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.05)',
        hovertemplate='Sharpe: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=rolling_vol,
        name='Rolling Vol',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='Vol: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)
    
    fig.update_layout(height=450)
    return _apply_dark_theme(fig)
