import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go

from core.data_engine import CONTRACTS, get_futures_data
from core.backtest_engine import FuturesBacktester
from core.strategies.ma_crossover import MACrossoverStrategy
from core.strategies.momentum import MomentumStrategy
from core.strategies.mean_reversion import MeanReversionStrategy
from core.analytics import calculate_performance_metrics, run_regression_analysis

from utils.styles import CSS_STYLES
from utils.visualization import (
    plot_equity_curves, plot_drawdowns, plot_signal_overlay,
    plot_return_distribution, plot_monthly_heatmap, plot_rolling_metrics
)

st.set_page_config(
    page_title="Futures Backtester & Risk Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)

st.title("Futures Backtesting & Risk Analytics Engine")
st.markdown("*A Python application for evaluating Momentum, Mean Reversion, and Moving-Average Crossover strategies.*")

st.sidebar.markdown("### 🌐 Data & Environment")

data_source = st.sidebar.radio(
    "Market Data Feed",
    ["Real Market Data (Yahoo Finance)", "High-Fidelity Synthetic Simulation"],
    index=0
)

contract_symbols = list(CONTRACTS.keys())
selected_symbol = st.sidebar.selectbox(
    "Target Futures Contract",
    contract_symbols,
    index=0,
    format_func=lambda x: f"{x} - {CONTRACTS[x]['name']}"
)

spec = CONTRACTS[selected_symbol]

today = datetime.now()
five_years_ago = today - timedelta(days=5*365)
start_date = st.sidebar.date_input("Start Date", five_years_ago)
end_date = st.sidebar.date_input("End Date", today)

if start_date >= end_date:
    st.sidebar.error("Error: Start Date must be prior to End Date.")

st.sidebar.markdown("### 💰 Account & Execution Model")
initial_capital = st.sidebar.number_input(
    "Initial Equity (USD)", 
    min_value=1000.0, 
    max_value=10000000.0, 
    value=spec['default_capital'], 
    step=5000.0
)

sizing_type = st.sidebar.selectbox(
    "Position Sizing Schema",
    ["fixed", "percent_equity"],
    format_func=lambda x: "Fixed Contracts count" if x == "fixed" else "Risk-Adjusted (% Account Equity Margin)"
)

if sizing_type == "fixed":
    sizing_value = st.sidebar.number_input(
        "Contracts Per Trade", 
        min_value=1.0, 
        max_value=100.0, 
        value=2.0, 
        step=1.0
    )
else:
    sizing_value = st.sidebar.slider(
        "Margin Allocation Weight (%)", 
        min_value=5.0, 
        max_value=50.0, 
        value=20.0, 
        step=1.0
    ) / 100.0

use_custom_costs = st.sidebar.checkbox("Override Default Fees & Slippage")
if use_custom_costs:
    commission = st.sidebar.number_input("Brokerage Fee ($ per contract/side)", min_value=0.0, value=spec['commission'], step=0.50)
    slippage = st.sidebar.number_input("Slippage Drag (points per trade)", min_value=0.0, value=spec['slippage'], step=0.01, format="%.4f")
else:
    commission = spec['commission']
    slippage = spec['slippage']

if "Synthetic" in data_source:
    st.sidebar.markdown("### 🎲 Simulation Settings")
    syn_vol = st.sidebar.slider("Annualized Volatility Target (%)", 5.0, 60.0, 20.0, 1.0) / 100.0
    syn_trend = st.sidebar.slider("Annualized Trend Drift (%)", -30.0, 30.0, 5.0, 1.0) / 100.0
    syn_regimes = st.sidebar.checkbox("Enable Regime Switching Paths", value=True)
else:
    syn_vol = 0.20
    syn_trend = 0.05
    syn_regimes = True

with st.spinner("Downloading and processing contract term structures..."):
    try:
        df = get_futures_data(
            symbol=selected_symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            use_synthetic=("Synthetic" in data_source),
            trend_strength=syn_trend,
            volatility_level=syn_vol,
            regime_switching=syn_regimes
        )
    except Exception as e:
        st.error(f"Failed to fetch market data: {e}. Falling back to synthetic simulation.")
        df = get_futures_data(
            symbol=selected_symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            use_synthetic=True,
            trend_strength=syn_trend,
            volatility_level=syn_vol,
            regime_switching=syn_regimes
        )

with st.expander("📝 View Contract Specifications", expanded=True):
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(f"**Contract:** {selected_symbol}")
        st.markdown(f"**Name:** {spec['name']}")
    with col_s2:
        st.markdown(f"**Sector:** {spec['sector']}")
        st.markdown(f"**Multiplier:** ${spec['multiplier']}/pt")
    with col_s3:
        st.markdown(f"**Tick Size:** {spec['tick_size']}")
        st.markdown(f"**Initial Margin:** {spec['margin_rate']*100:.1f}%")
    with col_s4:
        st.markdown(f"**Maint. Margin:** {spec['maintenance_margin_rate']*100:.1f}%")
        st.markdown(f"**Commissions:** ${commission:.2f}/side")

st.markdown('<div class="section-header">Strategy Configuration</div>', unsafe_allow_html=True)
strategy_name = st.selectbox(
    "Select Strategy to Analyze",
    ["Moving-Average Crossover (Baseline)", "Momentum Strategy (TSMOM)", "Mean Reversion Strategy"]
)

col_p1, col_p2, col_p3 = st.columns(3)

if strategy_name == "Moving-Average Crossover (Baseline)":
    with col_p1:
        fast_window = st.number_input("Fast MA Window (days)", min_value=2, max_value=100, value=10)
    with col_p2:
        slow_window = st.number_input("Slow MA Window (days)", min_value=5, max_value=250, value=50)
    with col_p3:
        ma_type = st.selectbox("Average Model", ["EMA", "SMA"])
    
    strategy = MACrossoverStrategy(fast_window=fast_window, slow_window=slow_window, ma_type=ma_type)

elif strategy_name == "Momentum Strategy (TSMOM)":
    with col_p1:
        lookback = st.number_input("Momentum Lookback (days)", min_value=5, max_value=250, value=60)
    with col_p2:
        use_adx = st.checkbox("Apply Trend Strength Filter (ADX)", value=True)
    with col_p3:
        adx_thresh = st.slider("ADX Filter Threshold", 10.0, 40.0, 20.0, 1.0) if use_adx else 20.0
        
    strategy = MomentumStrategy(lookback_window=lookback, adx_threshold=adx_thresh, use_adx_filter=use_adx)

else:
    with col_p1:
        mr_window = st.number_input("Rolling Mean Window (days)", min_value=5, max_value=150, value=20)
    with col_p2:
        entry_z = st.number_input("Entry Z-Score Limit (σ)", min_value=0.5, max_value=4.0, value=2.0, step=0.1)
    with col_p3:
        exit_z = st.number_input("Exit Z-Score Limit (σ)", min_value=-1.0, max_value=2.0, value=0.0, step=0.1)
        
    strategy = MeanReversionStrategy(rolling_window=mr_window, entry_threshold=entry_z, exit_threshold=exit_z)

signals = strategy.generate_signals(df)

backtester = FuturesBacktester(
    contract_symbol=selected_symbol,
    initial_capital=initial_capital,
    sizing_type=sizing_type,
    sizing_value=sizing_value,
    commission=commission,
    slippage=slippage
)
results = backtester.run(df, signals)

metrics = calculate_performance_metrics(results)
regression = run_regression_analysis(results)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance Dashboard", 
    "🎯 Signal Diagnostics", 
    "⚠️ Risk & CAPM Attribution", 
    "🎛️ Blended Portfolio Sandbox",
    "📝 Trade Ledger"
])

with tab1:
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            label="Net Return",
            value=f"{metrics['total_return']*100:.1f}%",
            delta=f"Bench: {metrics['benchmark_total_return']*100:.1f}%"
        )
        
    with kpi_col2:
        st.metric(
            label="Annualized Return (CAGR)",
            value=f"{metrics['cagr']*100:.2f}%",
            delta=f"Vol: {metrics['annualized_volatility']*100:.1f}%"
        )
        
    with kpi_col3:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics['sharpe_ratio']:.2f}",
            delta=f"Sortino: {metrics['sortino_ratio']:.2f}"
        )
        
    with kpi_col4:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics['max_drawdown']*100:.1f}%",
            delta=f"Duration: {metrics['max_drawdown_duration']} days",
            delta_color="inverse"
        )
        
    st.markdown('<div class="section-header">Performance Visualization</div>', unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        st.plotly_chart(plot_equity_curves(results), use_container_width=True)
        st.plotly_chart(plot_drawdowns(results), use_container_width=True)
    with col_chart2:
        st.markdown("<p style='text-align: center; color: #94a3b8; font-weight: bold;'>Year-on-Year Monthly Return Matrix</p>", unsafe_allow_html=True)
        fig_map = plot_monthly_heatmap(results)
        st.pyplot(fig_map)
        plt.close(fig_map)
        
        st.markdown("### 📊 Trade Efficiency Metrics")
        stat_df = pd.DataFrame({
            "Metric": [
                "Total Realized Trades", "Win Rate (%)", "Profit Factor", 
                "Avg. Win / Avg. Loss", "Max Leverage Reached", "Avg. Leverage Ratio",
                "Total Margin Violations"
            ],
            "Value": [
                f"{metrics['total_trades']}", f"{metrics['win_rate']*100:.1f}%", f"{metrics['profit_factor']:.2f}",
                f"{metrics['win_loss_ratio']:.2f}", f"{metrics['max_leverage']:.1f}x", f"{metrics['avg_leverage']:.2f}x",
                f"{metrics['total_margin_calls']}"
            ]
        })
        st.table(stat_df.set_index("Metric"))

with tab2:
    st.markdown('<div class="section-header">Trade & Signal Visual Validation</div>', unsafe_allow_html=True)
    st.info(
        "The chart below displays the underlying contract price action. "
        "Upward arrows show Long execution entries, downward arrows show Short execution entries, and 'X' represents flattening out of positions."
    )
    st.plotly_chart(plot_signal_overlay(results), use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Capital Asset Pricing Model (CAPM) Factor Attribution</div>', unsafe_allow_html=True)
    
    col_reg1, col_reg2 = st.columns([1, 1])
    with col_reg1:
        st.markdown(
        )
        
        if regression['p_value_alpha'] < 0.05:
            st.success("✨ **Statistically Significant Alpha:** The strategy has generated true excess returns not explained by benchmark beta.")
        else:
            st.warning("⚠️ **Insignificant Alpha:** The alpha coefficient is not statistically significant at 95% confidence. Returns might be dominated by beta or random variance.")
            
        st.markdown("### 📊 Distribution of Returns & Risk Limits")
        risk_df = pd.DataFrame({
            "Risk Metric (Daily Value-at-Risk / Shortfall)": [
                "Value at Risk (VaR) 95% - Historical",
                "Value at Risk (VaR) 99% - Historical",
                "Value at Risk (VaR) 95% - Parametric",
                "Value at Risk (VaR) 99% - Parametric",
                "Expected Shortfall (ES) 95% - Historical",
                "Expected Shortfall (ES) 99% - Historical"
            ],
            "Value (%)": [
                f"{metrics['var_95_historical']*100:.2f}%",
                f"{metrics['var_99_historical']*100:.2f}%",
                f"{metrics['var_95_parametric']*100:.2f}%",
                f"{metrics['var_99_parametric']*100:.2f}%",
                f"{metrics['expected_shortfall_95']*100:.2f}%",
                f"{metrics['expected_shortfall_99']*100:.2f}%"
            ]
        })
        st.table(risk_df.set_index("Risk Metric (Daily Value-at-Risk / Shortfall)"))
        
    with col_reg2:
        with st.expander("Show Raw OLS Regression Output Summary"):
            st.text(regression['regression_summary'])
            
        st.plotly_chart(plot_return_distribution(results), use_container_width=True)

    st.markdown('<div class="section-header">Rolling Risk Metrics</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_rolling_metrics(results), use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Multi-Strategy Portfolio Allocator</div>', unsafe_allow_html=True)
    st.markdown(
    )
    
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        w_crossover = st.slider("Moving Average Crossover Weight (%)", 0, 100, 30, 5)
    with col_w2:
        w_momentum = st.slider("Time-Series Momentum Weight (%)", 0, 100, 40, 5)
    with col_w3:
        w_reversion = st.slider("Mean Reversion Weight (%)", 0, 100, 30, 5)
        
    total_w = w_crossover + w_momentum + w_reversion
    
    if total_w != 100:
        st.error(f"Total weight must equal 100%. Current sum: {total_w}%")
    else:
        strat_xo = MACrossoverStrategy(fast_window=10, slow_window=50, ma_type='EMA')
        strat_mom = MomentumStrategy(lookback_window=60, use_adx_filter=True)
        strat_rev = MeanReversionStrategy(rolling_window=20, entry_threshold=2.0, exit_threshold=0.0)
        
        sig_xo = strat_xo.generate_signals(df)
        sig_mom = strat_mom.generate_signals(df)
        sig_rev = strat_rev.generate_signals(df)
        
        blended_signal = (
            (w_crossover / 100.0) * sig_xo + 
            (w_momentum / 100.0) * sig_mom + 
            (w_reversion / 100.0) * sig_rev
        )
        
        blend_backtester = FuturesBacktester(
            contract_symbol=selected_symbol,
            initial_capital=initial_capital,
            sizing_type=sizing_type,
            sizing_value=sizing_value,
            commission=commission,
            slippage=slippage
        )
        blend_results = blend_backtester.run(df, blended_signal)
        blend_metrics = calculate_performance_metrics(blend_results)
        
        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
        
        with comp_col1:
            st.metric(
                label="Blended Portfolio Return",
                value=f"{blend_metrics['total_return']*100:.1f}%",
                delta=f"vs Single: {metrics['total_return']*100:.1f}%"
            )
            
        with comp_col2:
            st.metric(
                label="Blended CAGR",
                value=f"{blend_metrics['cagr']*100:.2f}%",
                delta=f"Vol: {blend_metrics['annualized_volatility']*100:.1f}%"
            )
            
        with comp_col3:
            st.metric(
                label="Blended Sharpe Ratio",
                value=f"{blend_metrics['sharpe_ratio']:.2f}",
                delta=f"vs Single: {metrics['sharpe_ratio']:.2f}"
            )
            
        with comp_col4:
            st.metric(
                label="Blended Max Drawdown",
                value=f"{blend_metrics['max_drawdown']*100:.1f}%",
                delta=f"vs Single: {metrics['max_drawdown']*100:.1f}%",
                delta_color="inverse"
            )
            
        fig_blend = go.Figure()
        fig_blend.add_trace(go.Scatter(
            x=df.index, y=df['close'] * (initial_capital / df['close'].iloc[0]),
            name='Benchmark Index', line=dict(color='#7f7f7f', width=1.5, dash='dot')
        ))
        fig_blend.add_trace(go.Scatter(
            x=df.index, y=blend_results['equity'],
            name=f'Blended Portfolio ({w_crossover}/{w_momentum}/{w_reversion})',
            line=dict(color='#2ca02c', width=2.5)
        ))
        fig_blend.add_trace(go.Scatter(
            x=df.index, y=results['equity'],
            name=f'Selected Single Strategy ({strategy_name})',
            line=dict(color='#1f77b4', width=1.5, dash='dash')
        ))
        
        fig_blend.update_layout(
            title=dict(text="Portfolio Diversification Comparative Curves (USD)", font=dict(size=16, weight="bold")),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_blend, use_container_width=True)

with tab5:
    st.markdown('<div class="section-header">Historical Transaction Ledger</div>', unsafe_allow_html=True)
    
    trade_log = []
    positions = results['position'].values
    contracts = results['contracts'].values
    prices = results['close'].values
    dates = results.index
    capital = results['equity'].values
    costs = results['transaction_costs'].values
    
    active_pos = 0.0
    active_entry_price = 0.0
    
    for i in range(1, len(results)):
        prev_p = positions[i-1]
        curr_p = positions[i]
        
        if curr_p != prev_p:
            action = "FLATTEN"
            if curr_p > prev_p:
                action = "BUY / COVER" if prev_p != 0 else "BUY / ENTER LONG"
            elif curr_p < prev_p:
                action = "SELL / LIQUIDATE" if prev_p != 0 else "SELL / ENTER SHORT"
            
            direction = 1 if (curr_p > prev_p) else -1
            exec_price = prices[i] + (direction * slippage)
            
            qty = abs(contracts[i] - (contracts[i-1] if prev_p != 0 else 0))
            if qty == 0:
                qty = contracts[i] if curr_p != 0 else contracts[i-1]
                
            fee = costs[i]
            
            trade_pnl = 0.0
            if prev_p != 0 and (curr_p == 0 or np.sign(curr_p) != np.sign(prev_p)):
                trade_pnl = (prices[i] - active_entry_price) * spec['multiplier'] * prev_p * qty
                trade_pnl -= fee
            
            trade_log.append({
                "Date": dates[i].strftime('%Y-%m-%d'),
                "Action": action,
                "Size (Contracts)": round(qty, 2),
                "Execution Price": round(exec_price, 4),
                "Paid Fees ($)": round(fee, 2),
                "Realized PnL ($)": round(trade_pnl, 2),
                "Account Balance ($)": round(capital[i], 2),
                "Target Position": int(curr_p)
            })
            
            if curr_p != 0:
                active_entry_price = prices[i]
                active_pos = curr_p
                
    if trade_log:
        ledger_df = pd.DataFrame(trade_log)
        
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)
        
        csv_buffer = io.StringIO()
        ledger_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        st.download_button(
            label="📥 Download Full Transaction Ledger (CSV)",
            data=csv_bytes,
            file_name=f"{selected_symbol}_trade_ledger_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trades executed during the backtest window under current parameter rules.")
