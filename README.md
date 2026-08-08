# HedgeQuant Futures Engine ⚡

A high-performance, institutional-grade futures backtesting and risk analytics platform built with **Python**, **Pandas**, **NumPy**, **Statsmodels**, and **Streamlit**. 

This system allows traders and quantitative analysts to backtest three distinct algorithmic trading strategies, run advanced portfolio allocation sandboxes, and inspect deep risk attribution metrics including CAPM factor regression models.

---

## 🚀 Key Features

### 1. Multi-Strategy Suite
* **Moving-Average Crossover (Baseline):** Supports SMA & EMA options with configurable fast and slow windows.
* **Time-Series Momentum (TSMOM):** Standard trend-following momentum adjusted by rolling volatility and filtered with an Average Directional Index (ADX) filter to minimize range-bound whipsaws.
* **Mean Reversion (Z-Score):** Path-dependent hysteretic entry and exit thresholds based on rolling price standard deviations.

### 2. High-Fidelity Futures Accounting
* **Leverage & Margining:** Simulates Initial and Maintenance margin requirements based on contract-specific exchange parameters.
* **Execution Friction:** Accounts for contract-level round-turn commissions and point-based slippage drag.
* **Risk Liquidation:** Simulates forced liquidations (margin calls) when account equity falls below the maintenance margin limit.
* **Contract Specification Database:** Built-in specifications for S&P 500 E-mini (`ES=F`), Crude Oil (`CL=F`), Gold (`GC=F`), and 10-Year US Treasuries (`ZN=F`).

### 3. Quantitative Risk & Attribution
* **CAPM Regression:** Leverages `statsmodels` OLS regression to decompose strategy returns into Alpha and Beta components, displaying statistical significance (p-values) and model fit ($R^2$).
* **Value-at-Risk (VaR) & Expected Shortfall (ES):** Displays 95% and 99% confidence limits using both parametric (Gaussian) and historical simulation approaches.
* **Advanced Visualizations:** Premium dark-themed interactive Plotly line charts, drawdown curves, rolling risk diagnostics, and Matplotlib-generated monthly return heatmaps.

### 4. Interactive Blended Portfolio Sandbox
* Allocate weights dynamically across all three strategies to construct a blended multi-strategy portfolio.
* Visualize the correlation and diversification benefits (reduced max drawdowns, improved Sharpe ratios) of running a multi-strategy book.

---

## 📁 Directory Structure

```
futures-backtest-engine/
├── app.py                     # Streamlit application entrypoint & dashboard UI
├── requirements.txt           # Python library dependencies
├── test_runner.py             # Test suite validating math & engineering pipelines
├── core/
│   ├── __init__.py
│   ├── data_engine.py         # Data download (Yahoo Finance) & GARCH synthetic path generator
│   ├── backtest_engine.py     # Vectorized accounting, margin-limits, and commission module
│   ├── analytics.py           # Risk measures (VaR, ES) & CAPM regression model fitters
│   └── strategies/
│       ├── __init__.py
│       ├── base.py            # Base strategy interface
│       ├── ma_crossover.py    # Baseline moving average logic
│       ├── momentum.py        # Trend following with ADX filters
│       └── mean_reversion.py  # Rolling Z-score with path-dependent entries
└── utils/
    ├── __init__.py
    ├── styles.py              # CSS styling injected for dark Bloomberg terminal aesthetic
    └── visualization.py       # Custom interactive Plotly and Seaborn Heatmap routines
```

---

## 🛠️ Installation & Setup

1. **Install Dependencies:**
   Ensure you have Python 3.9+ and pip installed. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Verification Tests:**
   Verify the math, backtester, and regression models are functioning correctly before launching:
   ```bash
   python test_runner.py
   ```

3. **Launch the Dashboard:**
   Start the interactive Streamlit terminal:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Backtester Mathematical Methodology

### Daily Marking-to-Market
Account equity ($E_t$) is updated daily as:
$$E_t = E_{t-1} + N_{contracts} \times (P_t - P_{t-1}) \times M \times S_{t-1} - C_t$$

Where:
* $P_t$: Closing price of the underlying contract.
* $M$: Contract point multiplier (e.g., $50 for ES).
* $S_{t-1}$: Strategy signal (-1 for Short, 0 for Flat, +1 for Long).
* $C_t$: Transaction costs (fees + slippage points adjusted for contracts traded).

### Margin Call Liquidation
If the account equity drops below the maintenance margin:
$$E_t < N_{contracts} \times P_t \times M \times R_{maint}$$

The position is immediately liquidated (flattened) at the day's closing price, paying execution friction, and the account enters a neutral cash position.
