import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from typing import Dict, Any, Tuple

def calculate_performance_metrics(results_df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, Any]:
    equity = results_df['equity']
    returns = results_df['strategy_return']
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    n_days = len(results_df)
    years = n_days / 252.0
    
    final_equity = equity.iloc[-1]
    initial_equity = equity.iloc[0]
    cagr = (final_equity / initial_equity) ** (1 / max(years, 0.001)) - 1 if final_equity > 0 else -1.0
    
    ann_vol = returns.std() * np.sqrt(252)
    
    excess_returns = returns - daily_rf
    ann_excess_return = excess_returns.mean() * 252
    sharpe = ann_excess_return / ann_vol if ann_vol > 0 else 0.0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = ann_excess_return / downside_std if downside_std > 0 else 0.0
    
    cum_max = equity.cummax()
    drawdowns = (equity - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    is_in_dd = drawdowns < 0
    dd_durations = []
    current_dur = 0
    for val in is_in_dd:
        if val:
            current_dur += 1
        else:
            if current_dur > 0:
                dd_durations.append(current_dur)
            current_dur = 0
    if current_dur > 0:
        dd_durations.append(current_dur)
    max_dd_duration = max(dd_durations) if dd_durations else 0
    
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0
    
    positions = results_df['position']
    position_changes = positions.diff().fillna(0.0)
    
    trades = []
    active_trade = None
    
    df_trades = results_df.copy()
    df_trades['pos_shift'] = position_changes
    
    trade_returns = []
    trade_pnl = []
    trade_durations = []
    win_trades = 0
    loss_trades = 0
    
    trade_indices = results_df[results_df['transaction_costs'] > 0].index
    active_periods = results_df[results_df['position'] != 0]
    current_trade_equity = 0.0
    in_trade = False
    trade_start_idx = 0
    
    for i in range(1, len(results_df)):
        prev_pos = positions.iloc[i-1]
        curr_pos = positions.iloc[i]
        
        if prev_pos == 0 and curr_pos != 0:
            in_trade = True
            trade_start_idx = i
            current_trade_equity = equity.iloc[i-1]
        elif prev_pos != 0 and (curr_pos == 0 or curr_pos != prev_pos):
            if in_trade:
                trade_end_equity = equity.iloc[i]
                pnl = trade_end_equity - current_trade_equity
                ret = (trade_end_equity / current_trade_equity) - 1 if current_trade_equity > 0 else 0.0
                trade_returns.append(ret)
                trade_pnl.append(pnl)
                trade_durations.append(i - trade_start_idx)
                if pnl > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
                
                if curr_pos != 0:
                    trade_start_idx = i
                    current_trade_equity = equity.iloc[i-1]
                else:
                    in_trade = False
                    
    if in_trade:
        trade_end_equity = equity.iloc[-1]
        pnl = trade_end_equity - current_trade_equity
        ret = (trade_end_equity / current_trade_equity) - 1 if current_trade_equity > 0 else 0.0
        trade_returns.append(ret)
        trade_pnl.append(pnl)
        trade_durations.append(len(results_df) - trade_start_idx)
        if pnl > 0:
            win_trades += 1
        else:
            loss_trades += 1

    total_trades = len(trade_pnl)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    gross_profits = sum([p for p in trade_pnl if p > 0])
    gross_losses = abs(sum([p for p in trade_pnl if p < 0]))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else (gross_profits if gross_profits > 0 else 1.0)
    
    avg_win = np.mean([p for p in trade_pnl if p > 0]) if win_trades > 0 else 0.0
    avg_loss = np.mean([p for p in trade_pnl if p < 0]) if loss_trades > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 0 else 0.0
    
    historical_returns = returns.values
    
    var_95_hist = np.percentile(historical_returns, 5)
    var_99_hist = np.percentile(historical_returns, 1)
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    var_95_param = mean_ret + std_ret * stats.norm.ppf(0.05)
    var_99_param = mean_ret + std_ret * stats.norm.ppf(0.01)
    
    es_95 = historical_returns[historical_returns <= var_95_hist].mean() if len(historical_returns[historical_returns <= var_95_hist]) > 0 else var_95_hist
    es_99 = historical_returns[historical_returns <= var_99_hist].mean() if len(historical_returns[historical_returns <= var_99_hist]) > 0 else var_99_hist

    total_margin_calls = int(results_df['margin_call'].sum())
    max_leverage = results_df['leverage'].max()
    avg_leverage = results_df['leverage'].mean()
    
    return {
        'cagr': cagr,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'max_drawdown_duration': max_dd_duration,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'win_loss_ratio': win_loss_ratio,
        'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0,
        'var_95_historical': var_95_hist,
        'var_99_historical': var_99_hist,
        'var_95_parametric': var_95_param,
        'var_99_parametric': var_99_param,
        'expected_shortfall_95': es_95,
        'expected_shortfall_99': es_99,
        'total_margin_calls': total_margin_calls,
        'max_leverage': max_leverage,
        'avg_leverage': avg_leverage,
        'final_equity': final_equity,
        'total_return': (final_equity - initial_equity) / initial_equity,
        'benchmark_total_return': (results_df['benchmark_equity'].iloc[-1] - initial_equity) / initial_equity,
        'trade_pnl': trade_pnl,
        'trade_returns': trade_returns
    }

def run_regression_analysis(results_df: pd.DataFrame) -> Dict[str, Any]:
    y = results_df['strategy_return']
    x = results_df['benchmark_return']
    
    x_with_const = sm.add_constant(x)
    
    try:
        model = sm.OLS(y, x_with_const).fit()
        
        alpha_daily = model.params['const']
        alpha_annualized = alpha_daily * 252
        beta = model.params['benchmark_return']
        
        p_alpha = model.pvalues['const']
        p_beta = model.pvalues['benchmark_return']
        
        t_alpha = model.tvalues['const']
        t_beta = model.tvalues['benchmark_return']
        
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        f_statistic = model.fvalue
        
        return {
            'alpha_annualized': alpha_annualized,
            'alpha_daily': alpha_daily,
            'beta': beta,
            'p_value_alpha': p_alpha,
            'p_value_beta': p_beta,
            't_stat_alpha': t_alpha,
            't_stat_beta': t_beta,
            'r_squared': r_squared,
            'adjusted_r_squared': adj_r_squared,
            'f_statistic': f_statistic,
            'regression_summary': model.summary().as_text()
        }
    except Exception as e:
        print(f"Regression model failed: {e}")
        return {
            'alpha_annualized': 0.0,
            'alpha_daily': 0.0,
            'beta': 0.0,
            'p_value_alpha': 1.0,
            'p_value_beta': 1.0,
            't_stat_alpha': 0.0,
            't_stat_beta': 0.0,
            'r_squared': 0.0,
            'adjusted_r_squared': 0.0,
            'f_statistic': 0.0,
            'regression_summary': f"Error fitting CAPM regression model: {e}"
        }
