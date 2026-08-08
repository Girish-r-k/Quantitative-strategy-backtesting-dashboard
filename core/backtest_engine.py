import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from core.data_engine import CONTRACTS

class FuturesBacktester:
    def __init__(self, contract_symbol: str, initial_capital: float = 100000.0,
                 sizing_type: str = 'fixed', sizing_value: float = 1.0,
                 commission: float = None, slippage: float = None):
        
        self.contract_symbol = contract_symbol
        self.spec = CONTRACTS.get(contract_symbol, CONTRACTS['ES=F'])
        self.initial_capital = initial_capital
        self.sizing_type = sizing_type
        self.sizing_value = sizing_value
        
        self.commission = commission if commission is not None else self.spec['commission']
        self.slippage = slippage if slippage is not None else self.spec['slippage']
        self.multiplier = self.spec['multiplier']
        self.margin_rate = self.spec['margin_rate']
        self.maintenance_margin_rate = self.spec['maintenance_margin_rate']

    def run(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        results = df.copy()
        results['signal'] = signals.astype(float)
        
        n_days = len(results)
        equity = np.zeros(n_days)
        equity[0] = self.initial_capital
        
        positions = np.zeros(n_days)
        contracts_held = np.zeros(n_days)
        daily_pnl = np.zeros(n_days)
        tx_costs = np.zeros(n_days)
        margin_req = np.zeros(n_days)
        leverage = np.zeros(n_days)
        margin_call_flags = np.zeros(n_days, dtype=bool)
        
        close_prices = results['close'].values
        open_prices = results['open'].values
        signal_vals = results['signal'].values
        
        current_equity = self.initial_capital
        active_position = 0.0
        active_contracts = 0.0
        
        for i in range(1, n_days):
            price_today = close_prices[i]
            price_yesterday = close_prices[i-1]
            
            prev_sig = signal_vals[i-1]
            target_sig = signal_vals[i]
            
            if self.sizing_type == 'fixed':
                target_contracts = float(self.sizing_value)
            else:
                margin_per_contract = price_yesterday * self.multiplier * self.margin_rate
                if margin_per_contract > 0:
                    target_contracts = (current_equity * self.sizing_value) / margin_per_contract
                    target_contracts = max(0.0, np.floor(target_contracts))
                else:
                    target_contracts = 0.0
            
            pnl = 0.0
            costs = 0.0
            
            if active_position != 0:
                pnl = active_contracts * (price_today - price_yesterday) * self.multiplier * active_position
                
            if target_sig != active_position:
                trades_to_make = 0.0
                if active_position == 0:
                    trades_to_make = target_contracts
                elif target_sig == 0:
                    trades_to_make = active_contracts
                else:
                    trades_to_make = active_contracts + target_contracts
                    
                cost_per_contract = self.commission + (self.slippage * self.multiplier)
                costs = trades_to_make * cost_per_contract
                
                active_position = target_sig
                active_contracts = target_contracts if target_sig != 0 else 0.0
                
            current_equity += pnl - costs
            
            if current_equity <= 0:
                current_equity = 0.0
                active_position = 0.0
                active_contracts = 0.0
                
            maint_margin_req = 0.0
            if active_position != 0:
                maint_margin_req = active_contracts * price_today * self.multiplier * self.maintenance_margin_rate
                
            if maint_margin_req > 0 and current_equity < maint_margin_req:
                margin_call_flags[i] = True
                liquidation_cost = active_contracts * (self.commission + (self.slippage * self.multiplier))
                current_equity -= liquidation_cost
                current_equity = max(current_equity, 0.0)
                
                active_position = 0.0
                active_contracts = 0.0
                maint_margin_req = 0.0
                
            equity[i] = current_equity
            positions[i] = active_position
            contracts_held[i] = active_contracts
            daily_pnl[i] = pnl - costs
            tx_costs[i] = costs
            margin_req[i] = maint_margin_req
            
            notional_val = active_contracts * price_today * self.multiplier
            leverage[i] = notional_val / current_equity if current_equity > 0 else 0.0
            
        equity[0] = self.initial_capital
        positions[0] = 0.0
        contracts_held[0] = 0.0
        
        results['equity'] = equity
        results['position'] = positions
        results['contracts'] = contracts_held
        results['daily_pnl'] = daily_pnl
        results['transaction_costs'] = tx_costs
        results['margin_required'] = margin_req
        results['leverage'] = leverage
        results['margin_call'] = margin_call_flags
        
        results['strategy_return'] = results['equity'].pct_change().fillna(0.0)
        
        price_returns = results['close'].pct_change().fillna(0.0)
        results['benchmark_equity'] = self.initial_capital * (1 + price_returns).cumprod()
        results['benchmark_return'] = price_returns
        
        return results
