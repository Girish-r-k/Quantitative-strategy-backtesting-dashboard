import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from core.data_engine import CONTRACTS

class FuturesBacktester:
    """
    Realistic vector-assisted simulation engine designed specifically for futures.
    Models leverage, initial/maintenance margin requirements, contract multipliers,
    commissions, slippage, and auto-liquidation (margin call) events.
    """
    def __init__(self, contract_symbol: str, initial_capital: float = 100000.0,
                 sizing_type: str = 'fixed', sizing_value: float = 1.0,
                 commission: float = None, slippage: float = None):
        
        self.contract_symbol = contract_symbol
        self.spec = CONTRACTS.get(contract_symbol, CONTRACTS['ES=F'])
        self.initial_capital = initial_capital
        self.sizing_type = sizing_type  # 'fixed' (contracts) or 'percent_equity' (margin weight)
        self.sizing_value = sizing_value # number of contracts or decimal percentage of equity
        
        # Override default specs if custom values are provided
        self.commission = commission if commission is not None else self.spec['commission']
        self.slippage = slippage if slippage is not None else self.spec['slippage']
        self.multiplier = self.spec['multiplier']
        self.margin_rate = self.spec['margin_rate']
        self.maintenance_margin_rate = self.spec['maintenance_margin_rate']

    def run(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Runs the backtest simulation step-by-step over the price history.
        Ensures accurate path dependency for equity compounding and margin calls.
        """
        results = df.copy()
        results['signal'] = signals.astype(float)
        
        # Pre-allocate backtest metrics
        n_days = len(results)
        equity = np.zeros(n_days)
        equity[0] = self.initial_capital
        
        positions = np.zeros(n_days)          # -1, 0, 1
        contracts_held = np.zeros(n_days)     # integer/float count of contracts
        daily_pnl = np.zeros(n_days)
        tx_costs = np.zeros(n_days)
        margin_req = np.zeros(n_days)
        leverage = np.zeros(n_days)
        margin_call_flags = np.zeros(n_days, dtype=bool)
        
        # Extract numpy arrays for speed
        close_prices = results['close'].values
        open_prices = results['open'].values
        signal_vals = results['signal'].values
        
        current_equity = self.initial_capital
        active_position = 0.0
        active_contracts = 0.0
        
        for i in range(1, n_days):
            price_today = close_prices[i]
            price_yesterday = close_prices[i-1]
            
            # 1. Check for signal changes (rebalancing at the close of yesterday, executed today)
            prev_sig = signal_vals[i-1]
            target_sig = signal_vals[i]
            
            # Position sizing determination
            if self.sizing_type == 'fixed':
                target_contracts = float(self.sizing_value)
            else: # percent_equity: allocation to margin
                # Number of contracts we can buy = (Equity * SizingValue) / (Price * Multiplier * MarginRate)
                margin_per_contract = price_yesterday * self.multiplier * self.margin_rate
                if margin_per_contract > 0:
                    target_contracts = (current_equity * self.sizing_value) / margin_per_contract
                    target_contracts = max(0.0, np.floor(target_contracts))
                else:
                    target_contracts = 0.0
            
            # Calculate PnL from yesterday's close to today's close
            pnl = 0.0
            costs = 0.0
            
            if active_position != 0:
                pnl = active_contracts * (price_today - price_yesterday) * self.multiplier * active_position
                
            # If position changed, calculate slippage and commissions
            # Signal shifts: from active_position to target_sig
            if target_sig != active_position:
                # Number of contracts traded
                trades_to_make = 0.0
                if active_position == 0:
                    trades_to_make = target_contracts
                elif target_sig == 0:
                    trades_to_make = active_contracts
                else: # Direction reversal: e.g. Long to Short
                    trades_to_make = active_contracts + target_contracts
                    
                # Commission per contract + Slippage execution penalty
                # Slippage acts as a price drag: execution is worse than close price
                cost_per_contract = self.commission + (self.slippage * self.multiplier)
                costs = trades_to_make * cost_per_contract
                
                # Update positions for today
                active_position = target_sig
                active_contracts = target_contracts if target_sig != 0 else 0.0
                
            # 2. Update equity
            current_equity += pnl - costs
            
            # Prevent equity from going completely negative
            if current_equity <= 0:
                current_equity = 0.0
                active_position = 0.0
                active_contracts = 0.0
                
            # 3. Margin assessment (Maintenance Margin Check)
            maint_margin_req = 0.0
            if active_position != 0:
                # Maintenance margin = Contract Price * Multiplier * Maintenance Margin Rate
                maint_margin_req = active_contracts * price_today * self.multiplier * self.maintenance_margin_rate
                
            # Margin Call Liquidation trigger
            if maint_margin_req > 0 and current_equity < maint_margin_req:
                margin_call_flags[i] = True
                # Force liquidation: close position at today's close, pay transaction costs
                liquidation_cost = active_contracts * (self.commission + (self.slippage * self.multiplier))
                current_equity -= liquidation_cost
                current_equity = max(current_equity, 0.0)
                
                active_position = 0.0
                active_contracts = 0.0
                maint_margin_req = 0.0
                
            # Store values
            equity[i] = current_equity
            positions[i] = active_position
            contracts_held[i] = active_contracts
            daily_pnl[i] = pnl - costs
            tx_costs[i] = costs
            margin_req[i] = maint_margin_req
            
            # Leverage calculation: Notional Value / Equity
            notional_val = active_contracts * price_today * self.multiplier
            leverage[i] = notional_val / current_equity if current_equity > 0 else 0.0
            
        # Initial values override for row 0
        equity[0] = self.initial_capital
        positions[0] = 0.0
        contracts_held[0] = 0.0
        
        # Assign back to results dataframe
        results['equity'] = equity
        results['position'] = positions
        results['contracts'] = contracts_held
        results['daily_pnl'] = daily_pnl
        results['transaction_costs'] = tx_costs
        results['margin_required'] = margin_req
        results['leverage'] = leverage
        results['margin_call'] = margin_call_flags
        
        # Calculate strategy returns
        results['strategy_return'] = results['equity'].pct_change().fillna(0.0)
        
        # Calculate simple buy and hold return (scaled to initial capital)
        price_returns = results['close'].pct_change().fillna(0.0)
        results['benchmark_equity'] = self.initial_capital * (1 + price_returns).cumprod()
        results['benchmark_return'] = price_returns
        
        return results
