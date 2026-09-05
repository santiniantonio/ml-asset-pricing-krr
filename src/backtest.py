import logging
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from typing import List

from src.models import CustomKernelRidge
from src.portfolio import mean_variance_weights, evaluate_portfolio_performance, equal_weight_return

logger = logging.getLogger(__name__)

def run_walk_forward_backtest(
    df: pd.DataFrame, 
    features: List[str], 
    tickers: List[str], 
    kernel_type: str = 'linear', 
    train_window: int = 120, 
    val_window: int = 24, 
    opt_freq: int = 12,
    transaction_cost: float = 0.0010
) -> pd.DataFrame:
    """
    Executes a Walk-Forward (Rolling Window) backtest tracking Transaction Costs and Turnover.
    """
    unique_dates = sorted(df['Date'].unique())
    total_months = len(unique_dates)
    
    logger.info(f"--- INITIALIZING WALK-FORWARD SIMULATION: {kernel_type.upper()} KERNEL ---")
    logger.info(f"Window: {train_window}m | Optuna Freq: {opt_freq}m | T-Cost: {transaction_cost*10000:.0f} bps")
    
    results = []
    previous_weights = None
    best_params = {'lam': 1.0, 'gam': 1.0, 'c': 1.0, 'd': 2}
    
    for i in range(total_months - train_window):
        train_val_dates = unique_dates[i : i + train_window]
        test_date = unique_dates[i + train_window]
        
        dates_train = train_val_dates[:-val_window]
        dates_val = train_val_dates[-val_window:]
        
        df_train = df[df['Date'].isin(dates_train)]
        df_val = df[df['Date'].isin(dates_val)]
        df_test = df[df['Date'] == test_date].set_index('Ticker').reindex(tickers)
        
        R_train_matrix = df_train.pivot(index='Date', columns='Ticker', values='Return').fillna(0).reindex(columns=tickers, fill_value=0).values
        
        # 1. OPTIMIZATION PHASE (Optuna)
        if i % opt_freq == 0:
            def objective(trial: optuna.Trial) -> float:
                lam = trial.suggest_float('lam', 0.01, 100.0, log=True)
                gam = trial.suggest_float('gam', 0.01, 50.0, log=True)
                current_c, current_d = 1.0, 2
                
                if kernel_type == 'poly':
                    current_c = trial.suggest_float('c', 0.1, 10.0)
                    current_d = trial.suggest_int('d', 1, 3)
                elif kernel_type == 'rbf':
                    current_c = trial.suggest_float('c', 0.01, 10.0, log=True)
                    
                models = {}
                for tkr in tickers:
                    df_tkr_train = df_train[df_train['Ticker'] == tkr]
                    Z_train_tkr = df_tkr_train[features].values
                    R_train_target = df_tkr_train['Target_R_t+1'].values
                    mod = CustomKernelRidge(kernel_type=kernel_type, lambda_reg=lam, c=current_c, d=current_d)
                    if len(Z_train_tkr) > 0:
                        mod.fit(Z_train_tkr, R_train_target)
                    models[tkr] = mod
                    
                val_returns = []
                val_prev_weights = None
                
                for dt in dates_val:
                    df_dt = df_val[df_val['Date'] == dt].set_index('Ticker').reindex(tickers)
                    Z_val_dt = df_dt[features].fillna(0).values
                    R_real_dt = df_dt['Target_R_t+1'].fillna(0).values
                    
                    R_pred = [models[tkr].predict(Z_val_dt[j].reshape(1, -1))[0] if tkr in models and models[tkr].alpha is not None else 0 for j, tkr in enumerate(tickers)]
                    w_t = mean_variance_weights(R_train_matrix, np.array(R_pred), gamma=gam)
                    
                    # PRO TRICK: Calculate Net Return during validation to penalize high turnover
                    _, net_ret, _ = evaluate_portfolio_performance(w_t, val_prev_weights, R_real_dt, transaction_cost)
                    val_returns.append(net_ret)
                    val_prev_weights = w_t
                    
                returns_arr = np.array(val_returns)
                vol = np.std(returns_arr)
                return float((np.mean(returns_arr) / vol) * np.sqrt(12)) if vol > 0 else 0.0
            
            fixed_sampler = TPESampler(seed=42 + i) 
            study = optuna.create_study(direction='maximize', sampler=fixed_sampler)
            study.optimize(objective, n_trials=20) 
            
            best_params.update(study.best_params)
            if 'c' not in study.best_params: best_params['c'] = 1.0
            if 'd' not in study.best_params: best_params['d'] = 2
            logger.info(f"[{test_date.strftime('%Y-%m')}] Optuna Updated: lambda={best_params['lam']:.2f}, gamma={best_params['gam']:.2f}")

        # 2. FINAL TRAINING & OUT-OF-SAMPLE PREDICTION
        final_models = {}
        df_train_full = df[df['Date'].isin(train_val_dates)] 
        
        for tkr in tickers:
            df_tkr_train = df_train_full[df_train_full['Ticker'] == tkr]
            Z_train_full = df_tkr_train[features].values
            R_train_target = df_tkr_train['Target_R_t+1'].values
            
            mod = CustomKernelRidge(kernel_type=kernel_type, lambda_reg=best_params['lam'], c=best_params['c'], d=best_params['d'])
            if len(Z_train_full) > 0: 
                mod.fit(Z_train_full, R_train_target)
            final_models[tkr] = mod
            
        Z_test = df_test[features].fillna(0).values
        R_real = df_test['Target_R_t+1'].fillna(0).values
        
        R_pred = [final_models[tkr].predict(Z_test[j].reshape(1, -1))[0] if tkr in final_models and final_models[tkr].alpha is not None else 0 for j, tkr in enumerate(tickers)]
        
        R_train_full_matrix = df_train_full.pivot(index='Date', columns='Ticker', values='Return').fillna(0).reindex(columns=tickers, fill_value=0).values
        
        current_weights = mean_variance_weights(R_train_full_matrix, np.array(R_pred), gamma=best_params['gam'])
        
        # 3. RECORDING METRICS (Gross, Net, Turnover)
        gross_ml, net_ml, turnover = evaluate_portfolio_performance(current_weights, previous_weights, R_real, transaction_cost)
        ret_ew = equal_weight_return(R_real)
        
        results.append({
            'Date': test_date,
            'Gross_Return': gross_ml,
            'Net_Return': net_ml,
            'EW_Return': ret_ew,
            'Turnover': turnover,
            'Weights': current_weights
        })
        
        previous_weights = current_weights
        
    return pd.DataFrame(results)