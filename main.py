import logging
import warnings
import pandas as pd
import optuna

# Import modules
from src.data_loader import get_and_preprocess_data
from src.backtest import run_walk_forward_backtest
from src.visualization import (
    calculate_metrics, 
    plot_cumulative_returns, 
    plot_portfolio_weights, 
    plot_factor_importance_ic
)

# ---------------------------------------------------------
# SYSTEM CONFIGURATION & LOGGING
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Quantitative_Pipeline")

def main():
    logger.info("==================================================")
    logger.info("STARTING QUANTITATIVE ASSET PRICING PIPELINE")
    logger.info("==================================================")
    
    # 1. Data Processing
    tickers = ['MSFT', 'JPM', 'JNJ', 'PG', 'MCD', 'XOM', 'CAT', 'AAPL', 'T', 'WMT']
    df_panel, list_features, list_tickers = get_and_preprocess_data(tickers)
    
    # Transaction Costs: 10 basis points (0.10%)
    T_COST = 0.0010 
    
    logger.info("Executing Walk-Forward Backtests...")
    results_dict = {}

    # 2. Linear Baseline
    res_lin = run_walk_forward_backtest(
        df_panel, list_features, list_tickers, 
        kernel_type='linear', opt_freq=12, transaction_cost=T_COST
    )
    results_dict['Linear Model'] = res_lin

    # 3. Polynomial Kernel
    res_poly = run_walk_forward_backtest(
        df_panel, list_features, list_tickers, 
        kernel_type='poly', opt_freq=12, transaction_cost=T_COST
    )
    results_dict['Polynomial Model'] = res_poly

    # 4. Gaussian RBF Kernel
    res_rbf = run_walk_forward_backtest(
        df_panel, list_features, list_tickers, 
        kernel_type='rbf', opt_freq=12, transaction_cost=T_COST
    )
    results_dict['Gaussian RBF Model'] = res_rbf

    # 5. Visualizations
    logger.info("Generating analytical plots...")
    plot_cumulative_returns(results_dict, title="Out-of-Sample Returns (Net of Transaction Costs)")
    
    # Extract final weights for allocation plotting
    last_date_str = res_rbf['Date'].iloc[-1].strftime('%Y-%m')
    last_weights = res_rbf['Weights'].iloc[-1]
    plot_portfolio_weights(last_weights, list_tickers, date_str=last_date_str)
    
    plot_factor_importance_ic(df_panel, list_features)

    # 6. Final Performance Report (Gross vs Net)
    print("\n" + "="*100)
    print(f"OUT-OF-SAMPLE PERFORMANCE SUMMARY (Net of {T_COST*10000:.0f} bps Transaction Costs)")
    print("="*100)
    print(f"{'MODEL':<22} | {'GROSS RET':<12} | {'NET RET':<12} | {'ANNNUAL SHARPE':<12} | {'ANNUAL VOL':<12} | {'MEAN TURNOVER'}")
    print("-" * 100)
    for model_name, df_res in results_dict.items():
        # Metriche lorde
        _, gross_cum, _ = calculate_metrics(df_res['Gross_Return'])
        
        # Metriche nette
        net_sharpe, net_cum, vol_annua = calculate_metrics(df_res['Net_Return'])
        
        # Turnover medio mensile
        mean_turnover = df_res['Turnover'].mean() * 100
        
        print(f"{model_name:<22} | {gross_cum*100:>9.2f}% | {net_cum*100:>9.2f}% | {net_sharpe:>10.4f} | {vol_annua*100:>9.2f}% | {mean_turnover:>9.2f}% / mo")

    # Benchmark 1/N
    ew_sharpe, ew_cum, ew_vol = calculate_metrics(res_lin['EW_Return'])
    print("-" * 100)
    print(f"{'Equal-Weight (1/N)':<22} | {'-':>10} | {ew_cum*100:>9.2f}% | {ew_sharpe:>10.4f} | {ew_vol*100:>9.2f}% | {'0.00% / mo':>12}")
    print("="*100)
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()