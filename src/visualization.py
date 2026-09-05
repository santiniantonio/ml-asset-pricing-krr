import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

# Graphics configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'figure.figsize': (10, 6)})

def calculate_metrics(returns: pd.Series) -> Tuple[float, float, float]:
    """Calculates Sharpe Ratio, Cumulative Return, and Annualized Volatility."""
    if len(returns) == 0: 
        return 0.0, 0.0, 0.0
        
    mean_ret = np.mean(returns)
    volatility = np.std(returns)
    
    sharpe = (mean_ret / volatility) * np.sqrt(12) if volatility > 0 else 0.0
    cum_return = np.prod(1 + returns) - 1
    vol_annua = volatility * np.sqrt(12)
    
    return float(sharpe), float(cum_return), float(vol_annua)

def plot_cumulative_returns(results_dict: Dict[str, pd.DataFrame], title: str = "Out-of-Sample Cumulative Returns") -> None:
    """Plots the cumulative performance of tested models (Net vs Benchmark)."""
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (model_name, df_res), color in zip(results_dict.items(), colors):
        # Plot Net ML Return
        cum_net = (np.cumprod(1 + df_res['Net_Return']) - 1) * 100
        plt.plot(df_res['Date'], cum_net, linewidth=2.5, label=f"{model_name} (Net)", color=color)
        
    # Plot Equal Weight Benchmark (only once, taking it from the first df)
    first_df = list(results_dict.values())[0]
    cum_ew = (np.cumprod(1 + first_df['EW_Return']) - 1) * 100
    plt.plot(first_df['Date'], cum_ew, linewidth=2.5, label="Equal-Weight (1/N)", color='#d62728', linestyle='-')
        
    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_portfolio_weights(weights: np.ndarray, tickers: List[str], date_str: str) -> None:
    """Generates a bar chart of the Long/Short portfolio allocations."""
    plt.figure(figsize=(10, 5))
    colors = ['#2ca02c' if w > 0 else '#d62728' for w in weights]
    bars = plt.bar(tickers, weights, color=colors, alpha=0.85)
    plt.axhline(0, color='black', linewidth=1.2)
    
    for bar, weight in zip(bars, weights):
        yval = bar.get_height()
        offset = 0.05 * max(abs(weights)) if max(abs(weights)) > 0 else 0.05
        plt.text(bar.get_x() + bar.get_width()/2, 
                 yval + (offset if weight > 0 else -offset - 0.02), 
                 f"{weight:.2f}", ha='center', va='bottom' if weight > 0 else 'top', 
                 fontsize=9, fontweight='bold')
                 
    plt.title(f'Mean-Variance Allocation (100% Gross) - {date_str}', fontweight='bold')
    plt.xlabel('Asset')
    plt.ylabel('Weight Allocation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_factor_importance_ic(df: pd.DataFrame, features: List[str], target: str = 'Target_R_t+1') -> None:
    """Calculates and visualizes the Mean Information Coefficient (IC)."""
    monthly_ic = []
    
    for _, group in df.dropna(subset=[target] + features).groupby('Date'):
        if len(group) > 2: 
            corrs = group[features].corrwith(group[target], method='spearman')
            monthly_ic.append(corrs)
            
    mean_ic = pd.DataFrame(monthly_ic).mean()
    mean_ic_sorted = mean_ic.reindex(mean_ic.abs().sort_values().index)
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ca02c' if val > 0 else '#d62728' for val in mean_ic_sorted.values]
    
    bars = plt.barh(mean_ic_sorted.index, mean_ic_sorted.values, color=colors, alpha=0.85)
    plt.axvline(0, color='black', linewidth=1.2)
    
    for bar, val in zip(bars, mean_ic_sorted.values):
        offset = 0.005 if val > 0 else -0.005
        ha = 'left' if val > 0 else 'right'
        plt.text(val + offset, bar.get_y() + bar.get_height()/2, 
                 f"{val:.3f}", va='center', ha=ha, fontsize=10, fontweight='bold')
        
    plt.xlim(mean_ic_sorted.min() - 0.015, mean_ic_sorted.max() + 0.015)   
    plt.title('Feature Importance: Mean Information Coefficient (IC)', fontweight='bold')
    plt.xlabel('Mean Spearman IC (Correlation with Out-of-Sample Return)')
    plt.tight_layout()
    plt.show()