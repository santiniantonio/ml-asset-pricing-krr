# Quantitative Asset Pricing: Kernel Ridge Regression & Mean-Variance Optimization

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian_Optimization-blue)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

This repository contains a quantitative research pipeline for cross-sectional stock return prediction and portfolio optimization using machine learning. It leverages **Kernel Ridge Regression (KRR)** models dynamically coupled with a **Mean-Variance Portfolio Optimizer**, rigorously evaluated out-of-sample over a 21-year period (2003–2024).

The core objective of this project is to bridge the gap between theoretical Machine Learning alpha and operational reality by explicitly modeling market frictions, portfolio turnover, and cross-sectional data standardisation.

---

## 📈 Key Quantitative Features

*   **Strict Look-Ahead Bias Prevention:** All 8 predictive features (Momentum, Reversal, Volatility, Amihud Illiquidity, etc.) are cross-sectionally standardized at each time step ($t$).
*   **Walk-Forward Methodology:** Static train/test splits are abandoned in favor of a 120-month rolling window backtest, ensuring chronological time-series integrity.
*   **Friction-Aware Bayesian Optimization:** Hyperparameters ($\lambda, \gamma, c, d$) are dynamically tuned using the Tree-structured Parzen Estimator (TPE) via `Optuna`. Crucially, the optimization objective is the **Annualized Net Sharpe Ratio**, forcing the algorithm to organically penalize high-turnover parameter configurations.
*   **Transaction Costs Accounting:** All out-of-sample portfolio allocations are evaluated strictly net of a realistic 10 basis points (0.10%) transaction cost per unit of turnover.
*   **100% Gross Exposure Normalization:** Ensures mathematical fairness when comparing the ML models' active Long/Short allocations against the fully invested $1/N$ benchmark.

---

## 📊 Empirical Results (2003 - 2024)

The Walk-Forward backtest highlights a severe *bias-variance tradeoff* heavily compounded by market frictions. Over relatively short 10-year rolling windows, highly flexible non-linear models (Polynomial) suffer from noise-driven overfitting, resulting in massive portfolio turnover (69.40% / mo) that destroys theoretical alpha. The structurally rigid **Linear Kernel** generalizes significantly better, proving to be the most robust predictive model.

*Performance is evaluated strictly net of 10 bps transaction costs on a universe of 10 U.S. mega-cap equities.*

| Model | Gross Return | Net Return | Ann. Net Sharpe | Annual Volatility | Mean Turnover |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Kernel** | 1628.67% | **1396.80%** | **1.0121** | 13.10% | 55.27% / mo |
| **Polynomial Kernel** | 723.60% | 586.72% | 0.6987 | 14.04% | 69.40% / mo |
| **Gaussian RBF** | 1055.07% | 903.38% | 0.8661 | 13.21% | 53.93% / mo |
| *Equal-Weight (1/N)* | *-* | *2679.07%* | *1.2055* | *13.41%* | *0.00% / mo* |

> **Note on Benchmark Dominance:** The immense outperformance of the $1/N$ passive benchmark empirically demonstrates the profound impact of *survivorship bias*. Testing on a static universe of ex-post selected mega-winners severely handicaps active Long/Short ML models, as shorting assets structurally destined for exponential growth incurs heavy directional and friction losses.

---

## 📁 Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # yfinance download, feature engineering, and cross-sectional scaling
│   ├── models.py         # Custom implementation of Panel-Data Kernel Ridge Regression
│   ├── portfolio.py      # Mean-Variance optimization, Gross Exposure logic, and Turnover tracking
│   ├── backtest.py       # Walk-Forward rolling window and Optuna TPE integration
│   └── visualization.py  # Matplotlib/Seaborn graphics and performance metric calculations
├── main.py               # Main execution pipeline
├── Report.pdf            # Comprehensive LaTeX academic paper detailing the mathematical framework
└── README.md 
``` 

## 🚀 How to Run

1. Clone the repository and install the required dependencies (`pandas`, `numpy`, `yfinance`, `optuna`, `matplotlib`, `seaborn`).
2. Run the main pipeline from the terminal:
```bash
   python main.py
```
3. The script will dynamically download data from Yahoo Finance, run the 21-year rolling optimizations, display the analytical charts (Cumulative Returns, Factor IC, Portfolio Weights), and print the final net performance summary.

---

**Author:** Antonio Gabriele Santini | *Quantitative Finance UniTo*
