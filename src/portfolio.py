import numpy as np
from typing import Optional, Tuple

def mean_variance_weights(R_train_matrix: np.ndarray, R_pred: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """
    Constructs the regularized Mean-Variance portfolio.
    Implements 100% Gross Exposure normalization to ensure realistic capital allocation.
    """
    R_train_matrix = np.asarray(R_train_matrix, dtype=float)
    R_pred = np.asarray(R_pred, dtype=float).flatten()
    N = R_train_matrix.shape[1] 
    
    # Covariance Matrix Estimation
    Sigma_hat = np.cov(R_train_matrix, rowvar=False)
    
    # Regularization: Add gamma * Identity Matrix for invertibility
    Sigma_reg = Sigma_hat + (gamma * np.eye(N))
    
    # Calculate raw optimal weights
    w_t_raw = np.linalg.solve(Sigma_reg, R_pred)
    
    # Gross Exposure Normalization (100% Capital Utilization)
    sum_abs_weights = np.sum(np.abs(w_t_raw))
    if sum_abs_weights > 0:
        w_t_final = w_t_raw / sum_abs_weights
    else:
        w_t_final = w_t_raw
        
    return w_t_final

def equal_weight_return(R_realized: np.ndarray) -> float:
    """Calculates the return of a 1/N Equal-Weight Benchmark."""
    R_realized = np.asarray(R_realized, dtype=float)
    weights = np.ones(len(R_realized)) / len(R_realized)
    return float(np.dot(weights, R_realized))

def evaluate_portfolio_performance(
    current_weights: np.ndarray, 
    previous_weights: Optional[np.ndarray], 
    R_realized: np.ndarray, 
    transaction_cost: float = 0.0010
) -> Tuple[float, float, float]:
    """
    Calculates Gross Return, Net Return (after transaction costs), and Turnover.
    
    Args:
        current_weights (np.ndarray): Target portfolio weights for period t.
        previous_weights (Optional[np.ndarray]): Portfolio weights from period t-1.
        R_realized (np.ndarray): Actual realized returns in period t.
        transaction_cost (float): Brokerage fee per unit of turnover (Default: 0.001 = 10 bps).
        
    Returns:
        Tuple[float, float, float]: (gross_return, net_return, turnover)
    """
    gross_return = float(np.dot(current_weights, R_realized))
    
    if previous_weights is None:
        previous_weights = np.zeros_like(current_weights)
        
    # Turnover: Sum of absolute changes in allocations
    turnover = float(np.sum(np.abs(current_weights - previous_weights)))
    
    # Net return: Gross return minus the cost of rebalancing
    net_return = gross_return - (turnover * transaction_cost)
    
    return gross_return, net_return, turnover