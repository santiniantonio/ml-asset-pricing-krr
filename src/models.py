import numpy as np
from typing import Optional

class CustomKernelRidge:
    """
    Custom implementation of Kernel Ridge Regression tailored for panel data.
    Implements Linear, Polynomial, and Gaussian (RBF) kernels.
    """
    
    def __init__(
        self, 
        kernel_type: str = 'linear', 
        lambda_reg: float = 1.0, 
        c: float = 1.0, 
        d: int = 2
    ):
        """
        Args:
            kernel_type (str): 'linear', 'poly', or 'rbf'.
            lambda_reg (float): Ridge penalty parameter (L2 regularization).
            c (float): Kernel hyperparameter (constant for poly, bandwidth for rbf).
            d (int): Polynomial degree (used only if kernel_type='poly').
        """
        if kernel_type not in ['linear', 'poly', 'rbf']:
            raise ValueError(f"Unsupported kernel type: '{kernel_type}'")
            
        self.kernel_type = kernel_type
        self.lambda_reg = lambda_reg
        self.c = c
        self.d = d
        
        self.alpha: Optional[np.ndarray] = None
        self.Z_train: Optional[np.ndarray] = None
        
    def _compute_kernel(self, Z1: np.ndarray, Z2: np.ndarray) -> np.ndarray:
        """Calculates the Kernel matrix between two datasets."""
        if self.kernel_type == 'linear':
            return np.dot(Z1, Z2.T)
            
        elif self.kernel_type == 'poly':
            return (self.c + np.dot(Z1, Z2.T)) ** self.d
            
        elif self.kernel_type == 'rbf':
            Z1_sq = np.sum(Z1**2, axis=1).reshape(-1, 1)
            Z2_sq = np.sum(Z2**2, axis=1)
            dist_sq = Z1_sq + Z2_sq - 2 * np.dot(Z1, Z2.T)
            dist_sq = np.maximum(dist_sq, 0.0) # Numerical stability
            return np.exp(-self.c * dist_sq)
            
        return np.array([]) # Fallback strictly for type hinting
            
    def fit(self, Z_train: np.ndarray, R_train: np.ndarray) -> None:
        """
        Estimates the dual alpha coefficient vector on the training set.
        
        Args:
            Z_train (np.ndarray): Training features matrix (T x K).
            R_train (np.ndarray): Target returns vector (T x 1).
        """
        self.Z_train = np.asarray(Z_train, dtype=float)
        R_train = np.asarray(R_train, dtype=float)
        T = self.Z_train.shape[0]
        
        K = self._compute_kernel(self.Z_train, self.Z_train)
        inversion_matrix = K + (self.lambda_reg * np.eye(T))
        
        # Solve the linear system to find alpha
        self.alpha = np.linalg.solve(inversion_matrix, R_train)
        
    def predict(self, Z_test: np.ndarray) -> np.ndarray:
        """
        Predicts returns for a new set of characteristics.
        
        Args:
            Z_test (np.ndarray): Test features matrix.
            
        Returns:
            np.ndarray: Predicted returns vector.
        """
        if self.alpha is None or self.Z_train is None:
            raise RuntimeError("Model must be fitted before predicting.")
            
        Z_test = np.asarray(Z_test, dtype=float)
        K_test_train = self._compute_kernel(Z_test, self.Z_train)
        return np.dot(K_test_train, self.alpha)