"""
Implementation of time-series detrending via polynomials. The main function is `polynomial_detrending`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union

def polynomial_detrending(x: Union[np.ndarray, pd.Series], polynomial_order: int, sampling_period: float = 1.0, lambda2: float=0.0) -> tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series], np.ndarray]:
    """
    Detrend a time series using polynomial detrending.

    Args:
    - x (`np.ndarray | pd.Series`): Input time series data.
    - polynomial_order (`int`): Order of the polynomial for detrending.
    - sampling_period (`float`): Sampling period of the time series.
    - lambda2 (`float`): Regularization parameter.

    Returns:
    - x_trend (`np.ndarray | pd.Series`): The polynomial trend of the signal.
    - x_no_trend (`np.ndarray | pd.Series`): The detrended signal.
    - alpha (`np.ndarray`): The coefficients of the polynomial trend.
    """
    if isinstance(x, np.ndarray):
        x_trend, x_no_trend, alpha = polynomial_detrending_array(x, polynomial_order, sampling_period, lambda2)
    elif isinstance(x, pd.Series):
        x_trend, x_no_trend, alpha = polynomial_detrending_array(x.values, polynomial_order, sampling_period, lambda2)
        if x.name is not None:
            x_trend = pd.Series(x_trend, name=f'trend_of_{x.name}', index=x.index)
            x_no_trend = pd.Series(x_no_trend, name=f'detrended_{x.name}', index=x.index)
        else:
            x_trend = pd.Series(x_trend, name=f'trend', index=x.index)
            x_no_trend = pd.Series(x_no_trend, name=f'detrended', index=x.index)
        return x_trend, x_no_trend, alpha
    
def polynomial_detrending_array(x: np.ndarray, polynomial_order: int, sampling_period: float = 1.0, lambda2: float=0.0) -> tuple[np.ndarray, np.ndarray]:
    Beta = vectorized_compute_Beta(len(x), polynomial_order, T_s=sampling_period)
    alpha = np.linalg.inv(Beta.T @ Beta + lambda2 * np.eye(Beta.shape[1])) @ Beta.T @ x
    x_trend = Beta @ alpha
    x_no_trend = x - x_trend
    return x_trend, x_no_trend, alpha

def vectorized_compute_Beta(N, K, T_s=1):
    Beta = (np.arange(N)[:, None] * T_s)**np.arange(K+1)
    return Beta

def compute_Beta(N, K, T_s=1):
    Beta = np.zeros((N, K+1))
    for i in range(N):
        for j in range(K+1):
            Beta[i, j] = (i * T_s) ** j
    return Beta



