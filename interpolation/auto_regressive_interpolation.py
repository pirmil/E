"""
The functions `ar_interpolation_multiple_nan_seq` and `ar_interpolation` enable interpolation via an auto-regressive model of the data. Here are the specifities:
- `ar_interpolation` only works when there is a single sequence of `NaN` values but it is computationally optimized.
- `ar_interpolation_multiple_nan_seq` works regardless of the number of `NaN` values sequences but it is not computationally optimized.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.stattools import levinson_durbin
import sys
sys.path.append('.')
from interpolation.helper_func import format_series_for_output


def ar_interpolation_multiple_nan_seq(x: pd.Series, p: int, max_iter=20, polynomial_order=1, with_tqdm=False) -> tuple[pd.Series, pd.Series]:
    """
    #### This function is not computationally optimized.
    Perform autoregressive interpolation for multiple `NaN` sequences in a time series.
    
    Args:
    - x (`pd.Series`): The time series data with `NaN` values.
    - p (`int`): The order of the autoregressive process.
    - max_iter (`int`): Maximum number of iterations for the autoregressive interpolation.
    - polynomial_order (`int`): The order of the polynomial interpolation.
        
    Returns:
    - ar_interp (`pd.Series`): The interpolated time series using autoregressive interpolation.
    - poly_interpolation (`pd.Series`): The interpolated time series using polynomial interpolation.
    """
    na_mask = np.isnan(x.values).astype(bool)
    x_c, x_c_values, x_mean = center(x); del x
    x_c_hat_poly = x_c.interpolate(method='polynomial', order=polynomial_order)
    x_c_hat = x_c_hat_poly.copy()
    x_c_hat_values = x_c_hat.values
    for _ in tqdm(range(max_iter), disable=not(with_tqdm), desc="AR Interpolation"):
        _, ar_coefs, *_ = levinson_durbin(x_c_hat_values, nlags=p, isacov=False)
        B, d = compute_B_and_d(ar_coefs, p, x_c_values)
        x_c_hat_values[na_mask] = np.linalg.inv(B.T @ B) @ B.T @ (-d)
    ar_interp, poly_interpolation = format_series_for_output(x_c.name, x_c.index, 'ar', x_c_hat_values + x_mean, x_c_hat_poly.values + x_mean)
    return ar_interp, poly_interpolation

def ar_interpolation(x: pd.Series, p: int, max_iter=100, polynomial_order=1) -> tuple[pd.Series, pd.Series]:
    """
    #### This function is computationally optimized.
    Perform autoregressive interpolation for a single `NaN` sequence in a time series.
    
    Args:
    - x (`pd.Series`): The time series data with `NaN` values.
    - p (`int`): The order of the autoregressive process.
    - max_iter (`int`): Maximum number of iterations for the autoregressive interpolation.
    - polynomial_order (`int`): The order of the polynomial interpolation.
        
    Returns:
    - ar_interp (`pd.Series`): The interpolated time series using autoregressive interpolation.
    - poly_interpolation (`pd.Series`): The interpolated time series using polynomial interpolation.
    """
    nan_number = x.isna().sum()
    x_c, x_c_values, x_mean = center(x); del x
    first_nan_pos, na_mask = first_nan_positions(x_c_values)
    if len(first_nan_pos) >= 2:
        raise NotImplementedError(f"The function only supports one NaN sequence")
    if first_nan_pos[0] - p < 0:
        raise ValueError(f"Not enough non-NaN values for the order p={p}")
    x_c_hat_poly = x_c.interpolate(method='polynomial', order=polynomial_order)
    x_c_hat = x_c_hat_poly.copy()
    x_c_hat_values = x_c_hat.values
    for _ in range(max_iter):
        _, ar_coefs, *_ = levinson_durbin(x_c_hat_values, nlags=p, isacov=False)
        B, d = vectorized_compute_B_and_d(ar_coefs, nan_number, p, x_c_values, first_nan_pos[0])
        x_c_hat_values[na_mask.astype(bool)] = np.linalg.inv(B.T @ B) @ B.T @ (-d)
    ar_interp, poly_interpolation = format_series_for_output(x_c.name, x_c.index, 'ar', x_c_hat_values + x_mean, x_c_hat_poly.values + x_mean)
    return ar_interp, poly_interpolation



def center(x: pd.Series) -> tuple[pd.Series, float]:
    x_mean = np.nanmean(x.values)
    x_c = x.copy(); del x
    x_c = x_c - x_mean
    return x_c, x_c.values, x_mean

def first_nan_positions(x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    isna_mask = np.isnan(x_val).astype(float)
    first_nan_mask = np.maximum(0, np.diff(np.insert(isna_mask, 0, 0.0)))
    first_nan_indices = np.where(first_nan_mask == 1.)[0]
    return first_nan_indices, isna_mask

def vectorized_compute_B_and_d(ar_coefs: np.ndarray, nan_number: int, p: int, x_c_values: np.ndarray, nan_seq_start: int) -> np.ndarray:
    x_c_values_2 = np.nan_to_num(x_c_values, nan=0, copy=True) ; del x_c_values
    # Additive inverse of the coefficients and add the coeff of order 0
    ar_coefs = np.insert(-ar_coefs, 0, 1.0)
    res = np.zeros(p + 1)
    for delta in range(p + 1):
        l_range = np.arange(p - delta + 1)
        res[delta] = np.sum(ar_coefs[l_range] * ar_coefs[l_range + delta])
    B = vectorized_compute_B(res, nan_number)

    d = vectorized_compute_d(res, x_c_values_2, nan_seq_start, p, nan_number)

    return B, d

def compute_B_and_d(ar_coefs: np.ndarray, p: int, x_c_values: np.ndarray) -> np.ndarray:
    nan_indices = np.where(np.isnan(x_c_values))[0]
    nan_number = len(nan_indices)
    x_c_values_2 = np.nan_to_num(x_c_values, nan=0, copy=True) ; del x_c_values
    # Additive inverse of the coefficients and add the coeff of order 0
    ar_coefs = np.insert(-ar_coefs, 0, 1.0)
    B = np.zeros((nan_number, nan_number))
    for t in range(nan_number):
        for tp in range(nan_number):
            for l in range(p - np.abs(nan_indices[t]-nan_indices[tp]) + 1):
                B[t, tp] += ar_coefs[l] * ar_coefs[l+np.abs(nan_indices[t]-nan_indices[tp])]
    d = np.zeros(nan_number)
    for t in range(nan_number):
        for k in range(-p, p+1):
            if 0 <= nan_indices[t]-k < len(x_c_values_2):
                d[t] += B[0, np.abs(k)] * x_c_values_2[nan_indices[t]-k]
    return B, d
        
        
def vectorized_compute_B(res, n):
    indices = np.arange(n)
    diff_indices = np.abs(indices[:, None] - indices[None, :])
    valid_indices = np.logical_and(diff_indices < len(res), diff_indices >= 0)
    B = np.zeros((n, n))
    B[valid_indices] = res[diff_indices[valid_indices]]
    return B

def vectorized_compute_d(res, x_c_values_2, nan_seq_start, p, nan_number):
    k_range = np.arange(-p, p + 1)
    t_range = np.arange(nan_number)
    indices = t_range[:, None] + nan_seq_start - k_range[None, :]
    d = np.sum(res[np.abs(k_range)] * x_c_values_2[indices], axis=1)
    return d


def compute_d(res, x_c_values_2, nan_seq_start, p, nan_number):
    d = np.zeros(nan_number)
    for t in range(nan_number):
        for k in range(-p, p + 1):
            d[t] += res[np.abs(k)] * x_c_values_2[t + nan_seq_start - k]
    return d