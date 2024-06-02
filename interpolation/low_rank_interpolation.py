"""
The function `low_rank_interpolation` enables interpolation via the low-rank approach.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from interpolation.helper_func import format_series_for_output


def low_rank_interpolation(signal: pd.Series, window_length=None, K=None, polynomial_order=3, max_iter=100):
    """
    Perform low-rank interpolation on the given signal.

    Args:
    - signal (`pd.Series`): The input signal to be interpolated.
    - window_length (`int`): The length of the sliding window used for trajectory matrix construction.
    - K (`int`): The rank parameter for low-rank interpolation.
    - polynomial_order (`int`): The order of the polynomial interpolation used to initialize the low-rank interpolation.
    - max_iter (`int`): The maximum number of iterations for the low-rank interpolation algorithm.

    Returns:
    - lr_interpolation (`pd.Series`): The low-rank interpolated signal.
    - poly_interpolation (`pd.Series`): The signal interpolated using polynomial interpolation.
    """
    # The 1.5 coefficient might be to change
    if window_length is None:
        window_length = min(int(largest_nan_sequence(signal) * 1.5), len(signal) - 1)
    if K is None:
        K = int(0.9 * min(window_length, len(signal) - window_length + 1))
    elif K >= min(window_length, len(signal) - window_length + 1):
        raise ValueError(f"K is too large.")
    
    polynomial_interpolation = signal.interpolate(method='polynomial', order=polynomial_order).ffill().bfill()
    X = trajectory_matrix_from_signal(signal.values, window_length=window_length)
    X_hat_init = trajectory_matrix_from_signal(polynomial_interpolation.values, window_length=window_length)
    X_hat = low_rank_interpolation_trajectory_matrix(X, K=K, X_hat_init=X_hat_init, max_iter=max_iter)

    l_rank_interpolation = signal_from_trajectory_matrix_from_signal(X_hat, len(signal), window_length)

    lr_interpolation, poly_interpolation = format_series_for_output(signal.name, signal.index, 'lr', l_rank_interpolation, polynomial_interpolation)
    return lr_interpolation, poly_interpolation

def low_rank_interpolation_trajectory_matrix(X: np.ndarray, K: int, max_iter=100, X_hat_init=None) -> np.ndarray:
    """
    Args:
        X (`np.ndarray`): Trajectory matrix containing missing values
        K (`int`): Expected rank of the interpolated trajectory matrix. 
        CAVEAT: K must verify K < min(window_length, N - window_length + 1)

    Returns:
        X_hat (`np.ndarray`): Interpolated trajectory matrix
    """
    if X_hat_init is None:
        X_hat = np.nan_to_num(X, nan=np.nanmean(X))
    else:
        X_hat = X_hat_init
    W = 1.0 - np.isnan(X).astype(float)
    for _ in range(max_iter):
        U, S, Vt = np.linalg.svd(np.nan_to_num(X, nan=0, copy=True) + X_hat * (1 - W))
        # Rank-K approximation of the combination
        X_hat = U[:, :K] @ np.diag(S[:K]) @ Vt[:K, :]
    return X_hat

def find_consecutive_ones_positions(arr: np.ndarray) -> list[tuple[int, int]]:
    start_positions = np.where((arr == 1) & (np.concatenate(([0], arr[:-1])) == 0))[0]
    end_positions = np.where((arr == 1) & (np.concatenate((arr[1:], [0])) == 0))[0]
    return list(zip(start_positions, end_positions))

def nan_sequences(arr: np.ndarray) -> list[tuple[int, int]]:
    """
    Returns a list that informs about the start and the end of each sequence of NaN

    Args:
        arr (`np.ndarray`): Input array.

    Returns:
        ns (`list[tuple[int, int]]`): Each tuple contains informs about the start and the end of a sequence of NaN
    """
    arrnan = np.isnan(arr).astype(float)
    return find_consecutive_ones_positions(arrnan)


def largest_nan_sequence(signal: pd.Series) -> int:
    nan_seq_lengths = [b - a for (a, b) in nan_sequences(signal)]
    return max(nan_seq_lengths)

def trajectory_matrix_from_signal(x: np.ndarray, window_length: int, overlap_length: int=None):
    """
    Args:
        x (`np.ndarray`): Signal to process
        window_length (`int`): Window length
        K0 (`int`): Overlap length. Must be strictly less than window length

    Returns:
        X (`np.ndarray`): Trajectory matrix of x
    """
    if overlap_length is None:
        overlap_length = window_length - 1
    elif overlap_length >= window_length:
        raise ValueError(f"Window length must be strictly greater than overlap.")
    step = window_length - overlap_length
    X = np.lib.stride_tricks.sliding_window_view(x, (window_length,))[::step].T
    assert(X.shape==(window_length, 1 + np.floor((len(x) - window_length) / (window_length - overlap_length))))
    return X

def signal_from_trajectory_matrix_from_signal(X_hat, N, window_length) -> np.ndarray:
    if X_hat.shape != (window_length, N - window_length + 1):
        raise NotImplementedError(f'This function only supports trajectory matrices built with overlap_length=window_length-1.')
    idx = window_length if window_length <= X_hat.shape[1] else N - window_length + 1
    occurence = np.full(N, idx)
    occurence[:idx] = np.arange(1, idx+1)
    occurence[-idx:] = np.arange(idx, 0, -1)
    signal = np.zeros(N)
    for i in range(X_hat.shape[1]):
        signal[i:i+window_length] += X_hat[:, i]
    signal /= occurence
    return signal