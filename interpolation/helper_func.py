from __future__ import annotations
import numpy as np
import pandas as pd

def fill_with_nans(array: np.ndarray, nnan: int, seed: int = None) -> np.ndarray:
    nan_array = array.copy()
    del array
    if seed is not None:
        np.random.seed(seed)
    N = nan_array.size
    indices = np.random.choice(N, nnan, replace=False)
    nan_array[indices] = np.nan
    return nan_array

def format_series_for_output(signal_name: str, signal_index: pd.Index, method_name: str, method_interpolation: np.ndarray, polynomial_interpolation: np.ndarray) -> tuple[pd.Series, pd.Series]:
    if signal_name is None:
        name_lr = f'{method_name}_interpolation'
        name_poly = 'poly_interpolation'
    else:
        name_lr = f'{method_name}_interpolation_{signal_name}'
        name_poly = f'poly_interpolation_{signal_name}'
    met_interpolation = pd.Series(method_interpolation, index=signal_index, name=name_lr)
    pol_interpolation = pd.Series(polynomial_interpolation, index=signal_index, name=name_poly)
    return met_interpolation, pol_interpolation