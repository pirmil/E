"""
For more information, refer to the website https://otexts.com/fpp2/stationarity.html
"""
from __future__ import annotations
from typing import Union
import pandas as pd
import numpy as np


def first_order_diff(arr: Union[np.ndarray, pd.Series]):
    if isinstance(arr, pd.Series):
        return pd.Series(first_order_diff(arr.values), index=arr.index, name=arr.name)
    integrated = np.insert(np.diff(arr.astype(float)), 0, np.nan)
    return integrated

def difference(arr: Union[np.ndarray, pd.Series], p: int, with_log=False):
    """
    In practice, it is almost never necessary to go beyond second-order differences.

    https://otexts.com/fpp2/stationarity.html
    """
    if p==0: return np.log(arr) if with_log else arr
    if p > len(arr):
        raise ValueError(f"length of the time series must be greater than the order of difference")
    integ = np.log(arr.copy()) if with_log else arr.copy(); del arr
    for _ in range(p):
        integ = first_order_diff(integ)
    return integ

def seasonal_difference(arr: Union[np.ndarray, pd.Series], season: int, with_log=False):
    """
    In practice, it is almost never necessary to go beyond second-order differences.

    https://otexts.com/fpp2/stationarity.html
    """
    if isinstance(arr, pd.Series):
        return pd.Series(seasonal_difference(arr.values, season=season, with_log=with_log), index=arr.index, name=arr.name)
    arr_2 = np.log(arr) if with_log else arr
    return np.concatenate((np.full(season, np.nan), arr_2[season:] - arr_2[:-season]))

def difference_of_seasonal_difference(arr: Union[np.ndarray, pd.Series], p: int, season: int, with_log=False):
    if isinstance(arr, pd.Series):
        return pd.Series(difference_of_seasonal_difference(arr.values, p=p, season=season, with_log=with_log), index=arr.index, name=arr.name)
    return difference(seasonal_difference(arr, season, with_log=with_log), p=p, with_log=False)