"""
For more information, refer to the website https://otexts.com/fpp2/stationarity.html
"""
from __future__ import annotations
from typing import Union, Literal
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import kpss, adfuller
from pmdarima.arima.utils import ndiffs

def is_stationary(arr: Union[np.ndarray, pd.Series], test: Literal['kpss', 'adf']='kpss', level=0.05, verbose=True):
    if test == 'kpss':
        kpss_stat, p_value, lags, crit_values = kpss(arr)
    elif test == 'adf':
        results = adfuller(arr)
        p_value = results[1]
    if verbose: print(f"test: {test} - p-value: {p_value:.3f} - is stationary: {p_value > level}")
    return p_value > level

def ndiffs_needed(arr: Union[np.ndarray, pd.Series], test: Literal['kpss', 'adf', 'pp']='adf', max_diff=2, with_log=False):
    """
    Number of difference needed to make the signal / the log of the signal stationary. The number is limited to ``max_d``.
    """
    arr_2 = np.log(arr) if with_log else arr; del arr
    return ndiffs(arr_2, test=test, max_d=max_diff)