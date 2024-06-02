from __future__ import annotations
import pandas as pd
import numpy as np

def add_lagged_columns(X: pd.DataFrame, feature_name: str, lags: np.ndarray):
    for lag in lags:
        X[f'{feature_name}_lag_{lag}'] = X[feature_name].shift(lag)
    return X