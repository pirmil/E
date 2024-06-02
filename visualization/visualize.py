"""
Functions to visualize `pd.Series`:
- `NaN` values with the function `visualize_nan`
- scatter plots of the correlations to target with the function `scatter_correlation_to_target`
- heatmap of the correlation between variables and optionally the target with `heatmap`
- evolution of a feature between training and test set with `concat_and_plot_feature`
- distribution of the target with `plot_target_histo`
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Literal

def visualize_nan(ser: pd.Series, method: Literal['nearest_non_nan', 'fixed_length']='nearest_non_nan', margin=24, ncols=3, fill_value=None, nonan_ser: pd.Series=None, return_slices=False) -> list[tuple[int, int]]:
    """
    Args:
        - method (`str`): 
            - if 'nearest_non_nan', then  it considers the neighbour non-`NaN` (use it when `NaN` values are frequent)
            - if 'fixed_length', then it considers the `margin` neighbours on both sides (use it when `NaN` values are scarce)

    """
    print(f"{ser.isna().sum()} / {len(ser)} ({ser.isna().sum() / len(ser)*100:.2f}%) NaN values")
    nan_seq = nan_sequences(ser.values)
    print(f"Number of NaN sequences: {len(nan_seq)}")
    nan_seq_lengths = [b - a + 1 for (a, b) in nan_seq]
    print(f"Shortest NaN sequence: {np.min(nan_seq_lengths)} - Largest NaN sequence: {np.max(nan_seq_lengths)} - Average length of a sequence {np.mean(nan_seq_lengths):.2f}")
    if method=='nearest_non_nan':
        to_explore = tuples_to_explore_nearest_non_nan(nan_seq, len(ser))
    elif method=='fixed_length':
        to_explore = tuples_to_explore_fixed_length(nan_seq, len(ser), margin)
    if fill_value is not None:
        slices = plot_nan_occurences(ser.fillna(fill_value), to_explore, ncols, nonan_ser, return_slices)
    else:
        slices = plot_nan_occurences(ser, to_explore, ncols, nonan_ser, return_slices)
    if return_slices:
        return to_explore, slices
    else:
        return to_explore

def scatter_correlation_to_target(X_train: pd.DataFrame, y_train: pd.DataFrame, ncols=3):
    nplots = round_to_next_multiple_of_k(len(X_train.columns), ncols)
    nrows = nplots // ncols
    _, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
    for i in range(nrows):
        for j in range(ncols):
            column_index = i * ncols + j
            if column_index < nplots:  # Make sure not to exceed the number of columns
                try:
                    sns.scatterplot(x=X_train.iloc[:, column_index], y=y_train.iloc[:, 0], ax=ax[i, j])
                except:
                    break
            else:
                ax[i, j].axis('off')  # Turn off the empty subplot
    plt.tight_layout()
    plt.show()

def heatmap(X: pd.DataFrame, y: pd.DataFrame=None, annot=True, figsize=None) -> None:
    if figsize is None:
        plt.figure(figsize=(len(X.columns), len(X.columns)))
    else:
        plt.figure(figsize=figsize)
    if y is None:
        sns.heatmap(X.corr(), annot=annot, cmap='coolwarm', fmt=".2f", linewidths=.5)
    else:
        sns.heatmap(pd.concat((X, y), axis=1).corr(), annot=annot, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of X')
    plt.show()

def concat_and_plot_feature(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_name: str, fill_value=np.nan, plot_series=False, idx_start=0) -> None:
    pred_spot_price_train = np.append(X_train[feature_name].fillna(fill_value).values, np.full_like(X_test[feature_name].fillna(fill_value).values, fill_value))
    pred_spot_price_train[0] = pred_spot_price_train[-1] = 0 # trick to get a full plot
    pred_spot_price_test = np.append(np.full_like(X_train[feature_name].fillna(fill_value).values, fill_value), X_test[feature_name].fillna(fill_value).values)
    pred_spot_price_test[0] = pred_spot_price_test[-1] = 0 # trick to get a full plot
    plt.figure(figsize=(20, 5))
    plt.title(f'{feature_name}')
    if plot_series:
        concat_index = list(X_train.index) + list(X_test.index)
        pred_spot_price_train = pd.Series(pred_spot_price_train, index=concat_index)
        pred_spot_price_test = pd.Series(pred_spot_price_test, index=concat_index)
        pred_spot_price_train.iloc[idx_start:].plot(label='Tranining')
        pred_spot_price_test.iloc[idx_start:].plot(label='Test')
    else:
        plt.plot(pred_spot_price_train[idx_start:], label='Tranining')
        plt.plot(pred_spot_price_test[idx_start:], label='Test')
    if pd.notna(fill_value):
        plt.axhline(fill_value, label=f'fill_value={fill_value}', color='k')
    plt.legend()
    plt.show()



def plot_target_histo(y_train_raw: pd.DataFrame):
    y_values = y_train_raw.values.flatten().copy()
    mean_y = np.mean(y_values)
    std_y = np.std(y_values)
    median_y = np.median(y_values)
    percentile_95 = np.percentile(y_values, 95)
    percentile_5 = np.percentile(y_values, 5)
    positive_percentage = np.sum(y_values > 0) / len(y_values) * 100

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    hist1 = axs[0].hist(y_values, bins=500)
    axs[0].set_title('y_train_raw histogram')
    axs[0].set_xlabel('Values')
    axs[0].set_ylabel('Frequency')

    axs[0].text(0.05, 0.95, f"Mean: {mean_y:.2f}\nStd: {std_y:.2f}\nMedian: {median_y:.2f}\n95% Percentile: {percentile_95:.2f}\n5% Percentile: {percentile_5:.2f}\nPercentage of time it is positive: {positive_percentage:.2f}%",
                transform=axs[0].transAxes, fontsize=10, verticalalignment='top')

    hist2 = axs[1].hist(y_values, bins=500, density=True)
    axs[1].set_yscale('log')
    axs[1].set_title('Same histogram with Log-Scaled Bins')
    axs[1].set_xlabel('Values')
    axs[1].set_ylabel('Density (log scale)')

    plt.tight_layout()
    plt.show()

def round_to_next_multiple_of_k(number, k):
    return number if number % k == 0 else number + k - number % k

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

def tuples_to_explore_nearest_non_nan(pairs: list[tuple[int, int]], N: int) -> list[list[tuple[int, int]]]:
    if len(pairs)==0:
        return []
    low = 0
    up = None
    to_explore = []
    for i in range(len(pairs) - 1):
        up = pairs[i+1][0]
        to_explore.append([pairs[i], (low, up)])
        low = pairs[i][1] + 1
    to_explore.append([pairs[-1], (low, N)])
    return to_explore

def tuples_to_explore_fixed_length(pairs: list[tuple[int, int]], N: int, margin = 5) -> list[list[tuple[int, int]]]:
    if len(pairs)==0:
        return []
    to_explore = []
    for i, pair in enumerate(pairs):
        to_explore.append([pairs[i], (max(0, pair[0] - margin), min(N, pair[1] + margin + 1))])
    return to_explore

def plot_nan_occurences(ser: pd.Series, to_explore: list[list[tuple[int, int]]], ncols=3, nonan_ser: pd.Series=None, return_slices=False, LaTeX=False):
    if LaTeX:
        params = {
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}',
            "font.family": "serif", "font.serif": ["Computer Modern Roman"],
            'font.size': 12,
            'image.interpolation': 'none'
         }
        plt.rcParams.update(params)
    nplots = round_to_next_multiple_of_k(len(to_explore), k=ncols)
    if nplots==1:
        data_slice = ser.iloc[to_explore[0][1][0]:to_explore[0][1][1]]
        number_nans = to_explore[0][0][1] - to_explore[0][0][0] + 1
        data_slice.plot(label=f'{number_nans} / {len(data_slice)} ({number_nans / len(data_slice)*100:.2f}%) NaN values')
        if nonan_ser is not None:
            nonan_data_slice = nonan_ser.iloc[to_explore[0][1][0]:to_explore[0][1][1]]
            nonan_data_slice.plot(label=f'No NaN', alpha=0.6)
        plt.legend()
        plt.grid(True)
        if return_slices:
            return [data_slice]
        else:
            return None
    if return_slices: data_slices = []
    nrows = nplots // ncols
    _, ax = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
    for i in range(nrows):
        for j in range(ncols):
            column_index = i * ncols + j
            if column_index < nplots:  # Make sure not to exceed the number of columns
                try:
                    data_slice = ser.iloc[to_explore[column_index][1][0]:to_explore[column_index][1][1]]
                    if return_slices: data_slices.append(data_slice)
                    number_nans = to_explore[column_index][0][1] - to_explore[column_index][0][0] + 1
                    data_slice.plot(ax=ax[i, j], label=f'{number_nans} / {len(data_slice)} ({number_nans / len(data_slice)*100:.2f}%) NaN values')
                    if nonan_ser is not None:
                        nonan_data_slice = nonan_ser.iloc[to_explore[column_index][1][0]:to_explore[column_index][1][1]]
                        nonan_data_slice.plot(ax=ax[i, j], label=f'No NaN', alpha=0.6)
                    ax[i, j].legend()
                    ax[i, j].grid(True)
                except Exception as e:
                    print(f"{e.__class__.__name__} error at {column_index}")
                    break
            else:
                ax[i, j].axis('off')  # Turn off the empty subplot
    
    plt.tight_layout()
    plt.show()
    if return_slices: 
        return data_slices
    else:
        return None

