from __future__ import annotations
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from typing import List, Tuple
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
sys.path.append('..')
from stationarity.difference import difference_of_seasonal_difference, seasonal_difference, difference

def create_dir(path: str):
    if not os.path.isdir(path): 
        os.makedirs(path)

def print_shape(df: pd.DataFrame, name: str):
    print(f"{name}.shape: {df.shape}", end=' | ')

def print_shapes(dfs: List[pd.DataFrame], names: List[str]):
    assert len(dfs)==len(names), "There must be as many names as dataframes"
    for df, name in zip(dfs, names):
        print_shape(df, name)

def get_identifier_lstm(model, task, train_size, sequence_length, optimizer, epochs, batch_size, hash_hidden_sizes, lr, weight_decay, dropout, scaled_target, clip, n_features):
    task_name = 'cls' if task == 'classification' else 'reg'
    identifier = f"{model}_{task_name}_ts{int(train_size*100)}_sl{sequence_length}_{optimizer}_ep{epochs}_bs{batch_size}_hs{hash_hidden_sizes}_lr{lr}_wd{weight_decay}_do{dropout}_scal{scaled_target}_cl{clip}_feat{n_features}"
    return identifier

def get_identifier_cnn(model, task, train_size, optimizer, epochs, batch_size, lr, weight_decay, dropout, scaled_target, clip, n_features):
    task_name = 'cls' if task == 'classification' else 'reg'
    identifier = f"{model}_{task_name}_ts{int(train_size*100)}_{optimizer}_ep{epochs}_bs{batch_size}_lr{lr}_wd{weight_decay}_do{dropout}_scal{scaled_target}_cl{clip}_feat{n_features}"
    return identifier

def get_weights(train_val_weights_None: bool, y_train_true: pd.DataFrame, sequence_length: int):
    """
    When train_val_weights_None is True, the weighted accuracy should be equal to the accuracy

    Note that train_val_weights_None=True is only used to test whether the code is correct but it is not a default mode
    """
    train_val_weights = None if train_val_weights_None else y_train_true.values[sequence_length:, 0]
    return train_val_weights

def print_model_info(model: nn.Module):
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)

def check_binary_validity(y_binary: np.ndarray):
    assert np.all(np.logical_or(y_binary == 1, y_binary == -1)), "Array contains values other than 1 and -1"

def check_1d(y: np.ndarray):
    assert len(y.shape) == 1, "The array must be a 1D array"

def print_class_distribution(y_train_binary, y_val_binary, y_full_train_binary):
    check_binary_validity(y_train_binary)
    check_binary_validity(y_val_binary)
    check_binary_validity(y_full_train_binary)
    print(f"Recall that LSTM excludes the first sequence_length labels")
    print(f"Full Training (including the first sequence_length samples) - This should remain the same no matter the choice of clip and scale of the target:\n{len(y_full_train_binary)} samples | Class -1: {np.mean(y_full_train_binary==-1):.3f} | Class 1: {np.mean(y_full_train_binary==1):.3f}")
    print(f"Training: {len(y_train_binary)} samples | Class -1: {np.mean(y_train_binary==-1):.3f} | Class 1: {np.mean(y_train_binary==1):.3f}")
    print(f"Validation: {len(y_val_binary)} samples | Class -1: {np.mean(y_val_binary==-1):.3f} | Class 1: {np.mean(y_val_binary==1):.3f}")

def print_class_distribution_test(y_pred_binary_test, corrected_y_pred_binary_test, correction_percentile):
    check_binary_validity(y_pred_binary_test)
    check_binary_validity(corrected_y_pred_binary_test)   
    print(f"Test: {len(y_pred_binary_test)} samples | Class -1: {np.mean(y_pred_binary_test==-1):.3f} | Class 1: {np.mean(y_pred_binary_test==1):.3f}")
    print(f"Test (corrected {correction_percentile}-percentile) | Class -1: {np.mean(corrected_y_pred_binary_test==-1):.3f} | Class 1: {np.mean(corrected_y_pred_binary_test==1):.3f}")    

def print_accuracies(y_train_binary, y_val_binary, y_pred_train, y_pred_val, scaled_threshold, c_percentiles):
    check_1d(y_train_binary)
    check_1d(y_val_binary)
    check_1d(y_pred_train)
    check_1d(y_pred_val)
    check_binary_validity(y_train_binary)
    check_binary_validity(y_val_binary)
    y_pred_binary_train = custom_sign(y_pred_train, thr=scaled_threshold)
    y_pred_binary_val = custom_sign(y_pred_val, thr=scaled_threshold)

    print(f"Training Accuracy: {accuracy_score(y_train_binary, y_pred_binary_train):.3f} | Class -1: {np.mean(y_pred_binary_train==-1):.3f} | Class 1: {np.mean(y_pred_binary_train==1):.3f}")
    print("Training Confusion Matrix: \n", confusion_matrix(y_train_binary, y_pred_binary_train))
    for c_percentile in c_percentiles:
        ypbt = threshold_percentile(y_pred_train, c_percentile)
        print(f"Training Accuracy ({c_percentile}-percentile): {accuracy_score(y_train_binary, ypbt):.3f} | Class -1: {np.mean(ypbt==-1):.3f} | Class 1: {np.mean(ypbt==1):.3f}")
        
    print(f"Validation Accuracy: {accuracy_score(y_val_binary, y_pred_binary_val):.3f} | Class -1: {np.mean(y_pred_binary_val==-1):.3f} | Class 1: {np.mean(y_pred_binary_val==1):.3f}")
    print("Validation Confusion Matrix: \n", confusion_matrix(y_val_binary, y_pred_binary_val))
    for c_percentile in c_percentiles:
        ypbv = threshold_percentile(y_pred_val, c_percentile)
        print(f"Validation Accuracy ({c_percentile}-percentile): {accuracy_score(y_val_binary, ypbv):.3f} | Class -1: {np.mean(ypbv==-1):.3f} | Class 1: {np.mean(ypbv==1):.3f}")

def print_weighted_accuracies(y_train_binary, y_val_binary, y_pred_train, y_pred_val, scaled_threshold, c_percentiles, train_absweights=None, val_absweights=None):
    check_1d(y_train_binary)
    check_1d(y_val_binary)
    check_1d(y_pred_train)
    check_1d(y_pred_val)
    if train_absweights is not None:
        check_1d(train_absweights)
    if val_absweights is not None:
        check_1d(val_absweights)
    check_binary_validity(y_train_binary)
    check_binary_validity(y_val_binary)
    y_pred_binary_train = custom_sign(y_pred_train, thr=scaled_threshold)
    y_pred_binary_val = custom_sign(y_pred_val, thr=scaled_threshold)

    print(f"Training Weighted Accuracy: {weighted_accuracy_score(y_train_binary, y_pred_binary_train, train_absweights):.3f} | Class -1: {np.mean(y_pred_binary_train==-1):.3f} | Class 1: {np.mean(y_pred_binary_train==1):.3f}")
    for c_percentile in c_percentiles:
        ypbt = threshold_percentile(y_pred_train, c_percentile)
        print(f"Training Weighted Accuracy ({c_percentile}-percentile): {weighted_accuracy_score(y_train_binary, ypbt, train_absweights):.3f} | Class -1: {np.mean(ypbt==-1):.3f} | Class 1: {np.mean(ypbt==1):.3f}")
        
    print(f"Validation Weighted Accuracy: {weighted_accuracy_score(y_val_binary, y_pred_binary_val, val_absweights):.3f} | Class -1: {np.mean(y_pred_binary_val==-1):.3f} | Class 1: {np.mean(y_pred_binary_val==1):.3f}")
    for c_percentile in c_percentiles:
        ypbv = threshold_percentile(y_pred_val, c_percentile)
        print(f"Validation Weighted Accuracy ({c_percentile}-percentile): {weighted_accuracy_score(y_val_binary, ypbv, val_absweights):.3f} | Class -1: {np.mean(ypbv==-1):.3f} | Class 1: {np.mean(ypbv==1):.3f}")

def weighted_accuracy_score(y_true, y_pred, weights=None):
    if weights is None:
        return accuracy_score(y_true, y_pred)
    assert len(y_true)==len(y_pred)==len(weights), "True labels, predictions and weights must have the same length"
    assert np.all(weights>=0), "Weights must be positive"
    correct = (y_true==y_pred).astype(float)
    return np.sum(weights * correct) / np.sum(weights)

def get_y_pred_binary_test(y_pred_test, scaled_threshold, correction_percentile):
    y_pred_binary_test = custom_sign(y_pred_test, thr=scaled_threshold)
    corrected_y_pred_binary_test = threshold_percentile(y_pred_test, correction_percentile)
    return y_pred_binary_test, corrected_y_pred_binary_test

def visualize_features(X_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, y_train_ref: pd.DataFrame, save_path: str, LaTeX=False):
    if LaTeX:
        params = {
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}',
            "font.family": "serif", "font.serif": ["Computer Modern Roman"],
            'font.size': 12,
            'image.interpolation': 'none'
         }
        plt.rcParams.update(params)
    X_train_ref_2 = X_train_ref.copy(); del X_train_ref
    X_test_ref_2 = X_test_ref.copy(); del X_test_ref
    if not isinstance(X_train_ref_2.index[0], pd.Timestamp):
        X_train_ref_2.index = pd.to_datetime(X_train_ref_2.index)
        X_test_ref_2.index = pd.to_datetime(X_test_ref_2.index)
    with PdfPages(save_path) as pdf:
        for col in X_test_ref_2.columns:
            fig, ax = plt.subplots(figsize=(20, 3))
            ax.plot(X_train_ref_2[col], label=f'train_{col}')
            ax.plot(X_test_ref_2[col], label=f'test_{col}')
            ax.grid(True)
            ax.legend()
            pdf.savefig(fig)
            plt.close()
        fig, ax = plt.subplots(figsize=(20, 3))
        col = y_train_ref.columns[0]
        ax.plot(y_train_ref[col], label=f'train_{col} (target)')
        ax.grid(True)
        ax.legend()
        pdf.savefig(fig)
        plt.close()

def add_col_diff(X_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, col: str, p: int):
    X_train_ref[f'd{p}_{col}'] = difference(X_train_ref[col], p=p, with_log=False)
    X_test_ref[f'd{p}_{col}'] = difference(X_test_ref[col], p=p, with_log=False)
    return X_train_ref, X_test_ref

def add_col_seasonal_diff(X_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, col: str, season: int):
    X_train_ref[f'sd{season}_{col}'] = seasonal_difference(X_train_ref[col], season=season, with_log=False)
    X_test_ref[f'sd{season}_{col}'] = seasonal_difference(X_test_ref[col], season=season, with_log=False)
    return X_train_ref, X_test_ref

def add_col_diff_of_seasonal_diff(X_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, col: str, p: int, season: int):
    X_train_ref[f'd{p}sd{season}_{col}'] = difference_of_seasonal_difference(X_train_ref[col], p=p, season=season, with_log=False)
    X_test_ref[f'd{p}sd{season}_{col}'] = difference_of_seasonal_difference(X_test_ref[col], p=p, season=season, with_log=False)
    return X_train_ref, X_test_ref

def drop_cols(X_train: pd.DataFrame, X_test: pd.DataFrame, cols: List[str]):
    intersection = X_train.columns.intersection(cols)
    for col in intersection:
        X_train.drop(col, axis=1, inplace=True)
        X_test.drop(col, axis=1, inplace=True)
        print(f"Successfully dropped {col}!")
    return X_train, X_test

def get_scaler(scaler_name: str):
    if scaler_name == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_name == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    return scaler

def fill_NaN(X_train: pd.DataFrame, X_test: pd.DataFrame, fill_method='interpolation'):
    for col in X_train.columns:
        n_nan = X_train[col].isna().sum()
        if n_nan > 0: print(f"train_{col} contains {n_nan}/{len(X_train)} NaNs")
        n_nan = X_test[col].isna().sum()
        if n_nan > 0: print(f"test_{col} contains {n_nan}/{len(X_test)} NaNs")
    if fill_method == 'zero':
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)
        print(f"Successfully filled the missing values by replacing them with 0!")
    elif fill_method == 'interpolation':
        X_train.interpolate(method='linear', axis=0, inplace=True)
        X_test.interpolate(method='linear', axis=0, inplace=True)
        print(f"Successfully filled the missing values by linear interpolation!")
        nan_X_train = X_train.isna().sum().sum()
        nan_X_test = X_test.isna().sum().sum()
        print(f"After linear interpolation: {nan_X_train} NaNs in X_train | {nan_X_test} NaNs in X_test")
        if nan_X_test + nan_X_train > 0:
            X_train.fillna(0, inplace=True)
            X_test.fillna(0, inplace=True)
            print(f"Successfully filled the remaining missing values by replacing them with 0!")    
    return X_train, X_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_name: str):
    scaler = get_scaler(scaler_name)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    return X_train_scaled, X_test_scaled

def scale_target(y_train_ref: pd.DataFrame, scaler_name: str) -> Tuple[pd.DataFrame, float]:
    scaler = get_scaler(scaler_name)
    print(f"Min before scaling {y_train_ref.min().min():.2f} | Max before scaling {y_train_ref.max().max():.2f}")
    scaled_y_train = scaler.fit_transform(y_train_ref)
    scaled_y_train = pd.DataFrame(scaled_y_train, index=y_train_ref.index, columns=y_train_ref.columns)
    with warnings.catch_warnings(): # the scaler expects a dataframe with named columns
        warnings.simplefilter("ignore", category=UserWarning)
        scaled_threshold = scaler.transform([[0.0]])[0][0]
    print(f"Min after scaling {np.min(scaled_y_train):.2f} | Max after scaling {np.max(scaled_y_train):.2f} | scaler(0.0) = {scaled_threshold:.5f}")
    return scaled_y_train, scaled_threshold


def clip_target(y_train_ref: pd.DataFrame, k: float):
    """
    k is a float that lies in the range (0, 100) indicating the clip percentile
    """
    print(f"Min before clipping {y_train_ref.min().min():.2f} | Max before clipping {y_train_ref.max().max():.2f}")
    top_k_percentile = np.percentile(y_train_ref.values, 100-k)
    bottom_k_percentile = np.percentile(y_train_ref.values, k)
    clipped_y_train = pd.DataFrame(np.clip(y_train_ref.values, bottom_k_percentile, top_k_percentile), index=y_train_ref.index, columns=y_train_ref.columns)
    print(f"Min after clipping {clipped_y_train.min().min():.2f} | Max after clipping {clipped_y_train.max().max():.2f}")
    return clipped_y_train

def custom_sign(array, thr: float):
    result = np.zeros_like(array)
    result[array > thr] = 1
    result[array < thr] = -1
    return result

def threshold_percentile(array, percentile_class_1):
    threshold = np.percentile(array, 100 - percentile_class_1)    
    # Set values above the threshold to 1 and the rest to -1
    return np.where(array >= threshold, 1, -1)