from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Union, List, Tuple
from sklearn.model_selection import train_test_split

def test_data_lstm(X: np.ndarray, X_before: np.ndarray, sequence_length: int, y: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
    Xc = np.vstack((X_before[-sequence_length:], X)); del X_before, X
    X_lstm = []
    for i in range(len(Xc)-sequence_length):
        X_lstm.append(Xc[i:i+sequence_length])
    X_lstm = np.array(X_lstm)
    return X_lstm, y

def data_lstm(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    y_lstm = y[sequence_length:]; del y
    X_lstm = []
    for i in range(len(X)-sequence_length):
        X_lstm.append(X[i:i+sequence_length])
    X_lstm = np.array(X_lstm)
    return X_lstm, y_lstm


def get_lstm_data_from_X_train_X_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: Union[np.ndarray, None], sequence_length: int, train_size: float, shuffle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_test_lstm, y_test_lstm = test_data_lstm(X_test, X_train, sequence_length, y_test)
    X_train_val_lstm, y_train_val_lstm = data_lstm(X_train, y_train, sequence_length)
    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_train_val_lstm, y_train_val_lstm, train_size=train_size, shuffle=shuffle)
    return X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm, X_train_val_lstm, y_train_val_lstm

def get_lstm_data_from_X_train_X_test_no_shuffle(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: Union[np.ndarray, None], sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_test_lstm, y_test_lstm = test_data_lstm(X_test, X_train, sequence_length, y_test)
    X_train_val_lstm, y_train_val_lstm = data_lstm(X_train, y_train, sequence_length)
    return X_train_val_lstm, y_train_val_lstm, X_test_lstm, y_test_lstm

def get_lstm_data(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: Union[np.ndarray, None], X_test: np.ndarray, y_test: Union[np.ndarray, None], sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_lstm, y_train_lstm = data_lstm(X_train, y_train, sequence_length)
    X_val_lstm, y_val_lstm = test_data_lstm(X_val, X_train, sequence_length, y_val)
    X_test_lstm, y_test_lstm = test_data_lstm(X_test, np.vstack((X_train, X_val)), sequence_length, y_test)
    return X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm

def weighted_accuracy(y_train: pd.Series, y_hat_train: pd.Series) -> float:
    """
    Only stricly positive and strictly negative values can increase the weighted accuracy.

    If the predicted value is 0, then the weighted accuracy score does not increase.
    """
    correct_mask = ((y_train * y_hat_train) > 0).astype(bool)
    return y_hat_train[correct_mask].abs().sum() / y_hat_train.abs().sum()

def plot_loss_and_metrics(history, epochs, figsize=(20, 5), names: List[Literal['loss', 'binary_accuracy', 'weighted_accuracy']]=['loss'], save_path=None):
    plt.figure(figsize=figsize)
    for name in names:
        if isinstance(history, dict):
            train_loss = history[name]
            val_loss = history[f'val_{name}']
        else:
            train_loss = history.history[name]
            val_loss = history.history[f'val_{name}']
        plt.plot(range(1, epochs+1), train_loss, label=f'Training {name}')
        plt.plot(range(1, epochs+1), val_loss, label=f'Validation {name}')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, format='pdf')
        print(f"Figure saved as {save_path}")
    else:
        plt.show()

def check_same_index(X: pd.DataFrame, y: pd.DataFrame):
    assert X.index.equals(y.index), "X and X_testy must have the same index"

def get_cnn_data_from_df(df: pd.DataFrame):
    df['date'] = pd.to_datetime(df.index, utc=True).date
    grouped_df = df.groupby('date')
    day_dict = {}
    final_dates = []
    for date, group in grouped_df:
        if group.shape[0] == 24:
            final_dates.append(date)
            day_dict[date] = group.drop('date', axis=1).T
    array_3d = np.stack([dataframe.to_numpy() for dataframe in day_dict.values()])
    return array_3d, np.array(final_dates)

def check_same_dates(dates_X, dates_y):
    assert np.all(dates_X==dates_y), "Final dates must be the same for X and y"

def get_cnn_data_from_X_train_X_test_no_shuffle(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: Union[pd.DataFrame, None], train_weights: Union[pd.DataFrame, None], test_weights: Union[pd.DataFrame, None]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert input DataFrames into 3D numpy arrays for use in a Convolutional Neural Network (CNN).

    This function takes in train and test DataFrames and converts them into 3D numpy arrays without
    shuffling the data. The function assumes that the input DataFrames have a date-like index and
    drops any days that do not have data for all 24 hours.

    The resulting 3D numpy arrays have shapes:
        - X_array_3d.shape = n_days x n_features x 24
        - y_array_3d.shape = n_days x 1 x 24

    Parameters
    ----------
    X_train : pd.DataFrame
        Training input features DataFrame with a date-like index.
    y_train : pd.DataFrame
        Training target DataFrame with a date-like index.
    X_test : pd.DataFrame
        Test input features DataFrame with a date-like index.
    y_test : Union[pd.DataFrame, None]
        Test target DataFrame with a date-like index, or None if not provided.

    Returns
    -------
    X_train_array_3d : np.ndarray
        3D numpy array of training input features.
    y_train_array_3d : np.ndarray
        3D numpy array of training target data.
    train_dates : np.ndarray
        1D numpy array of dates from the training data.
    X_test_array_3d : np.ndarray
        3D numpy array of test input features.
    y_test_array_3d : Union[np.ndarray, None]
        3D numpy array of test target data, or None if not provided.
    test_weights_array_3d: Union[np.ndarray, None]
        3D numpy array of test weights, or None if not provided.
    test_dates : np.ndarray
        1D numpy array of dates from the test data.

    Raises
    ------
    ValueError
        If X_train and X_test do not have the same date-like index, or if the shapes of the resulting
    numpy arrays do not match the expected format.
    """
    check_same_index(X_train, y_train)
    X_train_val_array_3d, train_val_dates = get_cnn_data_from_df(X_train)
    y_train_val_array_3d, train_dates_y = get_cnn_data_from_df(y_train)
    y_train_val_array_3d, train_dates_y = get_cnn_data_from_df(y_train)
    check_same_dates(train_val_dates, train_dates_y)
    X_test_array_3d, test_dates = get_cnn_data_from_df(X_test)
    if train_weights is not None: 
        check_same_index(X_train, train_weights)
        train_val_weights_array_3d, train_dates_weights = get_cnn_data_from_df(train_weights)
        check_same_dates(train_val_dates, train_dates_weights)
    else:
        train_val_weights_array_3d = None
    if y_test is not None:
        check_same_index(X_test, y_test)
        y_test_array_3d, test_dates_y = get_cnn_data_from_df(y_test)
        check_same_dates(test_dates, test_dates_y)
    else:
        y_test_array_3d = None
    if test_weights is not None: 
        check_same_index(X_test, test_weights)
        test_weights_array_3d, test_dates_weights = get_cnn_data_from_df(test_weights)
        check_same_dates(test_dates, test_dates_weights)
    else:
        test_weights_array_3d = None
    return X_train_val_array_3d, y_train_val_array_3d, train_val_weights_array_3d, train_val_dates, X_test_array_3d, y_test_array_3d, test_weights_array_3d, test_dates