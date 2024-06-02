from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from typing import List
import warnings

import sys
sys.path.append('..')
from data_processing.data_import import Elmy_import
from prediction.se_attention_cnn_pytorch import SEAC
from prediction.lstm_pytorch import LSTMPyTorch, train_LSTM_regression, inference_regression_lstm
from prediction.cnn_pytorch import CNN, train_CNN_regression, inference_regression_cnn
from prediction.helper_func import get_lstm_data_from_X_train_X_test_no_shuffle, plot_loss_and_metrics, get_cnn_data_from_X_train_X_test_no_shuffle
from training.helper_func import visualize_features, print_shapes, get_identifier_lstm, get_identifier_cnn, scale_target, scale_features, drop_cols, add_col_diff, fill_NaN
from training.helper_func import add_col_diff_of_seasonal_diff, add_col_seasonal_diff, clip_target, get_weights, print_model_info, custom_sign, print_accuracies, print_weighted_accuracies, print_class_distribution, create_dir, get_y_pred_binary_test, print_class_distribution_test
from training.args2 import opts

def train_regression_lstm(identifier: str, X_train_ref: pd.DataFrame, y_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, y_test_ref: pd.DataFrame, train_size: float, epochs: int, batch_size: int, lr: float, weight_decay: float, sequence_length: int, hidden_sizes: List[int], dropout: float, train_val_weights_None: bool, metrics: List[str], optimizer: str, scaled_threshold: float, c_percentiles: List[float], percentile_c1: float, num_workers: int, random_val_set: bool, y_train_true: pd.DataFrame, save_true: bool):
    print_shapes([X_train_ref, y_train_ref, X_test_ref], ['X_train_ref', 'y_train_ref', 'X_test_ref'])
    train_val_weights = get_weights(train_val_weights_None, y_train_true, sequence_length)
    X_train_val_lstm, y_train_val_lstm, X_test_lstm, y_test_lstm = get_lstm_data_from_X_train_X_test_no_shuffle(X_train_ref.values, y_train_ref.values, X_test_ref.values, None, sequence_length)
    lstm = LSTMPyTorch(n_input=X_train_ref.shape[1], n_output=1, hidden_sizes=hidden_sizes, dropout=dropout)
    print_shapes([X_train_val_lstm, y_train_val_lstm, X_test_lstm], ['X_train_val_lstm', 'y_train_val_lstm', 'X_test_lstm'])
    print_model_info(lstm)
    del X_train_ref, X_test_ref, y_test_lstm
    lstm, history, train_indices, val_indices = train_LSTM_regression(lstm, X_train_val_lstm, y_train_val_lstm, train_size, 'mean_squared_error', optimizer, epochs, batch_size, metrics, lr, weight_decay, train_val_weights, scaled_threshold, num_workers, random_val_set)
    
    create_dir('../figures')
    plot_loss_and_metrics(history, epochs, names=['loss'], save_path=f"../figures/loss_{identifier}.pdf")
    plot_loss_and_metrics(history, epochs, names=metrics, save_path=f"../figures/acc_{identifier}.pdf")

    with torch.no_grad():
        actual_y_train = y_train_ref.values[sequence_length:, 0]
        y_train_binary = custom_sign(actual_y_train[train_indices], thr=scaled_threshold)
        y_val_binary = custom_sign(actual_y_train[val_indices], thr=scaled_threshold)
        y_full_train_binary = custom_sign(y_train_ref.values, thr=scaled_threshold)

        y_pred_train = inference_regression_lstm(lstm, X_train_val_lstm[train_indices], y_train_ref.index.values[train_indices])
        y_pred_val = inference_regression_lstm(lstm, X_train_val_lstm[val_indices], y_train_ref.index.values[val_indices])

        print_class_distribution(y_train_binary, y_val_binary, y_full_train_binary)
        print_accuracies(y_train_binary, y_val_binary, y_pred_train, y_pred_val, scaled_threshold, c_percentiles)
        if train_val_weights_None:
            train_absweights = None
            val_absweights = None
        else:
            train_absweights = np.abs(train_val_weights[train_indices])
            val_absweights = np.abs(train_val_weights[val_indices])
        print_weighted_accuracies(y_train_binary, y_val_binary, y_pred_train, y_pred_val, scaled_threshold, c_percentiles, train_absweights, val_absweights)

        y_pred_test = inference_regression_lstm(lstm, X_test_lstm, y_test_ref.index)
        y_pred_binary_test, corrected_y_pred_binary_test = get_y_pred_binary_test(y_pred_test, scaled_threshold, percentile_c1)
        print_class_distribution_test(y_pred_binary_test, corrected_y_pred_binary_test, percentile_c1)

        create_dir("../data/submission")
        save_path = f'../data/submission/pc1_{percentile_c1}_{identifier}.csv'
        pd.DataFrame(corrected_y_pred_binary_test, columns=y_test_ref.columns, index=y_test_ref.index).to_csv(save_path)
        print(f"Successfully saved the submission at path {save_path}")
        if save_true:
            save_path = f'../data/submission/true_{identifier}.csv'
            pd.DataFrame(y_pred_test.values, columns=y_test_ref.columns, index=y_test_ref.index).to_csv(save_path)
            print(f"Successfully saved the submission at path {save_path}")

def get_CNN(model_name, params):
    if model_name == 'CNN':
        return CNN(**params)
    elif model_name == 'SEAC':
        return SEAC(**params)


def train_regression_cnn(identifier: str, X_train_ref: pd.DataFrame, y_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, y_test_ref: pd.DataFrame, train_size: float, epochs: int, batch_size: int, lr: float, weight_decay: float, dropout: float, train_val_weights_None: bool, metrics: List[str], optimizer: str, scaled_threshold: float, c_percentiles: List[float], percentile_c1: float, num_workers: int, random_val_set: bool, y_train_true: pd.DataFrame, save_true: bool):
    print_shapes([X_train_ref, y_train_ref, X_test_ref], ['X_train_ref', 'y_train_ref', 'X_test_ref'])
    X_train_val_array_3d, y_train_val_array_3d, train_val_weights_array_3d, train_val_dates, X_test_array_3d, y_test_array_3d, test_weights_array_3d, test_dates = get_cnn_data_from_X_train_X_test_no_shuffle(X_train_ref, y_train_ref, X_test_ref, None, y_train_true, None)
    cnn = get_CNN(args.model, {"num_features": X_train_val_array_3d.shape[1], "dropout": dropout})
    print_shapes([X_train_val_array_3d, y_train_val_array_3d, X_test_array_3d], ['X_train_val_array_3d', 'y_train_val_array_3d', 'X_test_array_3d'])
    print_model_info(cnn)
    del X_train_ref, y_test_array_3d, test_weights_array_3d
    cnn, history, train_indices, val_indices = train_CNN_regression(cnn, X_train_val_array_3d, y_train_val_array_3d, train_size, 'mean_squared_error', optimizer, epochs, batch_size, metrics, lr, weight_decay, train_val_weights_array_3d, scaled_threshold, num_workers, random_val_set)
    
    create_dir('../figures')
    plot_loss_and_metrics(history, epochs, names=['loss'], save_path=f"../figures/loss_{identifier}.pdf")
    plot_loss_and_metrics(history, epochs, names=metrics, save_path=f"../figures/acc_{identifier}.pdf")

    with torch.no_grad():
        train_dates = train_val_dates[train_indices]
        val_dates = train_val_dates[val_indices]
        actual_y_train = y_train_ref[y_train_ref['date'].isin(train_dates)].drop('date', axis=1)
        actual_y_val = y_train_ref[y_train_ref['date'].isin(val_dates)].drop('date', axis=1)
        y_train_binary = custom_sign(actual_y_train.values[:, 0], thr=scaled_threshold)
        y_val_binary = custom_sign(actual_y_val.values[:, 0], thr=scaled_threshold)
        y_full_train_binary = custom_sign(y_train_ref.drop('date', axis=1).values[:, 0], thr=scaled_threshold)

        y_pred_train = inference_regression_cnn(cnn, X_train_val_array_3d[train_indices])
        y_pred_val = inference_regression_cnn(cnn, X_train_val_array_3d[val_indices])

        print_class_distribution(y_train_binary, y_val_binary, y_full_train_binary)
        print_accuracies(y_train_binary, y_val_binary, y_pred_train, y_pred_val, scaled_threshold, c_percentiles)
        if train_val_weights_None:
            train_absweights = None
            val_absweights = None
        else:
            actual_y_train_true = y_train_true[y_train_true['date'].isin(train_dates)].drop('date', axis=1).values[:, 0]
            actual_y_val_true = y_train_true[y_train_true['date'].isin(val_dates)].drop('date', axis=1).values[:, 0]
            train_absweights = np.abs(actual_y_train_true)
            val_absweights = np.abs(actual_y_val_true)
        print_weighted_accuracies(y_train_binary, y_val_binary, y_pred_train, y_pred_val, scaled_threshold, c_percentiles, train_absweights, val_absweights)

        actual_X_test = X_test_ref[X_test_ref['date'].isin(test_dates)]
        y_pred_test = inference_regression_cnn(cnn, X_test_array_3d)
        y_pred_test.index = actual_X_test.index
        y_pred_binary_test, corrected_y_pred_binary_test = get_y_pred_binary_test(y_pred_test, scaled_threshold, percentile_c1)    
        print_class_distribution_test(y_pred_binary_test, corrected_y_pred_binary_test, percentile_c1)

        create_dir("../data/submission")
        save_path = f'../data/submission/pc1_{percentile_c1}_{identifier}.csv'
        pd.DataFrame(corrected_y_pred_binary_test, columns=y_test_ref.columns, index=actual_X_test.index).to_csv(save_path)
        print(f"Successfully saved the submission at path {save_path}")
        if save_true:
            save_path = f'../data/submission/true_{identifier}.csv'
            pd.DataFrame(y_pred_test.values, columns=y_test_ref.columns, index=actual_X_test.index).to_csv(save_path)
            print(f"Successfully saved the submission at path {save_path}")

if __name__=='__main__':
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = opts()
    X_train_ref = Elmy_import('../data/processed/X_train.csv', with_date=False)
    y_train_ref = Elmy_import('../data/raw/y_train_raw.csv', target=True)
    y_train_true = y_train_ref.copy()
    X_test_ref = Elmy_import('../data/processed/X_test.csv', with_date=False)
    y_test_ref = Elmy_import('../data/raw/y_test_random.csv', target=True)

    X_train_ref, X_test_ref = drop_cols(X_train_ref, X_test_ref, ['Date', 'Date (UTC)'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if args.target_scaler:
            y_train_ref, scaled_threshold = scale_target(y_train_ref, args.target_scaler)
        else:
            scaled_threshold = 0.0

        if args.clip_percentile is not None:
            y_train_ref = clip_target(y_train_ref, args.clip_percentile)

        if len(args.keep) > 0:
            if max(len(args.d1), len(args.d2), len(args.sd24), len(args.d1_sd24), len(args.drop)) > 0:
                print(f"Warning: args.keep has the priority over args.d1, args.d2, args.sd24, args.d1_sd24 and args.drop so they will be ignored")
            X_train_ref = X_train_ref[args.keep]
            X_test_ref = X_test_ref[args.keep]
        else:
            for col in args.d1:
                X_train_ref, X_test_ref = add_col_diff(X_train_ref, X_test_ref, col, p=1)
            for col in args.d2:
                X_train_ref, X_test_ref = add_col_diff(X_train_ref, X_test_ref, col, p=2)
            for col in args.sd24:
                X_train_ref, X_test_ref = add_col_seasonal_diff(X_train_ref, X_test_ref, col, season=24)
            for col in args.d1_sd24:
                X_train_ref, X_test_ref = add_col_diff_of_seasonal_diff(X_train_ref, X_test_ref, col, p=1, season=24)
            X_train_ref, X_test_ref = drop_cols(X_train_ref, X_test_ref, args.drop)

        X_train_ref, X_test_ref = fill_NaN(X_train_ref, X_test_ref, args.fill_method)

        if args.features_scaler:
            X_train_ref, X_test_ref = scale_features(X_train_ref, X_test_ref, args.features_scaler)

        if args.model == 'LSTM':
            hash_hidden_sizes = '_'.join(map(str, args.hidden_sizes))
            identifier = get_identifier_lstm(args.model, args.task, args.train_size, args.sequence_length, args.optimizer, args.epochs, args.batch_size, hash_hidden_sizes, args.lr, args.weight_decay, args.dropout, args.target_scaler, args.clip_percentile, X_train_ref.shape[1])
        elif args.model in ['CNN', 'SEAC']:
            hash_hidden_sizes = '_'.join(map(str, args.hidden_sizes))
            identifier = get_identifier_cnn(args.model, args.task, args.train_size, args.optimizer, args.epochs, args.batch_size, args.lr, args.weight_decay, args.dropout, args.target_scaler, args.clip_percentile, X_train_ref.shape[1])         

        if args.visualize_features:
            create_dir('../figures')
            visualize_features(X_train_ref, X_test_ref, y_train_ref, f"../figures/feat_{identifier}.pdf")

        if args.model == 'LSTM':
            if args.task == 'classification':
                pass
            elif args.task == 'regression':
                train_regression_lstm(identifier, X_train_ref, y_train_ref, X_test_ref, y_test_ref, args.train_size, args.epochs, args.batch_size, args.lr, args.weight_decay, args.sequence_length, args.hidden_sizes, args.dropout, args.train_val_weights_None, args.metrics, args.optimizer, scaled_threshold, args.c_percentiles, args.pc1, args.num_workers, args.random_val_set, y_train_true, args.save_true)
        elif args.model in ['CNN', 'SEAC']:
            if args.task == 'classification':
                pass
            elif args.task == 'regression':
                train_regression_cnn(identifier, X_train_ref, y_train_ref, X_test_ref, y_test_ref, args.train_size, args.epochs, args.batch_size, args.lr, args.weight_decay, args.dropout, args.train_val_weights_None, args.metrics, args.optimizer, scaled_threshold, args.c_percentiles, args.pc1, args.num_workers, args.random_val_set, y_train_true, args.save_true)
        elif args.model == 'lgb':
            if args.task == 'classification':
                pass
        elif args.model == 'xgb':
            if args.task == 'classification':
                pass