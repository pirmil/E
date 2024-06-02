from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import warnings
import lightgbm as lgb
import xgboost as xgb

import sys
sys.path.append('..')
from data_processing.data_import import Elmy_import
from prediction.lstm_pytorch import LSTMPyTorch, train_LSTM_classification, inference_train_classification, inference_test_classification, train_LSTM_regression, inference_train_regression, inference_test_regression
from prediction.helper_func import get_lstm_data_from_X_train_X_test_no_shuffle, plot_loss_and_metrics
from training.helper_func import visualize_features, print_shape, get_prefix_and_suffix, scale_target, scale_data, drop_cols, add_col_diff, add_col_diff_of_seasonal_diff, add_col_seasonal_diff, clip_target
from training.args import opts

def train_regression(X_train_ref: pd.DataFrame, y_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, y_test_ref: pd.DataFrame, train_size: float, epochs: int, batch_size: int, lr: float, weight_decay: float, sequence_length: int, hidden_sizes: List[int], dropout: float, train_val_weights_None: bool, metrics: List[str], optimizer: str, save_path: str, submission_save_path: str, scaled_threshold: float, target_percentile: float, scaled_target: bool, clip_percentile: float, num_workers: int, random_val_set: bool, y_train_true: pd.DataFrame):
    for df, name in zip([X_train_ref, y_train_ref, X_test_ref], ['X_train_ref', 'y_train_ref', 'X_test_ref']):
        print_shape(df, name)
    loss = 'mean_squared_error'
    train_val_weights = None if train_val_weights_None else y_train_true.values[-len(y_train_ref) + sequence_length:, 0]

    X_train_val_lstm, y_train_val_lstm, X_test_lstm, y_test_lstm = get_lstm_data_from_X_train_X_test_no_shuffle(X_train_ref.values, y_train_ref.values, X_test_ref.values, None, sequence_length)
    for df, name in zip([X_train_val_lstm, y_train_val_lstm, X_test_lstm], ['X_train_val_lstm', 'y_train_val_lstm', 'X_test_lstm']):
        print_shape(df, name)
    lstm = LSTMPyTorch(n_input=X_train_ref.shape[1], n_output=1, hidden_sizes=hidden_sizes, dropout=dropout)
    print(f"Number of parameters: {sum(p.numel() for p in lstm.parameters() if p.requires_grad)}")
    print(lstm)
    del X_train_ref, X_test_ref, y_test_lstm
    lstm, history = train_LSTM_regression(lstm, X_train_val_lstm, y_train_val_lstm, train_size, loss, optimizer, epochs, batch_size, metrics, lr, weight_decay, train_val_weights, scaled_threshold, num_workers, random_val_set)
    hash_hidden_sizes = '_'.join(map(str, hidden_sizes))
    prefix, suffix = get_prefix_and_suffix(save_path, 'regression', train_size, sequence_length, optimizer, epochs, batch_size, hash_hidden_sizes, lr, weight_decay, dropout, scaled_target, clip_percentile, target_percentile, 'pdf')
    plot_loss_and_metrics(history, epochs, names=['loss'], save_path=f"{prefix}_loss_{suffix}")
    plot_loss_and_metrics(history, epochs, names=metrics, save_path=f"{prefix}_acc_{suffix}")

    with torch.no_grad():
        temp = 30
        y_hat_train = inference_train_regression(lstm, X_train_val_lstm, y_train_ref.index, sequence_length)
        y_hat_test = inference_test_regression(lstm, X_test_lstm, y_test_ref.index)
        
        weights = y_train_ref.values[-len(y_train_ref) + sequence_length:, 0]
        abs_weights = np.abs(weights)
        pos_class = (weights > scaled_threshold).astype(float)
        pred_pos_class = (y_hat_train.values > scaled_threshold).astype(float)
        true_mask = pos_class==pred_pos_class
        weighted_acc = abs_weights[true_mask]
        print(f"{'-'*temp} Training {'-'*temp}")
        print(f'Accuracy {accuracy_score(pos_class, pred_pos_class):.2f} {np.sum(true_mask) / len(true_mask):.2f} - Weighted accuracy {np.sum(weighted_acc) / np.sum(abs_weights):.2f}')
        neg_class = (weights < scaled_threshold).astype(float)
        pred_neg_class = (y_hat_train.values < scaled_threshold).astype(float)
        print(f"Percentage of positive class: {pred_pos_class.mean()*100:.2f}% - True Percentage {pos_class.mean()*100:.2f}% - True full percentage {(y_train_ref.values>scaled_threshold).mean()*100:.2f}%")
        print(f"Percentage of negative class: {pred_neg_class.mean()*100:.2f}% - True Percentage {neg_class.mean()*100:.2f}% - True full percentage {(y_train_ref.values<scaled_threshold).mean()*100:.2f}%")
        print(f"\n{'-'*temp} Test {'-'*temp}")
        pred_pos_class = (y_hat_test.values > scaled_threshold).astype(float)
        pred_neg_class = (y_hat_test.values < scaled_threshold).astype(float)
        print(f"Percentage of positive class: {pred_pos_class.mean()*100:.2f}%")
        print(f"Percentage of negative class: {pred_neg_class.mean()*100:.2f}%")

        print(f"\n{'-'*temp} Test after applying the target percentile {target_percentile}% {'-'*temp}")
        new_thr = np.percentile(y_hat_test.values, 100 - target_percentile) if target_percentile is not None else scaled_threshold
        pred_pos_class = (y_hat_test.values > new_thr).astype(float)
        pred_neg_class = (y_hat_test.values < new_thr).astype(float)
        non_def_class = (y_hat_test.values == new_thr).astype(float)
        print(f"Percentage of positive class: {pred_pos_class.mean()*100:.2f}%")
        print(f"Percentage of negative class: {pred_neg_class.mean()*100:.2f}%")
        print(f"Percentage of non-definite class: {non_def_class.mean()*100:.2f}% (will be assigned to positive)")
        
        print(f"\n{'-'*temp} Final submission {'-'*temp}")
        y_hat_test = pd.DataFrame(2 * (y_hat_test.values >= new_thr).astype(float) - 1.0, index=y_test_ref.index, columns=y_test_ref.columns).replace(0.0, -1.0)
        print(f"Final percentage of positive class: {(y_hat_test.iloc[:, 0]>0).mean()*100:.2f}% - negative class {(y_hat_test.iloc[:, 0]<0).mean()*100:.2f}%")
        prefix, suffix = get_prefix_and_suffix(submission_save_path, 'regression', train_size, sequence_length, optimizer, epochs, batch_size, hash_hidden_sizes, lr, weight_decay, dropout, scaled_target, clip_percentile, target_percentile, 'csv')
        y_hat_test.to_csv(f"{prefix}_{suffix}")

def get_y_train_classification(y_train_raw: pd.Series, scaled_threshold: float=0.0):
    y_train = (y_train_raw < scaled_threshold).astype(int)
    y_train = pd.concat((y_train, 1-y_train), axis=1)
    y_train.columns = ['neg_delta', 'pos_delta']
    return y_train

def train_classification(X_train_ref: pd.DataFrame, y_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, y_test_ref: pd.DataFrame, train_size: float, epochs: int, batch_size: int, lr: float, weight_decay: float, sequence_length: int, hidden_sizes: List[int], dropout: float, train_val_weights_None: bool, metrics: List[str], optimizer: str, save_path: str, submission_save_path: str, scaled_threshold: float, target_percentile: float, scaled_target: bool, clip_percentile: float, num_workers: int, random_val_set: bool, y_train_true: pd.DataFrame):
    for df, name in zip([X_train_ref, y_train_ref, X_test_ref], ['X_train_ref', 'y_train_ref', 'X_test_ref']):
        print_shape(df, name)
    n_classes = 2
    loss = 'binary_crossentropy_with_logits'
    y_train = get_y_train_classification(y_train_ref, scaled_threshold)
    train_val_weights = None if train_val_weights_None else y_train_true.values[-len(y_train_ref) + sequence_length:, 0]
    
    X_train_val_lstm, y_train_val_lstm, X_test_lstm, y_test_lstm = get_lstm_data_from_X_train_X_test_no_shuffle(X_train_ref.values, y_train.values, X_test_ref.values, None, sequence_length)
    for df, name in zip([X_train_val_lstm, y_train_val_lstm, X_test_lstm], ['X_train_val_lstm', 'y_train_val_lstm', 'X_test_lstm']):
        print_shape(df, name)
    lstm = LSTMPyTorch(n_input=X_train_ref.shape[1], n_output=n_classes, hidden_sizes=hidden_sizes, dropout=dropout)
    print(f"Number of parameters: {sum(p.numel() for p in lstm.parameters() if p.requires_grad)}")
    print(lstm)
    del X_train_ref, X_test_ref, y_test_lstm
    lstm, history = train_LSTM_classification(lstm, X_train_val_lstm, y_train_val_lstm, train_size, loss, optimizer, epochs, batch_size, metrics, lr, weight_decay, train_val_weights, num_workers, random_val_set)
    hash_hidden_sizes = '_'.join(map(str, hidden_sizes))
    prefix, suffix = get_prefix_and_suffix(save_path, 'classification', train_size, sequence_length, optimizer, epochs, batch_size, hash_hidden_sizes, lr, weight_decay, dropout, scaled_target, clip_percentile, target_percentile, 'pdf')
    plot_loss_and_metrics(history, epochs, names=['loss'], save_path=f"{prefix}_loss_{suffix}")
    plot_loss_and_metrics(history, epochs, names=metrics, save_path=f"{prefix}_acc_{suffix}")

    with torch.no_grad():
        temp = 30
        y_hat_train = inference_train_classification(lstm, X_train_val_lstm, y_train.index, sequence_length)
        y_hat_test = inference_test_classification(lstm, X_test_lstm, y_test_ref.index)
        print(f"{'-'*temp} Training {'-'*temp}")
        weights = y_train_ref.values[-len(y_train) + sequence_length:, 0]
        abs_weights = np.abs(weights)
        pos_class = np.argmax(y_train.values[-len(y_train) + sequence_length:], axis=1)
        pred_pos_class = y_hat_train.values
        print('Expect True:', np.all((weights>0)==pos_class))
        true_mask = pos_class==pred_pos_class
        weighted_acc = abs_weights[true_mask]
        print(f'Accuracy {accuracy_score(y_train.iloc[-len(y_train) + sequence_length:, 1], y_hat_train):.2f} {np.sum(true_mask) / len(true_mask):.2f} - Weighted accuracy {np.sum(weighted_acc) / np.sum(abs_weights):.2f} {np.sum((y_hat_train.values==np.argmax(y_train.values[-len(y_train) + sequence_length:], axis=1)) * np.abs(y_train_ref.iloc[-len(y_train) + sequence_length:, 0].values)) / np.sum(np.abs(y_train_ref.iloc[-len(y_train) + sequence_length:, 0].values)):.2f}')
        print(f"Percentage of positive class: {(y_hat_train==1).mean()*100:.2f}% - True Percentage {(y_train.iloc[-len(y_train) + sequence_length:, 1]==1).mean()*100:.2f}% - True full percentage {(y_train.iloc[:, 1]==1).mean()*100:.2f}%")
        print(f"Percentage of negative class: {(y_hat_train==0).mean()*100:.2f}% - True Percentage {(y_train.iloc[-len(y_train) + sequence_length:, 0]==1).mean()*100:.2f}% - True full percentage {(y_train.iloc[:, 1]==0).mean()*100:.2f}%")
        print(f"\n{'-'*temp} Test {'-'*temp}")
        y_hat_test.replace(0, -1, inplace=True)
        print(f"Percentage of positive class: {(y_hat_test==1).mean()*100:.2f}%")
        print(f"Percentage of negative class: {(y_hat_test==-1).mean()*100:.2f}%")

        print(f"\n{'-'*temp} Test after applying the target percentile {target_percentile}% {'-'*temp}")
        y_hat_test = inference_test_classification(lstm, X_test_lstm, y_test_ref.index, argmax=False)
        new_thr = np.percentile(y_hat_test.values, 100 - target_percentile) if target_percentile is not None else 0.5
        pred_pos_class = (y_hat_test.values > new_thr).astype(float)
        pred_neg_class = (y_hat_test.values < new_thr).astype(float)
        non_def_class = (y_hat_test.values == new_thr).astype(float)
        print(f"Percentage of positive class: {pred_pos_class.mean()*100:.2f}%")
        print(f"Percentage of negative class: {pred_neg_class.mean()*100:.2f}%")
        print(f"Percentage of non-definite class: {non_def_class.mean()*100:.2f}% (will be assigned to positive)")
        
        print(f"\n{'-'*temp} Final submission {'-'*temp}")
        y_hat_test = pd.DataFrame(2 * (y_hat_test.values >= new_thr).astype(float) - 1.0, index=y_test_ref.index, columns=y_test_ref.columns).replace(0.0, -1.0)
        print(f"Final percentage of positive class: {(y_hat_test.iloc[:, 0]>0).mean()*100:.2f}% - negative class {(y_hat_test.iloc[:, 0]<0).mean()*100:.2f}%")
        prefix, suffix = get_prefix_and_suffix(submission_save_path, 'classification', train_size, sequence_length, optimizer, epochs, batch_size, hash_hidden_sizes, lr, weight_decay, dropout, scaled_target, clip_percentile, target_percentile, 'csv')
        y_hat_test.to_csv(f"{prefix}_{suffix}")



def train_classification_xgb(X_train_ref: pd.DataFrame, y_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, y_test_ref: pd.DataFrame, train_size: float, num_round: int, lr: float, random_val_set: bool, num_leaves: int, target_percentile: float, submission_save_path: str):
    y_train_ref = (y_train_ref>=scaled_threshold).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X_train_ref.values, y_train_ref.values, train_size=train_size, shuffle=random_val_set)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'binary:logistic',  # Use 'binary:logistic' for binary classification
        'eval_metric': 'logloss',  # Evaluation metric
        'learning_rate': lr,
        'max_depth': num_leaves
    }

    num_boost_round = num_round

    xgb_classifier = xgb.train(params, dtrain, num_boost_round, evals=[(dval, 'validation')])

    # Generate predictions on the validation set
    y_pred_proba = xgb_classifier.predict(dval)
    # Convert probabilities to class labels
    new_thr = np.percentile(y_pred_proba, 100 - target_percentile) if target_percentile is not None else 0.5
    y_pred = [1 if p >= new_thr else 0 for p in y_pred_proba]

    # Compute accuracy score on validation set
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy on validation set:", accuracy)

    # Generate predictions on the test set
    dtest = xgb.DMatrix(X_test_ref.values)
    y_pred_test_proba = xgb_classifier.predict(dtest)
    y_pred_test = [1 if p >= new_thr else -1 for p in y_pred_test_proba]

    # Print some evaluation metrics on test set
    print(f'Test')
    print(f'Percentage of positive class: {np.sum(y_pred_test==1)/len(y_pred_test)*100:.2f}%')
    y_pred_test = pd.DataFrame(y_pred_test, index=y_test_ref.index, columns=y_test_ref.columns)
    y_pred_test.to_csv(f"{submission_save_path}/cls_xgb.csv")

    return xgb_classifier


def train_classification_lgb(X_train_ref: pd.DataFrame, y_train_ref: pd.DataFrame, X_test_ref: pd.DataFrame, y_test_ref: pd.DataFrame, train_size: float, num_round: int, lr: float, random_val_set: bool, num_leaves: int, target_percentile: float, submission_save_path: str):
    y_train_ref = 2.0 * (y_train_ref>=scaled_threshold).astype(int) - 1.0
    X_train, X_val, y_train, y_val = train_test_split(X_train_ref.values, y_train_ref.values, train_size=train_size, shuffle=random_val_set)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': num_leaves,
        'learning_rate': lr,
    }

    lgbClassifer = lgb.train(params, train_data, num_round, valid_sets=[test_data])
    y_hat_train = lgbClassifer.predict(X_train_ref.values)
    y_hat_test = lgbClassifer.predict(X_test_ref.values)
    new_thr = np.percentile(y_hat_test, 100 - target_percentile) if target_percentile is not None else 0.5
    y_hat_test = 2 * (y_hat_test >= new_thr) - 1
    print(np.sum(y_hat_test==-1), np.sum(y_hat_test==1))
    print(f'Train')
    print(f'Percentage of positive class: {np.sum(y_hat_train >= new_thr)/len(y_hat_train)*100:.2f}%')
    print(f'True percentage of positive: {np.mean(y_train_ref.values==1)*100:.2f}%')
    print(f'Accuracy: {accuracy_score(y_train_ref.values, 2 * (y_hat_train >= new_thr) - 1):.2f}')
    print(f'Accuracy validation: {accuracy_score(y_val, 2*(lgbClassifer.predict(X_val) >= new_thr)-1):.2f}')
    print(f'Test')
    print(f'Percentage of positive class: {np.sum(y_hat_test==1)/len(y_hat_test)*100:.2f}%')
    y_hat_test = pd.DataFrame(y_hat_test, index=y_test_ref.index, columns=y_test_ref.columns)
    y_hat_test.to_csv(f"{submission_save_path}/cls_lgb.csv")

if __name__=='__main__':
    args = opts()
    X_train_ref = Elmy_import('../data/processed/X_train.csv', with_date=False)
    y_train_ref = Elmy_import('../data/raw/y_train_raw.csv', target=True)
    y_train_true = y_train_ref.copy()
    X_test_ref = Elmy_import('../data/processed/X_test.csv', with_date=False)
    y_test_ref = Elmy_import('../data/raw/y_test_random.csv', target=True)

    X_train_ref, X_test_ref = drop_cols(X_train_ref, X_test_ref, ['Date', 'Date (UTC)'])

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.submission_save_path):
        os.makedirs(args.submission_save_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if args.scale_target:
            y_train_ref, scaled_threshold = scale_target(y_train_ref)
            print(f"Threshold after scaling: {scaled_threshold:.2f}")
        else:
            scaled_threshold = 0.0

        if args.clip_percentile is not None:
            y_train_ref = clip_target(y_train_ref, args.clip_percentile)

        for col in args.d1:
            X_train_ref, X_test_ref = add_col_diff(X_train_ref, X_test_ref, col, p=1)
        for col in args.d2:
            X_train_ref, X_test_ref = add_col_diff(X_train_ref, X_test_ref, col, p=2)
        for col in args.sd24:
            X_train_ref, X_test_ref = add_col_seasonal_diff(X_train_ref, X_test_ref, col, season=24)
        for col in args.d1_sd24:
            X_train_ref, X_test_ref = add_col_diff_of_seasonal_diff(X_train_ref, X_test_ref, col, p=1, season=24)

        X_train_ref, X_test_ref = drop_cols(X_train_ref, X_test_ref, args.drop)

        if args.scale_features:
            X_train_ref, X_test_ref = scale_data(X_train_ref, X_test_ref)

        if args.visualize_features:
            if args.model == 'LSTM':
                hash_hidden_sizes = '_'.join(map(str, args.hidden_sizes))
                prefix, suffix = get_prefix_and_suffix(args.save_path, args.task, args.train_size, args.sequence_length, args.optimizer, args.epochs, args.batch_size, hash_hidden_sizes, args.lr, args.weight_decay, args.dropout, args.scale_target, args.clip_percentile, args.target_percentile, 'pdf')
                save_path = f"{prefix}_feat_{suffix}"
            elif args.model == 'lgb':
                save_path = f"{args.save_path}/cls_lgb.pdf"
            visualize_features(X_train_ref, X_test_ref, y_train_ref, save_path)

        if args.model == 'LSTM':
            if args.task == 'classification':
                train_classification(X_train_ref, y_train_ref, X_test_ref, y_test_ref, args.train_size, args.epochs, args.batch_size, args.lr, args.weight_decay, args.sequence_length, args.hidden_sizes, args.dropout, args.train_val_weights_None, args.metrics, args.optimizer, args.save_path, args.submission_save_path, scaled_threshold, args.target_percentile, args.scale_target, args.clip_percentile, args.num_workers, args.random_val_set, y_train_true)
            else:
                train_regression(X_train_ref, y_train_ref, X_test_ref, y_test_ref, args.train_size, args.epochs, args.batch_size, args.lr, args.weight_decay, args.sequence_length, args.hidden_sizes, args.dropout, args.train_val_weights_None, args.metrics, args.optimizer, args.save_path, args.submission_save_path, scaled_threshold, args.target_percentile, args.scale_target, args.clip_percentile, args.num_workers, args.random_val_set, y_train_true)
        elif args.model == 'lgb':
            if args.task == 'classification':
                train_classification_lgb(X_train_ref, y_train_ref, X_test_ref, y_test_ref, args.train_size, args.num_round, args.lr, args.random_val_set, args.num_leaves, args.target_percentile, args.submission_save_path)
        elif args.model == 'xgb':
            if args.task == 'classification':
                train_classification_xgb(X_train_ref, y_train_ref, X_test_ref, y_test_ref, args.train_size, args.num_round, args.lr, args.random_val_set, args.num_leaves, args.target_percentile, args.submission_save_path)