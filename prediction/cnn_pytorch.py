from __future__ import annotations
from typing import Literal, Tuple, List, Union
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, num_features, input_size=24, verbose=True, dropout=0.5):
        super(CNN, self).__init__()
        self.num_features = num_features
        self.input_size = input_size
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        input_size_fc1 = self.infer_input_size_fc1(verbose)
        self.fc1 = nn.Linear(input_size_fc1, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 24)

    def infer_input_size_fc1(self, verbose):
        with torch.no_grad():
            tmp = torch.zeros((1, self.num_features, self.input_size))
            tmp = self.conv1(tmp)
            tmp = self.pool1(tmp)
            tmp = self.flatten(tmp)
            input_size_fc1 = tmp.shape[1]
            if verbose:
                print(f"Infered input size for fc1: {input_size_fc1}")
        return input_size_fc1

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

def train_CNN_regression(model: nn.Module, X_train_val_cnn: np.ndarray, y_train_val_cnn: np.ndarray, train_size: float, loss: Literal['binary_crossentropy', 'mean_squared_error', 'binary_crossentropy_with_logits'], optimizer: Literal['adam', 'sgd', 'rmsprop'], epochs: int, batch_size: int, metrics: List[Literal['accuracy']]=[], lr=0.01, weight_decay=0.0, train_val_weights: np.ndarray=None, threshold=0.0, num_workers=0, random_val_set=True) -> Tuple[nn.Module, dict]:
    """
    Some specifications:
    - The target is 24-dimensional. 
    - A prediction/target exceeding the threshold is considered as positive (class 1) 
    - A prediction less than the threshold is a negative prediction (class 0).
    """
    assert X_train_val_cnn.shape[2] == 24, "The sequences must be of length 24 because there are 24 hours in a day"
    n_hours_per_day = 24
    model.to(device)

    criterion = get_criterion(loss)
    optimizer = get_optimizer(model, optimizer, lr, weight_decay)

    # Convert data to PyTorch tensors
    train_loader, val_weights_tensor, X_val_tensor, y_val_tensor, train_indices, val_indices = get_loaders(X_train_val_cnn, y_train_val_cnn, train_val_weights, batch_size, train_size, num_workers, random_val_set)
    del X_train_val_cnn, y_train_val_cnn, train_val_weights

    y_val_tensor = torch.squeeze(y_val_tensor, dim=1)
    val_weights_tensor = torch.squeeze(val_weights_tensor, dim=1)
    
    history = {'loss': [], 'val_loss': []}

    with_accuracy = 'accuracy' in metrics
    with_weighted_accuracy = 'weighted_accuracy' in metrics
    for metric in metrics:
        history[metric] = []
        history[f'val_{metric}'] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_accuracy = epoch_weighted_accuracy = total_abs_weight = 0.0
        for X_batch, y_batch, abs_weights_batch in train_loader:
            X_batch, y_batch, abs_weights_batch = X_batch.to(device), torch.squeeze(y_batch.to(device), dim=1), torch.squeeze(abs_weights_batch.to(device), dim=1)
            optimizer.zero_grad()
            y_pred = torch.squeeze(model(X_batch), dim=1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss += loss.item()/len(train_loader)
                epoch_samples += y_batch.size(0)
                if with_accuracy or with_weighted_accuracy:
                    TP = torch.logical_and((y_batch > threshold), (y_pred > threshold))
                    TN = torch.logical_and((y_batch < threshold), (y_pred < threshold))
                    true_pred = torch.logical_or(TP, TN); del TP, TN
                if with_accuracy:
                    epoch_accuracy += true_pred.sum().item()
                if with_weighted_accuracy:
                    epoch_weighted_accuracy += (abs_weights_batch * true_pred).sum().item()
                    total_abs_weight += abs_weights_batch.sum().item()
        
        with torch.no_grad():
            if with_accuracy:
                history['accuracy'].append(epoch_accuracy / (epoch_samples * n_hours_per_day))
            if with_weighted_accuracy:
                history['weighted_accuracy'].append(epoch_weighted_accuracy / total_abs_weight)
            
            history['loss'].append(epoch_loss)

            # Evaluate on validation data
            model.eval()
            y_val_pred = torch.squeeze(model(X_val_tensor), dim=1)
            val_loss = criterion(y_val_pred, y_val_tensor).item()
            history['val_loss'].append(val_loss)
            if with_accuracy or with_weighted_accuracy:
                val_TP = torch.logical_and((y_val_tensor > threshold), (y_val_pred > threshold))
                val_TN = torch.logical_and((y_val_tensor < threshold), (y_val_pred < threshold))
                val_true_pred = torch.logical_or(val_TP, val_TN); del val_TP, val_TN
            if with_accuracy:
                history['val_accuracy'].append(val_true_pred.sum().item() / (y_val_tensor.size(0) * n_hours_per_day))
            if with_weighted_accuracy:
                val_weighted_correct = (val_weights_tensor * val_true_pred).sum().item()
                val_sum_weights = val_weights_tensor.sum().item()
                history['val_weighted_accuracy'].append(val_weighted_correct/val_sum_weights)

            message_epoch = f'loss {epoch_loss:.3f}'
            message_epoch_val = f'val_loss {val_loss:.3f}'
            for metric, with_metric in zip(['accuracy', 'weighted_accuracy'], [with_accuracy, with_weighted_accuracy]):
                if with_metric:
                    message_epoch = f"{message_epoch} | {metric} {history[metric][-1]:.3f}"
                    message_epoch_val = f"{message_epoch_val} | val_{metric} {history[f'val_{metric}'][-1]:.3f}"
            print(f"Epoch {epoch+1:04} || {message_epoch} || {message_epoch_val}")

    return model, history, train_indices, val_indices

def get_optimizer(model: nn.Module, name: Literal['adam', 'sgd', 'rmsprop'], lr: float, weight_decay: float):
    if name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def get_criterion(name: Literal['binary_crossentropy', 'mean_squared_error', 'binary_crossentropy_with_logits']):
    if name == 'binary_crossentropy':
        criterion = nn.BCELoss()
    elif name == 'mean_squared_error':
        criterion = nn.MSELoss()
    elif name == 'binary_crossentropy_with_logits':
        criterion = nn.BCEWithLogitsLoss()
    return criterion

def get_loaders(X_train_val_array_3d, y_train_val_array_3d, train_val_weights, batch_size, train_size, num_workers, random_val_set):
    all_indices = np.arange(len(X_train_val_array_3d))
    if random_val_set:
        np.random.shuffle(all_indices)
    train_indices = all_indices[:int(train_size * len(X_train_val_array_3d))]
    val_indices = all_indices[int(train_size * len(X_train_val_array_3d)):]
    X_train_tensor = torch.tensor(X_train_val_array_3d[train_indices], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_val_array_3d[train_indices], dtype=torch.float32)

    X_val_tensor = torch.tensor(X_train_val_array_3d[val_indices], dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_train_val_array_3d[val_indices], dtype=torch.float32, device=device)

    del X_train_val_array_3d, y_train_val_array_3d, all_indices

    if train_val_weights is None:
        train_weights_tensor = torch.tensor(np.ones(len(train_indices)), dtype=torch.float32)
        val_weights_tensor = torch.tensor(np.ones(len(val_indices)), dtype=torch.float32, device=device)
    else:
        train_weights_tensor = torch.tensor(train_val_weights[train_indices], dtype=torch.float32).abs()
        val_weights_tensor = torch.tensor(train_val_weights[val_indices], dtype=torch.float32, device=device).abs()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_weights_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_weights_tensor, X_val_tensor, y_val_tensor, train_indices, val_indices

def inference_regression_cnn(model: nn.Module, X_cnn: np.ndarray, name=''):
    model.eval()
    values = model(torch.tensor(X_cnn, dtype=torch.float32, device=device)).cpu().numpy().flatten()
    y_pred = pd.Series(values, name=name)
    return y_pred 

def inference_train_regression(model: nn.Module, X_train_val_lstm: np.ndarray, train_val_index: pd.Index, sequence_length: int, name='train_pred'):
    model.eval()
    values = model(torch.tensor(X_train_val_lstm, dtype=torch.float32, device=device))[:, 0]
    return pd.Series(values, name=name, index=train_val_index.values[sequence_length:])

def inference_test_regression(model: nn.Module, X_test_lstm: np.ndarray, test_index: pd.Index, name='test_pred'):
    model.eval()
    values = model(torch.tensor(X_test_lstm, dtype=torch.float32, device=device))[:, 0]
    return pd.Series(values, name=name, index=test_index)

def inference_train_classification(model: nn.Module, X_train_val_lstm: np.ndarray, train_val_index: pd.Index, sequence_length: int, name='train_pred', argmax=True):
    model.eval()
    if argmax:
        values = torch.argmax(model(torch.tensor(X_train_val_lstm, dtype=torch.float32, device=device)), dim=1)
    else:
        values = torch.softmax(model(torch.tensor(X_train_val_lstm, dtype=torch.float32, device=device)), dim=1)[:, 1] # probability of being positive only
    return pd.Series(values, name=name, index=train_val_index.values[sequence_length:])

def inference_test_classification(model: nn.Module, X_test_lstm: np.ndarray, test_index: pd.Index, name='test_pred', argmax=True):
    model.eval()
    if argmax:
        values = torch.argmax(model(torch.tensor(X_test_lstm, dtype=torch.float32, device=device)), dim=1)
    else:
        values = torch.softmax(model(torch.tensor(X_test_lstm, dtype=torch.float32, device=device)), dim=1)[:, 1] # probability of being positive only
    return pd.Series(values, name=name, index=test_index)