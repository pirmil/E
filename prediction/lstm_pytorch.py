from __future__ import annotations
from typing import Literal, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMPyTorch(nn.Module):
    def __init__(self, n_input: int, n_output: int, hidden_sizes: List[int], dropout: float=0.0):
        super(LSTMPyTorch, self).__init__()
        self.lstms = nn.ModuleList([nn.LSTM(input_size=n_input if i == 0 else hidden_sizes[i-1],
                                            hidden_size=hidden_size,
                                            batch_first=True) for i, hidden_size in enumerate(hidden_sizes)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_sizes[-1], n_output)

    def forward(self, x: Tensor) -> Tensor: 
        for lstm in self.lstms:
            x, _ = lstm(x)
            x = self.dropout(x)
        x = self.fc(x[:, -1, :]) # Taking the output from the last time step
        return x

def train_LSTM_classification(model: nn.Module, X_train_val_lstm: np.ndarray, y_train_val_lstm: np.ndarray, train_size: float, loss: Literal['binary_crossentropy', 'mean_squared_error', 'binary_crossentropy_with_logits'], optimizer: Literal['adam', 'sgd', 'rmsprop'], epochs: int, batch_size: int, metrics: List[Literal['accuracy']]=[], lr=0.01, weight_decay=0.0, train_val_weights: np.ndarray=None, num_workers=0, random_val_set=True) -> Tuple[nn.Module, dict]:
    if train_val_weights is not None and len(train_val_weights.shape) >= 2: raise ValueError(f"train_val_weights must be a 1D array")
    model.to(device)

    criterion = get_criterion(loss)
    optimizer = get_optimizer(model, optimizer, lr, weight_decay)

    # Convert data to PyTorch tensors
    train_loader, val_weights_tensor, X_val_tensor, y_val_tensor, train_indices, val_indices = get_loaders(X_train_val_lstm, y_train_val_lstm, train_val_weights, batch_size, train_size, num_workers, random_val_set)
    del X_train_val_lstm, y_train_val_lstm, train_val_weights

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
            X_batch, y_batch, abs_weights_batch = X_batch.to(device), y_batch.to(device), abs_weights_batch.to(device)
            optimizer.zero_grad()
            y_pred = torch.squeeze(model(X_batch), dim=1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss += loss.item() / len(train_loader)
                epoch_samples += y_batch.size(0)
                if with_accuracy or with_weighted_accuracy:
                    correct = torch.argmax(y_pred, dim=1) == torch.argmax(y_batch, dim=1)
                if with_accuracy:
                    epoch_accuracy += correct.sum().item()
                if with_weighted_accuracy:
                    epoch_weighted_accuracy += (abs_weights_batch * correct).sum().item()
                    total_abs_weight += abs_weights_batch.sum().item()
        
        with torch.no_grad():  
            if with_accuracy:
                history['accuracy'].append(epoch_accuracy / epoch_samples)
            if with_weighted_accuracy:
                history['weighted_accuracy'].append(epoch_weighted_accuracy / total_abs_weight)
            
            history['loss'].append(epoch_loss)

            # Evaluate on validation data
            model.eval()
            y_val_pred = torch.squeeze(model(X_val_tensor), dim=1)
            val_loss = criterion(y_val_pred, y_val_tensor).item()
            history['val_loss'].append(val_loss)
            if with_accuracy or with_weighted_accuracy:
                val_correct = torch.argmax(y_val_pred, dim=1) == torch.argmax(y_val_tensor, dim=1)
            if with_accuracy:
                history['val_accuracy'].append(val_correct.sum().item() / y_val_tensor.size(0))
            if with_weighted_accuracy:
                val_weighted_correct = (val_weights_tensor * val_correct).sum().item()
                val_sum_weights = val_weights_tensor.sum().item()
                history['val_weighted_accuracy'].append(val_weighted_correct/val_sum_weights)

            message_epoch = f'Epoch {epoch+1:04}/{epochs} - loss {epoch_loss:.3f} - val_loss {val_loss:.3f}'
            for metric, with_metric in zip(['accuracy', 'weighted_accuracy'], [with_accuracy, with_weighted_accuracy]):
                if with_metric:
                    message_epoch = f"{message_epoch} - {metric} {history[metric][-1]:.3f} - val_{metric} {history[f'val_{metric}'][-1]:.3f}"
            print(message_epoch)

    return model, history, train_indices, val_indices

def train_LSTM_regression(model: nn.Module, X_train_val_lstm: np.ndarray, y_train_val_lstm: np.ndarray, train_size: float, loss: Literal['binary_crossentropy', 'mean_squared_error', 'binary_crossentropy_with_logits'], optimizer: Literal['adam', 'sgd', 'rmsprop'], epochs: int, batch_size: int, metrics: List[Literal['accuracy']]=[], lr=0.01, weight_decay=0.0, train_val_weights: np.ndarray=None, threshold=0.0, num_workers=0, random_val_set=True) -> Tuple[nn.Module, dict]:
    """
    Some specifications:
    - The target is 1-dimensional. 
    - A prediction/target exceeding the threshold is considered as positive (class 1) 
    - A prediction less than the threshold is a negative prediction (class 0).
    """
    if train_val_weights is not None and len(train_val_weights.shape) >= 2: raise ValueError(f"train_val_weights must be a 1D array")
    model.to(device)

    criterion = get_criterion(loss)
    optimizer = get_optimizer(model, optimizer, lr, weight_decay)

    # Convert data to PyTorch tensors
    train_loader, val_weights_tensor, X_val_tensor, y_val_tensor, train_indices, val_indices = get_loaders(X_train_val_lstm, y_train_val_lstm, train_val_weights, batch_size, train_size, num_workers, random_val_set)
    del X_train_val_lstm, y_train_val_lstm, train_val_weights

    y_val_tensor = torch.squeeze(y_val_tensor, dim=1)
    
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
            X_batch, y_batch, abs_weights_batch = X_batch.to(device), torch.squeeze(y_batch.to(device), dim=1), abs_weights_batch.to(device)
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
                history['accuracy'].append(epoch_accuracy / epoch_samples)
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
                history['val_accuracy'].append(val_true_pred.sum().item() / y_val_tensor.size(0))
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

def get_loaders(X_train_val_lstm, y_train_val_lstm, train_val_weights, batch_size, train_size, num_workers, random_val_set):
    all_indices = np.arange(len(X_train_val_lstm))
    if random_val_set:
        np.random.shuffle(all_indices)
    train_indices = all_indices[:int(train_size * len(X_train_val_lstm))]
    val_indices = all_indices[int(train_size * len(X_train_val_lstm)):]
    X_train_tensor = torch.tensor(X_train_val_lstm[train_indices], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_val_lstm[train_indices], dtype=torch.float32)

    X_val_tensor = torch.tensor(X_train_val_lstm[val_indices], dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_train_val_lstm[val_indices], dtype=torch.float32, device=device)

    del X_train_val_lstm, y_train_val_lstm, all_indices

    if train_val_weights is None:
        train_weights_tensor = torch.tensor(np.ones(len(train_indices)), dtype=torch.float32)
        val_weights_tensor = torch.tensor(np.ones(len(val_indices)), dtype=torch.float32, device=device)
    else:
        train_weights_tensor = torch.tensor(train_val_weights[train_indices], dtype=torch.float32).abs()
        val_weights_tensor = torch.tensor(train_val_weights[val_indices], dtype=torch.float32, device=device).abs()
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_weights_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_weights_tensor, X_val_tensor, y_val_tensor, train_indices, val_indices

def inference_regression_lstm(model: nn.Module, X_lstm: np.ndarray, index: pd.Index, name=''):
    model.eval()
    values = model(torch.tensor(X_lstm, dtype=torch.float32, device=device)).cpu()[:, 0]
    return pd.Series(values, name=name, index=index)

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