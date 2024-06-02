"""
Processes data  of type `X_train`, `X_test` and enables to apply a simple LSTM pipeline using TensorFlow.

CAVEAT: y is always considered to be the same length as X. 
For example, assume that the taks is to predicted the close price given multiple features including the close price.
Then the data passed to `get_lstm_data` or `get_lstm_data_from_X_train_X_test` should be of the form

X_train                   | y_train
--------------------------|---------
close[0] open[0] high[0]  | close[0]
close[1] open[1] high[1]  | close[1]


"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import BinaryAccuracy
from typing import Literal, Union, List, Tuple
import pandas as pd

def lstm_model(sequence_length: int, n_input: int, n_output: int, hidden_sizes: List[int], regularization_strength: float=0.0, dropout: float=0.0) -> Sequential:
    model = Sequential()
    model.add(LSTM(hidden_sizes[0], return_sequences=True, input_shape=(sequence_length, n_input), kernel_regularizer=regularizers.l2(regularization_strength)))
    model.add(Dropout(dropout))
    for i in range(len(hidden_sizes)-2):
        model.add(LSTM(hidden_sizes[i+1], return_sequences=True, kernel_regularizer=regularizers.l2(regularization_strength)))
        model.add(Dropout(dropout))
    model.add(LSTM(hidden_sizes[-1], kernel_regularizer=regularizers.l2(regularization_strength))) # no return_sequences for the last LSTM layer
    model.add(Dense(n_output, kernel_regularizer=regularizers.l2(regularization_strength)))
    return model

def train_lstm(model: Sequential, X_train_lstm: np.ndarray, y_train_lstm: np.ndarray, X_val_lstm: np.ndarray, y_val_lstm: np.ndarray, loss: Literal['binary_crossentropy', 'mean_squared_error', 'binary_crossentropy_with_logits'], optimizer: Literal['adam', 'sgd', 'rmsprop'], epochs: int, batch_size: int, verbose: float=1.0, metrics: List[Literal['accuracy']]=[]) -> Tuple[Sequential]:
    li_metrics = []
    if 'accuracy' in metrics:
        li_metrics.append(BinaryAccuracy())
    if loss == 'binary_crossentropy_with_logits':
        loss = BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=li_metrics)
    history = model.fit(X_train_lstm, y_train_lstm, validation_data=(X_val_lstm, y_val_lstm), epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model, history

