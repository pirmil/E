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

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        Squeeze-and-Excitation (SE) block, used to selectively emphasize informative features and suppress less useful ones
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEAC(nn.Module):
    def __init__(self, num_features, input_size=24, verbose=True, dropout=0.5):
        super(SEAC, self).__init__()
        self.num_features = num_features
        self.input_size = input_size
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3)
        self.se1 = SEBlock(32)  # Adding SE block after the first conv layer
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        input_size_fc1 = self.infer_input_size_fc1(verbose)
        self.fc1 = nn.Linear(input_size_fc1, 24)
        #self.relu2 = nn.ReLU()
        #self.fc2 = nn.Linear(64, 24)

    def infer_input_size_fc1(self, verbose):
        with torch.no_grad():
            tmp = torch.zeros((1, self.num_features, self.input_size))
            tmp = self.conv1(tmp)
            tmp = self.se1(tmp)  # Apply SE block
            tmp = self.pool1(tmp)
            tmp = self.flatten(tmp)
            input_size_fc1 = tmp.shape[1]
            if verbose:
                print(f"Infered input size for fc1: {input_size_fc1}")
            return input_size_fc1

    def forward(self, x):
        x = self.conv1(x)
        x = self.se1(x)  # Apply SE block
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        #x = self.relu2(x)
        #x = self.fc2(x)
        return x