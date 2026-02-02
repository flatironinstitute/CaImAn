#!/usr/bin/env python

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from caiman.paths import caiman_datadir

class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x

def test_torch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        model = PyTorchCNN()
        print('PyTorch model created successfully')
    except:
        raise Exception(f'NN model could not be loaded.')

    dataset = TensorDataset(torch.randn(10, 1, 50, 50))
    loader = DataLoader(dataset, batch_size=32)
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                predictions = model(batch[0])
        print('PyTorch model deployed successfully')
        pass
    except:
        raise Exception('NN model could not be deployed.')

if __name__ == "__main__":
    test_torch()