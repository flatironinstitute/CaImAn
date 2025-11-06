#!/usr/bin/env python
"""
Contains the model architecture for cnn_model.pt and cnn_model_online.pt. The files 
cnn_model.pt and cnn_model_online.pt contain the model weights. The weight files are 
used to load the weights into the model architecture. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)

        # Flattening and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=6400, out_features=512)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        # Convolutional Block 1 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Convolutional block 2 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flattening and in_features layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
        