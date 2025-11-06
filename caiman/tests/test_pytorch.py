#!/usr/bin/env python

import numpy as np
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from caiman.paths import caiman_datadir
from caiman.pytorch_model_arch import PyTorchCNN

def test_torch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model')
        model_file = model_name + ".pt"
        model = PyTorchCNN()
        model.load_state_dict(torch.load(model_file))
    except:
        raise Exception(f'NN model could not be loaded.')

    dataset = TensorDataset(torch.randn(10, 1, 50, 50))
    loader = DataLoader(dataset, batch_size=32)
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                predictions = model(batch[0])
        pass 
    except:
        raise Exception('NN model could not be deployed.')

if __name__ == "__main__":
    test_torch()