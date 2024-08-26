#!/usr/bin/env python
import pkg_resources

from caiman.train.train_cnn_model_keras import cnn_model_keras, save_model_keras, load_model_keras
from caiman.train.train_cnn_model_pytorch import cnn_model_pytorch, train_test_split, train, validate, get_batch_accuracy, save_model_pytorch, load_model_pytorch

__version__ = pkg_resources.get_distribution('caiman').version