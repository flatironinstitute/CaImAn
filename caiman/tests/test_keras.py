#!/usr/bin/env python

import numpy as np
import os
import pickle

from caiman.keras_model_arch import keras_cnn_model_from_pickle
from caiman.paths import caiman_datadir

os.environ["KERAS_BACKEND"] = "torch"
try:
    import keras_core as keras
except ImportError:
    import keras

def test_keras(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model')
        model_file = model_name + ".pkl"
        with open(model_file, 'rb') as f:
            print(f"Using model {model_file}")
            pickle_data = pickle.load(f)

        loaded_model = keras_cnn_model_from_pickle(pickle_data, keras)
    except:
        raise Exception(f'NN model could not be loaded')

    A = np.random.randn(10, 50, 50, 1)
    try:
        predictions = loaded_model.predict(A, batch_size=32)
    except:
        raise Exception('NN model could not be deployed.')
