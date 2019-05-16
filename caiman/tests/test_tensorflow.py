#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:48:55 2019

@author: epnevmatikakis
"""

from caiman.paths import caiman_datadir
import os
import numpy as np
from keras.models import model_from_json


def test_tf():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ["KERAS_BACKEND"] = "tensorflow"

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model')
        model_file = model_name + ".json"

        with open(model_file, 'r') as json_file:
            print('USING MODEL:' + model_file)
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_name + '.h5')
        loaded_model.compile('sgd', 'mse')
    except:
        raise('NN model could not be loaded')

    A = np.random.randn(10, 50, 50, 1)
    try:
        predictions = loaded_model.predict(A, batch_size=32)
        pass
    except:
        raise('NN model could not be deployed')
