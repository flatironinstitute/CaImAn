#!/usr/bin/env python

import numpy as np
import os
import keras 
import json 
os.environ["KERAS_BACKEND"] = "torch"
from keras.saving import deserialize_keras_object
from keras.models import load_model 
use_keras = True

from caiman.paths import caiman_datadir

def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration string and returns a model instance.

    Args:
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A Keras model instance (uncompiled).
    """
    breakpoint() 
    model_config = json.loads(json_string)
    return deserialize_keras_object(model_config, custom_objects=custom_objects)  

def test_torch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model')
        if use_keras:
            """
            model_file = model_name + ".json"
            with open(model_file, 'r') as json_file:
                print('USING MODEL:' + model_file)
                loaded_model_json = json_file.read()

            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_name + '.h5')
            loaded_model.compile(optimizer='sgd', loss='mse')
            """

            model_file = model_name + ".keras"
            loaded_model = load_model(model_file, compile=False)
            loaded_model.compile(optimizer='sgd', loss='mse')
    except:
        raise Exception(f'NN model could not be loaded. use_keras = {use_keras}')

    A = np.random.randn(10, 50, 50, 1)
    try:
        if use_keras:
            predictions = loaded_model.predict(A, batch_size=32)
        pass 
    except:
        raise Exception('NN model could not be deployed. use_keras = ' + str(use_keras))

if __name__ == "__main__":
    test_torch()