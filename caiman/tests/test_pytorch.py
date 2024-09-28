#!/usr/bin/env python

import numpy as np
import os
import keras 

from caiman.paths import caiman_datadir
from caiman.utils.utils import load_graph

try:
    os.environ["KERAS_BACKEND"] = "torch"
    from keras.models import load_model 
    use_keras = True
except(ModuleNotFoundError):
    import torch 
    use_keras = False

def test_torch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model')
        if use_keras:
            model_file = model_name + ".keras"
            print('USING MODEL:' + model_file)

            loaded_model = load_model(model_file)
            loaded_model.compile('sgd', 'mse')
        elif use_keras == True: 
            model_file = model_name + ".pth"
            loaded_model = torch.load(model_file)
    except:
        raise Exception(f'NN model could not be loaded. use_keras = {use_keras}')

    A = np.random.randn(10, 50, 50, 1)
    try:
        if use_keras == False: 
            predictions = loaded_model.predict(A, batch_size=32)
        elif use_keras == True:
            A = torch.tensor(A, dtype=torch.float32)
            A = torch.reshape(A, (-1, A.shape[-1], A.shape[1], A.shape[2])) 
            with torch.no_grad():
                predictions = loaded_model(A)  
        pass 
    except:
        raise Exception('NN model could not be deployed. use_keras = ' + str(use_keras))

if __name__ == "__main__":
    test_torch()