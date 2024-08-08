#!/usr/bin/env python

import os
import numpy as np
import keras 
from keras.models import load_model 
from caiman.paths import caiman_datadir  

# try:
#    os.environ["KERAS_BACKEND"] = "torch"
#    from keras.models import load_model 
#    use_keras = False 
#except(ModuleNotFoundError):
import torch 
use_keras = True 
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

def test_torch():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model_test_1')
        if use_keras == False: 
            model_file = model_name + ".keras"
            print(f"USING MODEL (keras API): {model_file}")
            loaded_model = load_model(model_file, compile=False)
            loaded_model.compile(optimizer='sgd', loss='mse')
        elif use_keras == True: 
            model_file = model_name + ".pth"
            loaded_model = torch.load(model_file)
    except:
        raise Exception(f'NN model could not be loaded.')

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
        raise Exception('NN model could not be deployed')

if __name__ == "__main__":
    test_torch()