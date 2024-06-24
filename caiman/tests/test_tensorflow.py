#!/usr/bin/env python

import numpy as np
import os

from caiman.paths import caiman_datadir
from caiman.utils.utils import load_graph
from caiman import keras

#FIXME A lot of what this did is no longer relevant

if keras is not None:
    os.environ["KERAS_BACKEND"] = "tensorflow"
    model_from_json = keras.models.model_from_json
    use_keras = True
else:
    import tensorflow as tf
    use_keras = False

def test_tf():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        model_name = os.path.join(caiman_datadir(), 'model', 'cnn_model')
        if use_keras:
            model_file = model_name + ".json"
            with open(model_file, 'r') as json_file:
                print('USING MODEL:' + model_file)
                loaded_model_json = json_file.read()

            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_name + '.h5')
            loaded_model.compile('sgd', 'mse')
        else:
            model_file = model_name + ".h5.pb"
            loaded_model = load_graph(model_file)
    except:
        raise Exception(f'NN model could not be loaded. use_keras = {use_keras}')

    A = np.random.randn(10, 50, 50, 1)
    try:
        if use_keras:
            predictions = loaded_model.predict(A, batch_size=32)
        else:
            tf_in = loaded_model.get_tensor_by_name('prefix/conv2d_20_input:0')
            tf_out = loaded_model.get_tensor_by_name('prefix/output_node0:0')
            with tf.Session(graph=loaded_model) as sess:
                predictions = sess.run(tf_out, feed_dict={tf_in: A})
        pass
    except:
        raise Exception('NN model could not be deployed. use_keras = ' + str(use_keras))
