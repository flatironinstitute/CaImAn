#!/usr/bin/env python
import caiman as cm
from caiman.paths import caiman_datadir
from caiman.utils.utils import download_model, download_demo
from caiman.source_extraction.volpy.mrcnn import neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib
import numpy as np
import os
import tensorflow as tf

def mrcnn(img, size_range, weights_path):
    config = neurons.NeuronsConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one img at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.7
        IMAGE_RESIZE_MODE = "pad64"
        IMAGE_MAX_DIM = 512
        RPN_NMS_THRESHOLD = 0.7
        POST_NMS_ROIS_INFERENCE = 1000
    config = InferenceConfig()
    config.display()
    model_dir = os.path.join(caiman_datadir(), 'model')
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                  config=config)
        tf.keras.Model.load_weights(model.keras_model, weights_path, by_name=True)
        results = model.detect([img], verbose=1)
        r = results[0]
        selection = np.logical_and(r['masks'].sum(axis=(0,1)) > size_range[0] ** 2, 
                                   r['masks'].sum(axis=(0,1)) < size_range[1] ** 2)
        r['masks'] = r['masks'][:, :, selection]
        ROIs = r['masks'].transpose([2, 0, 1])   
    return ROIs

def test_mrcnn():
    weights_path = download_model('mask_rcnn')    
    summary_images = cm.load(download_demo('demo_voltage_imaging_summary_images.tif'))
    ROIs = mrcnn(img=summary_images.transpose([1, 2, 0]), size_range=[5, 22],
                                 weights_path=weights_path)
    assert ROIs.shape[0] == 14, 'fail to infer correct number of neurons'
    
    