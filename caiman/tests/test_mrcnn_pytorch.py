#!/usr/bin/env python

import numpy as np
import os
import torch

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.utils.utils import download_model, download_demo
from caiman.source_extraction.volpy.mrcnn import neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib

def mrcnn(img, size_range, weights_path):

    return 

def test_mrcnn():
    weights_path = download_model('mask_rcnn')    
    summary_images = cm.load(download_demo('demo_voltage_imaging_summary_images.tif'))
    ROIs = mrcnn(img=summary_images.transpose([1, 2, 0]), size_range=[5, 22],
                                 weights_path=weights_path)
    assert ROIs.shape[0] == 14, 'fail to infer correct number of neurons'