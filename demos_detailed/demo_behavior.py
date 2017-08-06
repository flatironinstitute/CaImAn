#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 11:43:39 2017

@author: agiovann
"""

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
import glob

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import copy

try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import sys
import numpy as np
import psutil
import glob
import os
import scipy
from ipyparallel import Client
# mpl.use('Qt5Agg')
import pylab as pl
pl.ion()
#%
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.behavior import behavior
from scipy.sparse import coo_matrix

#%%
params_movie = {'fname': ['Sue_2x_3000_40_-46.tif'],
                                            
               }

#%% load, rotate and eliminate useless pixels
m = cm.load('example_movies/movie_behavior.h5',is_behavior=True)
m = m.transpose([0,2,1])
m = m[:700,150:,:]
#%% visualize movie
m.play()
#%% select interesting portion of the FOV (draw a polygon on the figure that pops up, when done press enter)
mask = behavior.select_roi(np.median(m[::100],0),1)[0]
#%%
n_components = 5 # number of movement looked for
resize_fact = 0.5 # for computational efficiency movies are downsampled 
num_std_mag_for_angle = .6 # number of standard deviations above mean for the magnitude that are considered enough to measure the angle in polar coordinates
only_magnitude = False # if onlu interested in factorizing over the magnitude
method_factorization = 'dict_learn' # could also use nmf
max_iter_DL=-30 # number of iterations for the dictionary learning algorithm (Marial et al, 2010)

spatial_filter_, time_trace_, of_or = cm.behavior.behavior.extract_motor_components_OF(m, n_components, mask = mask,  
                        resize_fact= resize_fact, only_magnitude = only_magnitude,verbose = True, method_factorization = 'dict_learn', max_iter_DL=max_iter_DL)

#%%
mags, dircts, dircts_thresh, spatial_masks_thrs = cm.behavior.behavior.extract_magnitude_and_angle_from_OF(spatial_filter_, time_trace_, of_or, num_std_mag_for_angle = num_std_mag_for_angle, sav_filter_size =3, only_magnitude = only_magnitude)
