#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:14:31 2018

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
import pickle
from caiman.components_evaluation import select_components_from_metrics
from caiman.base.rois import nf_match_neurons_in_binary_masks
from caiman.utils.utils import apply_magic_wand
from caiman.base.rois import detect_duplicates_and_subsets
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
    print('Not launched under iPython')

import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import scipy
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.cluster import setup_cluster
import glob
#%%
print('******** IF COMPUTATION GETS STUCK BECAUSE OF MEMORY PROBLEMS, REDUCE n_processes')
backend_patch = 'local'
n_processes = 22 # reduce this number uf there are memory problems
#%%
regexps = []
for folder in ['J115','J123','K53']:
    rgx = os.path.abspath(folder)+'/*_.mmap'
    print(rgx)

    try:
        cm.stop_server()
        dview.terminate()
    except:
        print('No clusters to stop')

    c, dview, n_processes = setup_cluster(
            backend=backend_patch, n_processes=n_processes, single_thread=False)

    ffllss = list(glob.glob(rgx)[:])
    ffllss.sort()
    print(ffllss)
#    for fl in ffllss:
#        mmm = cm.load(fl)-min_mov
#        print([mmm.min(),mmm.max()])
    mm = cm.load(ffllss[0])
    min_mov = mm.min()
    fname_new = cm.save_memmap(ffllss, base_name='Yr', order='C',
                   border_to_0=0, dview = dview, n_chunks = 80, add_to_movie=min_mov)  # exclude borders




