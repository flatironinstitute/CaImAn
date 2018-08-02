#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using OnACID.
@author: Andrea Giovannucci @agiovann and Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing their data used in this demo.
"""

from copy import deepcopy
import glob
import numpy as np
import os
import pylab as pl
import scipy
import sys
from time import time

try:
    if __IPYTHON__:
        print('Detected iPython')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.utils.visualization import view_patches_bar
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import download_demo, load_object, save_object
from caiman.components_evaluation import evaluate_components_CNN
from caiman.motion_correction import motion_correct_iteration_fast
import cv2
from caiman.utils.visualization import plot_contours
from caiman.source_extraction.cnmf.online_cnmf import bare_initialization
from caiman.paths import caiman_datadir

#%%
def main():
    pass # For compatibility between running under Spyder and the CLI

#%%  download and list all files to be processed

    # folder inside ./example_movies where files will be saved
    fld_name = 'Mesoscope'
    download_demo('Tolias_mesoscope_1.hdf5', fld_name)
    download_demo('Tolias_mesoscope_2.hdf5', fld_name)
    download_demo('Tolias_mesoscope_3.hdf5', fld_name)

    # folder where files are located
    folder_name = os.path.join(caiman_datadir(), 'example_movies', fld_name)
    extension = 'hdf5'                                  # extension of files
    # read all files to be processed
    fnames = glob.glob(folder_name + '/*' + extension)

    # your list of files should look something like this
    print(fnames)

#%%   Set up some parameters

    fr = 15  # frame rate (Hz)
    decay_time = 0.5  # approximate length of transient event in seconds
    gSig = (3, 3)  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    min_SNR = 2.5  # minimum SNR for accepting new components
    rval_thr = 0.85  # correlation threshold for new component inclusion
    ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
    gnb = 2  # number of background components
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
    mot_corr = True  # flag for online motion correction
    max_shift = np.ceil(10.).astype('int')  # maximum allowed shift during motion correction

    # set up some additional supporting parameters needed for the algorithm (these are default values but change according to dataset characteristics)
    init_batch = 200  # number of frames for initialization (presumably from the first file)
    expected_comps = 300  # maximum number of expected components used for memory pre-allocation (exaggerate here)
    K = 2  # initial number of components
    epochs = 2  # number of passes over the data

    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'init_batch': init_batch,
                   'init_method': 'bare',
                   'normalize': True,
                   'expected_comps': expected_comps,
                   'K': K,
                   'epochs': epochs,
                   'show_movie': True}
    opts = cnmf.params.CNMFParams(params_dict=params_dict)

#%% fit online
    
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

#%% plot contours
    
    print(('Number of components:' + str(cnm.estimates.A.shape[-1])))
    Cn = cm.load(fnames[0], subindices=slice(0,500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn)
    
#%% view components
    cnm.estimates.view_components(img=Cn)

#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
