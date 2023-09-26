#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic demo for the CaImAn Online algorithm (OnACID) using CNMF initialization.
It demonstrates the construction of the params and online_cnmf objects and
the fit function that is used to run the algorithm.
For a more complete demo check the script demo_OnACID_mesoscope.py

@author: jfriedrich & epnev
"""
#%%
from IPython import get_ipython
import logging
import numpy as np
import os

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.paths import caiman_datadir

try:
    if __IPYTHON__:
        print("Detected iPython")
        ipython = get_ipython()
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
        ipython.run_line_magic('matplotlib', 'qt')
except NameError:
    pass

#%%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)
    # filename="/tmp/caiman.log"
# %%
def main():
    pass  # For compatibility between running under an IDE and the CLI

    # %% load data
    fname = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]

    # %% set up some parameters
    fr = 10  # frame rate (Hz)
    decay_time = .75  # approximate length of transient event in seconds
    gSig = [6, 6]  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    min_SNR = 1  # minimum SNR for accepting candidate components
    thresh_CNN_noisy = 0.65  # CNN threshold for candidate components
    gnb = 2  # number of background components
    init_method = 'cnmf'  # initialization method

    # set up CNMF initialization parameters
    init_batch = 400  # number of frames for initialization
    patch_size = 32  # size of patch
    stride = 3  # amount of overlap between patches
    K = 4  # max number of components in each patch
    params_dict = {'fr': fr,
                   'fnames': fname,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'nb': gnb,
                   'init_batch': init_batch,
                   'init_method': init_method,
                   'rf': patch_size//2,
                   'stride': stride,
                   'sniper_mode': True,
                   'thresh_CNN_noisy': thresh_CNN_noisy,
                   'K': K}
    opts = cnmf.params.CNMFParams(params_dict=params_dict)

    # %% fit with online object
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

    # %% plot contours
    logging.info('Number of components:' + str(cnm.estimates.A.shape[-1]))
    Cn = cm.load(fname[0], subindices=slice(0,500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn)

    # %% pass through the CNN classifier with a low threshold (keeps clearer neuron shapes and excludes processes)
    use_CNN = True
    if use_CNN:
        # threshold for CNN classifier
        opts.set('quality', {'min_cnn_thr': 0.05})
        cnm.estimates.evaluate_components_CNN(opts)
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)
    
    # %% plot results
    cnm.estimates.view_components(img=Cn, idx=cnm.estimates.idx_components)

# %%
# This is to mask the differences between running this demo in IDE
# versus from the CLI
if __name__ == "__main__":
    main()

# %%
