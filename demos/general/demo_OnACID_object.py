#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:03:31 2018
Basic demo of caiman online using the cnmf object
@author: epnevmatikakis
"""

import os

import numpy as np
import pylab as pl
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.paths import caiman_datadir

# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI

# %% load data

    fname = os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')

# %% set up some parameters

    # frame rate (Hz)
    fr = 10
    # approximate length of transient event in seconds
    decay_time = 1.
    # expected half size of neurons
    gSig = [6, 6]
    # order of AR indicator dynamics
    p = 1
    # minimum SNR for accepting new components
    min_SNR = 2.5
    # correlation threshold for new component inclusion
    rval_thr = 0.85
    # number of background components
    gnb = 3
    # number of frames for initialization (presumably from the first file)
    initbatch = 400
    # number of components during initialization
    K = 10

# %% create CNMF object

    cnm = cnmf.CNMF(2, k=K, gSig=gSig, p=p, update_num_comps=True,
                    rval_thr=rval_thr, gnb=gnb, min_SNR=min_SNR,
                    decay_time=decay_time, fr=fr)

# %% run CaImAn online

    cnm.fit_online(fname, epochs=2, init_batch=initbatch)

    print(('Number of components:' + str(cnm.A.shape[-1])))

# %% plot contours again correlation image
    pl.figure()
    CI = cm.load(fname, subindices=slice(0,1000)).local_correlations(swap_dim=False)
    cnm.plot_contours(img=CI)

# %% view traces
    cnm.view_components(None, cnm.dims)
# %% pass through the CNN classifier with a low threshold (keeps clearer neuron shapes and excludes processes)
    use_CNN = True
    if use_CNN:
        # threshold for CNN classifier
        thresh_cnn = 0.05
        from caiman.components_evaluation import evaluate_components_CNN
        predictions, final_crops = evaluate_components_CNN(
            cnm.A, cnm.dims, gSig,
            model_name=os.path.join(caiman_datadir(), 'model', 'cnn_model'))
        idx = np.where(predictions[:, 1] > thresh_cnn)[0].tolist()
        cnm.plot_contours(img=CI, idx=idx)
# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
