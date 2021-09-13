#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:48:17 2020
Pipeline for online analysis of fluorescence imaging data
Voltage dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
Calcium dataset courtesy of Sue Ann Koay and David Tank (Princeton University)
@author: @agiovann, @caichangjia, @cynthia
"""
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.python.client import device_lib
from time import time, sleep
from threading import Thread

import caiman as cm
from caiman.fiola.config import load_fiola_config, load_caiman_config
from caiman.fiola.fiolaparams import fiolaparams
from caiman.fiola.fiola import FIOLA

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

logging.info(device_lib.list_local_devices())
# if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1
    
#%% load movie and masks
folder_idx = 1
movie_folder = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data', 
                '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/demo_K53'][folder_idx]
name = ['demo_voltage_imaging.hdf5', 'k53.tif'][folder_idx]
mode = ['voltage', 'calcium'][folder_idx]

fnames = os.path.join(movie_folder, name)
mov = cm.load(os.path.join(movie_folder, name))
mask = cm.load(os.path.join(movie_folder, name.split('.')[0]+'_ROIs.hdf5'))
#mask = None

display_images = False
if display_images:
    plt.figure(); plt.imshow(mov.mean(0), vmax=np.percentile(mov.mean(0), 99.9)); plt.title('Mean img')
    if mask is not None:
        plt.figure(); plt.imshow(mask.mean(0)); plt.title('Masks')

#%% load configuration; set up FIOLA object
# In the first part, we will first show each part (motion correct, source separation and spike extraction) of 
# FIOLA separately in an offline manner. 
# Then in the second part, we will show the full pipeline and its real time frame-by-frame analysis performance
options = load_fiola_config(fnames, mode, mask) 
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)

#%% offline motion correction
fio.dims = mov.shape[1:]
template = mov.bin_median()
mc_mov, shifts, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], mov.min())

if display_images:
    plt.figure(); plt.plot(shifts);plt.legend(['x shifts', 'y shifts']); plt.title('shifts'); plt.show()
    moviehandle = cm.movie(mc_mov.copy()).reshape((-1, template.shape[0], template.shape[1]), order='F')
    moviehandle.play(gain=3, q_min=5, q_max=99.99, fr=400)

#%% optimize masks using hals or initialize masks with CaImAn
fio.params.data['init_method'] = 'weighted_masks'
if mode == 'voltage':
    if fio.params.data['init_method'] == 'binary_masks':
        fio.fit_hals(mc_mov, mask)
elif mode == 'calcium':
    # we don't need to optimize masks using hals as we are using spatial footprints from CaImAn
    if fio.params.data['init_method'] == 'weighted_masks':
        logging.info('use weighted masks from CaImAn')     
    elif fio.params.data['init_method'] == 'caiman':
    # if masks are not provided, we can use caiman for initialization
        fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict = load_caiman_config(fnames)
        _, _, mask = fio.fit_caiman_init(mc_mov[:fio.params.data['num_frames_init']], 
                                         fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict)
   
    mask_2D = cm.movie(mask).to_2D(order='C')
    Ab = mask_2D.T
    fio.Ab = Ab / norm(Ab, axis=0)
        
#%% source extraction (nnls)
# when FOV and number of neurons is large, use batch_size=1
trace = fio.fit_gpu_nnls(mc_mov, fio.Ab, batch_size=1) 

#%% offline spike detection (only available for voltage currently)
fio.saoz = fio.fit_spike_extraction(trace)

#%% put the result in fio.estimates object
fio.compute_estimates()

#%% show results
fio.corr = cm.movie(mc_mov).local_correlations(swap_dim=False)
if display_images:
    fio.view_components(fio.corr)

#%% Now we start the second part. It uses fit method to perform initialization 
# which prepare parameters, spatial footprints etc for real-time analysis
# Then we call fit_online to perform real-time analysis
options = load_fiola_config(fnames, mode, mask) 
params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)

if fio.params.data['init_method'] == 'caiman':
    # in caiman initialization it will save the input movie to the init_file_name from the beginning
    fio.params.mc_dict, fio.params.opts_dict, fio.params.quality_dict = load_caiman_config(fnames)

scope = [fio.params.data['num_frames_init'], fio.params.data['num_frames_total']]
fio.fit(mov[:scope[0]])

#%% now fit online
fio.pipeline.load_frame_thread = Thread(target=fio.pipeline.load_frame, 
                                        daemon=True, 
                                        args=(mov[scope[0]:scope[1], :, :],))
fio.pipeline.load_frame_thread.start()

start = time()
fio.fit_online()
logging.info(f'total time online: {time()-start}')
logging.info(f'time per frame online: {(time()-start)/(scope[1]-scope[0])}')

#%% put the result in fio.estimates object
fio.compute_estimates()

#%% visualize the result, the last component is the background
if display_images:
    fio.view_components(fio.corr)

        
#%% save the result
save_name = f'{os.path.join(movie_folder, name.split(".")[0])}_fiola_result'
np.save(os.path.join(movie_folder, save_name), fio.estimates)

#%%
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)