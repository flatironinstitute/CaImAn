#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using OnACID.
@author: Andrea Giovannucci @agiovann and Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing their data used in this demo.

KERAS_BACKEND=tensorflow; CUDA_VISIBLE_DEVICES=-1; spyder
"""

import os
import sys

#try:
#    here = os.path.dirname(os.path.realpath(__file__))
#    caiman_path = os.path.join(here, "..", "..")
#    print("Caiman path detected as " + caiman_path)
#    sys.path.append(caiman_path)
#except:
#    pass
import numpy as np
try:
    if __IPYTHON__:
        print('Detected iPython')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


from time import time
import caiman as cm
from caiman.utils.visualization import view_patches_bar
from caiman.utils.utils import download_demo, load_object, save_object
import pylab as pl
import scipy
from caiman.motion_correction import motion_correct_iteration_fast
import cv2
import caiman.source_extraction.cnmf as cnmf
from copy import deepcopy
#%%
try:
    import sys
    if 'pydevconsole' in sys.argv[0]:
        raise Exception('Running in PYCHARM')
    ID = sys.argv[1]
    ID = str(np.int(ID)+1)
    print('Processing ID:'+ str(ID))
    ploton = False
    save_results = True
    save_init = False     # flag for saving initialization object. Useful if you want to check OnACID with different parameters but same initialization
except:
    print('ID NOT PASSED')
    ID = 11
    ploton = False
    save_results = False
    save_init = False # flag for saving initialization object. Useful if you want to check OnACID with different parameters but same initialization

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'
base_folder_files = '/mnt/ceph/neuro/zebra/05292014Fish1-4/'
#%%
K = 100 #number of initialization neurons
min_num_trial = 50 # number of neuron candidates per trial
fls = [os.path.join(base_folder_files,'Plane' + str(ID) + '.stack.hdf5')]
mmm = cm.load(fls,subindices = 0)
dims = mmm.shape
K = np.maximum(K,np.round(600/1602720*np.prod(mmm.shape)).astype(np.int))
min_num_trial = np.maximum(min_num_trial,np.round(200/1602720*np.prod(mmm.shape)).astype(np.int))
# your list of files should look something like this
print(fls)
print([K,min_num_trial])
#%%   Set up some parameters
# frame rate (Hz)
fr = 2
# approximate length of transient event in seconds
decay_time = 1.5
# number of passes over the data
epochs = 1
# expected half size of neurons
gSig = (6,6)
# order of AR indicator dynamics
p = 1
# minimum SNR for accepting new components
min_SNR = 2.5
# correlation threshold for new component inclusion
rval_thr = 1
# spatial downsampling factor (increases speed but may lose some fine structure)
ds_factor = 2
# number of background components
gnb = 3
# recompute gSig if downsampling is involved
gSig = tuple((np.array(gSig) / ds_factor))#.astype('int'))
# flag for online motion correction
mot_corr = True
# maximum allowed shift during motion correction
max_shift = np.ceil(10. / ds_factor).astype('int')

# number of frames for initialization (presumably from the first file)
initbatch = 200
# maximum number of expected components used for memory pre-allocation (exaggerate here)
expected_comps = 3000


show_movie = False

params_dict = {    'fnames': fls,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'expected_comps' : expected_comps,
                   'init_batch': initbatch,
                   'init_method': 'bare',
                   'normalize': True,
                   'sniper_mode': True,
                   'K': K,
                   'epochs': epochs,
                   'max_shifts_online': max_shift,
                   'pw_rigid': False,
                   'dist_shape_update': True,
                   'min_num_trial': min_num_trial,
                   'show_movie': show_movie}

opts = cnmf.params.CNMFParams(params_dict=params_dict)

#%%
cnm = cnmf.online_cnmf.OnACID(params=opts)
cnm.fit_online()
#%% compute correlation image
compute_corr = False
if compute_corr:
    m = cm.load(fls)
    mc = m.motion_correct(10,10)[0]
    mp = (mc.computeDFF(3))
    Cn = cv2.resize(mp[0].local_correlations(eight_neighbours=True, swap_dim=False),dims[::-1][:-1])
    np.save(os.path.join(base_folder,'Zebrafish/results_analysis_online_Plane_CN_' + str(ID) + '.npy'), Cn)
else:
    try:
        Cn = np.load(os.path.join(base_folder,'Zebrafish/results_analysis_online_Plane_CN_' + str(ID) + '.npy'))
    except Exception as e:
        print(e)

#%% load only the first initbatch frames and possibly downsample them
if ploton:
    pl.imshow(Cn)
    pl.title('Correlation Image')
    pl.colorbar()

#%%  save results (optional)
if save_results:
    np.savez(os.path.join(base_folder, 'Zebrafish/results_analysis_online_1EPOCH_gSig6_equalized_Plane_NEW_' + str(ID) + '.npz'),
             Cn=Cn, Ab=cnm.estimates.Ab, Cf=cnm.estimates.C_on, b=cnm.estimates.b, f=cnm.estimates.f,
             dims=cnm.estimates.dims, tottime='SEE VARIABLES t_XX', noisyC=cnm.estimates.noisyC, shifts=cnm.estimates.shifts,
             num_comps=cnm.estimates.A.shape[-1], t_online=cnm.t_online, t_detect=cnm.t_detect, t_shapes=cnm.t_shapes)
    cnm.save(os.path.join(base_folder, 'Zebrafish/results_analysis_online_1EPOCH_gSig6_equalized_Plane_NEW_' + str(ID) + '.hdf5'))
#%%
pl.figure()
pl.stackplot(np.arange(len(cnm.t_detect)), 1e3*np.array(cnm.t_motion), 1e3 * (np.array(cnm.t_online) - np.array(cnm.t_detect) - np.array(cnm.t_shapes) - np.array(cnm.t_motion)),
              1e3 * np.array(cnm.t_detect), 1e3 * np.array(cnm.t_shapes))
pl.title('Processing time per frame')
pl.xlabel('Frame #')
pl.ylabel('Processing time [ms]')
pl.ylim([0, 1000])
pl.legend(labels=['motion', 'process', 'detect', 'shapes'])

#%%
cnm.estimates.plot_contours()
