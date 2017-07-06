# -*- coding: utf-8 -*-

"""
A demo script for processing calcium imaging data using CNMF-E
author: Pengcheng Zhou
email: zhoupc1988@gmail.com
created: 6/12/17
last edited: 
"""

"""
--------------------------------MODULES--------------------------------
"""
import os
import numpy as np
import json
import cv2
import h5py
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmfe import CNMFE

"""
--------------------------------CLASSES--------------------------------
"""

"""
-------------------------------FUNCTIONS-------------------------------
"""

"""
----------------------------------RUN----------------------------------
"""

# parameters
pars_envs = {'backend': 'local', #{'loca', 'SLURM'}
             'processes number': None,
             'client': None,
             'direct view': None,
             'memory factor': None,
             'single thread': False
             }
pars_data = {'file path': None,
             'dir name': None,
             'file name': None,
             'file type': None,     #{'avi', 'tif', 'hdf5'}
             'frame rate': None,
             'frame number': None,
             'row number': None,
             'column number': None,
             'z planes': None,
             'pixel size': None,
             'spatial downsampling factor': 1,
             'temporal downsampling factor': 1
             }
pars_motion_parallel = {'splits_rig': 28,
                        'num_splits_to_process_rig': None,
                        'splits_els': 28,
                        'num_splits_to_process_els': [14, None]}
pars_motion = {'run motion correction': True,
               'niter_rig': 1,
               'max_shifts': (6, 6),
               'num_splits_to_process_rig': None,
               'strides': (48, 48),
               'overlaps': (24, 24),
               'upsample_factor_grid': 4,
               'max_deviation_rigid': 3,
               'shifts_opencv': True,
               'min_mov': 0,
               'nonneg_movie': False,
               'parallel': pars_motion_parallel}
pars_neuron = {'neuron diameter': 16, #unit: pixel
               'gaussian width': 4, #unit: pixel
               'maximum neuron number': None,
               'do merge': True,
               'merge method': ['correlation', 'distance'], # can use only
               # one method
               'merge threshold': [0.001, 0.85], # [minimum spatial
               # correlation, minimum temporal correlation]
               'minimum distance': 3, # unit: pixel,
               'alpha (sparse NMF)': None
              }

pars_initialization = {'use patch': True,
                       'method': 'greedyROI', #{'greedyROI',
                       # 'greedyROI_corr', 'gredyROI_endoscope', 'sparseNMF'}
                       'maximum neuron number': 5,
                       'minimum local correlation': 0.85,
                       'minimum peak-to-noise ratio': 10 }
pars_spatial = {'use patch': True,
                'patch size': 64,
                'overlap size': None,
                'spatial downsampling factor (patch)': 1,
                'temporal downsampling factor (patch)': 1
                }
pars_deconvolution = {'model': 'ar2',  # {'ar1', 'ar2'}
                      'method': 'constrained foopsi', #{'foopsi',
                      # 'constrained foopsi', 'thresholded foopsi'}
                      'algorithm': 'oasis' #{'oasis', 'cvxpy'}
                      }
pars_temporal = {'run deconvolution': True,
                 'deconvolution options': pars_deconvolution,
                 'algorithm': 'hals',
                 'iteration number': 2}
pars_background = {'size factor': 1.5, # {size factor} = {The radius of the
                   # ring}/{neuron diameter}
                   'downsampling factor': 2,
                   'background rank': 1,    #number of background components
                   }
pars_export = {}
options_caiman = {'data': pars_data,
                  'envs': pars_envs,
                  'neuron': pars_neuron,
                  'motion': pars_motion,
                  'initialization': pars_initialization,
                  'spatial': pars_spatial,
                  'temporal': pars_temporal,
                  'background': pars_background,
                  'export': pars_export}


# data info
print('\n'+'-'*25+'Loading data'+'-'*25)
file_path = '/home/zhoupc/Downloads/64_20170328_13.tif'
if os.path.exists(file_path):
    file_folder, file_name = os.path.split(file_path)
    temp, file_ext = os.path.splitext(file_path)

    # create a folder for saving results
    result_folder = temp + '_caiman'
    if os.path.exists(result_folder):
        print('The result folder has been created')
    else:
        os.mkdir(result_folder)
    print('The results can be found in folder %s\n' % result_folder)
else:
    print('data does not exist. Please check the file path! ')
fname = file_path

# load data and figure out its dimension
Y_raw = cm.load(file_path)
T, nrow, ncol = Y_raw.shape
pars_data['frame number'] = T
pars_data['row number'] = nrow
pars_data['column number'] = ncol

# save the parameters
with open('default_options.json', 'w') as f:
    json.dump(options_caiman, f)

neuron = CNMFE(packed_pars=options_caiman)
print('-'*25+'Done'+'-'*25+'\n')

# setup computing environement
client_running = neuron.set_envs()

# run motion correction
if pars_motion['run motion correction']:
    print('\n'+'-'*25+'Run motion correction '+'-'*25)

    fname_high_pass = os.path.join(result_folder, 'Y_high_pass.mmap')
    if os.path.exists(fname_high_pass):
        Y_high_pass = np.memmap(fname_high_pass, mode='r', dtype=np.float32,
                                shape=Y_raw.shape)
        print('spatially filtered data exists!')
    else:
        print('\napply spatial filtering to roughly remove the background')
        Y_high_pass = np.memmap(fname_high_pass, mode='w+', dtype=np.float32,
                            shape=Y_raw.shape)
        psf = neuron.gen_filter_kernel()
        len_dots = np.min([50, T])
        print('-'*len_dots)
        temp = np.floor(np.linspace(0, T, len_dots+1))
        k = 0
        for i in np.arange(T):
            Y_high_pass[i] = cv2.filter2D(Y_raw[i].astype(np.float32), -1, psf)
            if i >= temp[k]:
                print('.', end='', flush=True)
                k += 1
        print('\nDone!\n')
    fname_high_pass_npy = os.path.join(result_folder, 'Y_high_pass.npy')
    np.save(fname_high_pass_npy, Y_high_pass)

    print('\nCompute spatial shifts using the spatially filtered data')
    mc = MotionCorrect(fname_high_pass_npy, min_mov=0, dview=neuron.dview,
                       packed_pars=options_caiman)
    #
    print('\nApply the spatial shifts to the raw video data ')
    # run rigid motion correction
    mc.motion_correct_rigid(save_movie=True)
    Y_rig = cm.load(mc.file_tot_rig)
    #
    #
    #
    # print('-'*25+'Done'+'-'*25+'\n')



















