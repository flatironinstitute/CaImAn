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
import caiman as cm
import json
from caiman.motion_correction import MotionCorrect
#import caiman.source_extraction.cnmfe as cnmfe

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
pars_data = {'file path': None,
             'dir name': None,
             'file name': None,
             'file type': None,     #{'avi', 'tif', 'hdf5'}
             'frame rate': None,
             'frame number': None,
             'row number': None,
             'column number': None,
             'z planes': None,
             'pixel size': None}
pars_motion_parallel = {'splits_rig': 28,
                        'num_splits_to_process_rig': None,
                        'splits_els': 28,
                        'num_splits_to_process_els': [14, None]}
pars_motion = {'run motion correction': True,
               'niter_rig': 1,
               'max_shifts': (6, 6),
               'num_splits_to_process_rig': None,
               'strides': (48, 48),
               'overlap': (24, 24),
               'upsample_factor_grid': 4,
               'max_deviation_rigid': 3,
               'parallel': pars_motion_parallel}

pars_init = {}
pars_spatial = {}
pars_temporal = {}
pars_background = {}
pars_export = {}
options_caiman = {'data': pars_data,
                  'motion': pars_motion,
                  'initialization': pars_init,
                  'update spatial': pars_spatial,
                  'update temporal': pars_temporal,
                  'update background': pars_background,
                  'export results': pars_export}
# setting up computing environment
print('\n'+'-'*25+'Setting up computing environment'+'-'*25)
backend = 'local'       #{'local', 'SLURM'}
client_run, dview, n_processes = cm.cluster.setup_cluster(backend=backend, n_processes=None, single_thread=False)
print('-'*25+'Done'+'-'*25+'\n')

# data info
print('\n'+'-'*25+'Loading data'+'-'*25)
file_path = '/home/zhoupc/Downloads/64_20170328_13.tif'
if os.path.exists(file_path):
    file_folder, file_name = os.path.split(file_path)
    temp, file_ext = os.path.splitext(file_path)

    # create a folder for saving results
    result_folder = temp + 'caiman'
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

print('-'*25+'Done'+'-'*25+'\n')

# run motion correction
if pars_motion['run motion correction']:
    print('\n'+'-'*25+'Run motion correction '+'-'*25)
    mc = MotionCorrect(fname, min_mov=0, dview=dview, packed_pars=pars_motion )

    # run rigid motion correction
    mc.motion_correct_rigid(save_movie=True)
    Y_rig = cm.load(mc.file_tot_rig)



    print('-'*25+'Done'+'-'*25+'\n')



















