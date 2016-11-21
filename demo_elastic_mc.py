# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:29:15 2016

@author: agiovann
"""

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic(u'load_ext autoreload')
        get_ipython().magic(u'autoreload 2')
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
#%%
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import motion_correct_online
from caiman.cluster import apply_to_patch
#%%
#backend='SLURM'
backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    # roughly number of cores on your machine minus 1
    n_processes = np.maximum(np.int(psutil.cpu_count()), 1)
print 'using ' + str(n_processes) + ' processes'
#%% start cluster for efficient computation
single_thread = False

if single_thread:
    dview = None
else:
    try:
        c.close()
    except:
        print 'C was not existing, creating one'
    print "Stopping  cluster to avoid unnencessary use of memory...."
    sys.stdout.flush()
    if backend == 'SLURM':
        try:
            cm.stop_server(is_slurm=True)
        except:
            print 'Nothing to stop'
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cm.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:
        cm.stop_server()
        cm.start_server()
        c = Client()

    print 'Using ' + str(len(c)) + ' processes'
    dview = c[:len(c)]
#%% 
Yr,dim,T=cm.load_memmap('Yr_d1_64_d2_128_d3_1_order_C_frames_6764_.mmap')    
res = apply_to_patch(Yr,(T,)+dim,None,128,8, motion_correct_online, 200, max_shift_w=15, max_shift_h=15, save_base_name='test_mmap',  init_frames_template=100, show_movie=False, remove_blanks=True,n_iter=2,show_template=False)    
#%%
[pl.plot(np.reshape(np.array(r[0][0]).T,(4,-1)).T) for r in res]
#%%
pl.imshow(res[0][0][2],cmap='gray',vmin=200,vmax=500)
#%%
mr,dim_r,T=cm.load_memmap('test_mmap_d1_49_d2_114_d3_1_order_C_frames_6764_.mmap')
#%% video
res_p = apply_to_patch(mr,(T,)+dim_r,None,16,8, motion_correct_online, 0, max_shift_w=3, max_shift_h=3,\
         save_base_name='test_patch_',  init_frames_template=100, show_movie=False, remove_blanks=False,n_iter=2)
#%%
res_p = apply_to_patch(mr,(T,)+dim_r,dview,24,8, motion_correct_online, 0,\
 max_shift_w=2, max_shift_h=2, save_base_name=None,  init_frames_template=100,\
 show_movie=False, remove_blanks=False,n_iter=2)
#%%
for idx,r in enumerate(res_p):
    pl.subplot(2,5,idx+1)
#    pl.plot(np.reshape(np.array(r[0][0]).T,(4,-1)).T)
    pl.plot(np.array(r[0][0]))
#%%
imgtot = np.zeros(np.prod(dim_r))
for idx,r in enumerate(res_p):    
#    pl.subplot(2,5,idx+1)
    img = r[0][2]
#    lq,hq = np.percentile(img,[10,90])
#    pl.imshow(img,cmap='gray',vmin=lq,vmax=hq) 
    imgtot[r[1]] = np.maximum(img.T.flatten(),imgtot[r[1]])

m=cm.load('test_mmap_d1_49_d2_114_d3_1_order_C_frames_6764_.mmap')

lq,hq = np.percentile(imgtot,[1,97])
pl.subplot(2,1,1)
pl.imshow(np.reshape(imgtot,dim_r,order='F'),cmap='gray',vmin=lq,vmax=hq,interpolation='none')    
pl.subplot(2,1,2)
lq,hq = np.percentile(res[0][0][2],[10,99])
pl.imshow(np.mean(m,0),cmap='gray',vmin=lq,vmax=hq,interpolation='none')
