# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
#%%
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')


try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import caiman as cm
import numpy as np
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise

#%%
#backend='SLURM'
backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    # roughly number of cores on your machine minus 1
    n_processes = np.maximum(np.int(psutil.cpu_count()), 1)
print(('using ' + str(n_processes) + ' processes'))
#%% start cluster for efficient computation
single_thread = False
if single_thread:
    dview = None
else:
    try:
        c.close()
    except:
        print('C was not existing, creating one')
    print("Stopping  cluster to avoid unnencessary use of memory....")
    sys.stdout.flush()
    if backend == 'SLURM':
        try:
            cm.stop_server(is_slurm=True)
        except:
            print('Nothing to stop')
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cm.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:
        cm.stop_server()
        cm.start_server()
        c = Client()

    print(('Using ' + str(len(c)) + ' processes'))
    dview = c[:len(c)]
#%% set parameters and create template by rigid motion correction
t1 = time.time()
#fname = 'M_FLUO_4.tif'
#max_shifts = (8,8)
#num_iter = 2

#fname = 'Sue_2x_3000.tif'
fname = 'Sue_2x_3000_40_-46.tif'
max_shifts = (6,6)
num_iter = 1

splits = 56 # for parallelization split the movies in  num_splits chuncks across time
num_splits_to_process = None # if none all the splits are processed and the movie is saved
shifts_opencv = True
save_movie_rigid = True
t1 = time.time()
fname_tot_rig, total_template_rig, templates_rig, shifts_rig = cm.motion_correction.motion_correct_batch_rigid(fname, max_shifts, dview = dview, splits = splits ,num_splits_to_process = num_splits_to_process, num_iter = num_iter,  template = None, shifts_opencv = shifts_opencv , save_movie_rigid = save_movie_rigid)
t2 = time.time() - t1
print(t2)
pl.imshow(total_template_rig,cmap = 'gray',vmax = np.percentile(total_template_rig,95))     
#%%
pl.plot(shifts_rig)
#%%
m_rig = cm.load(fname_tot_rig)
#%%
add_to_movie = - np.min(total_template_rig)
print(add_to_movie)
#%% visualize movies
m_rig.resize(1,1,.2).play(fr = 30, gain = 20, magnification=2, offset = add_to_movie)
#%% visualize templates
cm.movie(np.array(templates_rig)).play(fr = 10, gain = 5, magnification=2, offset = add_to_movie)
#%%
## for 512 512 this seems good
t1 = time.time()
new_templ = total_template_rig.copy()

if new_templ.shape == (512,512):
    
    strides = (128,128)# 512 512 
    overlaps = (32,32)

    newoverlaps = None
    newstrides = None
    
elif new_templ.shape == (256,256) or  new_templ.shape == (170,170):
    strides = (48,48)# 512 512 
    overlaps = (24,24)

#    strides = (16,16)# 512 512 
    newoverlaps = None
    newstrides = None

elif new_templ.shape == (64,128):
    strides = (32,32)#
    overlaps = (16,16)

    newoverlaps = None
    newstrides = None
else:
    raise Exception('Unknown size, set manually') 

shifts_opencv = True
save_movie = True
splits = 56
num_splits_to_process = None
upsample_factor_grid = 4
max_deviation_rigid = 3
add_to_movie = -np.min(total_template_rig)
num_iter = 1
for num_splits_to_process in [28,None]:
    fname_tot_els, total_template_wls, templates_els, x_shifts_els, y_shifts_els, coord_shifts_els  = cm.motion_correction.motion_correct_batch_pwrigid(fname, max_shifts, strides, overlaps, add_to_movie, newoverlaps = None,  newstrides = None,
                                             dview = dview, upsample_factor_grid = upsample_factor_grid, max_deviation_rigid = max_deviation_rigid,
                                             splits = splits ,num_splits_to_process = num_splits_to_process, num_iter = num_iter,
                                             template = new_templ, shifts_opencv = shifts_opencv, save_movie = save_movie)
    new_templ = total_template_wls
#%%
pl.subplot(2,1,1)
pl.plot(x_shifts_els)
pl.subplot(2,1,2)
pl.plot(y_shifts_els)
#%%
m_els = cm.load(fname_tot_els)
cm.concatenate([m_rig.resize(1,1,.2),m_els.resize(1,1,.2)],axis = 1).play(fr = 50, gain = 20,magnification=2, offset = add_to_movie) 
#%% compute metrics for the results
final_size =  np.subtract(np.subtract(new_templ.shape, max_shifts),(1,1))
winsize = 75
swap_dim = False
resize_fact_flow = .2
tmpl, correlations, flows_orig, norms,smoothness  = cm.motion_correction.compute_metrics_motion_correction(fname_tot_els,final_size[0],final_size[1],swap_dim, winsize=winsize , play_flow=False, resize_fact_flow=resize_fact_flow)
tmpl, correlations, flows_orig, norms,smoothness  = cm.motion_correction.compute_metrics_motion_correction(fname_tot_rig,final_size[0],final_size[1],swap_dim, winsize=winsize , play_flow=False, resize_fact_flow=resize_fact_flow)
tmpl, correlations, flows_orig, norms,smoothness  = cm.motion_correction.compute_metrics_motion_correction(fname,final_size[0],final_size[1],swap_dim, winsize=winsize , play_flow=False, resize_fact_flow=resize_fact_flow)

#%% plot the results of metrics
fls = [fname_tot_els[:-4]+ '_metrics.npz',fname_tot_rig[:-4]+ '_metrics.npz',fname[:-4]+ '_metrics.npz']
for cnt,fl in enumerate(fls):
    with np.load(fl) as ld:
#        print(ld.keys())
#        pl.figure()
        print(fl)
        print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) + ' ; ' + str(ld['smoothness']) )
        pl.subplot(len(fls),4,1+4*cnt)
        try:
            mean_img = np.mean(cm.load(fl[:-12]+'mmap'),0)[12:-12,12:-12]
        except:
            try:
                mean_img = np.mean(cm.load(fl[:-12]+'.tif'),0)[12:-12,12:-12]    
            except:
                mean_img = np.mean(cm.load(fl[:-12]+'hdf5'),0)[12:-12,12:-12]        
        lq,hq = np.nanpercentile(mean_img,[.5,99.5])
        pl.imshow(mean_img,vmin = lq,vmax = hq)
    #        pl.plot(ld['correlations'])
        
        pl.subplot(len(fls),4,4*cnt+2)
        pl.imshow(ld['img_corr'],vmin = 0,vmax =.25)
#        pl.colorbar()
        pl.subplot(len(fls),4,4*cnt+3)
#       
        pl.plot(ld['norms'])
        pl.subplot(len(fls),4,4*cnt+4)
        flows = ld['flows']
        pl.imshow(np.mean(np.sqrt(flows[:,:,:,0]**2+flows[:,:,:,1]**2),0),vmin = 0, vmax = 0.3)
        pl.colorbar()
        