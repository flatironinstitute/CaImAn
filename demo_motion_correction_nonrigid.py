# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
#%%
import cv2
try:
    cv2.setNumThreads(1)
except:
    print 'Open CV is naturally single threaded'
    
    
try:
    if __IPYTHON__:
        print 1
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic(u'load_ext autoreload')
        get_ipython().magic(u'autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import caiman as cm
import numpy as np
import time
import pylab as pl
import psutil
import sys
import os
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
from caiman.motion_correction import tile_and_correct#, motion_correction_piecewise
#%% in parallel
def tile_and_correct_wrapper(params):
    
    from skimage.external.tifffile import imread
    import numpy as np
    import cv2
    try:
        cv2.setNumThreads(1)
    except:
        1 #'Open CV is naturally single threaded'
        
    from caiman.motion_correction import tile_and_correct
    
    img_name,  out_fname,idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie,max_deviation_rigid,upsample_factor_grid, newoverlaps, newstrides  = params
    
    
    
    imgs = imread(img_name,key = idxs)
    mc = np.zeros(imgs.shape,dtype = np.float32)
    shift_info = []
    for count, img in enumerate(imgs): 
        if count % 10 == 0:
            print count
        mc[count],total_shift,start_step,xy_grid = tile_and_correct(img, template, strides, overlaps,max_shifts, add_to_movie=add_to_movie, newoverlaps = newoverlaps, newstrides = newstrides,\
                upsample_factor_grid= upsample_factor_grid, upsample_factor_fft=10,show_movie=False,max_deviation_rigid=max_deviation_rigid)
        shift_info.append([total_shift,start_step,xy_grid])
    outv = np.memmap(out_fname,mode='r+', dtype=np.float32, shape=shape_mov, order='F')
    outv[:,idxs] = np.reshape(mc,(len(imgs),-1),order = 'F').T
    
    return shift_info,idxs
#%%
def motion_correction_piecewise(fname,num_splits, strides, overlaps, add_to_movie=0, template = None, max_shifts = (12,12),max_deviation_rigid = 3,newoverlaps = None, newstrides = None,\
                                upsample_factor_grid = 4, order = 'F',c = None):
    '''

    '''

    with TiffFile(fname) as tf:
        d1,d2 = tf[0].shape
        T = len(tf)    
        idxs = np.array_split(range(T),num_splits)
    
    
    
    if template is None:
        raise Exception('Not implemented')
    
    
    
    shape_mov =  (d1*d2,T)
    base_name = fname[:-4]
    dims = d1,d2
    
    fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(1 if len(dims) == 2 else dims[2]) + '_order_' + str(order) + '_frames_' + str(T) + '_.mmap'
    fname_tot = os.path.join(os.path.split(fname)[0],fname_tot) 
    
    big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32, shape=shape_mov, order=order)
    
    pars = []
    
    for idx in idxs:
            pars.append([fname,big_mov.filename,idx,shape_mov, template, strides, overlaps, max_shifts, np.array(add_to_movie,dtype = np.float32),max_deviation_rigid,upsample_factor_grid, newoverlaps, newstrides ])

    t1 = time.time()
    if c is not None:
        res = c[:].map_sync(tile_and_correct_wrapper,pars)
    else:
        res = map(tile_and_correct_wrapper,pars)

    print time.time()-t1    
    
    return fname_tot, res
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
#%% set parameters and create template by rigid motion correction
#fname = 'k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif'
fname = 'M_FLUO_t.tif'
#fname = 'M_FLUO_4.tif'

m = cm.load(fname)

t1  = time.time()
mr,sft,xcr,template = m[:].copy().motion_correct(18,18,template=None)
t2  = time.time() - t1
print t2
add_to_movie = - np.min(m)
template = cm.motion_correction.bin_median(mr)
np.save('template_gc.npy',template)
#%%
## for 512 512 this seems good
overlaps = (16,16)
if template.shape == (512,512):
    strides = (128,128)# 512 512 
    #strides = (48,48)# 128 64
elif template.shape == (64,128):
    strides = (48,48)# 512 512
else:
    raise Exception('Unknown size, set manually')    

num_splits = 28 # for parallelization split the movies in  num_splits chuncks across time
newstrides = None
upsample_factor_grid = 4
fname_tot, res = motion_correction_piecewise(fname,num_splits, strides, overlaps,\
                            add_to_movie=add_to_movie, template = template, max_shifts = (12,12),max_deviation_rigid = 3,\
                            newoverlaps = None, newstrides = newstrides,\
                            upsample_factor_grid = upsample_factor_grid, order = 'F',c = c)
#%%
#%
total_shifts = []
start_steps = []
xy_grids = []
mc = np.zeros(m.shape)
for count,img in enumerate(np.array(m)):
    if count % 10  == 0:
        print count
    mc[count],total_shift,start_step,xy_grid = tile_and_correct(img, template, strides, overlaps,(12,12), newoverlaps = None, \
                newstrides = newstrides, upsample_factor_grid=upsample_factor_grid,\
                upsample_factor_fft=10,show_movie=False,max_deviation_rigid=2,add_to_movie=add_to_movie)
    
    total_shifts.append(total_shift)
    start_steps.append(start_step)
    xy_grids.append(xy_grid)

#%%
#mc = cm.load('k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034_d1_512_d2_512_d3_1_order_F_frames_3000_.mmap')

#mc = cm.load('M_FLUO_4_d1_64_d2_128_d3_1_order_F_frames_4620_.mmap')
mc = cm.load('M_FLUO_t_d1_64_d2_128_d3_1_order_F_frames_6764_.mmap')

#%%
mc.resize(1,1,.2).play(gain=10,fr = 30, offset = 0,magnification=1.)
#%%
m.resize(1,1,.2).play(gain=10,fr = 30, offset = 0,magnification=1.)
#%%
cm.concatenate([mr.resize(1,1,.5),mc.resize(1,1,.5)],axis=1).play(gain=10,fr = 100, offset = 0,magnification=3.)

#%%
import h5py
with  h5py.File('sueann_corrected.mat') as f:
    mef = np.array(f['M2'])

mef = cm.movie(mef.transpose([0,2,1]))    

#%%
cm.concatenate([mef.resize(1,1,.15),mc.resize(1,1,.15)],axis=1).play(gain=30,fr = 80, offset = 300,magnification=1.)
#%%
T,d1,d2 = np.shape(m)
shape_mov = (d1*d2,m.shape[0])

Y = np.memmap('M_FLUO_4_d1_64_d2_128_d3_1_order_F_frames_4620_.mmap',mode = 'r',dtype=np.float32, shape=shape_mov, order='F')
mc = cm.movie(np.reshape(Y,(d2,d1,T),order = 'F').transpose([2,1,0]))
mc.resize(1,1,.25).play(gain=10.,fr=50)
#%%
pl.plot(np.reshape(np.array(total_shifts),(len(total_shifts),-1))) 
#%%
m_raw = np.nanmedian(m,0)
m_rig = np.nanmedian(mr,0)
m_el = np.nanmedian(mc,0)
m_ef = np.nanmedian(mef,0)
#%%
import scipy
r_raw = []
r_rig = []
r_el = []
r_ef = []
max_shft = 12
for fr_id in range(m.shape[0]):
    fr = m[fr_id].copy()[max_shft:-max_shft,max_shft :-max_shft]
    templ_ = m_raw.copy()[max_shft:-max_shft,max_shft :-max_shft]
    r_raw.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0]) 
    
    fr = mr[fr_id].copy()[max_shft:-max_shft,max_shft :-max_shft]
    templ_ = m_rig.copy()[max_shft:-max_shft,max_shft :-max_shft]    
    r_rig.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0]) 

    fr = mc[fr_id].copy()[max_shft:-max_shft,max_shft :-max_shft]
    templ_ = m_el.copy()[max_shft:-max_shft,max_shft :-max_shft]    
    r_el.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0])        
    if 0:
        fr = mef[fr_id].copy()[max_shft:-max_shft,max_shft :-max_shft]
        templ_ = m_ef.copy()[max_shft:-max_shft,max_shft :-max_shft]    
        r_ef.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0])        

#%%
import pylab as pl
vmax=150
max_shft = 3
pl.subplot(2,3,1)
pl.scatter(r_raw,r_rig)
pl.plot([0,1],[0,1],'r--')
pl.xlabel('raw')
pl.ylabel('rigid')
pl.xlim([0,1])
pl.ylim([0,1])
pl.subplot(2,3,2)
pl.scatter(r_rig,r_el)
pl.plot([0,1],[0,1],'r--')
pl.ylabel('pw-rigid')
pl.xlabel('rigid')
pl.xlim([0,1])
pl.ylim([0,1])
#%



pl.subplot(2,3,4);
pl.title('rigid mean')
pl.imshow(np.nanmean(mr,0)[max_shft:-max_shft,max_shft:-max_shft],cmap='gray', vmax = vmax,interpolation = 'none');

pl.axis('off')
pl.subplot(2,3,5);
pl.imshow(np.nanmean(mc,0)[max_shft:-max_shft,max_shft:-max_shft],cmap='gray',vmax = vmax,interpolation = 'none')
pl.title('pw-rigid mean')
pl.axis('off')


if 0:
    pl.subplot(2,3,3);
    pl.scatter(r_el,r_ef)
    pl.plot([0,1],[0,1],'r--')
    pl.ylabel('pw-rigid')
    pl.xlabel('pw-rigid eft')
    pl.xlim([0,1])
    pl.ylim([0,1])
    
    pl.subplot(2,3,6);
    pl.imshow(np.nanmean(mef,0)[max_shft:-max_shft,max_shft:-max_shft],cmap='gray',vmax = vmax,interpolation = 'none')
    pl.title('pw-rigid eft mean')
    pl.axis('off')

#%%

pl.plot(r_raw)
pl.plot(r_rig)
pl.plot(r_el)
pl.plot(r_ef)
#%%
mc = cm.movie(mc)
mc[np.isnan(mc)] = 0
#%% play movie
(mc+add_to_movie).resize(1,1,.25).play(gain=10.,fr=50)
#%% compute correlation images
ccimage = m.local_correlations(eight_neighbours=True,swap_dim=False)
ccimage_rig = mr.local_correlations(eight_neighbours=True,swap_dim=False)
ccimage_els = mc.local_correlations(eight_neighbours=True,swap_dim=False)
ccimage_ef = mef.local_correlations(eight_neighbours=True,swap_dim=False)
#%% check correlation images
pl.subplot(2,2,1)
pl.imshow(ccimage)
pl.subplot(2,2,2)
pl.imshow(ccimage_rig)
pl.subplot(2,2,3)
pl.imshow(ccimage_els)
pl.subplot(2,2,4)
pl.imshow(ccimage_ef)
