# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
#%%
try:
    if __IPYTHON__:
        print 1
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic(u'load_ext autoreload')
        get_ipython().magic(u'autoreload 2')
except NameError:
    print('Not IcPYTHON')
    pass

import caiman as cm
import numpy as np
from caiman.motion_correction import motion_correction_piecewise, tile_and_correct
import time
import pylab as pl
import cv2
from skimage.external.tifffile import TiffFile
cv2.setNumThreads(1)
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
fname = 'k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif'
fname = 'M_FLUO_4.tif'
m = cm.load(fname)

t1  = time.time()
mr,sft,xcr,template = m[:500].copy().motion_correct(18,18,template=None)
t2  = time.time() - t1
print t2
add_to_movie = - np.min(m)
template = cm.motion_correction.bin_median(mr)
#%%
## for 512 512 this seems good
overlaps = (16,16)
strides = (128,128)# 512 512 
strides = (48,48)# 128 64

num_splits = 28 # for parallelization split the movies in  num_splits chuncks across time
newstrides = None
upsample_factor_grid = 4
fname_tot, res = motion_correction_piecewise(fname,num_splits, strides, overlaps,\
                            add_to_movie=add_to_movie, template = template, max_shifts = (12,12),max_deviation_rigid = 3,\
                            newoverlaps = None, newstrides = newstrides,\
                            upsample_factor_grid = upsample_factor_grid, order = 'F',dview = dview)
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
mc = cm.load('M_FLUO_4_d1_64_d2_128_d3_1_order_F_frames_4620_.mmap')

mc.resize(1,1,1).play(gain=100,fr = 50, offset = 100,magnification=3)

#%%
T,d1,d2 = np.shape(m)
shape_mov = (d1*d2,m.shape[0])

Y = np.memmap('M_FLUO_4_d1_64_d2_128_d3_1_order_F_frames_4620_.mmap',mode = 'r',dtype=np.float32, shape=shape_mov, order='F')
mc = cm.movie(np.reshape(Y,(d2,d1,T),order = 'F').transpose([2,1,0]))
mc.resize(1,1,.25).play(gain=10.,fr=50)
#%%
cv2.setNumThreads(14)
t1 = time.time()
res = map(tile_and_correct_wrapper,pars)  
print time.time()-t1
#%% plot shifts per patch
pl.plot(np.reshape(np.array(total_shifts),(len(total_shifts),-1))) 
#%%
m_raw = np.nanmean(m,0)
m_rig = np.nanmean(mr,0)
m_el = np.nanmean(mc,0)
#%%
import scipy
r_raw = []
r_rig = []
r_el = []
for fr_id in range(m.shape[0]):
    fr = m[fr_id]
    templ_ = m_raw.copy()
    templ_[np.isnan(fr)]=0
    fr[np.isnan(fr)]=0
    r_raw.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0]) 
    fr = mr[fr_id]
    templ_ = m_rig.copy()
    templ_[np.isnan(fr)]=0
    fr[np.isnan(fr)]=0
    r_rig.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0]) 
    fr = mc[fr_id]
    templ_ = m_el.copy()
    templ_[np.isnan(fr)]=0
    fr[np.isnan(fr)]=0
    r_el.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0])        

#%%
import pylab as pl
pl.subplot(2,2,1)
pl.scatter(r_raw,r_rig)
pl.plot([0,1],[0,1],'r--')
pl.xlabel('raw')
pl.ylabel('rigid')
pl.xlim([0,1])
pl.ylim([0,1])
pl.subplot(2,2,2)
pl.scatter(r_rig,r_el)
pl.plot([0,1],[0,1],'r--')
pl.ylabel('pw-rigid')
pl.xlabel('rigid')
pl.xlim([0,1])
pl.ylim([0,1])
#%
pl.subplot(2,2,3);
pl.title('rigid mean')
pl.imshow(np.nanmean(mr,0),cmap='gray');

pl.axis('off')
pl.subplot(2,2,4);
pl.imshow(np.nanmean(mc,0),cmap='gray')
pl.title('pw-rigid mean')
pl.axis('off')


#%%
mc = cm.movie(mc)
mc[np.isnan(mc)] = 0
#%% play movie
(mc+add_to_movie).resize(1,1,.25).play(gain=10.,fr=50)
#%% compute correlation images
ccimage = m.local_correlations(eight_neighbours=True,swap_dim=False)
ccimage_rig = mr.local_correlations(eight_neighbours=True,swap_dim=False)
ccimage_els = mc.local_correlations(eight_neighbours=True,swap_dim=False)
#%% check correlation images
pl.subplot(3,1,1)
pl.imshow(ccimage)
pl.subplot(3,1,2)
pl.imshow(ccimage_rig)
pl.subplot(3,1,3)
pl.imshow(ccimage_els)
#%%
pl.subplot(2,1,1)
pl.imshow(np.mean(mr,0))
pl.subplot(2,1,2)
pl.imshow(np.mean(mc,0))


#%% TEST EFY
#import scipy
#ld =scipy.io.loadmat('comparison_1.mat')
#locals().update(ld)
#d1,d2 = np.shape(I)
#grid_size = tuple(grid_size[0].astype(np.int))
#mot_uf = mot_uf[0][0].astype(np.int)
#overlap_post =  overlap_post[0][0].astype(np.int)
#shapes = tuple(np.add(grid_size, overlap_post))
#overlaps = (overlap_post,overlap_post)
#strides = np.subtract(shapes,overlaps)
##newshapes = tuple(np.add(grid_size/mot_uf, 2*overlap_post))
#newstrides = np.add(tuple(np.divide((d1,d2),sf_up.shape[:2])),-1)
#newshapes = np.add(np.multiply(newstrides,2),+3)
#
#    
#O_and,sf_a,start_steps, xy_a  = tile_and_correct_faster(I, temp, shapes,overlaps,max_shifts = (15,15),newshapes = newshapes, newstrides= newstrides,show_movie=False,max_deviation_rigid=2, add_to_movie = 0, upsample_factor_grid=4)
#print np.nanmean(O_and),np.nanmean(I),np.mean(O[O!=0])
##%
#
#vmin,vmax = np.percentile(sf_up[:,:,0,0],[1,99])
#pl.subplot(2,1,1)
#img_sh = np.zeros( np.add(xy_a[-1],1))
#for s,g in  zip (sf_a,xy_a): 
#    img_sh[g] = s[0]  
#    
#pl.imshow(img_sh,interpolation = 'none',vmin = vmin,vmax = vmax)
#pl.subplot(2,1,2)
#pl.imshow(sf_up[:,:,0,0],interpolation = 'none',vmin = vmin,vmax = vmax)
##%
#newtemp = temp.copy()
##newtemp[np.isnan(O_and)] = 0
#O_and[np.isnan(O_and)] = 0
#print  scipy.stats.pearsonr(O_and.flatten(),newtemp.flatten())
#newtemp = temp.copy()
##newtemp[O==0] = 0
#
#print scipy.stats.pearsonr(O.flatten(),newtemp.flatten())
##%
#lq,hq = np.percentile(O,[1,99])
#pl.subplot(2,1,1)
#pl.imshow(O_and,vmin = lq, vmax =hq)
#pl.subplot(2,1,2)
#pl.imshow(O,vmin = lq, vmax =hq)