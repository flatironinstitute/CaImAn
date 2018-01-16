#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
from __future__ import print_function
#%%
from builtins import range
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import caiman as cm
import numpy as np
from caiman.motion_correction import tile_and_correct
import time
import pylab as pl
#%% set parameters and create template
#m = cm.load('M_FLUO_t.tif')
m = cm.load('M_FLUO_4.tif')[:1000]
#m = cm.load('k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif')[:1000]
#m = cm.load('CA1_green.tif')
#m = cm.load('ExampleStack1.tif')
t1 = time.time()
mr, sft, xcr, template = m.copy().motion_correct(18, 18, template=None)
t2 = time.time() - t1
print(t2)
add_to_movie = - np.min(m)

#%%
t1 = time.time()
# granule
#shapes = (32+16,32+16)
overlaps = (16, 16)
strides = (32, 32)
#newshapes = (32,32)
newstrides = None
# Sue Ann
#overlaps = (32,32)
#strides = (128,128)

#%
total_shifts = []
start_steps = []
xy_grids = []
mc = np.zeros(m.shape)
for count, img in enumerate(np.array(m)):
    if count % 10 == 0:
        print(count)
    mc[count], total_shift, start_step, xy_grid = tile_and_correct(img, template, strides, overlaps, (12, 12), newoverlaps=None,
                                                                   newstrides=newstrides, upsample_factor_grid=4,
                                                                   upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=add_to_movie)

    total_shifts.append(total_shift)
    start_steps.append(start_step)
    xy_grids.append(xy_grid)

#%%
pl.plot(np.reshape(np.array(total_shifts), (len(total_shifts), -1)))
#%%
m_raw = np.nanmean(m, 0)
m_rig = np.nanmean(mr, 0)
m_el = np.nanmean(mc, 0)
#%%
import scipy
r_raw = []
r_rig = []
r_el = []
for fr_id in range(m.shape[0]):
    fr = m[fr_id]
    templ_ = m_raw.copy()
    templ_[np.isnan(fr)] = 0
    fr[np.isnan(fr)] = 0
    r_raw.append(scipy.stats.pearsonr(fr.flatten(), templ_.flatten())[0])
    fr = mr[fr_id]
    templ_ = m_rig.copy()
    templ_[np.isnan(fr)] = 0
    fr[np.isnan(fr)] = 0
    r_rig.append(scipy.stats.pearsonr(fr.flatten(), templ_.flatten())[0])
    fr = mc[fr_id]
    templ_ = m_el.copy()
    templ_[np.isnan(fr)] = 0
    fr[np.isnan(fr)] = 0
    r_el.append(scipy.stats.pearsonr(fr.flatten(), templ_.flatten())[0])

#%%
import pylab as pl
pl.subplot(2, 2, 1)
pl.scatter(r_raw, r_rig)
pl.plot([0, 1], [0, 1], 'r--')
pl.xlabel('raw')
pl.ylabel('rigid')
pl.xlim([0, 1])
pl.ylim([0, 1])
pl.subplot(2, 2, 2)
pl.scatter(r_rig, r_el)
pl.plot([0, 1], [0, 1], 'r--')
pl.ylabel('pw-rigid')
pl.xlabel('rigid')
pl.xlim([0, 1])
pl.ylim([0, 1])
#%
pl.subplot(2, 2, 3)
pl.title('rigid mean')
pl.imshow(np.mean(mr, 0), cmap='gray', vmax=300)

pl.axis('off')
pl.subplot(2, 2, 4)
pl.imshow(np.mean(mc, 0), cmap='gray', vmax=300)
pl.title('pw-rigid mean')
pl.axis('off')


#%%
mc = cm.movie(mc)
mc[np.isnan(mc)] = 0
mc.resize(1, 1, .25).play(gain=10., fr=50)
#%%
ccimage = m.local_correlations(eight_neighbours=True, swap_dim=False)
ccimage_rig = mr.local_correlations(eight_neighbours=True, swap_dim=False)
ccimage_els = mc.local_correlations(eight_neighbours=True, swap_dim=False)
#%%
pl.subplot(3, 1, 1)
pl.imshow(ccimage)
pl.subplot(3, 1, 2)
pl.imshow(ccimage_rig)
pl.subplot(3, 1, 3)
pl.imshow(ccimage_els)
#%%
pl.subplot(2, 1, 1)
pl.imshow(np.mean(mr, 0))
pl.subplot(2, 1, 2)
pl.imshow(np.mean(mc, 0))


#%% TEST EFY
#import scipy
#ld =scipy.io.loadmat('comparison_1.mat')
# locals().update(ld)
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
# print np.nanmean(O_and),np.nanmean(I),np.mean(O[O!=0])
# %
#
#vmin,vmax = np.percentile(sf_up[:,:,0,0],[1,99])
# pl.subplot(2,1,1)
#img_sh = np.zeros( np.add(xy_a[-1],1))
# for s,g in  zip (sf_a,xy_a):
#    img_sh[g] = s[0]
#
#pl.imshow(img_sh,interpolation = 'none',vmin = vmin,vmax = vmax)
# pl.subplot(2,1,2)
#pl.imshow(sf_up[:,:,0,0],interpolation = 'none',vmin = vmin,vmax = vmax)
# %
#newtemp = temp.copy()
##newtemp[np.isnan(O_and)] = 0
#O_and[np.isnan(O_and)] = 0
# print  scipy.stats.pearsonr(O_and.flatten(),newtemp.flatten())
#newtemp = temp.copy()
##newtemp[O==0] = 0
#
# print scipy.stats.pearsonr(O.flatten(),newtemp.flatten())
# %
#lq,hq = np.percentile(O,[1,99])
# pl.subplot(2,1,1)
#pl.imshow(O_and,vmin = lq, vmax =hq)
# pl.subplot(2,1,2)
#pl.imshow(O,vmin = lq, vmax =hq)
