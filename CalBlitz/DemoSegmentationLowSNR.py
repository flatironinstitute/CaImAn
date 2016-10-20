# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015
@author: agiovann
"""
#% add required packages
import h5py
import calblitz as cb
import time
import pylab as pl
import numpy as np
import cv2
#% set basic ipython functionalities
#pl.ion()
#%load_ext autoreload
#%autoreload 2


#%%
filename='ac_001_001.tif'
frameRate=30;
start_time=0;
#%% load and motion correct movie (see other Demo for more details)
m=cb.load(filename, fr=frameRate,start_time=start_time);
template_before=np.mean(m,0)
m=m-np.min(np.mean(m,axis=0))
m=m.gaussian_blur_2D(kernel_size_x=5,kernel_size_y=5,kernel_std_x=1,kernel_std_y=1,borderType=cv2.BORDER_REPLICATE)
#%% automatic parameters motion correction
max_shift_h=15;
max_shift_w=15;
m,shifts,xcorrs,template=m.motion_correct(max_shift_w=max_shift_w,max_shift_h=max_shift_h, num_frames_template=None, template = None,method='opencv')
pl.plot(shifts)
#%% apply shifts to the original movie to not loose resolution
m=cb.load(filename, fr=frameRate,start_time=start_time);
m=m.apply_shifts(shifts,interpolation='linear', method='opencv')
max_h,max_w= np.max(shifts,axis=0)
min_h,min_w= np.min(shifts,axis=0)
m=m.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)
#%%
template_after=np.mean(m,0)
pl.subplot(2,1,1)
pl.title('Before motion correction')
pl.imshow(template_before,cmap=pl.cm.gray)
pl.subplot(2,1,2)
pl.title('After motion correction')
pl.imshow(template_after,cmap=pl.cm.gray)
#%%
(m-np.min(m)).play(backend='opencv',fr=100,gain=10.,magnification=1)
