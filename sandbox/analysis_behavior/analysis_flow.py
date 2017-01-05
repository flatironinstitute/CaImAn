# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
from __future__ import print_function
from builtins import str
from builtins import range
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF

"""
try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
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
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.behavior import behavior
#%%
from scipy.sparse import coo_matrix
#%%
def get_nonzero_subarray(arr,mask):
    
    x, y = mask.nonzero()   
    
    return arr.toarray()[x.min():x.max()+1, y.min():y.max()+1]
#%%
import cv2
def to_polar(x,y):
    mag, ang = cv2.cartToPolar(x, y)
    return mag,ang
#%%
m = cm.load_movie_chain(glob.glob('*.avi'),fr=100)
#%%
mask = coo_matrix(behavior.select_roi(m.mean(0),1)[0])
#%
ms = [get_nonzero_subarray(mask.multiply(fr),mask) for fr in m]
ms = np.dstack(ms)
ms = cm.movie(ms.transpose([2,0,1]))
#%%
of_or = cm.behavior.behavior.compute_optical_flow(ms[:15000],do_show=False,polar_coord=False) 
#min1,min0 = np.min(of[1]),np.min(of[0])
#of[1] -= min1
#of[0] -= min0
of = of_or - np.min(of_or)
#%%
spatial_filter, time_trace, norm_fact = cm.behavior.behavior.extract_components(of[:,:15000,:,:],n_components=1,verbose = True,normalize_std=False,max_iter=400)
x,y = scipy.signal.medfilt(time_trace[0],kernel_size=[1,1]).T
spatial_mask = spatial_filter[0]
spatial_mask[spatial_mask<(np.max(spatial_mask[0])*.99)] = np.nan
ofmk = of_or*spatial_mask[None,None,:,:]
#%%
mag,dirct = to_polar(x-cm.components_evaluation.mode_robust(x),y-cm.components_evaluation.mode_robust(y))
range_ = np.std(np.nanpercentile(np.sqrt(ofmk[0]**2+ofmk[1]**2),95,(1,2)))
mag = (mag/np.std(mag))*range_
mag = scipy.signal.medfilt(mag.squeeze(),kernel_size=1)
dirct = scipy.signal.medfilt(dirct.squeeze(),kernel_size=1).T

dirct[mag<1]=np.nan
#mag[mag<.25]=np.nan


pl.plot(mag,'.-')
pl.plot(dirct,'.-')
#%%
spatial_filter, time_trace, norm_fact = cm.behavior.behavior.extract_components(of[:,:2000,:,:],n_components=1,verbose = True)
mag,dirct = scipy.signal.medfilt(time_trace[0],kernel_size=[1,1]).T
mag,dirct = time_trace[0].T

dirct[mag<.25]=np.nan
pl.plot(mag,'.-')
pl.plot(dirct,'.-')