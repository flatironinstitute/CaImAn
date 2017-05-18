#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:17:33 2017

@author: agiovann
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
import glob
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
#%%
import caiman as cm
import numpy as np
import os
import glob
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
#%%
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.utils import download_demo
#%%
with np.load('/opt/local/privateCaImAn/SUE_MAY15_UPDATE_SHAPES_NOSTATS_SHIFTS20/results_analysis_online_SUE_LAST_117k.npz') as ld:
    locals().update(ld)
    Ab = Ab[()]
    
#%%
base_folder = '/opt/local/privateCaImAn/SUE_MAY15_UPDATE_SHAPES_NOSTATS_SHIFTS20'
files_to_compare = ['results_analysis_online_SUE_6000.npz','results_analysis_online_SUE_10000.npz','results_analysis_online_SUE_114000.npz']

base_folder = '/opt/local/privateCaImAn/JEFF_MAY_14_AFT_BETTER_INIT_UPDATES_NO_STATS/'
files_to_compare = ['results_analysis_online_JEFF_init_.npz','results_analysis_online_JEFF_LAST_6000.npz','results_analysis_online_JEFF_LAST_90000.npz']

count = 0
As = []
bs = []
Cs = []
fs = []
Cns = []
noisyCs = []
num_frames = []
for file_ in files_to_compare:
    print(file_)
    with np.load(os.path.join(base_folder,file_)) as ld:
        locals().update(ld)
        Ab = Ab[()]
        A,b = Ab[:,:-1],Ab[:,-1].toarray()
        C, f = Cf[:-1,:], Cf[-1:,:]        
        As.append(A)
        bs.append(b)
        Cs.append(C)
        fs.append(f)
        Cns.append(Cn)
        noisyCs.append(noisyC)
        num_frames.append(np.where(~np.isnan(noisyC.sum(0)))[0][-1]+1)
        count += 1
        pl.subplot(1,3,count)
        crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9, vmax =.75)
#        pl.xlim([200,400]);pl.ylim([200,400])
#        pl.subplot(2,3,2*count)
#        pl.imshow(A.sum(0).reshape(dims,))

#%%
pl.figure()
count = 0
idx_neuro = 10
neuron_groups = [[180],[183,277],[183,277,709]]
for A,b,C,f, Cn,ftc,noisyC, nfr,ngrp in zip(As,bs,Cs,fs,Cns,files_to_compare, noisyCs, num_frames,neuron_groups):
    count+=1
    a = A.tocsc()[:,np.array(ngrp)-1]
    pl.subplot(3,3,count)
#    pl.imshow(Cn,vmax = 0.7)
    crd = cm.utils.visualization.plot_contours(a, Cn, thr=0.9, vmax =.7, colors = 'r')

    pl.ylabel('Correlation Image')
    pl.xlim([200,400]);pl.ylim([200,400])
#    pl.colorbar()
    pl.title('#frames:'+str(nfr))
    pl.subplot(3,3,3+count)
    pl.ylabel('Spatial Components')
    a = a.sum(1).reshape(dims,order='F')
    a[a==0] = np.nan
    pl.imshow(A.tocsc().sum(1).reshape(dims,order='F'),vmax = .2)
    pl.imshow(a,vmax = .1, alpha =.6, cmap = 'hot')
    
    pl.xlim([200,400]);pl.ylim([200,400])
#    pl.colorbar()
    pl.subplot(3,3,6+count)
    if count == 3:
        pl.plot(np.mean(np.reshape(C[np.array(ngrp)-1,:nfr],(3,10,-1),order='F'),axis=1).T)
    else:
        pl.plot(C[np.array(ngrp)-1,:nfr].T)
#    if nfr<=3000:
#        pl.plot(noisyC[ngrp,:nfr].T)
#    else:
#        pl.plot(noisyC[ngrp,:nfr].T)
    pl.xlabel('frames')
    pl.ylabel('A.U.')
#    pl.legend(['Denoised','Raw'])
    
    
