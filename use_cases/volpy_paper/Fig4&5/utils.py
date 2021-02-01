#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:31:28 2020

@author: nel
"""

#%%
import numpy as np
import scipy.io
import os
from scipy.signal import find_peaks

#%%
def normalize(data):
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns))
    data_norm = data/std 
    return data_norm

def flip_movie(m):
    mm = -m.copy()
    mm = mm - mm.min(axis=0)    
    return mm

def load_gt(folder):
    gt_files = [file for file in os.listdir(folder) if 'SimResults' in file]
    gt_file = gt_files[0]
    gt = scipy.io.loadmat(os.path.join(folder, gt_file))
    gt = gt['simOutput'][0][0]['gt']
    spikes = gt['ST'][0][0][0]
    spatial = gt['IM2'][0][0]
    temporal = gt['trace'][0][0][:,:,0]
    spatial[spatial <= np.median(spatial) * 5] = 0
    spatial[spatial > 0] = 1
    spatial = spatial.transpose([2, 0, 1])
    return spatial, temporal, spikes

def extract_spikes(traces, threshold):
    spikes = []
    for cc in traces:
        cc = normalize(cc)
        spk = find_peaks(cc, threshold)[0]
        spikes.append(spk)
    return spikes

    