#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 20:20:27 2020

@author: nel
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from caiman.source_extraction.volpy.spikepursuit import signal_filter
from caiman.base.rois import extractROIsFromPCAICA
import numpy as np
import caiman as cm
import os

#%%
def run_pca_ica(fnames):
    m = cm.load(fnames)
    
    # run pca-ica
    output, _ = m.IPCA_stICA(componentsICA=15, mu=0.05)
    masks = output.copy()
    masks = np.array(extractROIsFromPCAICA(masks)[0])
    masks = masks / np.linalg.norm(masks, ord='fro', axis=(1,2))[:, np.newaxis, np.newaxis]
    spatial = masks.copy()
    
    plt.imshow(spatial.sum(0));plt.show()

    # from masks recover signal
    temporal = m.extract_traces_from_masks(masks)
    temporal = -signal_filter(temporal.T, freq=15, fr=400).T
    
    result = {'spatial':spatial, 'temporal':temporal}
    save_path = os.path.join(os.path.split(fnames)[0], 'pca-ica', f'pca-ica_{os.path.split(fnames)[1][:-5]}')
    np.save(save_path, result)

#%%
if __name__ == 'main':
    n_cells = 64
    for idx in range(n_cells):
        plt.figure(); plt.imshow(spatial[idx,:,:]);plt.colorbar()
        plt.figure(); plt.plot(temporal[:, idx])
        
    #%%
    row = 4; col = 4
    fig, ax = plt.subplots(row, col)
    for idx in range(n_cells):
        h = int(idx/col)
        w = idx - h * col
        ax[h,w].imshow(spatial[idx,:,:])
        ax[h,w].get_yaxis().set_visible(False)
        ax[h,w].get_xaxis().set_visible(False)
        ax[h,w].spines['right'].set_visible(False)
        ax[h,w].spines['top'].set_visible(False)  
        ax[h,w].spines['left'].set_visible(False)
        ax[h,w].spines['bottom'].set_visible(False)
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/other/pca-ica_demo_spatial.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/other/pca-ica_fish_spatial_mu_0.05.pdf')
       
    fig, ax = plt.subplots(row, col)
    for idx in range(n_cells):
        h = int(idx/col)
        w = idx - h * col
        ax[h,w].plot(temporal[:2000,idx], linewidth=0.1)
        ax[h,w].get_yaxis().set_visible(False)
        ax[h,w].get_xaxis().set_visible(False)
        ax[h,w].spines['right'].set_visible(False)
        ax[h,w].spines['top'].set_visible(False)  
        ax[h,w].spines['left'].set_visible(False)
        ax[h,w].spines['bottom'].set_visible(False)
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/other/pca-ica_demo_temporal.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/other/pca-ica_fish_temporal_mu_0.05.pdf')

#%%




