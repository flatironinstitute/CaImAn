#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:29:14 2020
This produces Fig 5c,d
@author: agiovann & caichangjia
"""
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.volpy.spikepursuit import signal_filter
from demo_voltage_simulation import run_volpy
from scipy.signal import find_peaks
from skimage import measure
import matplotlib
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/multiple_neurons'
SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/real_data'
names = np.array(sorted(os.listdir(ROOT_FOLDER)))
names = names[np.array([1, 0, 2])]

#%% Some functions
def normalize(data):
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns))
    data_norm = data/std 
    return data_norm

def compute_SNR(s1, s2, height=3.5, do_plot=True):
    s1 = normalize(s1)
    s2 = normalize(s2)
    pks1 = find_peaks(s1, height)[0]
    pks2 = find_peaks(s2, height)[0]
    both_found = np.intersect1d(pks1,pks2)

    if do_plot:
        plt.figure()
        plt.plot(s1)
        plt.plot(s2)
        plt.plot(both_found, s1[both_found],'o')
        plt.plot(both_found, s2[both_found],'o')
        plt.legend(['volpy','caiman','volpy_peak','caiman_peak'])
    
    print(len(both_found))
    print([np.mean(s1[both_found]),np.mean(s2[both_found])])
    snr = [np.mean(s1[both_found]),np.mean(s2[both_found])]
    if len(both_found) < 10:
        snr = [np.nan, np.nan]
    #print([np.std(s1[both_found]),np.std(s2[both_found])])
    return snr

def compute_SNR_S(S, height=3.5, do_plot=True, volpy_peaks=None):
    pks = {}
    dims = S.shape
    for idx, s in enumerate(S):
        S[idx] = normalize(s)
        pks[idx] = set(find_peaks(s, height)[0])
    
    if volpy_peaks is not None:
        pks[0] = set(volpy_peaks)
        print('use_volpy_peaks')
        
    found = np.array(list(set.intersection(pks[0], pks[1], pks[2], pks[3])))    

    if do_plot:
        plt.figure()
        plt.plot(S.T)
        plt.plot(found, S[0][found],'o')
        plt.legend(['volpy','caiman','mean_roi','sgpmd'])
    
    print(len(found))
    snr = np.mean(S[:, found], 1)
    print(np.round(snr, 2))
    if len(found) < 10:
        snr = [np.nan] * dims[0]
    #print([np.std(s1[both_found]),np.std(s2[both_found])])
    return snr

#%% This produces Fig 5c
idx_list = list(range(s1.shape[0]))

colorsets = plt.cm.tab10(np.linspace(0,1,10))
colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]

scope = [0, 20000]
if s1.shape[0] > 300:
    s1 = s1[np.newaxis,:]
    s2 = s2[np.newaxis,:]
    s3 = s3[np.newaxis,:]
    s4 = s4[np.newaxis,:]

dims = s1.shape

fig, ax = plt.subplots(dims[0],1)
ax = [ax]
for idx in idx_list:
    ax[idx].plot(normalize(s1[idx]), 'c', linewidth=0.5, color='black')
    ax[idx].plot(normalize(s2[idx]), 'c', linewidth=0.5, color='red')
    ax[idx].plot(normalize(s3[idx]), 'c', linewidth=0.5, color='orange')
    ax[idx].plot(normalize(s4[idx]), 'c', linewidth=0.5, color='green')
    if idx == 0:
        ax[idx].legend(['volpy', 'caiman', 'mean', 'sgpmd'])
    
    height = 3.5
    #pks1 = find_peaks(normalize(s1[idx]), height)[0]
    pks1 = vpy['spikes'][idx_volpy[idx]]
    pks2 = find_peaks(normalize(s2[idx]), height)[0]
    pks3 = find_peaks(normalize(s3[idx]), height)[0]
    pks4 = find_peaks(normalize(s4[idx]), height)[0]
    
    add = 15
    
    ax[idx].vlines(pks1, add+3.5, add+4, color='black')
    ax[idx].vlines(pks2, add+2.5, add+3, color='red')
    ax[idx].vlines(pks3, add+1.5, add+2, color='orange')
    ax[idx].vlines(pks4, add+0.5, add+1, color='green')
    
    #ax[idx].set_xlim([3000, 5000])
    #ax[idx].set_xlim([500, 1000])
    ax[idx].set_xlim([8000, 9000])
    
    
    
    if idx<dims[0]-1:
        ax[idx].get_xaxis().set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        ax[idx].spines['top'].set_visible(False) 
        ax[idx].spines['bottom'].set_visible(False) 
        ax[idx].spines['left'].set_visible(False) 
        ax[idx].set_yticks([])
    
    if idx==dims[0]-1:
        ax[idx].legend()
        ax[idx].spines['right'].set_visible(False)
        ax[idx].spines['top'].set_visible(False)  
        ax[idx].spines['left'].set_visible(True) 
        ax[idx].set_xlabel('Frames')
    
    ax[idx].set_ylabel('o')
    ax[idx].get_yaxis().set_visible(True)
    ax[idx].yaxis.label.set_color(colorsets[np.mod(idx,9)])
    plt.savefig(os.path.join(SAVE_FOLDER, 'IVQ32_S2_FOV1_temporal_v1.3.pdf'))
    #plt.savefig(os.path.join(SAVE_FOLDER, 'FOV4_50um_temporal_v1.3.pdf'))
    #plt.savefig(os.path.join(SAVE_FOLDER, '06152017_FIsh1-2_temporal_v1.3.pdf'))
#%%
Cn = mov[0]
vmax = np.percentile(Cn, 99)
vmin = np.percentile(Cn, 5)
plt.figure()
plt.imshow(Cn, interpolation='None', vmax=vmax, vmin=vmin, cmap=plt.cm.gray)
plt.title('Neurons location')
d1, d2 = Cn.shape
#cm1 = com(mask.copy().reshape((N,-1), order='F').transpose(), d1, d2)
colors='yellow'
for n, idx in enumerate(idx_list):
    contours = measure.find_contours(mask[idx], 0.5)[0]
    plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(n,9)])
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/FOV4_50um_footprints.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/06152017Fish1-2_footprints.pdf')   
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/multiple_neurons/IVQ32_S2_FOV1_footprints.pdf')   
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/IVQ32_S2_FOV1_spatial.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/06152017_FIsh1-2_spatial.pdf')
    #plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/figure3/FOV4_50um_spatial.pdf')

#%% This produces Fig 5d
snr1 = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/FOV4_50um_snr_v1.3.npy')
snr2 = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/06152017Fish1-2_snr_v1.3.npy')
snr3 = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/IVQ32_S2_FOV1_snr_v1.3.npy')

labels = ['L1', 'TEG', 'HPC']
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
mean = np.stack([snr1.mean(0),snr2.mean(0), snr3])
std = np.stack([snr1.std(0),snr2.std(0), [np.nan]*4])


fig, ax = plt.subplots()
rects1 = ax.bar(x - 3* width/2, mean[:,0], width, yerr=std[:,0], label='VolPy', color='black')
rects2 = ax.bar(x - 1*width/2, mean[:,1], width, yerr=std[:,1], label='CaImAn', color='red')
rects3 = ax.bar(x + 1*width/2, mean[:, 2], width,yerr=std[:,2], label='mean_roi', color='yellow')
rects4 = ax.bar(x + 3*width/2, mean[:, 3], width,yerr=std[:,3], label='sgpmd', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SNR')
ax.set_title('SNR for different methods on real data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/real_data/snr_real_data_v1.3.pdf')

#%% 
methods = ['VolPy', 'CaImAn', 'MeanROI', 'SGPMD']
df1 = pd.DataFrame({})
for idx, data in enumerate([snr1, snr2, snr3]):
    if len(data.shape) < 2:
        data = data[None, :]
    for idx2, s2 in enumerate(methods):
        rr = data[:,idx2]
        if len(rr) < 8:
            rr = list(rr)+ [np.nan] * (8-len(rr))
        df1[labels[idx] + '_' + s2] = rr

df2 = pd.DataFrame({'files':labels,'VolPy_mean':mean[:,0], 'CaImAn_mean':mean[:,1], 
                   'MeanROI_mean':mean[:,2], 'SGPMD-NMF_mean':mean[:,3], 
                   'VolPy_std':std[:,0], 'CaImAn_std':std[:,1], 
                   'stdROI_std':std[:,2], 'SGPMD-NMF_std':std[:,3]})

dfs = [df1,df2]
text = 'Spike to noise ratio (SpNR) for each considered algorithm and dataset type'
fig_name = 'Fig 5d'
excel_name = os.path.join(excel_folder, 'volpy_data.xlsx')
multiple_dfs(dfs, fig_name, excel_name, 2, text)        


###################################################################################################################################
#%% Below includes steps to run VolPy, load VolPy, CaImAn and SGPMD results
context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
flip_signal = False                            # Important!! Flip signal or not, True for Voltron indicator, False for others
hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
threshold_method = 'adaptive_threshold'                   # 'simple' or 'adaptive_threshold'
min_spikes= 30                                # minimal spikes to be found
threshold = 4                               # threshold for finding spikes, increase threshold to find less spikes
do_plot = True                               # plot detail of spikes, template for the last iteration
ridge_bg= 0.01                                 # ridge regression regularizer strength for background removement, larger value specifies stronger regularization 
sub_freq = 20                                 # frequency for subthreshold extraction
weight_update = 'ridge'                       # 'ridge' or 'NMF' for weight update
n_iter = 2

options={'context_size': context_size,
           'flip_signal': flip_signal,
           'hp_freq_pb': hp_freq_pb,
           'threshold_method': threshold_method,
           'min_spikes':min_spikes,
           'threshold': threshold,
           'do_plot':do_plot,
           'ridge_bg':ridge_bg,
           'sub_freq': sub_freq,
           'weight_update': weight_update,
           'n_iter': n_iter}

#%%
for idx, name in enumerate(names):
    name = names[idx]
    file = ['06152017Fish1-2_portion', 'FOV4_50um', 'IVQ32_S2_FOV1_processed'][idx]
    fr = [300, 400, 1000][idx]
    folder = os.path.join(ROOT_FOLDER, name)
    path = os.path.join(folder, 'volpy')
    fnames = os.path.join(path, f'{file}.hdf5')
    run_volpy(fnames, options=options, do_motion_correction=False, do_memory_mapping=False, fr=fr)    

#%% VolPy, CaImAn and SGPMD file
idx_file = 2
fr = [400, 300, 1000][idx_file]
files = ['FOV4_50um', '06152017Fish1-2_portion', 'IVQ32_S2_FOV1_processed']
fnames = [os.path.join(ROOT_FOLDER, names[i], files[i]+'.hdf5')for i in range(3)][idx_file]
mask_name = [os.path.join(ROOT_FOLDER, names[i], files[i]+'_ROIs.hdf5')for i in range(3)][idx_file]
mov = cm.load(fnames)
mask = cm.load(mask_name)

fl = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/FOV4_50um/volpy/volpy_FOV4_50um_adaptive_threshold_4_ridge_bg_0.01.npy', 
      '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/06152017Fish1-2/volpy/volpy_06152017Fish1-2_portion_adaptive_threshold_4_ridge_bg_0.01.npy',
      '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/multiple_neurons/IVQ32_S2_FOV1/volpy/volpy_IVQ32_S2_FOV1_processed_adaptive_threshold_4_ridge_bg_0.01.npy'][idx_file]
fl2 = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/caiman_FOV4_50um.npy',
       '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons/caiman_06152017Fish1-2.npy',
       '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/caiman_IVQ32_S2_FOV1.npy'][idx_file]
fl3 = ['/home/nel/Code/volpy_test/invivo-imaging/test_data/L1_all/output/cell_traces.tif',
       '/home/nel/Code/volpy_test/invivo-imaging/test_data/TEG_small/output/cell_traces.tif', 
       '/home/nel/Code/volpy_test/invivo-imaging/test_data/HPC/output/nmf_traces.tif'][idx_file]
fl3_spatial = ['/home/nel/Code/volpy_test/invivo-imaging/test_data/L1_all/output/spatial_footprints.tif', 
               '/home/nel/Code/volpy_test/invivo-imaging/test_data/TEG_small/output/spatial_footprints.tif',
               '/home/nel/Code/volpy_test/invivo-imaging/test_data/HPC/output/spatial_footprints.tif'][idx_file]

#%% HPC data
vpy  = np.load(fl, allow_pickle=True)[()]
#idx_list = np.where(np.array([vpy['spikes'][i].shape[0] for i in range(len(vpy['spikes']))])>50)[0]
s1 = vpy['ts'][0]
idx_volpy = [0]
plt.figure(); plt.plot(s1) # after whitened matched filter
plt.figure(); plt.imshow(vpy['weights'][0])

caiman_estimates = np.load(fl2, allow_pickle=True)[()]
idx = 2
plt.figure();plt.plot(caiman_estimates.C[idx])    
s2 = signal_filter(caiman_estimates.C, freq=15, fr=fr)[idx]
plt.plot(s2);
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape(caiman_estimates.dims, order='F'))

idx = 1
s3 = signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr)
plt.plot(s3)

idx = 1
s4 = signal_filter(cm.load(fl3).T, freq=15, fr=fr)[idx]
s4_spatial = cm.load(fl3_spatial)
plt.plot(s4)

sf = 101
s4 = np.append(np.array([0]*sf), s4)
S = np.vstack([s1, s2, s3, s4])

snr_all = compute_SNR_S(S, height=3.5, volpy_peaks=vpy['spikes'][0])
np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/IVQ32_S2_FOV1_snr_v1.3.npy', snr_all)

#%% TEG
fr = 300
vpy  = np.load(fl, allow_pickle=True)[()]
idx_list = np.where(np.array([vpy['spikes'][i].shape[0] for i in range(len(vpy['spikes']))])>30)[0]
idx_list = [0, 5]
for idx in idx_list:
    #idx = idx_list[2]
    s1 = vpy['ts'][idx]
    plt.figure(); plt.plot(s1) # after whitened matched filter
    plt.figure(); plt.imshow(vpy['weights'][idx])
idx_volpy = [5, 4]
s1 = vpy['ts'][np.array([5,4])]

caiman_estimates = np.load(fl2, allow_pickle=True)[()]
plt.figure();plt.plot(caiman_estimates.C[idx])    
s2 = signal_filter(caiman_estimates.C, freq=15, fr=fr)[idx]
plt.plot(s2);
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape(caiman_estimates.dims, order='F'))
s2 = signal_filter(caiman_estimates.C[np.array([13, 9])], freq=15, fr=fr)

idx = 0
s3 = -signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr)
plt.plot(s3)

s3 = np.array([-signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr) for idx in [0, 5]])

s4 = signal_filter(cm.load(fl3).T[:,np.array([1,0])].T, freq=15, fr=fr)
s4_spatial = cm.load(fl3_spatial)
plt.plot(s4.T)

sf = 100
s4 = np.hstack((np.zeros((2, 100)), s4))
snr_all = []
#idx = 0
for idx in [0,1]:
    S = np.vstack([s1[idx], s2[idx], s3[idx], s4[idx]])
    snr_all.append(compute_SNR_S(S, height=3.5, volpy_peaks=vpy['spikes'][idx_volpy[idx]]))
np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/06152017Fish1-2_snr_v1.3.npy', snr_all)


#%% L1
fr = 400
vpy  = np.load(fl, allow_pickle=True)[()]
idx_list = np.where(np.array([vpy['spikes'][i].shape[0] for i in range(len(vpy['spikes']))])>100)[0]
#idx_list = [0, 5, 6]
idx_list = np.array([ 0,  3, 12, 16, 18, 20, 26, 27])
for idx in idx_list:
    #idx = idx_list[2]
    print(idx)
    s1 = vpy['ts'][idx]
    plt.figure(); plt.plot(s1);plt.show() # after whitened matched filter
    plt.figure(); plt.imshow(vpy['weights'][idx]);plt.show()
#plt.imshow(vpy['weights'][idx_list].sum(0), cmap='gray')
idx_volpy = np.array([13,  0,  4, 12, 16,  5, 14, 18])
s1 = vpy['ts'][np.array(idx_volpy)]
#s1_spatial = vpy['weights'][np.array(idx_volpy)]
#s1_spatial = s1_spatial/s1_spatial.max((1,2))[:, np.newaxis, np.newaxis]

caiman_estimates = np.load(fl2, allow_pickle=True)[()]
plt.figure();plt.plot(caiman_estimates.C[idx])    
s2 = signal_filter(caiman_estimates.C, freq=15, fr=fr)[idx]
plt.plot(s2);
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape(caiman_estimates.dims, order='F'))
idx_caiman = np.array([161,  41, 168, 167, 172,  96, 101, 163])
s2 = signal_filter(caiman_estimates.C[idx_caiman], freq=15, fr=fr)


s3 = -signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr)
plt.plot(s3)
s3 = np.array([-signal_filter(mov[:,mask[idx]>0].mean(1), freq=15, fr=fr) for idx in idx_list])

idx_sgpmd = np.array([ 9, 13,  3,  1,  7,  4,  8,  5])
s4 = signal_filter(cm.load(fl3).T[:,idx_sgpmd].T, freq=15, fr=fr)
s4_spatial = cm.load(fl3_spatial)
s4_spatial = s4_spatial.reshape([512,128, 20], order='F')
s4_spatial = s4_spatial / s4_spatial.max((0,1))
s4_spatial = s4_spatial.transpose([2, 0, 1])
plt.plot(s4.T)

mask0 = np.array(mask[idx_list])
mask1 = vpy['weights']
mask1[mask1<0] = 0
mask1 = mask1 / mask1.max((1,2))[:, np.newaxis, np.newaxis]
        
from caiman.base.rois import nf_match_neurons_in_binary_masks
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        mask0, mask1, thresh_cost=1, min_dist=10, print_assignment=True,
        plot_results=True, Cn=mov[0], labels=['viola', 'cm'])    

sf = 100
s4 = np.hstack((np.zeros((s4.shape[0], sf)), s4))
s4 = s4[:,:10000]
s1 = s1[:,:10000]
s2 = s2[:,:10000]
s3 = s3[:,:10000]

snr_all = []
#idx = 0
for idx in range(s1.shape[0]):
    print(idx)
    S = np.vstack([s1[idx], s2[idx], s3[idx], s4[idx]])
    
    snr_all.append(compute_SNR_S(S, height=3.5, volpy_peaks=vpy['spikes'][idx_volpy[idx]]))
np.save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_result/multiple_neurons/FOV4_50um_snr_v1.3.npy', snr_all)

#%% function for matching spatial footprints
idx = 0
plt.figure();plt.plot(signal_filter(caiman_estimates.C, freq=15, fr=400)[idx])
plt.figure();plt.imshow(caiman_estimates.A[:,idx].toarray().reshape((512, 128), order='F'))
    
mask0 = np.array(mask[idx_list])
mask1 = caiman_estimates.A.toarray()[:, caiman_estimates.idx].reshape((512, 128, -1), order='F').transpose([2, 0, 1])
mask1[mask1>0.02] = 1
mask1 = np.float32(mask1)
plt.figure();plt.imshow(mask0.sum(0));plt.colorbar();plt.show()
        
from caiman.base.rois import nf_match_neurons_in_binary_masks
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        mask0, mask1, thresh_cost=1, min_dist=10, print_assignment=True,
        plot_results=True, Cn=mov[0], labels=['viola', 'cm'])    

