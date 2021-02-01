#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:36:12 2020
This file creates figure 4 c,d and e
sim3_10 - sim3_16: simulation with different spike amplitude
sim4_1 - sim4_15: simulation with different overlaps
@author: nel
"""
import numpy as np
import os
import sys
#sys.path.append('/home/nel/Code/NEL_LAB/volpy/figures/figure3_performance')

import scipy.io
import shutil

import caiman as cm
from caiman.source_extraction.volpy.spikepursuit import signal_filter
from caiman.base.rois import nf_match_neurons_in_binary_masks
from demo_voltage_simulation import run_volpy
from caiman_on_voltage import run_caiman
from pca_ica import run_pca_ica
from utils import normalize, flip_movie, load_gt, extract_spikes
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from match_spikes import match_spikes_greedy, compute_F1
from scipy.signal import find_peaks
from simulation_sgpmd import run_sgpmd_demixing

ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation'
SAVE_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/simulation'
#ROOT_FOLDER = '/home/nel/data/voltage_data/simul_electr/kaspar/Ground truth data/'

#%% Fig4 c left, F1 score with best threshold on non-overlapping neurons
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/thresh_best_v2.0'
files = np.array(sorted(os.listdir(result_folder)))[np.array([-1, 0, 3, 5, 2, 1])]#[np.array([7, 1, 4, 6])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
data_points = []
for idx, results in enumerate(result_all):
    try:
        if idx == 0:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15', linewidth = 3)
        else:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='15')
    except:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='.', markersize='15')    
    #plt.errorbar(x, [np.array(result['F1']).sum()/10 for result in results], 
    #                 [np.std(np.array(result['F1'])) for result in results], 
    #                 solid_capstyle='projecting', capsize=3)
    data_points.append([np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']])
    plt.legend(['VolPy', 'CaImAn', 'SGPMD-NMF', 'Suite2p', 'PCA-ICA', 'MeanROI'])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score with best threshold')

#plt.savefig(os.path.join(SAVE_FOLDER, 'F1_VolPy_vs_CaImAn_best_threshold_v5.0.pdf'))

#%%
labels = ['VolPy', 'CaImAn', 'SGPMD-NMF', 'Suite2p', 'PCA-ICA', 'MeanROI']
results = result_all.copy()
df1 = pd.DataFrame({})
for idx1, s1 in enumerate(labels):
    for idx2, s2 in enumerate(x):
        rr = results[idx1].item()['result'][idx2]['F1'].copy()
        if len(rr) < 10:
            rr = rr+ [np.nan] * (10-len(rr))
        df1[s1 + '_' + str(s2)] = rr

df2 = pd.DataFrame({'spike amplitude':x,'VolPy':data_points[0], 'CaImAn':data_points[1], 
                   'SGPMD-NMF':data_points[2], 'Suite2p':data_points[3], 
                   'PCA-ICA':data_points[4], 'MeanROI':data_points[5]})

dfs = [df1,df2]
text = 'Average F1 score against ground truth in function of spike amplitude. All algorithms (including VolPy) were evaluated with the optimal threshold.'
fig_name = 'Fig 4c left'
excel_name = os.path.join(excel_folder, 'volpy_data.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%% Fig4 c right, F1 score with automatic threshold on non-overlapping neurons for VolPy and SpikePursuit
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/thresh_best_v2.0'
files = np.array(sorted(os.listdir(result_folder)))[np.array([6, 4])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
cmap = plt.get_cmap("tab10")
colors = [0,1,2]
data_points = []
for idx, results in enumerate(result_all):
    try:
        if idx == 0:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='10',color=cmap(colors[idx]), linewidth=3)
            data_points.append([np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']])
        elif idx == 1:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='10',color=cmap(colors[idx]), linestyle='--')
            data_points.append([np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']])
        else:
            plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results.item()['result']], marker='.', markersize='10',color=cmap(colors[idx]))
    except:
        plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in results], marker='.', markersize='10', color=cmap(colors[idx]), linestyle='--')        #plt.errorbar(x, [np.array(result['F1']).sum()/10 for result in results], 
        data_points.append([np.array(result['F1']).sum()/len(result['F1']) for result in results])
    #          [np.std(np.array(result['F1'])) for result in results], 
    #                 solid_capstyle='projecting', capsize=3)
    
    plt.legend(['VolPy_adaptive_threshold', 'SpikePursuit',  'VolPy_simple_3.0'])
    #plt.legend([file for file in files])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score with best threshold')

#plt.savefig(os.path.join(SAVE_FOLDER, 'F1_VolPy_vs_SpikePursuit_best_threshold_v5.0.pdf'))

#%% 
labels = ['VolPy_adaptive_threshold', 'SpikePursuit']
results = result_all.copy()
df1 = pd.DataFrame({})
for idx1, s1 in enumerate(labels):
    for idx2, s2 in enumerate(x):
        try:
            rr = results[idx1].item()['result'][idx2]['F1'].copy()
        except:
            rr = results[idx1][idx2]['F1'].copy()
        if len(rr) < 10:
            rr = rr+ [np.nan] * (10-len(rr))
        df1[s1 + '_' + str(s2)] = rr

df2 = pd.DataFrame({'spike amplitude':x,'VolPy_adaptive_threshold':data_points[0], 'SpikePursuit':data_points[1]})
dfs = [df1,df2]
text = 'Comparison with SpikePursuit, adaptive threshold in both cases'
fig_name = 'Fig 4c right'
excel_name = os.path.join(excel_folder, 'volpy_data.xlsx')
# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)
    
#%% Fig4 d, SpNR on non-overlapping neurons for VolPy and SpikePursuit
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/spnr_v2.0'
files = np.array(sorted(os.listdir(result_folder)))[np.array([7, 0, 3, 5, 2, 1])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
data_points = []

for idx, results in enumerate(result_all):
    plt.plot(x, [np.array(result).sum()/len(result) for result in results],  marker='.', markersize=10) 
    data_points.append([np.array(result).sum()/len(result) for result in results])
    plt.legend(['VolPy', 'CaImAn', 'SGPMD-NMF', 'Suite2p', 'PCA-ICA', 'MeanROI'])
    plt.xlabel('spike amplitude')
    plt.ylabel('SpNR')
    plt.title('SpNR')
    
#plt.savefig(os.path.join(SAVE_FOLDER, 'SpNR_VolPy_vs_CaImAn_v5.0.pdf'))

#%%
labels = ['VolPy', 'CaImAn', 'SGPMD-NMF', 'Suite2p', 'PCA-ICA', 'MeanROI']
results = result_all.copy()
df1 = pd.DataFrame({})
for idx1, s1 in enumerate(labels):
    for idx2, s2 in enumerate(x):
        rr = results[idx1][idx2].copy()
        if len(rr) < 10:
            rr = rr+ [np.nan] * (10-len(rr))
        df1[s1 + '_' + str(s2)] = rr

df2 = pd.DataFrame({'spike amplitude':x,'VolPy':data_points[0], 'CaImAn':data_points[1], 
                   'SGPMD-NMF':data_points[2], 'Suite2p':data_points[3], 
                   'PCA-ICA':data_points[4], 'MeanROI':data_points[5]})

dfs = [df1,df2]
text = 'Spike-to-noise ratio (SpNR) in function of spike amplitude'
fig_name = 'Fig 4d'
excel_name = os.path.join(excel_folder, 'volpy_data.xlsx')
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%% Fig4 e Overlapping neurons for volpy with different overlapping areas
names = [f'sim4_{i}' for i in range(1, 16)]
distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]
area = []

for idx, dist in enumerate(distance):  
    select = np.arange(idx * 3, (idx + 1) * 3)
    v_result_all = []
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        spatial, temporal, spikes = load_gt(folder)
        percent = (np.where(spatial.sum(axis=0) > 1)[0].shape[0])/(np.where(spatial.sum(axis=0) > 0)[0].shape[0])
        area.append(percent)   
        
area = np.round(np.unique(area), 2)[::-1]

result_all = {}
data_points = []
distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]
x = [round(0.075 + 0.05 * i, 3) for i in range(3)] 
for dist in distance:   
    result_folder = f'/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result_overlap/{dist}'
    files = np.array(sorted(os.listdir(result_folder)))#[0]#[np.array([5, 0, 1, 2, 3, 4, 6])]
    files = [file for file in files if 'adaptive' in file]
    result_all[files[0]] = np.load(os.path.join(result_folder, files[0]), allow_pickle=True)
    #result_all[files[1]] = np.load(os.path.join(result_folder, files[1]), allow_pickle=True)
    
    
for idx, key in enumerate(result_all.keys()):
    results = result_all[key]
    print(results)
    plt.plot(x, [np.array(result['F1']).sum()/2 for result in results.item()['result']], marker='.',markersize=15, label=f'{area[idx]:.0%}')
    data_points.append([np.array(result['F1']).sum()/2 for result in results.item()['result']])
    plt.legend()
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('VolPy F1 score with different overlapping areas')
    
#plt.savefig(os.path.join(SAVE_FOLDER, 'F1_overlapping_VolPy_v5.0.pdf'))

#%%
labels = ['35%', '26%', '19%', '6%', '0%']
results = result_all.copy()
df1 = pd.DataFrame({})
for idx1, s1 in enumerate(results.keys()):
    for idx2, s2 in enumerate(x):
        rr = results[s1].item()['result'][idx2]['F1'].copy()
        df1[labels[idx1] + '_' + str(s2)] = rr

df2 = pd.DataFrame({'spike amplitude':x,'35%':data_points[0], '26%':data_points[1], 
                   '19%':data_points[2], '6%':data_points[3], '0%':data_points[4]})

dfs = [df1,df2]
text = 'Evaluation of VolPy~on overlapping neurons. Average F1 score detecting spikes in function of spike amplitude and overlap between two neurons'
fig_name = 'Fig 4e'
excel_name = os.path.join(excel_folder, 'volpy_data.xlsx')
# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)        

################################################################################################################
#%% Below includes steps to generate simulation movies, prepares files, run different algorithms
# Generate simulation movie
plt.imshow(m[0]);plt.colorbar()
plt.imshow(m[0, 170:220, 75:125]); plt.colorbar()
mm = m[5000:10000, 170:220, 77:127]
plt.imshow(mm.mean(0)); plt.colorbar()
mm.play()
mm = mm.transpose([1,2,0])
scipy.io.savemat('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/sim3/sim3_raw.mat', 
                 {'data':mm, "sampleRate":400.0})

#%% Move file in folders
names = [f'sim3_{i}' for i in range(10, 17)]
names = [f'sim4_{i}' for i in range(10, 16)]
for name in names:
    try:
        os.makedirs(os.path.join(ROOT_FOLDER, name))
        print('make folder')
    except:
        print('already exist')
    files = [file for file in os.listdir(ROOT_FOLDER) if '.mat' in file and name[4:] in file]
    for file in files:
        shutil.move(os.path.join(ROOT_FOLDER, file), os.path.join(ROOT_FOLDER, name, file))

#%% save in .hdf5
#names = [f'sim3_{i}' for i in range(10, 17)]
#names = [f'sim4_{i}' for i in range(1, 4)]
for name in names:
    fnames_mat = f'/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/{name}/{name}.mat'
    m = scipy.io.loadmat(fnames_mat)
    m = cm.movie(m['dataAll'].transpose([2, 0, 1]))
    fnames = fnames_mat[:-4] + '.hdf5'
    m.save(fnames)
    
#%% 
for name in names:
    try:
        """
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'caiman'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'volpy'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'pca-ica'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'sgpmd'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'spikepursuit'))
        """
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'suite2p'))
        print('make folder')
    except:
        print('already exist')
    
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, f'{name}.hdf5')
    m  = cm.load(fnames)
    mm = flip_movie(m)
    mm.save(os.path.join(folder, 'caiman', name+'_flip.hdf5'))
    
#%%
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, f'{name}.hdf5')
    m  = cm.load(fnames)
    mm = m.transpose([1, 2, 0])    
    scipy.io.savemat(os.path.join(ROOT_FOLDER, name, 'spikepursuit', f'{name}.mat'), 
                 {'data':mm, "sampleRate":400.0})

    
#%% move to volpy folder
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    volpy_folder = os.path.join(folder, 'volpy')
    files = os.listdir(folder)
    files = [file for file in files if '.hdf5' not in file and '.mat' not in file]
    files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    for file in files:
        shutil.move(os.path.join(folder, file), os.path.join(volpy_folder, file))
        
#%% 
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    volpy_folder = os.path.join(folder, 'volpy')
    files = os.listdir(folder)
    files = [file for file in files if '.hdf5' in file and 'flip' not in file]
    files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    for file in files:
        shutil.copyfile(os.path.join(folder, file), os.path.join(volpy_folder, file))
        
#%%
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    sgpmd_folder = os.path.join(folder, 'sgpmd')
    s2p_folder = os.path.join(folder, 'suite2p')
    files = os.listdir(sgpmd_folder)
    files = [file for file in files if 'sim' in file]
    for file in files:
        shutil.copyfile(os.path.join(sgpmd_folder, file), os.path.join(s2p_folder, file))
        
#%%
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    s_folder = os.path.join(ROOT_FOLDER, name, 'sgpmd')
    file = f'{name}.hdf5'
    m = cm.load(os.path.join(folder, file))
    m.save(os.path.join(s_folder, name+'.tif'))
    
#%%
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    spatial, temporal, spikes = load_gt(folder)  
    #ROIs = spatial.transpose([1,2,0])
    ROIs = spatial.copy()
    
    volpy_folder = os.path.join(folder, 'volpy')
    np.save(os.path.join(volpy_folder, 'ROIs_gt'), ROIs)
    
#%% volpy params
#for ridge_bg in [0.5, 0.1, 0.01, 0.001, 0]:
    context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
    flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    threshold_method = 'adaptive_threshold'                   # 'simple' or 'adaptive_threshold'
    min_spikes= 30                                # minimal spikes to be found
    threshold = 3.0                               # threshold for finding spikes, increase threshold to find less spikes
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

    #%% run volpy
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in names:
        folder = os.path.join(ROOT_FOLDER, name)
        volpy_folder = os.path.join(folder, 'volpy')
        fnames = os.path.join(volpy_folder, f'{name}.hdf5')
        run_volpy(fnames, options=options, do_motion_correction=False, do_memory_mapping=False)
    
#%% run caiman
#names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, 'caiman', f'{name}_flip.hdf5')
    run_caiman(fnames)
    
#%% run pca-ica
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, f'{name}.hdf5')
    run_pca_ica(fnames)
    
#%% run sgpmd need invivo environment
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    run_sgpmd_demixing(folder)

################################################################################################################
#%% Below inlcudes steps to compute F1 score for different algorithms against the ground truth
thresh_list = np.arange(2.0, 4.1, 0.1)
#thresh_list = np.array([3.0])
#thresh_list = np.round(thresh_list, 2)

#%% VolPy
#for ridge_bg in [0.5, 0.1, 0.01, 0.001, 0]:
names = [f'sim4_{i}' for i in range(1, 16)]
distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]

for idx, dist in enumerate(distance):  
    select = np.arange(idx * 3, (idx + 1) * 3)
    v_result_all = []
    snr_all = []
    best_thresh_all = []
    #names = [f'sim3_{i}' for i in range(10, 17)]
    #for name in np.array(names):#[select]:
    for name in np.array(names)[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        spatial, temporal, spikes = load_gt(folder)
        summary_file = os.listdir(os.path.join(folder, 'volpy'))
        summary_file = [file for file in summary_file if 'summary' in file][0]
        summary = cm.load(os.path.join(folder, 'volpy', summary_file))
        
        v_folder = os.path.join(folder, 'volpy')
        v_files = sorted([file for file in os.listdir(v_folder) if 'adaptive_threshold_3.0' in file and f'bg_{ridge_bg}.npy' in file])#f'simple_3.0' in file and f'bg_{ridge_bg}.npy' in file]) ##f'simple_3.0' in file and f'bg_{ridge_bg}.npy' in file]) # 'simple_3.0' #'adaptive' in file and '0.1'
        #v_files = sorted([file for file in os.listdir(v_folder) if 'adaptive' in file])
        #v_files = sorted([file for file in os.listdir(v_folder) if 'NMF' in file])
        v_file = v_files[0]
        v = np.load(os.path.join(v_folder, v_file), allow_pickle=True).item()
        
        v_spatial = v['weights'].copy()
        v_temporal = v['ts'].copy()
        v_ROIs = v['ROIs'].copy()
        v_ROIs = v_ROIs * 1.0
        v_templates = v['templates'].copy()
        v_spikes = v['spikes'].copy()
        
        #plt.figure(); plt.suptitle(f'distance:{dist}');plt.subplot(1,2,1);plt.imshow(v_spatial[0]); plt.subplot(1,2,2);plt.imshow(v_spatial[1]);
        #plt.figure(); plt.suptitle(f'distance:{dist}');plt.subplot(1,2,1);plt.imshow(v_spatial[0]); plt.subplot(1,2,2);plt.imshow(v_spatial[1]);
        
#%%     
        if name == 'sim3_10':
            spatial = spatial[np.array([0, 1, 3, 4, 5, 8, 9])]
            spikes = spikes[np.array([0, 1, 3, 4, 5, 8, 9])]


        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                spatial, v_ROIs, thresh_cost=0.7, min_dist=10, print_assignment=True,
                plot_results=True, Cn=summary[2], labels=['gt', 'volpy'])   
        
        #plt.savefig(os.path.join(SAVE_FOLDER, 'simulation_Mask_RCNN_result_spikeamp_0.1_corr.pdf'))
        
        v_temporal = v_temporal[tp_comp]
        v_templates = v_templates[tp_comp]
        v_spikes = v_spikes[tp_comp]
        v_spatial = v_spatial[tp_comp]
        
        n_cells = len(tp_comp)
        v_result = {'F1':[], 'precision':[], 'recall':[]}        
        for thresh in thresh_list:
            rr = {'F1':[], 'precision':[], 'recall':[]}
            if 'simple' in v_file:
                v_spikes = extract_spikes(v_temporal, threshold=thresh)
                print('simple')
            elif 'adaptive' in v_file:
                #v_spikes = extract_spikes(v_temporal, threshold=thresh)
                print('adaptive')
                pass
            for idx in range(n_cells):
                s1 = spikes[idx].flatten()
                s2 = v_spikes[idx]
                idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
                F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
                rr['F1'].append(F1)
                rr['precision'].append(precision)
                rr['recall'].append(recall)
            if not v_result['F1']:
                v_result = rr
                best_thresh = thresh
                new_F1 = np.array(rr['F1']).sum()/n_cells
                print(f'update: thresh:{thresh}, best_F1:{new_F1}, new_F1:{new_F1}')
            else:
                best_F1 = np.array(v_result['F1']).sum()/n_cells
                new_F1 = np.array(rr['F1']).sum()/n_cells
                if new_F1 > best_F1:
                    print(f'update: thresh:{thresh}, best_F1:{best_F1}, new_F1:{new_F1}')
                    v_result = rr
                    best_thresh = thresh
        v_result_all.append(v_result)
        best_thresh_all.append(best_thresh)
        print(f'best threshold:{best_thresh_all}')       
        print(name)
        print(f'missing {10-len(tp_comp)} neurons')
        print(f"volpy average 10 neurons:{np.array(v_result['F1']).sum()/n_cells}")    

        snr = []
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            #s2 = v_spikes[idx]
            #t1 = temporal[:,idx]
            t2 = v_temporal[idx]
            t2 = normalize(t2)
            ss = np.mean(t2[s1-1])
            snr.append(ss)
        
        snr_all.append(snr)
    
    v_save_result = {'result':v_result_all, 'thresh':best_thresh_all}
    #np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'volpy_thresh_3.0'), v_save_result)
    #np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'volpy_thresh_best_adaptive_thresholding'), v_save_result)
    #np.save(os.path.join(ROOT_FOLDER, 'result', 'spnr_v2.0', f'volpy_thresh_adaptive'), snr_all)
    #np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'volpy_adaptive'), v_save_result)
    #np.save(os.path.join(ROOT_FOLDER, 'result', 'spnr_v2.0', f'volpy_adaptive'), snr_all)
    np.save(os.path.join(ROOT_FOLDER, 'result_overlap', f'{dist}', f'volpy_{dist}_thresh_adaptive'), v_save_result)
    
    
#%% CaImAn
#for threshold in np.arange(2.5, 4.1, 0.1):
#    threshold = np.round(threshold, 2)
#names = [f'sim4_{i}' for i in range(1, 16)]
#distance = [f'dist_{i}' for i in [5, 6, 7, 10, 15]]

for idx, dist in enumerate(distance):  
    select = np.arange(idx * 3, (idx + 1) * 3)
    c_result_all = []
    snr_all = []
    best_thresh_all = []
    #names = [f'sim3_{i}' for i in range(10, 17)]
    for name in np.array(names):#[select]:
        folder = os.path.join(ROOT_FOLDER, name)
        spatial, temporal, spikes = load_gt(folder)
        summary_file = os.listdir(os.path.join(folder, 'volpy'))
        summary_file = [file for file in summary_file if 'summary' in file][0]
        summary = cm.load(os.path.join(folder, 'volpy', summary_file))
    
    #%%
        c_folder = os.path.join(folder, 'caiman')
        caiman_files = [file for file in os.listdir(c_folder) if 'caiman_sim' in file][0]
        c = np.load(os.path.join(c_folder, caiman_files), allow_pickle=True).item()
        c_spatial = c.A.toarray().copy()
        c_spatial = c_spatial.reshape([50, 50, c_spatial.shape[1]], order='F').transpose([2, 0, 1])
        c_spatial_p = c_spatial.copy()
        for idx in range(len(c_spatial_p)):
            c_spatial_p[idx][c_spatial_p[idx] < c_spatial_p[idx].max() * 0.15] = 0
            c_spatial_p[idx][c_spatial_p[idx] >= c_spatial_p[idx].max() * 0.15] = 1
        c_temporal = c.C.copy()    
       
        #%%
        if name == 'sim3_10':
            spatial = spatial[np.array([0, 1, 3, 4, 5, 8, 9])]
            spikes = spikes[np.array([0, 1, 3, 4, 5, 8, 9])]
        
        plt.figure()
        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
                spatial, c_spatial_p, thresh_cost=0.7, min_dist=10, print_assignment=True,
                plot_results=True, Cn=summary[0], labels=['gt', 'caiman'])    
        
        #%%
        c_temporal = c_temporal[tp_comp]
        c_spatial = c_spatial[tp_comp]   
        #plt.figure(); plt.suptitle(f'distance:{dist}');plt.subplot(1,2,1);plt.imshow(c_spatial[0]); plt.subplot(1,2,2);plt.imshow(c_spatial[1]);
        
        #c_temporal_p = c_temporal_p[tp_comp]
        
        #%%
        #idx=0
        #plt.plot(c_temporal[idx])
        #plt.plot(signal_filter(c_temporal, freq=15, fr=400)[idx])
        
        #%%    
        n_cells = len(tp_comp)
        c_result = {'F1':[], 'precision':[], 'recall':[]}        
        for thresh in thresh_list:
            rr = {'F1':[], 'precision':[], 'recall':[]}
            c_temporal_p = signal_filter(c_temporal, freq=15, fr=400)
            c_spikes = extract_spikes(c_temporal_p, threshold=thresh)
            for idx in range(n_cells):
                s1 = spikes[idx].flatten()
                s2 = c_spikes[idx]
                idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
                F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
                rr['F1'].append(F1)
                rr['precision'].append(precision)
                rr['recall'].append(recall)
            if not c_result['F1']:
                c_result = rr
                best_thresh = thresh
                new_F1 = np.array(rr['F1']).sum()/n_cells
                print(f'update: thresh:{thresh}, best_F1:{new_F1}, new_F1:{new_F1}')
            else:
                best_F1 = np.array(c_result['F1']).sum()/n_cells
                new_F1 = np.array(rr['F1']).sum()/n_cells
                if new_F1 > best_F1:
                    print(f'update: thresh:{thresh}, best_F1:{best_F1}, new_F1:{new_F1}')
                    c_result = rr
                    best_thresh = thresh
        c_result_all.append(c_result)
        best_thresh_all.append(best_thresh)
        
        print(f'best threshold:{best_thresh_all}')
        print(name)
        print(f'missing {10-len(tp_comp)} neurons')
        print(f"caiman average 10 neurons:{np.array(c_result['F1']).sum()/n_cells}")
        snr = []
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            #s2 = c_spikes[idx]
            #t1 = temporal[:,idx]
            t2 = c_temporal_p[idx]
            t2 = normalize(t2)
            ss = np.mean(t2[s1-1])
            snr.append(ss)
        
        snr_all.append(snr)
    c_save_result = {'result':c_result_all, 'thresh':best_thresh_all}
    np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'caiman_thresh_best'), c_save_result)
    np.save(os.path.join(ROOT_FOLDER, 'result', 'spnr_v2.0', f'caiman_thresh_best'), snr_all)
        #np.save(os.path.join(ROOT_FOLDER, 'result', 'caiman_threshold', f'caiman_thresh_{np.round(threshold, 2)}'), c_result_all)
    
        #np.save(os.path.join(folder, 'caiman_F1.npy'), c_result)
        #np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'caiman_thresh_{np.round(threshold, 2)}'), c_result_all)
    #np.save(os.path.join(ROOT_FOLDER, 'result_overlap', dist, f'caiman_{dist}'), c_result_all)

#%% MeanROI
m_result_all = []
snr_all = []
best_thresh_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    spatial, temporal, spikes = load_gt(folder)  
    if name == 'sim3_10':
        spatial = spatial[np.array([0, 1, 3, 4, 5, 8, 9])]
        spikes = spikes[np.array([0, 1, 3, 4, 5, 8, 9])] 

    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))
    mov = mov.reshape([mov.shape[0], -1], order='F')
    spatial_F = [np.where(sp.reshape(-1, order='F')>0) for sp in spatial]
    m_temporal = np.array([-mov[:, sp].mean((1,2)) for sp in spatial_F])
    
#%%
    """
    idx = 2
    plt.plot(m_temporal[idx,:1000])
    spk = spikes[idx][spikes[idx]<1000]
    plt.vlines(spk, ymin=m_temporal[idx,:1000].max(), ymax=m_temporal[idx,:1000].max()+10 )
    plt.savefig(os.path.join(SAVE_FOLDER, 'eg_trace_amplitude_0.15.pdf'))
    """
#%% 
    m_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(spatial)
    for thresh in thresh_list:
        rr = {'F1':[], 'precision':[], 'recall':[]}
        m_temporal_p = signal_filter(m_temporal, freq=15, fr=400)
        m_spikes = extract_spikes(m_temporal_p, threshold=thresh)
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s2 = m_spikes[idx]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        if not m_result['F1']:
            m_result = rr
            best_thresh = thresh
            new_F1 = np.array(rr['F1']).sum()/n_cells
            print(f'update: thresh:{thresh}, best_F1:{new_F1}, new_F1:{new_F1}')
        else:
            best_F1 = np.array(m_result['F1']).sum()/n_cells
            new_F1 = np.array(rr['F1']).sum()/n_cells
            if new_F1 > best_F1:
                print(f'update: thresh:{thresh}, best_F1:{best_F1}, new_F1:{new_F1}')
                m_result = rr
                best_thresh = thresh
    m_result_all.append(m_result)
    best_thresh_all.append(best_thresh)
    
    if len(tp_comp) < 10:
        print(f'missing {10-tp_comp} neurons')
    print(f"mean roi average 10 neurons:{np.array(m_result['F1']).sum()/n_cells}")
    
    snr = []
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        #s2 = v_spikes[idx]
        #t1 = temporal[:,idx]
        t2 = m_temporal_p[idx]
        t2 = normalize(t2)
        ss = np.mean(t2[s1-1])
        snr.append(ss)
    
    snr_all.append(snr)
    
m_save_result = {'result':m_result_all, 'thresh':best_thresh_all}
np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'mean_roi_thresh_best'), m_save_result)
np.save(os.path.join(ROOT_FOLDER, 'result', 'spnr_v2.0', f'mean_roi_thresh_3.0'), snr_all)
    

    
#%% PCA-ICA
p_result_all = []
best_thresh_all = []
snr_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    p_folder = os.path.join(ROOT_FOLDER, name, 'pca-ica')
    
    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))
    spatial, temporal, spikes = load_gt(folder)    

    file = os.listdir(p_folder)[0]
    p = np.load(os.path.join(p_folder, file), allow_pickle=True).item()

    p_spatial = p['spatial'].copy()   
    p_temporal = p['temporal'].copy().T
    
    p_spatial_p = p_spatial.copy()
    for idx in range(len(p_spatial_p)):
        p_spatial_p[idx][p_spatial_p[idx] < p_spatial_p[idx].max() * 0.15] = 0
        p_spatial_p[idx][p_spatial_p[idx] >= p_spatial_p[idx].max() * 0.15] = 1
        
    if name == 'sim3_10':
        spatial = spatial[np.array([0, 1, 3, 4, 5, 8, 9])]
        spikes = spikes[np.array([0, 1, 3, 4, 5, 8, 9])]
    
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        spatial, p_spatial_p, thresh_cost=0.7, min_dist=10, print_assignment=True,
        plot_results=True, Cn=mov[0], labels=['gt', 'pca_ica'])    

    p_temporal = p_temporal[tp_comp]
    p_spatial = p_spatial[tp_comp]   


    p_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(tp_comp)
    for thresh in thresh_list:
        rr = {'F1':[], 'precision':[], 'recall':[]}
        p_temporal_p = signal_filter(p_temporal, freq=15, fr=400)
        p_spikes = extract_spikes(p_temporal_p, threshold=thresh)
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s2 = p_spikes[idx]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        if not p_result['F1']:
            p_result = rr
            best_thresh = thresh
            new_F1 = np.array(rr['F1']).sum()/n_cells
            print(f'update: thresh:{thresh}, best_F1:{new_F1}, new_F1:{new_F1}')
        else:
            best_F1 = np.array(p_result['F1']).sum()/n_cells
            new_F1 = np.array(rr['F1']).sum()/n_cells
            if new_F1 > best_F1:
                print(f'update: thresh:{thresh}, best_F1:{best_F1}, new_F1:{new_F1}')
                p_result = rr
                best_thresh = thresh
    p_result_all.append(p_result)
    best_thresh_all.append(best_thresh)
        
    print(name)
    print(f'missing {10-len(tp_comp)} neurons')
    print(f"pca-ica average 10 neurons:{np.array(p_result['F1']).sum()/n_cells}")
    print(best_thresh_all)
    
    snr = []
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        #s2 = v_spikes[idx]
        #t1 = temporal[:,idx]
        t2 = p_temporal[idx]
        t2 = normalize(t2)
        ss = np.mean(t2[s1-1])
        snr.append(ss)
    
    snr_all.append(snr)
    
p_save_result = {'result':p_result_all, 'thresh':best_thresh_all}
np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'pca_ica_thresh_best'), p_save_result)
np.save(os.path.join(ROOT_FOLDER, 'result', 'spnr_v2.0', f'pca_ica_thresh_3.0'), snr_all)
 
#%% SGPMD
#for threshold in np.arange(2.5, 4.1, 0.1):
#    print(threshold)    
s_result_all = []
best_thresh_all = []
snr_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    s_folder = os.path.join(folder, 'sgpmd', 'output')
    s_spatial = cm.load(os.path.join(s_folder, 'cell_spatial_footprints.tif'))
    s_spatial = s_spatial.reshape([48, 48, -1], order='F').transpose([2, 0, 1])
    s_spatial1 = np.zeros((s_spatial.shape[0], 50, 50))
    s_spatial1[:, 1:49, 1:49] = s_spatial    
    s_spatial = s_spatial1.copy()
    
    s_spatial_p = s_spatial.copy()
    for idx in range(len(s_spatial_p)):
        s_spatial_p[idx][s_spatial_p[idx] < s_spatial_p[idx].max() * 0.15] = 0
        s_spatial_p[idx][s_spatial_p[idx] >= s_spatial_p[idx].max() * 0.15] = 1
    s_temporal = cm.load(os.path.join(s_folder, 'cell_traces.tif'))
    
    # load gt and mov
    spatial, temporal, spikes = load_gt(folder)
    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))
    
    if name == 'sim3_10':
        spatial = spatial[np.array([0, 1, 3, 4, 5, 8, 9])]
        spikes = spikes[np.array([0, 1, 3, 4, 5, 8, 9])]
    
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
            spatial, s_spatial_p, thresh_cost=0.7, min_dist=10, print_assignment=True,
            plot_results=True, Cn=mov[0], labels=['gt', 'sgpmd'])    
    
    s_temporal = s_temporal[tp_comp]
    s_spatial = s_spatial[tp_comp]   
    #s_temporal_p = s_temporal_p[tp_comp]

    s_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(tp_comp)
    for thresh in thresh_list:
        rr = {'F1':[], 'precision':[], 'recall':[]}
        s_temporal_p = signal_filter(s_temporal, freq=15, fr=400)
        s_spikes = extract_spikes(s_temporal_p, threshold=thresh)
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s1 = np.delete(s1, np.where([s1<100])[0]) 
            s2 = s_spikes[idx] + 100
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        if not s_result['F1']:
            s_result = rr
            best_thresh = thresh
            new_F1 = np.array(rr['F1']).sum()/n_cells
            print(f'update: thresh:{thresh}, best_F1:{new_F1}, new_F1:{new_F1}')
        else:
            best_F1 = np.array(s_result['F1']).sum()/n_cells
            new_F1 = np.array(rr['F1']).sum()/n_cells
            if new_F1 > best_F1:
                print(f'update: thresh:{thresh}, best_F1:{best_F1}, new_F1:{new_F1}')
                s_result = rr
                best_thresh = thresh
    s_result_all.append(s_result)
    best_thresh_all.append(best_thresh)
    
    print(name)
    print(f'missing {10-len(tp_comp)} neurons')
    print(f"sgpmd average 10 neurons:{np.array(s_result['F1']).sum()/n_cells}")
    print(best_thresh_all)
    
    snr = []
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        s1 = np.delete(s1, np.where([s1<100])[0]) 
        s1 = s1 -100
        #s2 = v_spikes[idx]
        #t1 = temporal[:,idx]
        t2 = s_temporal_p[idx]
        t2 = normalize(t2)
        ss = np.mean(t2[s1-1])
        snr.append(ss)    
    snr_all.append(snr)
    
s_save_result = {'result':s_result_all, 'thresh':best_thresh_all}

np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'sgpmd_thresh_best'), s_save_result)
np.save(os.path.join(ROOT_FOLDER, 'result', 'spnr_v2.0', f'sgpmd_thresh_3.0'), snr_all)
    
#%% SpikePursuit
sp_result_all = []
snr_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    sp_folder = os.path.join(folder, 'spikepursuit')
    output = scipy.io.loadmat(os.path.join(sp_folder, f'spikepursuit_{name}.mat'))['output'][0]
    sp_spikes = []
    sp_temporal = []
    
    if name == 'sim3_10':
        for i in np.array([0, 1, 3, 4, 5, 8, 9]):
            sp_spikes.append(output[i]['spikeTimes'][0][0].flatten().astype(np.int16))
            sp_temporal.append(output[i]['yFilt'][0][0].flatten())
    else:
        for i in range(10):
            sp_spikes.append(output[i]['spikeTimes'][0][0].flatten().astype(np.int16))
            sp_temporal.append(output[i]['yFilt'][0][0].flatten())
    
    # load gt and mov
    spatial, temporal, spikes = load_gt(folder)
    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))
    
    if name == 'sim3_10':
        spatial = spatial[np.array([0, 1, 3, 4, 5, 8, 9])]
        spikes = spikes[np.array([0, 1, 3, 4, 5, 8, 9])]
        
    sp_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(spatial)
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        s2 = sp_spikes[idx]
        idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
        F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
        sp_result['F1'].append(F1)
        sp_result['precision'].append(precision)
        sp_result['recall'].append(recall)
    
    if len(tp_comp) < 10:
        print(f'missing {10-tp_comp} neurons')
    print(f"spikepursuit average 10 neurons:{np.array(sp_result['F1']).sum()/n_cells}")
    
    sp_result_all.append(sp_result)

    snr = []
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        #s2 = v_spikes[idx]
        #t1 = temporal[:,idx]
        t2 = -sp_temporal[idx]
        t2 = normalize(t2)
        ss = np.mean(t2[s1-1])
        snr.append(ss)
    
    snr_all.append(snr)

np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'spikepursuit'), sp_result_all)
np.save(os.path.join(ROOT_FOLDER, 'result', 'spnr_v2.0', f'spikepursuit'), snr_all)

#%% Suite2p
s2p_result_all = []
best_thresh_all = []
snr_all = []
names = [f'sim3_{i}' for i in range(10, 17)]
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    spatial, temporal, spikes = load_gt(folder)
    movie_file = [file for file in os.listdir(folder) if 'hdf5' in file and 'flip' not in file][0]
    mov = cm.load(os.path.join(folder, movie_file))

    s2p_folder = os.path.join(folder, 'suite2p', 'suite2p', 'plane0')
    #output = np.load(os.path.join(s2p_folder, f'F.npy'), allow_pickle=True).item()
    #s2p_temporal = np.load(os.path.join(s2p_folder, f'Fneu.npy'))
    s2p_temporal = -np.load(os.path.join(s2p_folder, f'F.npy'))
    s2p_spatials = []
    #np.load(os.path.join(s2p_folder, f'iscell.npy'))
    #np.load(os.path.join(s2p_folder, f'spks.npy'))
    stats = np.load(os.path.join(s2p_folder, f'stat.npy'), allow_pickle=True)
    spatial_pixs = [[stat['ypix'],stat['xpix']] for stat in stats]
    
    for spatial_pix in spatial_pixs:
        s2p_spatial = np.zeros([mov.shape[1], mov.shape[2]])
        s2p_spatial[spatial_pix[0], spatial_pix[1]] = 1
        s2p_spatials.append(s2p_spatial)
    
    s2p_spatial = np.array(s2p_spatials)
    
    if name == 'sim3_10':
        spatial = spatial[np.array([0, 1, 3, 4, 5, 8, 9])]
        spikes = spikes[np.array([0, 1, 3, 4, 5, 8, 9])]

    # load gt and mov
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
            spatial, s2p_spatial, thresh_cost=0.7, min_dist=10, print_assignment=True,
            plot_results=True, Cn=mov[0], labels=['gt', 's2p'])    
    
    #
    s2p_temporal = s2p_temporal[tp_comp]
    s2p_spatial = s2p_spatial[tp_comp]   
    #s2p_temporal_p = s2p_temporal_p[tp_comp]

    s2p_result = {'F1':[], 'precision':[], 'recall':[]}
    n_cells = len(spatial)
    for thresh in thresh_list:
        rr = {'F1':[], 'precision':[], 'recall':[]}
        s2p_temporal_p = signal_filter(s2p_temporal, freq=15, fr=400)
        s2p_spikes = extract_spikes(s2p_temporal_p, threshold=thresh)
        for idx in range(n_cells):
            s1 = spikes[idx].flatten()
            s2 = s2p_spikes[idx]
            idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=4)
            F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
            rr['F1'].append(F1)
            rr['precision'].append(precision)
            rr['recall'].append(recall)
        if not s2p_result['F1']:
            s2p_result = rr
            best_thresh = thresh
            new_F1 = np.array(rr['F1']).sum()/n_cells
            print(f'update: thresh:{thresh}, best_F1:{new_F1}, new_F1:{new_F1}')
        else:
            best_F1 = np.array(s2p_result['F1']).sum()/n_cells
            new_F1 = np.array(rr['F1']).sum()/n_cells
            if new_F1 > best_F1:
                print(f'update: thresh:{thresh}, best_F1:{best_F1}, new_F1:{new_F1}')
                s2p_result = rr
                best_thresh = thresh
    s2p_result_all.append(s2p_result)
    best_thresh_all.append(best_thresh)
    
    print(name)
    print(f'missing {10-len(tp_comp)} neurons')
    print(f"s2p average 10 neurons:{np.array(s2p_result['F1']).sum()/n_cells}")
    print(best_thresh_all)

    
    snr = []
    for idx in range(n_cells):
        s1 = spikes[idx].flatten()
        #s2 = v_spikes[idx]
        #t1 = temporal[:,idx]
        t2 = s2p_temporal_p[idx]
        t2 = normalize(t2)
        ss = np.mean(t2[s1-1])
        snr.append(ss)
    
    snr_all.append(snr)

s2p_save_result = {'result':s2p_result_all, 'thresh':best_thresh_all}
np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_best_v2.0', f'suite2p_thresh_best'), s2p_save_result)
np.save(os.path.join(ROOT_FOLDER, 'result', 'spnr_v2.0', f'suite2p_thresh_3.0'), snr_all)

#np.save(os.path.join(ROOT_FOLDER, 'result', 'thresh_3.0', f'suite2p_thresh_{np.round(thresh, 2)}'), s2p_result_all)
    
    
#np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/sim3_16/suite2p/suite2p/ops1.npy', allow_pickle=True).item()
#%%
plt.plot(normalize(s_temporal[0])); plt.plot(normalize(s_temporal_p[0]));

#%% F1 score for different background coefficients
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/volpy_ridge_bg'
files = np.array(os.listdir(result_folder))[np.array([1, 2,0,3,  4])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
for results in result_all:
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], marker='.', markersize=10)
    #plt.errorbar(x, [np.array(result['F1']).sum()/10 for result in results], 
    #                 [np.std(np.array(result['F1'])) for result in results], 
    #                 solid_capstyle='projecting', capsize=3)
    
    plt.legend([file for file in files])
    plt.xlabel('spike amplitude')
    plt.ylabel('F1 score')
    plt.title('F1 score at different bg')

#plt.savefig(os.path.join(SAVE_FOLDER, 'VolPy_ridge_vs_linear_regression.pdf'))

#%% F1 with precision/recall
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
result_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/simulation/result/thresh_3.0'
files = np.array(sorted(os.listdir(result_folder)))[np.array([4,0,3,2,1,-1])]
result_all = [np.load(os.path.join(result_folder, file), allow_pickle=True) for file in files]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']
for idx, results in enumerate(result_all):
    plt.plot(x, [np.array(result['F1']).sum()/10 for result in results], color=colors[idx])
    plt.plot(x, [np.array(result['precision']).sum()/10 for result in results], color=colors[idx], alpha=0.7, linestyle=':')
    plt.plot(x, [np.array(result['recall']).sum()/10 for result in results], color=colors[idx], alpha=0.7, linestyle='-.')
    
    li = ['volpy_3.0', 'caiman', 'mean_roi', 'pca_ica', 'sgpmd', 'volpy_adaptive']
    li = sum([[i, i+'_pre', i+'_rec'] for i in li], [])
    plt.legend(li)
    plt.xlabel('spike amplitude')
    plt.ylabel('F1/precision/recall')
    plt.title('F1/precision/recall of different methods')   

