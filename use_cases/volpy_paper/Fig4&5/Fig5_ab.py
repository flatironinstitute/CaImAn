# -*- coding: utf-8 -*-
"""
This file produces Fig5 a,b and Fig S5 for volpy paper.
@author: caichangjia
"""
#%%
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os
import scipy.signal
import shutil
import imageio
from match_spikes import match_spikes_greedy, compute_F1
from scipy import io

from utils import normalize, flip_movie
from demo_voltage_simulation import run_volpy
import caiman as cm
from caiman.base.rois import nf_read_roi_zip
from caiman.source_extraction.volpy.spikepursuit import signal_filter

ROOT_FOLDER = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/ephys_voltage'
names = os.listdir(ROOT_FOLDER)
names = [name for name in names if 'Fish' in name or 'Mouse' in name]
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/ephys_voltage/result_v2'
save_folder_graph = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/test_data/ephys_voltage/result_graph_v2'

#%% functions for match spikes for Fig 5a
def match_ephys_voltage(t1, t2, s1, s2, timepoint, max_dist=None, scope=None, hline=None):
    """ Match spikes from ground truth data and spikes from inference.
    Args:
        t1, t2: ephys/voltage signal
        s1, s2: ephys/voltage spikes
        timepoint: map between gt signal and inference signal (normally not one to one)
        scope: the scope of signal need matching
        hline: threshold of gt signal
    return: F1, precision and recall
    """
    # Adjust signal and spikes to the scope
    height = np.max(np.array(t1.max(), t2.max()))
    t1 = t1[scope[0]:scope[1]]
    s1 = s1[np.where(np.logical_and(s1>scope[0], s1<scope[1]))]-scope[0]    
    t2 = t2[np.where(np.multiply(timepoint>=scope[0],timepoint<scope[1]))[0]]
    s2 = np.array([timepoint[i] - scope[0] for i in s2 if timepoint[i]>=scope[0] and timepoint[i]<scope[1]])
    time = [i-scope[0] for i in timepoint if i < scope[1] and i >=scope[0]]
    
    # Find matches and compute metric
    idx1, idx2 = match_spikes_greedy(s1, s2, max_dist=max_dist)
    F1, precision, recall = compute_F1(s1, s2, idx1, idx2)
    print(f'F1:{F1}, precision:{precision}, recall:{recall}')
    
    # Plot signals and spikes
    plt.plot(t1, color='b', label='ephys')
    plt.plot(s1, 1.2*height*np.ones(s1.shape),color='b', marker='.', ms=2, fillstyle='full', linestyle='none')
    plt.plot(time, t2, color='orange', label='VolPy')
    plt.plot(s2, 1*height*np.ones(len(s2)),color='orange', marker='.', ms=2, fillstyle='full', linestyle='none')
    plt.hlines(hline, 0, len(t1), linestyles='dashed', color='gray')
    ax = plt.gca()
    ax.locator_params(nbins=7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i in range(len(idx1)):
        plt.plot((s1[idx1[i]], s2[idx2[i]]),(1.15*height, 1.05*height), color='gray',alpha=0.5, linewidth=1)

    return F1, precision, recall 

#%% Johannes's data
name = names[1]
if 'Fish2-2' in name: # TEG 2
    fr = 300
    dims = (32, 64)
    s_dims = (32, 64)
    scope = [0, 600000]
    max_dist = 60

    hline = -0.6
    m_height = 0.5
    c_height = 0.65
    v_idx = 0
    c_idx = 0
    s_idx = 1
    s_height = 0.55

elif 'Fish1-1' in name: # TEG 1
    fr = 300
    dims = (44, 128)
    s_dims = (44,120)
    scope = [0, 200000]
    max_dist = 60
    hline = -0.4
    m_height = 0.3
    c_height = 0.65
    v_idx = 7
    c_idx = 5
    s_idx = 1
    s_height = 0.45
    
elif 'Mouse' in name: # L1 1
    fr = 500
    dims = (80, 128)
    s_dims = (80,120)
    scope = [2000000,3000000]
    max_dist = 200
    hline = 0.5
    m_height = 0.5
    c_height = 0.5
    v_idx = 2
    c_idx = -5
    s_idx = 1
    s_height = 0.5
    
#%% Load ground truth signal
folder = os.path.join(ROOT_FOLDER, name)
path = os.path.join(folder, 'gt')
ephys = np.load(os.path.join(path, 'ephys.npy'))
frame_timing = np.load(os.path.join(path,'frame_timing.npy'))
timepoint = frame_timing
imaging_signal = np.load(os.path.join(path, 'imaging_signal.npy'))
ephys = ephys/np.max(ephys)
plt.plot(ephys);plt.hlines(hline, 0, len(ephys), linestyles='dashed', color='gray')
espikes = scipy.signal.find_peaks(-ephys, -hline, distance=int(6000*0.01))[0]

# VolPy result
path = os.path.join(folder, 'volpy')
files = os.listdir(path)
files = [file for file in files if 'volpy' in file if 'adaptive' in file and '.npy' in file][0]
v = np.load(os.path.join(path, files), allow_pickle=True).item()
v_temporal = v['ts'][v_idx][:len(frame_timing)]
v_temporal = v_temporal - np.mean(v_temporal)
v_spikes = v['spikes'][v_idx].copy()
v_spikes = np.delete(v_spikes, np.where(v_spikes>len(frame_timing))).astype(np.int32)
v_spatial = v['weights'][v_idx].copy()
v_temporal = v_temporal/np.max(v_temporal)
plt.figure(); plt.plot(v_temporal); plt.figure(); plt.imshow(v_spatial)

# Mean ROI result
folder = os.path.join(ROOT_FOLDER, name)
masks = nf_read_roi_zip(os.path.join(folder, 'mask.zip'), dims=dims)
masks_m = masks[0].reshape(-1, order='F')
mov_path = os.listdir(os.path.join(folder, 'volpy'))
mov_path = [p for p in mov_path if 'F_frames' in p][0]
m = cm.load(os.path.join(folder, 'volpy', mov_path))
mm = m.reshape((m.shape[0], -1), order='F')
m_temporal = (mm[:, masks_m>0]).mean(1)
m_temporal = signal_filter(-m_temporal[np.newaxis, :], freq=15, fr=fr)[0]
m_temporal = m_temporal / m_temporal.max()
m_spikes = scipy.signal.find_peaks(m_temporal, m_height, distance=int(fr*0.01))[0]
m_spikes = np.delete(m_spikes, np.where(m_spikes>len(frame_timing))).astype(np.int32)
plt.plot(m_temporal)

# CaImAn result
folder = os.path.join(ROOT_FOLDER, name)
path = os.path.join(folder, 'caiman')
files = os.listdir(path)
files = [file for file in files if 'caiman' in file and '.npy' in file][0]
c = np.load(os.path.join(path, files), allow_pickle=True).item()
c_temporal = signal_filter(c.C, freq=15, fr=fr)[c_idx]
c_temporal = c_temporal / c_temporal.max()
c_spatial = c.A.toarray().reshape((dims[0], dims[1], -1), order='F')[:,:, c_idx]
c_spikes = scipy.signal.find_peaks(c_temporal, c_height, distance=int(fr*0.01))[0]
c_spikes = np.delete(c_spikes, np.where(c_spikes>len(frame_timing))).astype(np.int32)
plt.figure(); plt.imshow(c_spatial);plt.figure();plt.plot(c_temporal)

# SGPMD-NMF result
folder = '/home/nel/Code/volpy_test/invivo-imaging/test_data/'
path_temporal = os.path.join(folder, name, 'output', 'cell_traces.tif')
path_spatial = os.path.join(folder, name, 'output', 'cell_spatial_footprints.tif')
s_spatial = imageio.imread(path_spatial)[:, s_idx].reshape(s_dims, order='F')
s_temporal = imageio.imread(path_temporal)[s_idx]
s_temporal = np.concatenate((np.array([0]*100), s_temporal))[:len(frame_timing)]
s_temporal = signal_filter(s_temporal, freq=15, fr=fr)
s_temporal = s_temporal / np.max(s_temporal)
s_spikes = scipy.signal.find_peaks(s_temporal, s_height, distance=int(fr*0.01))[0]
plt.figure();plt.imshow(s_spatial);plt.figure();plt.plot(s_temporal, label='pmd')

#%% This produces Fig 5a fish data result
for i in range(1):
    method = ['volpy', 'mean_roi', 'caiman', 'sgpmd'][i]
    
    t1 = ephys; s1 = espikes
    dictionary = {'volpy':[v_temporal, v_spikes, v_spatial], 'mean_roi':[m_temporal, m_spikes, None], 
                  'caiman':[c_temporal, c_spikes, c_spatial], 'sgpmd':[s_temporal, s_spikes, s_spatial]}
    
    F1, precision, recall = match_ephys_voltage(t1=ephys, s1=espikes, t2=dictionary[method][0], 
                                                     s2=dictionary[method][1], timepoint=timepoint, 
                                                     max_dist=max_dist, scope=scope, hline=hline) 
    
    result = {'F1':F1, 'precision':precision, 'recall':recall}
    plt.savefig(os.path.join(ROOT_FOLDER, name, method, f'{method}_{name}_adaptive_temporal_trace.pdf'))
    try:
        plt.figure(); plt.imshow(dictionary[method][2]); plt.savefig(os.path.join(ROOT_FOLDER, name, method, f'{method}_{name}_spatial.pdf'))
    except:
        print('no spatial footprint')
    plt.close()
    np.save(os.path.join(save_folder, f'{method}_{name}_adaptive.npy'), result)

#%% This code produces spatial footprints in Fig 5
plt.imshow(v_spatial)
plt.savefig(os.path.join(path, f'volpy_{name}_adaptive_spatial_footprint.pdf'))   
#%% Kaspar's data 
name = names[2]
fr = 500
dims = (80, 128)
s_dims = (80,120)
max_dist = 200
hline = 0.5
m_height = 0.5
c_height = 0.5
v_idx = 2
c_idx = -5
s_idx = 1
s_height = 0.5
scope = [2000000,3000000]

#
folder = os.path.join(ROOT_FOLDER, name)
path = os.path.join(folder, 'gt')
f = io.loadmat(os.path.join(path, os.listdir(path)[0]))
frames = np.where(np.logical_and(f['read_starts'][0]>scope[0], f['read_starts'][0]<scope[1]))[0]
timepoint = np.array([f['read_starts'][0][frames[i]]-scope[0] for i in range(frames.shape[0])])
ephys = f['v'][0][scope[0]:scope[-1]]
ephys = ephys - np.mean(ephys)
ephys = ephys/np.max(ephys)
espikes = scipy.signal.find_peaks(ephys, 0.5, distance=20000*0.01)[0]

#
path = os.path.join(folder, 'volpy')
files = os.listdir(path)
files = [file for file in files if 'volpy' in file and 'adaptive' in file and '.npy' in file][0]
v = np.load(os.path.join(path, files), allow_pickle=True).item()
v_temporal = v['ts'][v_idx]#[:len(frame_timing)]
v_temporal = v_temporal - np.mean(v_temporal)
v_spikes = v['spikes'][v_idx].copy()
v_spatial = v['weights'][v_idx].copy()
v_temporal = v_temporal/np.max(v_temporal)
plt.figure(); plt.imshow(v_spatial); plt.figure(); plt.plot(v_temporal)

#
folder = os.path.join(ROOT_FOLDER, name)
masks = nf_read_roi_zip(os.path.join(folder, 'mask.zip'), dims=dims)
masks_m = masks[0].reshape(-1, order='F')
mov_path = os.listdir(os.path.join(folder, 'volpy'))
mov_path = [p for p in mov_path if 'F_frames' in p][0]
m = cm.load(os.path.join(folder, 'volpy', mov_path))
mm = m.reshape((m.shape[0], -1), order='F')
m_temporal = (mm[:, masks_m>0]).mean(1)
m_temporal = signal_filter(-m_temporal[np.newaxis, :], freq=15, fr=fr)[0]
m_temporal = m_temporal / m_temporal.max()
plt.plot(m_temporal)
m_spikes = scipy.signal.find_peaks(m_temporal, m_height, distance=int(fr*0.01))[0]

#
folder = os.path.join(ROOT_FOLDER, name)
path = os.path.join(folder, 'caiman')
files = os.listdir(path)
files = [file for file in files if 'caiman' in file and '.npy' in file][0]
c = np.load(os.path.join(path, files), allow_pickle=True).item()
c_temporal = signal_filter(c.C, freq=15, fr=fr)[c_idx]
c_temporal = c_temporal / c_temporal.max()
c_spatial = c.A.toarray().reshape((dims[0], dims[1], -1), order='F')
c_spatial = c_spatial[:,:,c_idx]
#plt.imshow(c_spatial[:,:, c_idx])
c_spikes = scipy.signal.find_peaks(c_temporal, c_height, distance=int(fr*0.01))[0]
#c_spikes = np.delete(c_spikes, np.where(c_spikes>len(frame_timing))).astype(np.int32)
plt.figure(); plt.imshow(c_spatial); plt.figure(); plt.plot(c_temporal)

#
folder = '/home/nel/Code/volpy_test/invivo-imaging/test_data/'
path_temporal = os.path.join(folder, name, 'output', 'cell_traces.tif')
path_spatial = os.path.join(folder, name, 'output', 'cell_spatial_footprints.tif')
s_spatial = imageio.imread(path_spatial)[:, s_idx].reshape(s_dims, order='F')
s_temporal = imageio.imread(path_temporal)[s_idx]
s_temporal = signal_filter(s_temporal, freq=15, fr=fr)
s_temporal = np.concatenate((np.array([0]*100), s_temporal))
s_temporal = s_temporal / np.max(s_temporal)
s_spikes = scipy.signal.find_peaks(s_temporal, s_height, distance=int(fr*0.01))[0]

plt.figure(); plt.imshow(s_spatial); plt.figure(); plt.plot(s_temporal, label='pmd')

#%% This produces Fig 5a mouse data result
scope = [0, 1000000]

for i in range(4):
    method = ['volpy', 'mean_roi', 'caiman', 'sgpmd'][i]
    
    t1 = ephys; s1 = espikes
    dictionary = {'volpy':[v_temporal, v_spikes, v_spatial], 'mean_roi':[m_temporal, m_spikes, None], 
                  'caiman':[c_temporal, c_spikes, c_spatial], 'sgpmd':[s_temporal, s_spikes, s_spatial]}
    
    F1, precision, recall = match_ephys_voltage(t1=ephys, s1=espikes, t2=dictionary[method][0], 
                                                     s2=dictionary[method][1], timepoint=timepoint, 
                                                     max_dist=max_dist, scope=scope, hline=hline) 
    
    result = {'F1':F1, 'precision':precision, 'recall':recall}
    plt.savefig(os.path.join(ROOT_FOLDER, name, method, f'{method}_{name}_adaptive_temporal_trace.pdf'))
    try:
        plt.figure(); plt.imshow(dictionary[method][2]); plt.savefig(os.path.join(ROOT_FOLDER, name, method, f'{method}_{name}_spatial.pdf'))
    except:
        print('no spatial footprint')
    plt.close()
    np.save(os.path.join(save_folder, f'{method}_{name}_adaptive.npy'), result)

#%% Fig 5b performance of VolPy, CaImAn, Mean-ROI and SGPMD on voltage data with gt
files = os.listdir(save_folder)
v = []; m = []; c = []; s = []
v_f1 = []; m_f1 = []; c_f1 = []; s_f1 = []

for name in np.array(names)[np.array([1, 0, 2])]:
    for file in files:
        if name in file:
            print(name)
            if 'volpy' in file and 'adaptive' in file:
                temp = np.load(os.path.join(save_folder, file), allow_pickle=True).item()
                v.append(temp)
                v_f1.append(temp['F1'])
            if 'caiman' in file:
                temp = np.load(os.path.join(save_folder, file), allow_pickle=True).item()
                c.append(temp)
                c_f1.append(temp['F1'])

            if 'mean_roi' in file:
                temp = np.load(os.path.join(save_folder, file), allow_pickle=True).item()
                m.append(temp) 
                m_f1.append(temp['F1'])

            if 'sgpmd' in file:
                temp = np.load(os.path.join(save_folder, file), allow_pickle=True).item()

                s.append(temp)
                s_f1.append(temp['F1'])

labels = ['Fish1', 'Fish2', 'Mouse']
labels = ['TEG_1', 'TEG_2', 'L1_1']

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
F1 = np.stack([v_f1, c_f1, m_f1, s_f1])

fig, ax = plt.subplots()
rects1 = ax.bar(x - 3* width/2, F1[0], width, label='VolPy', color='black')
rects2 = ax.bar(x - 1*width/2, F1[1], width, label='CaImAn', color='red')
rects3 = ax.bar(x + 1*width/2, F1[2], width, label='mean_roi', color='yellow')
rects4 = ax.bar(x + 3*width/2, F1[3], width, label='sgpmd', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 score')
ax.set_title('F1 score for voltage data with ephys gt')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/voltage_ephys/F1_voltage_with_ephys_v1.5.pdf')
#%%    
df1 = pd.DataFrame({'files':labels,'VolPy':F1[0], 'CaImAn':F1[1], 
                   'MeanROI':F1[2], 'SGPMD-NMF':F1[3]})

dfs = [df1]
text = 'We compared the performance of VolPy, CaImAn, MeanROI, and SGPMD-NMF in retrieving spikes on the three neurons'
fig_name = 'Fig 5b'
excel_name = os.path.join(excel_folder, 'volpy_data.xlsx')
# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)  

###################################################################################################################
#%% Below includes steps to prepare files and run VolPy and CaImAn
for name in names:
    try:
        #os.makedirs(os.path.join(ROOT_FOLDER, name, 'caiman'))
        #os.makedirs(os.path.join(ROOT_FOLDER, name, 'volpy'))
        #os.makedirs(os.path.join(ROOT_FOLDER, name, 'sgpmd'))
        #os.makedirs(os.path.join(ROOT_FOLDER, name, 'mean_roi'))
        os.makedirs(os.path.join(ROOT_FOLDER, name, 'viola'))
        print('make folder')
    except:
        print('already exist')

#%% move file to volpy
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    path = os.path.join(folder, 'volpy')
    files = os.listdir(folder)
    files = [file for file in files if 'registered.tif' in file or 'Session1' in file]
    files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    for file in files:
        shutil.copyfile(os.path.join(folder, file), os.path.join(path, file))

#%% move file to caiman
for name in names:
    folder = os.path.join(ROOT_FOLDER, name)
    path = os.path.join(folder, 'caiman')
    files = os.listdir(folder)
    files = [file for file in files if 'registered.tif' in file or 'Session1' in file]
    files = [file for file in files if os.path.isfile(os.path.join(folder, file))]
    for file in files:
        shutil.copyfile(os.path.join(folder, file), os.path.join(path, file))
 
#%%
context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
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

#%% run volpy
for name in names:
    if 'Fish' in name:
        fr = 300
        file = 'registered'
    else:
        fr = 400
        file = 'Session1'
        
    folder = os.path.join(ROOT_FOLDER, name)
    path = os.path.join(folder, 'volpy')
    fnames = os.path.join(path, f'{file}.tif')
    run_volpy(fnames, options=options, do_motion_correction=False, do_memory_mapping=False, fr=fr)     
    
#%% flip caiman movie
for name in names:
    if 'Fish' in name:
        fr = 300
        file = 'registered'
    else:
        fr = 400
        file = 'Session1'

    folder = os.path.join(ROOT_FOLDER, name)
    fnames = os.path.join(folder, f'{file}.tif')
    m  = cm.load(fnames)
    mm = flip_movie(m)
    mm.save(os.path.join(folder, 'caiman', file+'_flip.hdf5'))
    


#%% This code produces Fig S5
sf = {}
sf[name] = [c_spatial[:,:,c_idx], s_spatial]
nn = ['TEG1', 'TEG2', 'L1']
mm = ['CaImAn', 'SGPMD-NMF']
fig, axes = plt.subplots(nrows=2, ncols=3)
for i in range(2):
    for j in range(3):
        axes[i,j].imshow(sf[names[j]][i])
        axes[i, j].title.set_text(f'{nn[j]}/{mm[i]}')
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/supp/ephys_spatial_footprint.pdf')

#%% This code produces Vid S4
m.fr = 300
mmm = (-m[1000:]).computeDFF(0.5)[0]
mmm.size()
mmm.play(magnification=4, fr=300)
m1 = mmm.resize(1,1,0.3)
m1.play(magnification=4)
m1[:3000].save('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/supp/TEG2_DFF.tif')

import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Figures/plos/original_files/supp/video.mp4',fourcc, 300, (m.shape[1],m.shape[2]))



###########################################################################################################################################
#%% Transform into a format compatible with Marton's data
sweep_time=None
e_t = np.arange(0, scope[1])
e_sg = ephys[:scope[1]]
e_sp = etime[etime<scope[1]]
e_sub = None
v_t = timepoint[timepoint<scope[1]]
v_sg = trace[timepoint<scope[1]]
v_sp = v_t[spikes[timepoint[spikes]<scope[1]]]
v_sub = None
save_name = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/09282017Fish1-1_output.npz'
np.savez(save_name, sweep_time=sweep_time, e_sg=e_sg, v_sg=v_sg, 
         e_t=e_t, v_t=v_t, e_sp=e_sp, v_sp=v_sp, e_sub=e_sub, v_sub=v_sub)

sweep_time=None
e_t = np.arange(0, 1000000)
e_sg = ephys
e_sp = etime
e_sub = None
v_t = timepoint
v_sg = trace
v_sp = v_t[spikes[timepoint[spikes]<scope[1]]]
v_sub = None
#save_name = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/Mouse_Session_1.npz'
#np.savez(save_name, sweep_time=sweep_time, e_sg=e_sg, v_sg=v_sg, 
#         e_t=e_t, v_t=v_t, e_sp=e_sp, v_sp=v_sp, e_sub=e_sub, v_sub=v_sub)

#%% Run VIOLA
vi_idx = 0
folder = os.path.join(ROOT_FOLDER, name)
path = os.path.join(folder, 'viola')
files = os.listdir(path)
files = [file for file in files if 'viola' in file and '.npy' in file]

for file in files:
    print(file)
    vi = np.load(os.path.join(path, file), allow_pickle=True).item()
    vi_temporal = vi.t_s[vi_idx]
    vi_spatial = vi.weights[vi_idx]
    vi_spikes = vi.spikes[0]
    plt.figure(); plt.imshow(vi_spatial); plt.figure(); plt.plot(vi_temporal)    
        
    scope = [0, 1000000]
    t1 = ephys; s1 = espikes
    F1, precision, recall = match_ephys_voltage(t1=ephys, s1=espikes, t2=vi_temporal, 
                                                         s2=vi_spikes, timepoint=timepoint, 
                                                         max_dist=max_dist, scope=scope, hline=hline) 









