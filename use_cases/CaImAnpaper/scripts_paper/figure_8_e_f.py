#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The scripts reads the results of the CaImAn online algorithm on the annotated
datasets and produces two subplots about its computational speed (Fig. 8):
i) The processing speed in frames per second for each dataset
ii) The allocation of computing time for each frame
For more information see the companion paper.
@author: epnevmatikakis
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/timings_online_fast'

datasets = ['N.03.00.t/', 'N.04.00.t/', 'N.02.00/', 'YST/', 'N.00.00/', 'N.01.01/', 'K53/', 'J115/', 'J123/']
files = glob.glob(os.path.join(base_folder, '*.npz'))
files.sort()
#%% gather timings
T = []
N = []
F1 = []
PR = []
RC = []
T_online = []
T_detect = []
T_shapes = []
dims = []

for (dataset, file) in zip(datasets, files):
    with np.load(file) as fl:
        D = fl['all_results'][()][dataset]
        dims.append(D['A'].shape[0])
        T.append(D['C'].shape[-1])
        N.append(D['C'].shape[0])
        F1.append(D['f1_score'])
        PR.append(D['precision'])
        RC.append(D['recall'])
        T_detect.append(np.array(D['t_detect']))
        T_shapes.append(np.array(D['t_shapes']))
        T_online.append(np.array(D['t_online'])-T_detect[-1]-T_shapes[-1])
        
#%% plot frames per second as a function of # of neurons
T_tot = [np.sum(T_detect[ind][1:T[ind]-200]+T_shapes[ind][1:T[ind]-200]+T_online[ind][1:T[ind]-200]) for ind in range(len(N))]
fps = [(T[ind]-201)/T_tot[ind] for ind in range(len(N))]
pps = np.array(dims)*np.array(fps)

ds_ind = [0,2,3,6,7,8]   #  indeced of datasets that were spatially downsampled
nd_ind = [1,4,5]
area = 40*np.ones(len(N))
area[nd_ind] *= 2

_, ax = plt.subplots()
#ax.scatter(np.array(N)[ds_ind], np.array(pps)[ds_ind], s=np.array(area)[ds_ind], label='with downsampling')
#ax.scatter(np.array(N)[nd_ind], np.array(pps)[nd_ind], s=np.array(area)[nd_ind], label='w/o downsampling')
ax.scatter(np.array(N)[ds_ind], np.array(fps)[ds_ind], s=np.array(area)[ds_ind], label='N_p ~ 6e4')
ax.scatter(np.array(N)[nd_ind], np.array(fps)[nd_ind], s=np.array(area)[nd_ind], label='N_p ~ 2.5e5')
ax.legend()
plt.xlabel('Number of neurons')
plt.ylabel('Frames per second')
plt.title('Processing speed')
plt.tight_layout()

#%% plot processing time allocation per frame

ind = 8
skip = 10
_, ax2 = plt.subplots()
label=['process', 'detect','shapes']
ax2.stackplot(np.arange(T_detect[ind].shape[0])[2::skip], 1e3*T_online[ind][2::skip], 1e3*T_detect[ind][2::skip], 1e3*T_shapes[ind][2::skip], labels=label)
ax2.legend(loc=2)
plt.ylim([0, 100])
plt.xlabel('Frame #')
plt.ylabel('Processing time [ms]')
plt.title('Processing time per frame for dataset ' + datasets[ind][:-1] )
plt.tight_layout()
