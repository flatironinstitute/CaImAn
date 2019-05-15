#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for exploring performance of CaImAn online as a function of:
i) min_SNR: minimum trace SNR for accepting candidate components
ii) thresh_CNN_noisy: minimum CNN threshold for accepting candidate components
iii) min_num_trial: number of candidate components per frame
The scripts loads the pre-computed results for combinations over a small grid
over these parameters and produces a Figure showing the best overall
performance as well as the best average performance for parameters more 
suitable to short or long datasets. The best performance for each dataset
is also ploted. See the companion paper for more details.
@author: epnevmatikakis
"""

import numpy as np
import os
import matplotlib.pyplot as plt
# %% list and sort files

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE'

with np.load(os.path.join(base_folder, 'all_records_grid_online.npz'), allow_pickle=True) as ld:
    records_online = ld['records']
    records_online = [list(rec) for rec in records_online]

# %% extract values

datasets = [rec[0] for rec in records_online[:9]]
inds = [rec[1:4] for rec in records_online[::9]]
RC_arr = np.array([float(rec[4]) for rec in records_online]).reshape(len(inds), 9)
PR_arr = np.array([float(rec[5]) for rec in records_online]).reshape(len(inds), 9)
F1_arr = np.array([float(rec[6]) for rec in records_online]).reshape(len(inds), 9)

#%% bar plot

colors = ['r','b','g','m']
n_groups = len(datasets)
datasets_names = [ds[:-1] + '\n' + str(ln) for (ds,ln) in zip(datasets, lengths)]
ind_i = [np.argmax(f) for f in F1_arr.T]
ind_mean = np.argmax(F1_arr.mean(1))
ind_small = np.argmax(F1_arr[:,[0,1,3,4,5]].mean(1))
ind_large = np.argmax(F1_arr[:,6:].mean(1))

#F1_mean = F1_arr[ind_max]
F1_small = F1_arr[ind_small]
F1_large = F1_arr[ind_large]
F1_mean = F1_arr[ind_mean]
F1_max = F1_arr.max(0)

PR_mean = PR_arr[ind_mean]
PR_small = PR_arr[ind_small]
PR_large = PR_arr[ind_large]
PR_max = np.array([PR_arr[ind_i[i],i] for i in range(len(datasets))])

RC_mean = RC_arr[ind_mean]
RC_small = RC_arr[ind_small]
RC_large = RC_arr[ind_large]
RC_max = np.array([RC_arr[ind_i[i],i] for i in range(len(datasets))])

# create plot
plt.subplots()
index = np.arange(n_groups)
bar_width = 0.18
opacity = 1
plt.subplot(3,1,1)
rects0 = plt.bar(index, F1_small, bar_width,
                 alpha=opacity,
                 color=colors[0],
                 label='low threshold')
 
rects1 = plt.bar(index + bar_width, F1_large, bar_width,
                 alpha=opacity,
                 color=colors[1],
                 label='high threshold')
 
rects2 = plt.bar(index + 2*bar_width, F1_mean, bar_width,
                 alpha=opacity,
                 color=colors[2],
                 label='mean')

rects3 = plt.bar(index + 3*bar_width, F1_max, bar_width,
                 alpha=opacity,
                 color=colors[3],
                 label='max')


 
plt.xlabel('Dataset')
plt.ylabel('$F_1$ Scores')
ax1 = plt.gca()
ax1.set_ylim([0.6,0.85])
plt.xticks(index + bar_width, datasets_names)
plt.legend()
 
plt.tight_layout()

plt.subplot(3,1,2)
rects0 = plt.bar(index, PR_small, bar_width,
                 alpha=opacity,
                 color=colors[0],
                 label='low threshold')
 
rects1 = plt.bar(index + bar_width, PR_large, bar_width,
                 alpha=opacity,
                 color=colors[1],
                 label='high threshold')
 
rects2 = plt.bar(index + 2*bar_width, PR_mean, bar_width,
                 alpha=opacity,
                 color=colors[2],
                 label='mean')

rects3 = plt.bar(index + 3*bar_width, PR_max, bar_width,
                 alpha=opacity,
                 color=colors[3],
                 label='max')


plt.xlabel('Dataset')
plt.ylabel('Precision')
ax2 = plt.gca()
ax2.set_ylim([0.55,0.95])
plt.xticks(index + bar_width, datasets_names)
 
plt.tight_layout()

plt.subplot(3,1,3)
rects0 = plt.bar(index, RC_small, bar_width,
                 alpha=opacity,
                 color=colors[0],
                 label='low threshold')
 
rects1 = plt.bar(index + bar_width, RC_large, bar_width,
                 alpha=opacity,
                 color=colors[1],
                 label='high threshold')
 
rects2 = plt.bar(index + 2*bar_width, RC_mean, bar_width,
                 alpha=opacity,
                 color=colors[2],
                 label='mean')

rects3 = plt.bar(index + 3*bar_width, RC_max, bar_width,
                 alpha=opacity,
                 color=colors[3],
                 label='max')


 
plt.xlabel('Dataset')
plt.ylabel('Recall')
ax2 = plt.gca()
ax2.set_ylim([0.425,0.9])
plt.xticks(index + bar_width, datasets_names)
 
plt.tight_layout()
plt.show()

plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}

plt.rc('font', **font)

#%% print the parameter combinations

print('Low threshold vals: (min_SNR, CNN_thr, num_comp)= ' + str(inds[ind_small]))
print('High threshold vals: (min_SNR, CNN_thr, num_comp)= ' + str(inds[ind_large]))
print('Best overall vals: (min_SNR, CNN_thr, num_comp)= ' + str(inds[ind_mean]))
for (dataset, ind_mx) in zip(datasets, ind_i):
    print('Best value for dataset ' + str(dataset) + ' was obtained for parameters (min_SNR, CNN_thr, num_comp)= ' + str(inds[ind_mx]))
