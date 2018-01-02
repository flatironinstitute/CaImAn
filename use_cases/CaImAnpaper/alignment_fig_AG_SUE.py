# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:09:24 2017
Create alignment figure for CaImAn paper
@author: epnevmatikakis
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
import numpy as np
import pylab as plt
import caiman as cm
#%%
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from glob import glob

files = ['/mnt/ceph/neuro/Sue/k53/20160530/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
         '/mnt/ceph/neuro/Sue/k53/20160531/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
         '/mnt/ceph/neuro/Sue/k53/20160603/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz']

#%% load data
data = []
Cns = []
for fl in files:
    print(fl)
    with np.load(fl) as ld:
        A_ = ld['A'][()].toarray()
        idx_ = np.where(ld['cnn_preds']>.75)[0]
#        idx_ = ld['idx_components']
        data.append(A_[:,idx_])
        Cns.append(ld['Cn'])

A1, A2, A3 = data
Cn1, Cn2, Cn3 = Cns
#%%
plt.subplot(1,3,1)
crd_good = cm.utils.visualization.plot_contours(
    A1, Cn1, thr=.96, vmax=0.5)
plt.subplot(1,3,2)
crd_good = cm.utils.visualization.plot_contours(
    A2, Cn2, thr=.96, vmax=0.5)
plt.subplot(1,3,3)
crd_good = cm.utils.visualization.plot_contours(
    A3, Cn2, thr=.96, vmax=0.5)
#%% normalize matrices
A1 = csc_matrix(A1 / A1.sum(0))
A2 = csc_matrix(A2 / A2.sum(0))
A3 = csc_matrix(A3 / A3.sum(0))
#%% match consecutive pairs

from caiman.base.rois import register_ROIs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

dims = 512, 512
match1_12, match2_12, mis1_12, mis2_12, perf_12 = register_ROIs(
    A1, A2, dims, plot_results=True, template1=Cn1, template2=Cn2)
plt.figure()

match2_23, match3_23, mis2_23, mis3_23, perf_23 = register_ROIs(
    A2, A3, dims, plot_results=True, template1=Cn2, template2=Cn3)

plt.figure()
match1_13, match3_13, mis1_13, mis3_13, perf_13 = register_ROIs(
    A1, A3, dims, plot_results=True, template1=Cn1, template2=Cn3)
#%%
match2_12 = list(match2_12)
match2_23 = list(match2_23)
# ROIs in session 2 that are registered against both session 1 and session 3
ind2_12_23 = list(set(match2_12).intersection(match2_23))

ind1_12_23 = [match1_12[match2_12.index(x)] for x in ind2_12_23 ]
ind3_12_23 = [match3_23[match2_23.index(x)] for x in ind2_12_23 ]
#%%
ind_1_tot = np.array(np.unique(np.setdiff1d(list(match1_12)+list(match1_13),ind1_12_23)))
ind_2_tot = np.array(np.unique(np.setdiff1d(list(match2_12)+list(match2_23),ind2_12_23)))
ind_3_tot = np.array(np.unique(np.setdiff1d(list(match3_13)+list(match3_23),ind3_12_23)))
#%% make figure
def norm_nrg(a_):

    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order='F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn = cumEn/cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order='F')
#%%

Cn = np.reshape(A1.sum(axis=1) + A2.sum(axis=1) +
                A3.sum(axis=1), (512, 512), order='F')
plt.figure()
masks_1 = np.reshape(A1.toarray(), dims + (-1,),
                     order='F').transpose(2, 0, 1)
masks_2 = np.reshape(A2.toarray(), dims + (-1,),
                     order='F').transpose(2, 0, 1)
masks_3 = np.reshape(A3.toarray(), dims + (-1,),
                     order='F').transpose(2, 0, 1)
#        try : #Plotting function
level = 0.95

plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 10}
plt.rc('font', **font)
lp, hp = np.nanpercentile(Cn, [5, 98])
plt.imshow(Cn, vmin=lp, vmax=hp, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='b', linewidths=1)
 for mm in masks_1[ind1_12_23]]
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_2[ind2_12_23 ]]
[plt.contour(norm_nrg(mm), levels=[level], colors='y', linewidths=1)
 for mm in masks_3[ind3_12_23]]
# plt.legend(('Day1','Day2','Day3'))
plt.title('Matched components across multiple days')
plt.axis('off')

day1 = mlines.Line2D([], [], color='b', label='Day 1')
day2 = mlines.Line2D([], [], color='r', label='Day 2')
day3 = mlines.Line2D([], [], color='y', label='Day 3')
plt.legend(handles=[day1, day2, day3], loc=4)

plt.show()
#%%
plt.imshow(Cn, vmin=lp, vmax=hp, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='b', linewidths=1)
 for mm in masks_1[ind_1_tot]]
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_2[ind_2_tot]]
[plt.contour(norm_nrg(mm), levels=[level], colors='y', linewidths=1)
 for mm in masks_3[ind_3_tot]]
# plt.legend(('Day1','Day2','Day3'))
plt.title('Matched components across multiple days')
plt.axis('off')

day1 = mlines.Line2D([], [], color='b', label='Day 1')
day2 = mlines.Line2D([], [], color='r', label='Day 2')
day3 = mlines.Line2D([], [], color='y', label='Day 3')
plt.legend(handles=[day1, day2, day3], loc=4)

plt.show()