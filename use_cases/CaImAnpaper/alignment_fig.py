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

#%%

from scipy.io import loadmat
from scipy.sparse import csc_matrix
from glob import glob

files = glob('/Users/epnevmatikakis/Documents/Ca_datasets/Taxidis/MultiDay/ROI*.mat')

#%% load data

data1 = loadmat(files[2])
data2 = loadmat(files[3])
data3 = loadmat(files[4])
A1 = data1['ROI_keep']
A2 = data2['ROI_keep']
A3 = data3['ROI_keep']

#%% normalize matrices

A1 = csc_matrix(A1/A1.sum(0))
A2 = csc_matrix(A2/A2.sum(0))
A3 = csc_matrix(A3/A3.sum(0))

#%% match consecutive pairs

from caiman.base.rois import register_ROIs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

dims = 512, 512
match1_12,match2_12,mis1_12,mis2_12,perf_12 = register_ROIs(A1, A2, dims, plot_results=True, template1 = data1['Cn'], template2 = data2['Cn'])
plt.figure()
match2_23,match3_23,mis2_23,mis3_23,perf_23 = register_ROIs(A2, A3, dims, plot_results=True, template1 = data2['Cn'], template2 = data3['Cn'])

#%% 
match2_12 = list(match2_12)
match2_23 = list(match2_23)
ind_2 = list(set(match2_12).intersection(match2_23)) # ROIs in session 2 that are registered against both session 1 and session 3
ind_1 = [match1_12[match2_12.index(x)] for x in ind_2]
ind_3 = [match3_23[match2_23.index(x)] for x in ind_2]

#%% make figure

def norm_nrg(a_):
    
    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1,order = 'F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims,order = 'F')

Cn = np.reshape(A1.sum(axis=1)+A2.sum(axis=1)+A3.sum(axis=1),(512,512),order='F')
plt.figure();
masks_1 = np.reshape(A1.toarray(),dims+(-1,),order='F').transpose(2,0,1) > 0
masks_2 = np.reshape(A2.toarray(),dims+(-1,),order='F').transpose(2,0,1) > 0
masks_3 = np.reshape(A3.toarray(),dims+(-1,),order='F').transpose(2,0,1) > 0
#        try : #Plotting function
level = 0.98
plt.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 10}
plt.rc('font', **font)
lp,hp = np.nanpercentile(Cn,[5,98])
plt.imshow(Cn,vmin=lp,vmax=hp, cmap = 'gray')
[plt.contour(norm_nrg(mm),levels=[level],colors='b',linewidths=2) for mm in masks_1[ind_1]]
[plt.contour(norm_nrg(mm),levels=[level],colors='r',linewidths=2) for mm in masks_2[ind_2]] 
[plt.contour(norm_nrg(mm),levels=[level],colors='y',linewidths=2) for mm in masks_3[ind_3]] 
#plt.legend(('Day1','Day2','Day3'))
plt.title('Matched components across multiple days')
plt.axis('off')

day1 = mlines.Line2D([], [], color='b',label='Day 1')
day2 = mlines.Line2D([], [], color='r',label='Day 2')
day3 = mlines.Line2D([], [], color='y',label='Day 3')
plt.legend(handles=[day1,day2,day3],loc=4)

plt.show()