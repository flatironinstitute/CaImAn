#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file produces the result of fig1 e,f in the VolPy paper.
One needs to run the demo_pipeline_voltage_imaging.py on L1.04.50 data to get
volpy estimates object for running the code.
@author: caichangjia
"""
#%%
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage.measurements import center_of_mass
import caiman as cm
from caiman.base.rois import com

colorsets = plt.cm.tab10(np.linspace(0,1,10))
colorsets = colorsets[[0,1,2,3,4,5,6,8,9],:]

#%% Delete based on number of spikes, locality
estimates = vpy.estimates.copy()
del1 = np.std(np.array(estimates['num_spikes'])[:,-4:], axis=1)
del2 = np.array(estimates['locality'])
del3 = np.array(estimates['low_spikes'])
select = np.multiply(np.multiply((del1<200), (del2>0)), (del3<2))
delarray = np.array([12, 28])
select[delarray] = 0
select = select>0

A = ROIs.copy()[select]
C = np.stack(estimates['trace_processed'], axis=0).copy()[select]
spike = [estimates['spikes'][i] for i in np.where(select>0)[0]]

#%% Seperate components
def neuron_number(A, N, n, n_group):    
    l = [center_of_mass(a)[1] for a in A]
    li = np.argsort(l)
    if N != n*n_group:
        li = np.append(li, np.ones((1,n*n_group-N),dtype=np.int8)*(-1))
    mat = li.reshape((n_group, n), order='F')
    return mat

N = A.shape[0]
n = N
n_group = int(np.ceil(N/n))
mat = neuron_number(A, N, n, n_group)

#%% Visualize spatial footprint
Cn = summary_image[:,:,2]  
A = A.astype(np.float64)
root_dir = '/home/nel/data/voltage_data/volpy_paper/figure1'

def plot_neuron_contour(A, N, n, n_group, Cn, save_path):
    number=0
    number1=0
    for i in range(n_group):
        plt.figure()    
        vmax = np.percentile(Cn, 97)
        vmin = np.percentile(Cn, 5)
        plt.imshow(Cn, interpolation='None', vmax=vmax, vmin=vmin, cmap=plt.cm.gray)
        plt.title('Neurons location')
        d1, d2 = np.shape(Cn)
        cm1 = com(A.copy().reshape((N,-1), order='F').transpose(), d1, d2)
        colors='yellow'
        for j in range(n):
            index = mat[i,j]
            print(index) 
            img = A[index]
            img1 = img.copy()
            #img1[img1<np.percentile(img1[img1>0],15)] = 0
            #img1 = connected_components(img1)
            img2 = np.multiply(img, img1)
            contours = measure.find_contours(img2, 0.5)[0]
            #img2=img.copy()
            img2[img2 == 0] = np.nan
            if index != -1:
                plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(number,9)])
                plt.text(cm1[index, 1]+0, cm1[index, 0]-0, str(number), color=colors)
                number=number+1 
        plt.savefig(os.path.join(root_dir, 'neuron_contour{}-{}.pdf'.format(number1,number-1)))
        number1=number
        
plot_neuron_contour(A, N, n, n_group, Cn, root_dir)

#%% Visualize corresponding traces
CZ = C[:,:20000]
CZ = (CZ-CZ.mean(axis=1)[:,np.newaxis])/CZ.std(axis=1)[:,np.newaxis]
#CZ = CZ/CZ.max(axis=1)[:,np.newaxis]

def plot_neuron_signal(A, N, n, n_group, Cn, save_path):     
    number=0
    number1=0
    for i in range(n_group):
        fig, ax = plt.subplots((mat[i,:]>-1).sum(),1)
        length = (mat[i,:]>-1).sum()
        for j in range(n):
            if j==0:
                ax[j].set_title('Signals')
            if mat[i,j]>-1:
                index = mat[i,j]
                T = C.shape[1]
                ax[j].plot(np.arange(20000), CZ[index], 'c', linewidth=0.1, color=colorsets[np.mod(number,9)])
                ax[j].scatter(spike[index],
                 np.max(CZ[index]+0.1) * np.ones(spike[index].shape),
                 color=colorsets[np.mod(number,9)], marker='o',s=0.1)
                ax[j].autoscale(tight=True)
                ax[j].text(-500, 0, f'{number}', horizontalalignment='center',
                     verticalalignment='center')
                ax[j].set_ylim([CZ.min(), CZ.max()+1])
                if j==0:
                    ax[j].text(-30, 3000, 'neuron', horizontalalignment='center', 
                      verticalalignment='center')
                if j<length-1:
                    ax[j].axis('off')
                if j==length-1:
                    ax[j].spines['right'].set_visible(False)
                    ax[j].spines['top'].set_visible(False)  
                    ax[j].spines['left'].set_visible(True) 
                    ax[j].get_yaxis().set_visible(True)
                    ax[j].set_xlabel('Frames')
                number = number + 1
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, 'neuron_signal{}-{}.pdf'.format(number1,number-1)))
        number1=number    
        
plot_neuron_signal(A, N, n, n_group, Cn, root_dir)




