# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:47:13 2015

@author: epnevmatikakis
"""
#%%
import pims
import numpy as np
import scipy.io as sio
from arpfit import arpfit
from greedyROI2d import greedyROI2d
from sklearn.decomposition import ProjectedGradientNMF
from update_spatial_components import update_spatial_components
from update_temporal_components import update_temporal_components
from merge_rois import mergeROIS
#%%
#frm=pims.open('demoMovie.tif')
#mov = np.array(pims.open('demoMovie.tif')) 
Ymat = sio.loadmat('Y.mat')
Y = Ymat['Y']*1.

#%%
sizeY = np.shape(Y)
d1 = sizeY[0]
d2 = sizeY[1]
T = sizeY[-1]

#%% greedy initialization
nr = 30
Ain,Cin,center = greedyROI2d(Y, nr = nr, gSig = [4,4], gSiz = [9,9])

#%% arpfit

active_pixels = np.squeeze(np.nonzero(np.sum(Ain,axis=1)))
Yr = np.reshape(Y,(d1*d2,T),order='F')
p = 2;
P = arpfit(Yr,p=2,pixels = active_pixels)

#%% nmf

Y_res = Yr - np.dot(Ain,Cin)

model = ProjectedGradientNMF(n_components=1, init='random',
                             random_state=0)
model.fit(np.maximum(Y_res,0)) 

fin = model.components_.squeeze()

#%% update spatial components

A,b = update_spatial_components(Yr, Cin, fin, Ain, d1=d1, d2=d2, sn = P['sn'])

#%% update temporal components
C,f,Y_res,Pnew = update_temporal_components(Yr,A,b,Cin,fin,ITER=2)
#%%
A_m,C_m,nr_m,merged_ROIs,P_m=mergeROIS(Y_res,A.tocsc(),b,np.array(C),f,d1,d2,Pnew,sn=P['sn'])