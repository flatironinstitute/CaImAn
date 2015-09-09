# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:47:13 2015

@author: epnevmatikakis
"""
#%%
import pims
import numpy as np
import scipy.io as sio
from greedyROI2d import greedyROI2d

#%%
#frm=pims.open('demoMovie.tif')
#mov = np.array(pims.open('demoMovie.tif')) 
Ymat = sio.loadmat('/Users/epnevmatikakis/Desktop/Y.mat')
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

from arpfit import arpfit
active_pixels = np.squeeze(np.nonzero(np.sum(Ain,axis=1)))
Yr = np.reshape(Y,(d1*d2,T),order='F')
p = 2;
P = arpfit(Yr,p=2,pixels = active_pixels)

#%% nmf

Y_res = Yr - np.dot(Ain,Cin)

from sklearn.decomposition import ProjectedGradientNMF
model = ProjectedGradientNMF(n_components=1, init='random',
                             random_state=0)
model.fit(np.maximum(Y_res,0)) 

fin = model.components_.squeeze()

#%% update spatial components
from update_spatial_components import update_spatial_components

A,b = update_spatial_components(Yr, Cin, fin, Ain, d1=d1, d2=d2, sn = P['sn'])

#%% update temporal components

from update_temporal_components import update_temporal_components

C,f,Y_res,Pnew = update_temporal_components(Yr,A,b,Cin,fin,ITER=2)