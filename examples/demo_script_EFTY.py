# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:47:13 2015

@author: epnevmatikakis
"""
#%%
%load_ext autoreload
%autoreload 2

import sys
import numpy as np
import scipy.io as sio
from ca_source_extraction import temporal, spatial, deconvolution, utilities
#%%
from initialization import greedyROI2d, hals_2D
from spatial import update_spatial_components
from temporal import update_temporal_components
from merging import mergeROIS
from utilities import *
#%%
from matplotlib import pyplot as plt
from time import time
import pylab as pl
from scipy.sparse import coo_matrix
import scipy
from sklearn.decomposition import ProjectedGradientNMF
#%%
import calblitz as cb
#%%
#try: 
#    pl.ion()
#    %load_ext autoreload
#    %autoreload 2
#except:
#    print "Probably not a Ipython interactive environment" 

#%% 
m=cb.load('demoMovie.tif',fr=8); 
T,h,w=np.shape(m)

#%% 
Y=np.asarray(m)
Y = np.transpose(Y,(1,2,0))
#Ymat = sio.loadmat('Y.mat')
#Y = Ymat['Y']*1.
d1,d2,T = np.shape(Y)

a = sio.loadmat('demo_movie_test.mat')
Y_test=a['Y']

assert scipy.linalg.norm(Y-Y_test)==0
#%%
#a = sio.loadmat('tmp.mat')

Ain_test=a['Ain']
Cin_test=a['Cin']
bin_test=a['bin']
fin_test=np.squeeze(a['fin'])
center_test=a['center']
#%%
#
#Ain=Ain_test
#Cin=Cin_test
#b_in=bin_test
#f_in=fin_test
#center=center_test-1
#%%
nr = 30
t1 = time()

Ain,Cin,_, b_in, f_in = greedyROI2d(Y, nr = nr, gSig = [4,4], gSiz = [9,9], use_median = False)
t_elGREEDY = time()-t1
Ain, Cin, b_in, f_in = hals_2D(Y, Ain, Cin, b_in, f_in,maxIter=10);


center = com(Ain,d1,d2);

Cn = local_correlations(Y)
plt.subplot(2,1,1)
plt1 = plt.imshow(Cn,interpolation='none')
plt.colorbar()
plt.scatter(x=center[:,1], y=center[:,0], c='m', s=40)
plt.axis((-0.5,d2-0.5,-0.5,d1-0.5))
plt.gca().invert_yaxis()
plt.subplot(2,1,2)

plt2 = plt.imshow(Cn,interpolation='none')
plt.colorbar()
plt.scatter(x=center_test[:,1]-1, y=center_test[:,0]-1, c='m', s=40)
plt.axis((-0.5,d2-0.5,-0.5,d1-0.5))
plt.gca().invert_yaxis()
#%% 
plt.subplot(1,2,1)
crd = plot_contours(coo_matrix(Ain[:,::-1]),Cn,thr=0.9)
plt.subplot(1,2,2)
crd = plot_contours(coo_matrix(Ain_test[:,::-1]),Cn,thr=0.9)
#%%  matlab input
Ain=Ain_test
Cin=Cin_test
b_in=bin_test
f_in=fin_test
center=center_test-1
#%%
active_pixels = np.squeeze(np.nonzero(np.sum(Ain,axis=1)))
Yr = np.reshape(Y,(d1*d2,T),order='F')
p = 1;
P = arpfit(Yr,p=1,pixels = active_pixels)
Y_res = Yr - np.dot(Ain,Cin)
model = ProjectedGradientNMF(n_components=1, init='random', random_state=0)
model.fit(np.maximum(Y_res,0))

fin = model.components_.squeeze()
#%%
t1 = time()
A,b,Cin = update_spatial_components(Yr, Cin, fin, Ain, d1=d1, d2=d2, sn = P['sn'],dist=1,max_size=5,min_size=3)
t_elSPATIAL = time() - t1
#%%
crd = plot_contours(A,Cn2,thr=0.9,cmap=pl.cm.gray)
#%%
t1 = time()
C,f,Y_res,Pnew = update_temporal_components(Yr,A,b,Cin,fin,ITER=2,deconv_method = 'spgl1')
t_elTEMPORAL2 = time() - t1
#%%
t1 = time()
A_sp=A.tocsc();
A_m,C_m,nr_m,merged_ROIs,P_m=mergeROIS(Y_res,A_sp,b,np.array(C),f,d1,d2,Pnew,sn=P['sn'],thr=.9,deconv_method='spgl1',min_size=3,max_size=8,dist=3)
t_elMERGE = time() - t1
#%%
crd = plot_contours(A_m,Cn2,thr=0.9)
#%%
A2,b2,C_m_ = update_spatial_components(Yr, C_m, f, A_m, d1=d1, d2=d2, sn = P['sn'],dist=2,max_size=5,min_size=3)
C2,f2,Y_res2,Pnew2 = update_temporal_components(Yr,A2,b2,C_m_,f,ITER=2,deconv_method = 'spgl1')
#%%
crd = plot_contours(A2,Cn2,thr=0.9,cmap=pl.cm.gray)

#%%


A_or, C_or, srt = order_components(A2,C2)
C_df = extract_DF_F(Yr,A2,C2)
crd = plot_contours(coo_matrix(A_or[:,::-1]),Cn,thr=0.9)
#%%

Mn=np.std(m,axis=0)
crd = plot_contours(coo_matrix(A_or[:,::-1]),Mn,thr=0.9)
#%%

view_patches(Yr,coo_matrix(A_or),C_or,b,f,d1,d2)