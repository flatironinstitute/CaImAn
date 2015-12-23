# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:47:13 2015

@author: epnevmatikakis
"""
#%%
try:
    %load_ext autoreload
    %autoreload 2
except:
    print 'NOT IPYTHON'
import sys
import numpy as np
import scipy.io as sio
import ca_source_extraction as cse
#%%
from ca_source_extraction.initialization import initialize_components,arpfit
from ca_source_extraction.spatial import update_spatial_components
from ca_source_extraction.temporal import update_temporal_components
from ca_source_extraction.merging import mergeROIS
from ca_source_extraction.pre_processing import preprocess_data
from ca_source_extraction.utilities import local_correlations,plot_contours,view_patches,order_components,extract_DF_F
#%%
from matplotlib import pyplot as plt
from time import time
import pylab as pl
from scipy.sparse import coo_matrix
import scipy
from sklearn.decomposition import NMF
#%% TYPE DECONVOLUTION
deconv_type='debug'
#deconv_type='spgl1'

#%%
import tifffile
t = tifffile.TiffFile('movies/demoMovie.tif') 
Y = t.asarray() 
Y = np.transpose(Y,(1,2,0))*1.
d1,d2,T=Y.shape
#%% PREPROCESS DATA
Yr = np.reshape(Y,(d1*d2,T),order='F')
Yr,sn,g=preprocess_data(Yr)
Y=np.reshape(Yr,(d1,d2,T),order='F')
#%%
nr = 30 # number of expected neurons
gSig=[4,4] #
gSiz=[9,9] #
ssub=1 # space subsampling
tsub=1 # time subsampling

t1 = time()
Ain, Cin, b_in, f_in, center=initialize_components(Y, K=nr, gSig=gSig, gSiz=gSiz, ssub=ssub, tsub=tsub, nIter = 5, use_median = False, kernel = None)                                                    
print time() - t1 # 
#%%
Cn = local_correlations(Y)
plt2 = plt.imshow(Cn,interpolation='none')
plt.colorbar()
plt.scatter(x=center[:,1], y=center[:,0], c='m', s=40)
crd = plot_contours(coo_matrix(Ain[:,::-1]),Cn,thr=0.9)
plt.axis((-0.5,d2-0.5,-0.5,d1-0.5))
plt.gca().invert_yaxis()
#%%
t1 = time()
A,b,Cin = update_spatial_components(Yr, Cin, f_in, Ain, d1=d1, d2=d2, sn = sn,dist=2,max_size=8,min_size=3,n_processes=1)
t_elSPATIAL = time() - t1
print t_elSPATIAL# 
#%%
crd = plot_contours(A,Cn,thr=0.9)
#%%
t1 = time()
C,f,Y_res,Pnew,S = update_temporal_components(Yr,A,b,Cin,f_in,ITER=2,deconv_method = deconv_type)
t_elTEMPORAL2 = time() - t1
print t_elTEMPORAL2 # took 98 sec
#%%
t1 = time()
A_sp=A.tocsc();
A_m,C_m,nr_m,merged_ROIs,P_m,S_m=mergeROIS(Y_res,A_sp,b,np.array(C),f,S,d1,d2,Pnew,sn=sn,thr=.85,deconv_method=deconv_type,min_size=3,max_size=8,dist=2)
t_elMERGE = time() - t1
print t_elMERGE
#%%
crd = plot_contours(A_m,Cn,thr=0.9)
#%%
t1 = time()
A2,b2,C_m_ = update_spatial_components(Yr, C_m, f, A_m, d1=d1, d2=d2, sn = sn,dist=2,max_size=8,min_size=3)
C2,f2,Y_res2,Pnew2,S = update_temporal_components(Yr,A2,b2,C_m_,f,ITER=2,deconv_method = deconv_type)
print time() - t1 # 100 seconds
#%%
crd = plot_contours(A2,Cn,thr=0.9)
#%%
A_or, C_or, srt = order_components(A2,C2)
#C_df = extract_DF_F(Yr,A2,C2)
crd = plot_contours(coo_matrix(A_or[:,::-1]),Cn,thr=0.9)
#%%q
view_patches(Yr,coo_matrix(A_or),C_or,b,f,d1,d2)