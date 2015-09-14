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
from arpfit import arpfit
from sklearn.decomposition import ProjectedGradientNMF
from update_spatial_components import update_spatial_components
from update_temporal_components import update_temporal_components
from matplotlib import pyplot as plt
from time import time
from merge_rois import mergeROIS
from scipy.sparse import coo_matrix
#import libtiff
from utilities import *

#%%
#frm=pims.open('demoMovie.tif')
#mov = np.array(pims.open('demoMovie.tif')) 
Ymat = sio.loadmat('Y.mat')
Y = Ymat['Y']*1.

#t = libtiff.TiffFile('demoMovie.tif') 
#t = libtiff.TiffFile('/Users/eftychios/Documents/_code/calcium_paper_code/datasets/clay/2014-04-05-003.tif')
#tt = t.get_tiff_array() 
#Y2 = tt[:]*1.
#Y = np.transpose(Y2,(1,2,0))

#%%
sizeY = np.shape(Y)
d1 = sizeY[0]
d2 = sizeY[1]
T = sizeY[-1]
##%%
#Y = np.array(pims.open('demoMovie.tif')) 
#Y=np.transpose(Y,(1,2,0))
#d1,d2,T=Y.shape
#%% greedy initialization
nr = 30
t1 = time()
Ain,Cin,center = greedyROI2d(Y, nr = nr, gSig = [4,4], gSiz = [9,9])
t_elGREEDY = time()-t1

#%% plot centers
Cn = local_correlations(Y)
fig = plt.figure()
plt1 = plt.imshow(Cn,interpolation='none')
plt.colorbar()

plt.scatter(x=center[:,1], y=center[:,0], c='m', s=40)
plt.axis((-0.5,d2-0.5,-0.5,d1-0.5))
plt.gca().invert_yaxis()
fig.suptitle('Component centers found with greedy initialization') 
#%% arpfit

active_pixels = np.squeeze(np.nonzero(np.sum(Ain,axis=1)))
Yr = np.reshape(Y,(d1*d2,T),order='F')
p = 2;
P = arpfit(Yr,p=2,pixels = active_pixels)

#%% nmf

Y_res = Yr - np.dot(Ain,Cin)
model = ProjectedGradientNMF(n_components=1, init='random', random_state=0)
model.fit(np.maximum(Y_res,0))

fin = model.components_.squeeze()

#%% update spatial components

t1 = time()
A,b = update_spatial_components(Yr, Cin, fin, Ain, d1=d1, d2=d2, sn = P['sn'])
t_elSPATIAL = time() - t1

#%% 
#t1 = time()
#C,f,Y_res,Pnew = update_temporal_components(Yr,A,b,Cin,fin,ITER=2)
#t_elTEMPORAL1 = time() - t1

#%%  solving using spgl1 for deconvolution (necessary files not in repo yet)
t1 = time()
C,f,Y_res,Pnew = update_temporal_components(Yr,A,b,Cin,fin,ITER=2,deconv_method = 'spgl1')
t_elTEMPORAL2 = time() - t1

#%%
t1 = time()
A_m,C_m,nr_m,merged_ROIs,P_m=mergeROIS(Y_res,A.tocsc(),b,np.array(C),f,d1,d2,Pnew,sn=P['sn'])
t_elMERGE = time() - t1

#%%
A2,b2 = update_spatial_components(Yr, C_m, f, A_m, d1=d1, d2=d2, sn = P['sn'])
C2,f2,Y_res2,Pnew2 = update_temporal_components(Yr,A2,b2,C_m,f,ITER=2,deconv_method = 'spgl1')

#%% order components

A_or, C_or, srt = order_components(A2,C2)
C_df = extract_DF_F(Yr,A2,C2)

#%% plot ordered components

view_patches(Yr,coo_matrix(A_or),C_or,b,f,d1,d2)
#%%
crd = plot_contours(coo_matrix(A_or),Cn,thr=0.9)
#SAVE TO FILE
#np.savez('preprocess_analysis',Y_res=Y_res,A=A.todense(),b=b,C=C,f=f,d1=d1,d2=d2,P=P,Pnew=Pnew,sn=P['sn'])

#%% %%%%%%%%%%%%% ANDREA NEEDS TO FIX THIS %%%%%%%%%%%%%%%%%%%
##SAVE TO FILE
#np.savez('preprocess_analysis',Y_res=Y_res,A=A.todense(),b=b,C=C,f=f,d1=d1,d2=d2,P=P,Pnew=Pnew,sn=P['sn'])
#
#import numpy as np
#from scipy.sparse import csc_matrix,coo_matrix
#vars_=np.load('preprocess_analysis.npz')
#
#Y_res=vars_['Y_res']
#A=coo_matrix(vars_['A'])
#b=vars_['b']
#C=vars_['C']
#f=vars_['f']
#d1=vars_['d1']
#d2=vars_['d2']
#P=vars_['P']
#Pnew=vars_['Pnew']
#sn=vars_['sn']
##%%
#from merge_rois import mergeROIS
#A_m,C_m,nr_m,merged_ROIs,P_m=mergeROIS(Y_res,A.tocsc(),b,C,f,d1,d2,Pnew,sn=sn)
#A_m,C_m,nr_m,merged_ROIs,P_m=mergeROIS(Y_res,A.tocsc(),b,C,f,d1,d2,Pnew,sn=P['sn'])
