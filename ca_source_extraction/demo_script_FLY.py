# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:47:13 2015

@author: epnevmatikakis
"""
#%%
import sys
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
import pylab as pl
#import libtiff
from utilities import *
from scipy.sparse import coo_matrix
import calblitz as cb
try: 
    pl.ion()
    %load_ext autoreload
    %autoreload 2
except:
    print "Probably not a Ipython interactive environment" 


#%%
m=cb.load('SinglePlane.mat',fr=.5); 

T,h,w=np.shape(m)


#%%
#m,shifts,xcorrs,template=m.motion_correct(max_shift_w=5,max_shift_h=5, num_frames_template=None, template = None,method='opencv')
#max_h,max_w=np.ceil(np.nanmax(shifts,axis=0))
#min_h,min_w=np.floor(np.nanmin(shifts,axis=0))
#m=m[:,-min_h:-max_h,-min_w:-max_w]
#%%
Cn2=np.nanmedian(m,axis=0)

#%%
#if np.min(m) <=0:
#    print 'Removing baseline'
#    m=m-np.min(m)+np.float16(1)
#else:
#    m=m+np.float16(0.01)
#    
#m,mbl=m.computeDFF(secsWindow=10,method='delta_f_over_sqrt_f',quantilMin=8)

#%%
#Y=np.load('M_FLUO_mc.npz')['mov'] 
#Y=Y[:,5:-5,5:-5]
#%%
Y=np.asarray(m)
Y = np.transpose(Y,(1,2,0))
#Ymat = sio.loadmat('Y.mat')
#Y = Ymat['Y']*1.
d1,d2,T = np.shape(Y)

#%%
nr = 20 # neuron numbers

t1 = time()

Ain,Cin,center = greedyROI2d(Y, nr = nr, gSig = [4,4], gSiz = [9,9], use_median = False)
t_elGREEDY = time()-t1

#%% plot centers
Cn = local_correlations(Y)
plt1 = plt.imshow(Cn,interpolation='none')
plt.colorbar()

plt.scatter(x=center[:,1], y=center[:,0], c='m', s=40)
plt.axis((-0.5,d2-0.5,-0.5,d1-0.5))
plt.gca().invert_yaxis()
#%%
crd = plot_contours(coo_matrix(Ain[:,::-1]),Cn,thr=0.9)
#%%
active_pixels = np.squeeze(np.nonzero(np.sum(Ain,axis=1)))
Yr = np.reshape(Y,(d1*d2,T),order='F')
p = 2;
P = arpfit(Yr,p=1,pixels = active_pixels)
Y_res = Yr - np.dot(Ain,Cin)
model = ProjectedGradientNMF(n_components=1, init='random', random_state=0)
model.fit(np.maximum(Y_res,0))

fin = model.components_.squeeze()
#%%
t1 = time()
A,b,Cin = update_spatial_components(Yr, Cin, fin, Ain, d1=d1, d2=d2, sn = P['sn'],dist=2,max_size=8,min_size=3)
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
A_m,C_m,nr_m,merged_ROIs,P_m=mergeROIS(Y_res,A_sp,b,np.array(C),f,d1,d2,Pnew,sn=P['sn'],thr=.7,deconv_method='spgl1',min_size=3,max_size=8,dist=2)
t_elMERGE = time() - t1
#%%
crd = plot_contours(A_m,Cn2,thr=0.9)
#%%
A2,b2,C_m_ = update_spatial_components(Yr, C_m, f, A_m, d1=d1, d2=d2, sn = P['sn'],dist=3,max_size=8,min_size=3)
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
#%% not ordered 
view_patches(Yr,coo_matrix(A2),C2,b,f,d1,d2)
#%%

view_patches(Yr,coo_matrix(A_or),C_or,b,f,d1,d2)
#%%
pl.plot(C2.T[:,0])
pl.plot(C2.T[:,1]*5)
pl.plot(C2.T[:,2]*5)


ff=f-np.percentile(f,20);
pl.plot(ff.T*5000,'k')
pl.legend(['neuron_1','neuron_2','neuron_3','trigger'])
#%%
Mn=np.std(m,axis=0)
crd = plot_contours(A2,Mn,thr=0.8)