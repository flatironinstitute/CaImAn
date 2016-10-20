# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015
@author: agiovann
"""
%load_ext autoreload
%autoreload 2
import ca_source_extraction as cse
import h5py
import calblitz as cb
import time
import pylab as pl
import numpy as np
#% set basic ipython functionalities
#pl.ion()
#%load_ext autoreload
#%autoreload 2


#%%
filename='movies/demoMovie_CTX.tif'
frameRate=15.62;
start_time=0;
#%%
filename_py=filename[:-4]+'.npz'
filename_hdf5=filename[:-4]+'.hdf5'
filename_mc=filename[:-4]+'_mc.npz'

#%% load and motion correct movie (see other Demo for more details)
m=cb.load(filename, fr=frameRate,start_time=start_time);
#m=m[:,20:35,20:35]
[T,d1,d2]=m.shape
#%%
#m_star=m.local_correlations_movie(window=50)
m_star=m;
m_star=m-m.zproject(method='median')

m_star=np.maximum(m_star,0)

comps,traces=m_star.NonnegativeMatrixFactorization(n_components=30)
#nnm=NMF(n_components=3)
#traces=nnm.fit_transform(np.maximum(m_star.to_2D(),0))
#comps=nnm.components_
for comp in comps:
    pl.imshow(np.reshape(comp,(d1,d2),order='F'))
    pl.pause(1)    
#%%
comps,_,_=m_star.partition_FOV_KMeans(n_clusters=3)
pl.imshow(comps)

