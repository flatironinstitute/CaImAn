# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:42:50 2016

@author: agiovann
"""

import sys
import numpy as np
import scipy.io as sio

sys.path.append('../SPGL1_python_port')
import ca_source_extraction as cse
#%
from matplotlib import pyplot as plt
from time import time
import pylab as pl
from scipy.sparse import coo_matrix
import scipy
from sklearn.decomposition import NMF
import tempfile
import os
import tifffile
import subprocess
import time as tm
from time import time
import calblitz as cb
#% for caching
caching=1

if caching:
    import tempfile
    import shutil
    import os

#% TYPE DECONVOLUTION
#deconv_type='debug'
deconv_type='spgl1'
large_data=False
remove_baseline=False
window_baseline=700
quantile_baseline=.1


#%    
if caching:
    Y=np.load('Y.npy',mmap_mode='r')
    Yr=np.load('Yr.npy',mmap_mode='r')        
else:
    Y=np.load('Y.npy')
    Yr=np.load('Yr.npy') 
    
d1,d2,T=Y.shape
#%%
with np.load('temp_pre_initialize.npz') as ld:
    locals().update(ld)
spatial_params=spatial_params.item()
temporal_params=temporal_params.item()
init_params=init_params.item()
#%%
t1 = time()
Ain, Cin, b_in, f_in, center=cse.initialize_components(Y, **init_params)                                                    
print time() - t1 #
plt2 = plt.imshow(Cn,interpolation='None')
plt.colorbar()
plt.scatter(x=center[:,1], y=center[:,0], c='m', s=40)
crd = cse.plot_contours(coo_matrix(Ain[:,::-1]),Cn,thr=0.9)
plt.axis((-0.5,d2-0.5,-0.5,d1-0.5))
plt.gca().invert_yaxis()
pl.show()
#%%
#A,b,Cin = cse.update_spatial_components_parallel(Yr, Cin, f_in, Ain, sn=sn, **spatial_params)
#C,f,Y_res,S,bl,c1,neurons_sn,g = cse.update_temporal_components_parallel(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**temporal_params)
#%%
#cse.view_patches(Yr,A,C,b,f,d1,d2,secs=0)
