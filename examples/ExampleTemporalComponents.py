# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:03:56 2015

@author: agiovann
"""
from scipy import io as sio
from update_temporal_components import update_temporal_components
import pylab as pl
import numpy as np
import time
#pl.ion()
#%load_ext autoreload
#%autoreload 2
#%%
efty_params = sio.loadmat('temporal_workspace.mat',struct_as_record=False) # load as structure matlab like

Y=efty_params['Yr']*1.0
C_in=efty_params['Cin']*1.0
f_in=efty_params['fin']*1.0
A=efty_params['A']*1.0
P=efty_params['P'][0,0] # necessary because of the way it is stored
P_new=efty_params['Pnew'][0,0] # necessary because of the way it is stored

b=efty_params['b']*1.0
f=efty_params['f']*1.0
C=efty_params['C']*1.0
Y_res=Y-np.dot(np.hstack((A.todense(),b)),np.vstack((C_in,f)))
#Y_res_out=efty_params['Y_res']*1.0
#demo_=np.load('demo_post_spatial.npz')
#Y=demo_['Y']
##A=demo_['A_out']
#b=demo_['b_out']
#fin=demo_['f_in']
#Cin=demo_['C_in'];
#g=demo_['g']
#sn=demo_['sn']
#d1=demo_['d1']
#d2=demo_['d2']
#P=demo_['P']
#%% 
#TODO: test with restimate_g=True
#TODO: test reordering of list
start=time.time()
C_out,f_out,Y_res_out,P_=update_temporal_components(Y,A,b,C_in,f_in,ITER=2,method='constrained_foopsi',g=None,bas_nonneg=False,p=2,fudge_factor=1);
print time.time()-start

kkk
#%%
#np.savez('after_temporal.npz',P_=P_)
##%%
#P_=np.load('after_temporal.npz')['arr_3']
#thr=0.85
#mx=50
#d1=P.d1
#d2=P.d2
#sn=P.sn
#from scipy.sparse import spdiags,coo_matrix,csgraph
#import scipy
#import numpy as np
#import cPickle as pickle
#from constrained_foopsi_AG import constrained_foopsi
#import random
#from scipy import linalg
#from update_spatial_components import update_spatial_components
