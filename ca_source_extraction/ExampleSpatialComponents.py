# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:17:13 2015

@author: agiovann
"""
#%load_ext autoreload
#%autoreload 2
import scipy.io as sio
import numpy as np
from update_spatial_components import update_spatial_components
#%% load example data
#efty_params = sio.loadmat('/Users/agiovann/Dropbox/preanalyzed data/ExamplesDataAnalysis/Andrea/GranuelCells/efty_params.mat',struct_as_record=False) # load as structure matlab like
#efty_params = sio.loadmat('demo_workspace.mat',struct_as_record=False) # load as structure matlab like
efty_params = sio.loadmat('demo_workspace_post.mat',struct_as_record=False) # load as structure matlab like

#%%
Y=efty_params['Yr']*1.0
C=efty_params['Cin']*1.0
f=efty_params['fin']*1.0
A_in=efty_params['Ain']*1.0
P=efty_params['P'][0,0] # necessary because of the way it is stored
A=efty_params['A']*1.0
b=efty_params['b']*1.0
#%%
A=sio.loadmat('Amat.mat')['A']
#%%
#A_out,b_out=update_spatial_components(Y,C,f,A_in,d1=d1,d2=d2,sn=sn)
print Y.shape
print type(Y)
print C.shape
print type(C)
print f.shape
print type(f)
print A_in.shape
print type(A_in)


#%%
#A_out,b_out=update_spatial_components(Y,C,f,A_in,d1=P.d1,d2=P.d2,g=P.g,sn=P.sn)
A_out,b_out=update_spatial_components(Y,C,f,A_in,d1=P.d1,d2=P.d2,sn=P.sn)

#%%
np.sum(np.abs(A.todense()-A_out.todense()))/np.sum(np.abs(A.todense()))
pl.imshow(A.todense(),aspect='auto',interpolation='none')
pl.figure()
pl.imshow(A_out.todense(),aspect='auto',interpolation='none')
#%%
#np.savez('demo_post_spatial',Y=Y,b_out=b_out,C_in=C,f_in=f,d1=P.d1,d2=P.d2,g=P.g,sn=P.sn,P=P)
##%%
#import cPickle as pickle
#import numpy as np
#import scipy.sparse


with open('demo_post.dat', 'wb') as outfile:
    pickle.dump(A_out, outfile, pickle.HIGHEST_PROTOCOL)