# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:03:56 2015

@author: agiovann
"""

from scipy import io as sio
from update_temporal_components import update_temporal_components
#%%
efty_params = sio.loadmat('workplace_full.mat',struct_as_record=False) # load as structure matlab like

Y=efty_params['Yr']*1.0
Cin=efty_params['Cin']*1.0
fin=efty_params['fin']*1.0
A=efty_params['A']*1.0
P=efty_params['P'][0,0] # necessary because of the way it is stored
A=efty_params['A']*1.0
b=efty_params['b']*1.0
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


C,f,Y_res,P_=update_temporal_components(Y,A,b,Cin,fin,ITER=1,restimate_g=True,method='constrained_foopsi',g=P.g,bas_nonneg=False,p=2,fudge_factor=1);
    