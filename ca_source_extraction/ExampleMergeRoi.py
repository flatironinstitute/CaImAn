# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:15:02 2015

@author: agiovann
"""

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
from merge_rois import mergeROIS
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
#P=demo_['P

#%%
P_=np.load('after_temporal.npz')['arr_3']
thr=0.8
mx=50
d1=P.d1
d2=P.d2
sn=P.sn
#%%
A_m,C_m,nr_m,merged_ROIs,P_m=mergeROIS(Y_res,A,b,C,f,d1,d2,P_,sn=sn)
#%%
efty_params_after = sio.loadmat('workspace_after_merge.mat',struct_as_record=False) # load as structure matlab like
A_ef=efty_params_after['Am']*1.0
C_ef=efty_params_after['Cm']*1.0
merged_ROIs_ef=efty_params_after['merged_ROIs']*1.0
P_new_ef=efty_params_after['P'][0,0]
#%%
np.sum(np.abs(A_m-A_ef).todense())/np.sum(np.abs(A_m).todense())
np.sum(np.abs(C_m-C_ef))/np.sum(np.abs(C_m))
#%%
display_merging=1;
for roi in merged_ROIs:

    ln = len(roi)
    pl.figure()
    set(gcf,'Position',[300,300,(ln+2)*300,300]);
    for j in range(ln):
        subplot(1,ln+2,j)
        pl.imshow(np.reshape(A[:,roi[j]],(d1,d2),order='F')) 
    #    title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
    
        subplot(1,ln+2,ln+1)
        pl.imshow(np.reshape(A_m[:,nr_m-len(roi)+i),(d1,d2),order='F'));
        #title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight; 
        subplot(1,ln+2,ln+2)
        traces=scipy.linalg.solve(np.diag(np.max(C[roi,:],axis=1)),C[roi,:]).T
        pl.plot(range(T),traces); 
        pl.plot(1:T,Cm(nr_m-length(merged_ROIs)+i,:)/max(Cm(nr_m-length(merged_ROIs)+i,:)),'--k')

#title('Temporal Components','fontsize',16,'fontweight','bold')
#drawnow;