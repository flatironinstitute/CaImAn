#%%
try:
#    %load_ext autoreload
#    %autoreload 2
    print 1
except:
    print 'NOT IPYTHON'

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
#plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse

#sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import calblitz as cb
#%% load and motion correct
preprocess=1
if preprocess:
    Yr=cb.load('k37_20160110_RSM_150um_65mW_zoom2p2_00001_00004.tif',fr=30)
    t1 = time()
    Yr,shifts,xcorrs,template=Yr.motion_correct(max_shift_w=10, max_shift_h=10, num_frames_template=500, template=None, method='opencv')    
    Yr.save('k37_20160110_RSM_150um_65mW_zoom2p2_00001_00004.hdf5')    
    Yr=Yr-np.percentile(Yr,1)     # needed to remove 
    Yr = np.transpose(Yr,(1,2,0)) 
    d1,d2,T=Yr.shape
    Yr=np.reshape(Yr,(d1*d2,T),order='F')
    np.save('Yr',np.asarray(Yr))
    print time() - t1

_,d1,d2=np.shape(cb.load('k37_20160110_RSM_150um_65mW_zoom2p2_00001_00004.hdf5',subindices=range(3)))
Yr=np.load('Yr.npy',mmap_mode='r')  
d,T=Yr.shape      
Y=np.reshape(Yr,(d1,d2,T),order='F')
#%%compute local correlation
Cn = cse.utilities.local_correlations(Y)
#%%
n_processes = np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
#print 'using ' + str(n_processes) + ' processes'
p=2 # order of the AR model (in general 1 or 2)
print "Stopping  cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server()

#%%
options = cse.utilities.CNMFSetParms(Y,p=p,gSig=[7,7],K=300,ssub=2)
cse.utilities.start_server(options['spatial_params']['n_processes'])

#%% PREPROCESS DATA AND INITIALIZE COMPONENTS
t1 = time()
Yr,sn,g=cse.pre_processing.preprocess_data(Yr,**options['preprocess_params'])
Atmp, Ctmp, b_in, f_in, center=cse.initialization.initialize_components(Y, **options['init_params'])                                                    
print time() - t1

#%% Refine manually component by clicking on neurons 
refine_components=True
if refine_components:
    Ain,Cin = cse.utilities.manually_refine_components(Y,options['init_params']['gSig'],coo_matrix(Atmp),Ctmp,Cn,thr=0.9)
else:
    Ain,Cin = Atmp, Ctmp
#%% plot estimated component
crd = cse.utilities.plot_contours(coo_matrix(Ain),Cn,thr=0.9)  
pl.show()
#%% UPDATE SPATIAL COMPONENTS
pl.close()
t1 = time()
A,b,Cin = cse.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])
t_elSPATIAL = time() - t1
print t_elSPATIAL 
plt.figure()
crd = cse.utilities.plot_contours(A,Cn,thr=0.9)
#%% update_temporal_components
pl.close()
t1 = time()
options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
options['temporal_params']['fudge_factor'] = 0.96 
C,f,S,bl,c1,neurons_sn,g,YrA = cse.temporal.update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
t_elTEMPORAL = time() - t1
print t_elTEMPORAL 
#%% merge components corresponding to the same neuron
t1 = time()
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merging.merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'], options['spatial_params'], bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, mx=50, fast_merge = True)
t_elMERGE = time() - t1
print t_elMERGE  

#%%
plt.figure()
crd = cse.plot_contours(A_m,Cn,thr=0.9)
#%% refine spatial and temporal 
pl.close()
t1 = time()
A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
print time() - t1
#%%
A_or, C_or, srt = cse.utilities.order_components(A2,C2)
#cse.utilities.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2,d1,d2,YrA = YrA[srt,:], secs=1)
cse.utilities.view_patches_bar(Yr,coo_matrix(A_or),C_or,b2,f2, d1,d2, YrA=YrA[srt,:])  
#plt.show(block=True) 
plt.show()  
 
#%%

plt.figure()
crd = cse.utilities.plot_contours(A_or,Cn,thr=0.9)

#%% STOP CLUSTER
pl.close()
cse.utilities.stop_server()
#%% save results
np.savez('results_analysis_004.npz',A=A.todense(),b=b,Cin=Cin,f_in=f_in, Ain=Ain, sn=sn,C=C,bl=bl,f=f,c1=c1,S=S, neurons_sn=neurons_sn, g=g,A2=A2.todense(),C2=C2,b2=b2,S2=S2,f2=f2,bl2=bl2,c12=c12, neurons_sn2=neurons_sn2, g21=g21,YrA=YrA,d1=d1,d2=d2)
#%% reload and show analysis results
import sys
import numpy as np
import ca_source_extraction as cse
from scipy.sparse import coo_matrix
import pylab as pl
import calblitz as cb


_,d1,d2=np.shape(cb.load('k37_20160110_RSM_150um_65mW_zoom2p2_00001_00004.hdf5',subindices=range(3)))
Yr=np.load('Yr.npy',mmap_mode='r')  
d,T=Yr.shape      
Y=np.reshape(Yr,(d1,d2,T),order='F')

with np.load('results_analysis_004.npz')  as ld:
      locals().update(ld)

A2=coo_matrix(A2)      
A_or, C_or, srt = cse.utilities.order_components(A2,C2)
#cse.utilities.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2,d1,d2,YrA = YrA[srt,:], secs=1)
cse.utilities.view_patches_bar(Yr,coo_matrix(A_or),C_or,b2,f2, d1,d2, YrA=YrA[srt,:])  
#plt.show(block=True) 
plt.show()    