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
#%%
n_processes = np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
#print 'using ' + str(n_processes) + ' processes'
p=2 # order of the AR model (in general 1 or 2)

#%% start cluster for efficient computation
print "Stopping  cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server()

#%% LOAD MOVIE AND MAKE DIMENSIONS COMPATIBLE WITH CNMF
reload=0
filename='demoMovie.tif'
#%%
t = tifffile.TiffFile(filename) 
Yr = t.asarray().astype(dtype=np.float32) 
Yr = np.transpose(Yr,(1,2,0))
d1,d2,T=Yr.shape
Yr=np.reshape(Yr,(d1*d2,T),order='F')
#np.save('Y',Y)
np.save('Yr',Yr)
#Y=np.load('Y.npy',mmap_mode='r')
Yr=np.load('Yr.npy',mmap_mode='r')        
Y=np.reshape(Yr,(d1,d2,T),order='F')
#%% new memmap
if 0:
    fname_new=cse.utilities.save_memmap([filename],base_name='Ytest')
    Yr,d1,d2,T=cse.utilities.load_memmap(fname_new)
    Y=np.reshape(Yr,(d1,d2,T),order='F')
#%%
Cn = cse.utilities.local_correlations(Y)
#n_pixels_per_process=d1*d2/n_processes # how to subdivide the work among processes

#%%
options = cse.utilities.CNMFSetParms(Y,p=p,gSig=[4,4],K=30)
cse.utilities.start_server(options['spatial_params']['n_processes'])

#%% PREPROCESS DATA AND INITIALIZE COMPONENTS
t1 = time()
Yr,sn,g,psx = cse.pre_processing.preprocess_data(Yr,**options['preprocess_params'])
Atmp, Ctmp, b_in, f_in, center=cse.initialization.initialize_components(Y, **options['init_params'])                                                    
print time() - t1

#%% Refine manually component by clicking on neurons 
refine_components=False
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


