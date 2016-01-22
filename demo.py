#%%
try:
    #%load_ext autoreload
    #%autoreload 2
    print 1
except:
    print 'NOT IPYTHON'

import sys
import numpy as np
#sys.path.append('../SPGL1_python_port')
import ca_source_extraction as cse
#%
from matplotlib import pyplot as plt
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
#% for caching
import psutil
#%%
n_processes = np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
#print 'using ' + str(n_processes) + ' processes'
p=2 # order of the AR model (in general 1 or 2)

#%% start cluster for efficient computation
print "(Stopping)  and restarting cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server()

#%% LOAD MOVIE AND MAKE DIMENSIONS COMPATIBLE WITH CNMF
reload=0
filename='movies/demoMovie.tif'
t = tifffile.TiffFile(filename) 
Y = t.asarray().astype(dtype=np.float32) 
Y = np.transpose(Y,(1,2,0))
d1,d2,T=Y.shape
Yr=np.reshape(Y,(d1*d2,T),order='F')
np.save('Y',Y)
np.save('Yr',Yr)
Y=np.load('Y.npy',mmap_mode='r')
Yr=np.load('Yr.npy',mmap_mode='r')        
d1,d2,T=Y.shape
Cn = cse.local_correlations(Y)
#n_pixels_per_process=d1*d2/n_processes # how to subdivide the work among processes

#%%
options = cse.utilities.CNMFSetParms(Y,p=p,gSig=[4,4])
cse.utilities.start_server(options['spatial_params']['n_processes'])

#%% PREPROCESS DATA AND INITIALIZE COMPONENTS
t1 = time()
Yr,sn,g=cse.preprocess_data(Yr,**options['preprocess_params'])
Ain, Cin, b_in, f_in, center=cse.initialize_components(Y, **options['init_params'])                                                    
print time() - t1
#%% 
plt2 = plt.figure(); plt.imshow(Cn,interpolation='None')
plt.colorbar()
plt.scatter(x=center[:,1], y=center[:,0], c='m', s=40)
crd = cse.plot_contours(coo_matrix(Ain),Cn,thr=0.9)
plt.axis((0,d2-1,0,d1-1))
plt2.suptitle('Spatial Components found with initialization', fontsize=16)
plt.gca().invert_yaxis()
plt.show()
  
#%% UPDATE SPATIAL COMPONENTS
t1 = time()
A,b,Cin = cse.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])
t_elSPATIAL = time() - t1
print t_elSPATIAL 
plt.figure()
crd = cse.plot_contours(A,Cn,thr=0.9)
#%% update_temporal_components
t1 = time()
C,f,S,bl,c1,neurons_sn,g = cse.update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
t_elTEMPORAL2 = time() - t1
print t_elTEMPORAL2 
#%% merge components corresponding to the same neuron
t1 = time()
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'], options['spatial_params'], bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, mx=50, fast_merge = True)
t_elMERGE = time() - t1
print t_elMERGE  

#%%
plt.figure()
crd = cse.plot_contours(A_m,Cn,thr=0.9)
#%% refine spatial and temporal 
t1 = time()
A2,b2,C2 = cse.update_spatial_components(Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
#C2,f2,Y_res2,S2,bl2,c12,neurons_sn2,g21 = cse.update_temporal_components_parallel(Yr,A2,b2,C2,f,bl=bl_m,c1=c1_m,sn=sn_m,g=g_m,**temporal_params)
C2,f2,S2,bl2,c12,neurons_sn2,g21 = cse.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
print time() - t1
#%%
A_or, C_or, srt = cse.order_components(A2,C2)
cse.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2, d1,d2,secs=0)    
#%%
plt.figure()
crd = cse.plot_contours(A_or,Cn,thr=0.9)
#%% STOP CLUSTER
cse.utilities.stop_server()
