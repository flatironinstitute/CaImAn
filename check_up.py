#%%
import sys
import numpy as np
import scipy.io as sio
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
#% for caching
import tempfile
import shutil
import os
#%%
n_processes=4 # roughly number of cores on your machine minus 2

p=2 # order of the AR model (in general 1 or two)

#%% start cluster for efficient computation
print "(Stopping)  and restarting cluster to avoid unnencessary use of memory...."

sys.stdout.flush()  
proc_1=subprocess.Popen(["ipcluster stop"],shell=True)
tm.sleep(5)

sys.stdout.flush()    
proc_2=subprocess.Popen(["ipcluster start -n " + str(n_processes)],shell=True) 
tm.sleep(5)
#%% LOAD MOVIE AND MAKE DIMENSIONS COMPATIBLE WITH CNMF
try: 
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
    n_pixels_per_process=d1*d2/n_processes # how to subdivide the work among processes
    #%% prepare parameters
    preprocess_params={ 'sn':None, 'g': None, 'noise_range' : [0.25,0.5], 'noise_method':'logmexp',
                        'n_processes':n_processes, 'n_pixels_per_process':n_pixels_per_process,   
                        'compute_g':False, 'p':p,   
                        'lags':5, 'include_noise':False, 'pixels':None}
    init_params = { 
                        'K':30,'gSig':[4,4],'gSiz':[9,9], 
                        'ssub':1,'tsub':1,
                        'nIter':5, 'kernel':None,
                        'maxIter':5
                        }
    spatial_params = {               
                        'dims':(d1, d2), 'dist':3,   'method' : 'ellipse',               
                        'n_processes':n_processes,'n_pixels_per_process':n_pixels_per_process,
                        'backend':'ipyparallel'
                    }
    temporal_params = {
                        'ITER':2, 'method':'cvxpy', 'p':p,
                        'n_processes':n_processes,'backend':'ipyparallel',
                        'bas_nonneg':True,  
                        'noise_range':[.25,.5], 'noise_method':'logmexp', 
                        'lags':5, 'fudge_factor':1., 
                        'verbosity':False
                    }
    #%% PREPROCESS DATA
    t1 = time()
    Yr,sn,g,_=cse.preprocess_data(Yr,**preprocess_params)
    Ain, Cin, b_in, f_in, center=cse.initialize_components(Y, **init_params)                                                    
    print time() - t1  
    
      
    #%%
    t1 = time()
    A,b,Cin = cse.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **spatial_params)
    t_elSPATIAL = time() - t1
    print t_elSPATIAL 
    #%% update_temporal_components
    t1 = time()
    C,f,Y_res,S,bl,c1,neurons_sn,g = cse.update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**temporal_params)
    t_elTEMPORAL2 = time() - t1
    print t_elTEMPORAL2 # took 98 sec   
    #%% merge components corresponding to the same neuron
    t1 = time()
    # A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Y_res,A,b,C,f,S,sn,temporal_params, spatial_params, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, mx=50)
    t_elMERGE = time() - t1
    print t_elMERGE  
except:
    print 'test failed'
    raise
#%% STOP CLUSTER
finally:
    print "Stopping Cluster...."
    sys.stdout.flush()  
    proc_2=subprocess.Popen(["ipcluster stop"],shell=True)
    tm.sleep(5)
