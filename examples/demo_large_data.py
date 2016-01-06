#%%
try:
    %load_ext autoreload
    %autoreload 2
    print 1
except:
    print 'NOT IPYTHON'
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
#%% for caching
caching=1

if caching:
    import tempfile
    import shutil
    import os

#%% TYPE DECONVOLUTION
#deconv_type='debug'
deconv_type='spgl1'
large_data=False
#%%

filename='movies/demoMovie.tif'
if 1:
    t = tifffile.TiffFile(filename) 
    Y = t.asarray() 
    Y = np.transpose(Y,(1,2,0))*1.
    d1,d2,T=Y.shape
    np.save('Y',Y)
    np.save('Yr',np.reshape(Y,(d1*d2,T),order='F'))

if caching:
    Y=np.load('Y.npy',mmap_mode='r')
    Yr=np.load('Yr.npy',mmap_mode='r')        
else:
    Y=np.load('Y.npy')
    Yr=np.load('Yr.npy') 
    
d1,d2,T=Y.shape
#%%
if not large_data:
    Cn = cse.local_correlations(Y)
else:
    Cn=np.mean(Y,axis=2) 
#%%
n_processes=4
n_pixels_per_process=d1*d2/n_processes
p=2;

preprocess_params={ 'sn':None, 'g': None, 'noise_range' : [0.25,0.5], 'noise_method':'logmexp',
                    'n_processes':n_processes, 'n_pixels_per_process':n_pixels_per_process,   
                    'compute_g':False, 'p':p,   
                    'lags':5, 'include_noise':False, 'pixels':None}
init_params = { 
                    'K':30,'gSig':[4,4],'gSiz':[9,9], 
                    'ssub':1,'tsub':1,
                    'nIter':5, 'use_median':False, 'kernel':None
                    }
#%% PREPROCESS DATA
t1 = time()
Yr,sn,g=cse.preprocess_data(Yr,**preprocess_params)
print time() - t1 # 
#%%
#'sn' : sn,
spatial_params = {               
                'd1':d1,  'd2':d2, 'dist':3,   'method' : 'ellipse',               
                'n_processes':n_processes,'n_pixels_per_process':n_pixels_per_process,'backend':'ipyparallel',
                'memory_efficient':False
                }
temporal_params = {
#                'bl':None,  'c1':None, 'g':None, 'sn' : None,
                'ITER':2, 'method':deconv_type, 'p':p,
                'n_processes':n_processes,'backend':'ipyparallel',
                'memory_efficient':False,                                
                'bas_nonneg':True,  
                'noise_range':[.25,.5], 'noise_method':'logmexp', 
                'lags':5, 'fudge_factor':1., 
                'verbosity':False
                }
#%%
t1 = time()
Ain, Cin, b_in, f_in, center=cse.initialize_components(Y, **init_params)                                                    
print time() - t1 # 
#%%
plt2 = plt.imshow(Cn,interpolation='None')
plt.colorbar()
plt.scatter(x=center[:,1], y=center[:,0], c='m', s=40)
crd = cse.plot_contours(coo_matrix(Ain[:,::-1]),Cn,thr=0.9)
plt.axis((-0.5,d2-0.5,-0.5,d1-0.5))
plt.gca().invert_yaxis()

#%% start cluster for efficient computation
print "Starting Cluster...."
sys.stdout.flush()    
p=subprocess.Popen(["ipcluster start -n " + str(n_processes)],shell=True) 
tm.sleep(10)    
#%%
t1 = time()
A,b,Cin = cse.update_spatial_components_parallel(Yr, Cin, f_in, Ain, sn=sn, **spatial_params)
t_elSPATIAL = time() - t1
print t_elSPATIAL 
#%%
#if 1:
#    np.savez('tmp_temporal.npz',Yr=Yr,A=A.todense(),b=b,Cin=Cin,f_in=f_in,temporal_params=temporal_params)
#else:
#    with np.load('tmp_temporal.npz') as fl:
#        Yr=fl['Yr']
#        A=fl['A']
#        b=fl['b']
#        Cin=fl['Cin']
#        f_in=fl['f_in']
#        temporal_params=fl['temporal_params']
       
    
#%%
crd = cse.plot_contours(A,Cn,thr=0.9)
#%%
#t1 = time()
#Axxx,bxxx,Cinxxx = cse.update_spatial_components(Yr, Cin, f_in, Ain, d1=d1, d2=d2, sn = sn,dist=2,max_size=8,min_size=3)
#t_elSPATIAL = time() - t1
#print t_elSPATIAL#        
#
#pl.figure()
#crd = cse.plot_contours(Axxx,np.mean(np.reshape(Yr,(d1,d2,np.shape(Yr)[-1]),order='F'),axis=2),thr=0.95)
#%% restart cluster
if caching:
    print "Stopping  and restarting cluster to avoid unnencessary use of memory...."
    sys.stdout.flush()  
    q=subprocess.Popen(["ipcluster stop"],shell=True)
    tm.sleep(5)
    print "restarting cluster..."
    sys.stdout.flush() 
    p=subprocess.Popen(["ipcluster start -n " + str(n_processes)],shell=True) 
    tm.sleep(10)
#%% update_temporal_components
t1 = time()
C,f,Y_res,S,bl,c1,neurons_sn,g = cse.update_temporal_components_parallel(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**temporal_params)
t_elTEMPORAL2 = time() - t1
print t_elTEMPORAL2 # took 98 sec    
#%%
#t1 = time()
#Cxx,fxx,Y_resxx,Pnewxx,Sxx = cse.update_temporal_components(Yr,A,b,Cin,f_in,**temporal_params)
#t_elTEMPORAL2 = time() - t1
#print t_elTEMPORAL2 # took 98 sec 
#%%
#if 1:
#    np.savez('tmp_temporal_2.npz',Yr=Yr,Y_res=Y_res,A=A.todense(),b=b,C=C,f=f,S=S,bl=bl,c1=c1,sn=sn,g=g,temporal_params=temporal_params,spatial_params=spatial_params)
#else:
#    with np.load('tmp_temporal_2.npz') as fl:
#        Yr=fl['Yr']
#        Y_res=fl['Y_res']
#        A=coo_matrix(fl['A'])
#        b=fl['b']
#        C=fl['C']
#        f=fl['f']
#        S=fl['S']
#        bl=fl['bl']
#        c1=fl['c1']
#        sn=fl['sn']
#        g=fl['g']
#        temporal_params=fl['temporal_params']
#        spatial_params=fl['spatial_params']
#%%
t1 = time()
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.mergeROIS_parallel(Y_res,A,b,C,f,S,sn,temporal_params, spatial_params, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, mx=50)
t_elMERGE = time() - t1
print t_elMERGE  
#%% STOP CLUSTER
print "Stopping Cluster...."
sys.stdout.flush()  
q=subprocess.Popen(["ipcluster stop"],shell=True)
tm.sleep(5)
p.terminate()
q.terminate()