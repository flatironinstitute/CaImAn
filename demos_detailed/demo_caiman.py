from __future__ import print_function
##%%
from builtins import str
from builtins import range
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:       
    print('Not IPYTHON')    
    pass

import sys
import numpy as np
from time import time
from scipy.sparse import coo_matrix
import psutil
import glob
import os
import scipy
from ipyparallel import Client
#mpl.use('Qt5Agg')
import pylab as pl
pl.ion()
#%%
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.source_extraction import cnmf
#%%
#backend='SLURM'
backend='local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()),1) # roughly number of cores on your machine minus 1
print(('using ' + str(n_processes) + ' processes'))
#%% start cluster for efficient computation
single_thread=False

if single_thread:
    dview=None
else:    
    try:
        c.close()
    except:
        print('C was not existing, creating one')
    print("Stopping  cluster to avoid unnencessary use of memory....")
    sys.stdout.flush()  
    if backend == 'SLURM':
        try:
            stop_server(is_slurm=True)
        except:
            print('Nothing to stop')
        #todocument
        slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cm.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)        
    else:
        cm.stop_server()
        cm.start_server()        
        c=Client()

    print(('Using '+ str(len(c)) + ' processes'))
    dview=c[:len(c)]
#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames=[]
base_folder='./example_movies/' # folder containing the demo files
for file in glob.glob(os.path.join(base_folder,'*.tif')):
    if file.endswith("ie.tif"):
        fnames.append(os.path.abspath(file))

fnames.sort()
if len(fnames)==0:
    #todocument raise
    raise Exception("Could not find any tiff file")

print(fnames)  
fnames=fnames
##%%
#idx_x=slice(12,500,None)
#idx_y=slice(12,500,None)
#idx_xy=(idx_x,idx_y)
add_to_movie=0 # of movie too negative need to add a baseline
downsample_factor=1 # use .2 or .1 if file is large and you want a quick answer
idx_xy=None
base_name='Yr'
name_new=cm.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy,add_to_movie=add_to_movie )
#todocument sort
name_new.sort()
print(name_new)

#%%
#todocument return
fname_new=cm.save_memmap_join(name_new,base_name='Yr', n_chunks=12, dview=dview)
#%%
Yr,dims,T=cm.load_memmap(fname_new)
Y=np.reshape(Yr,dims+(T,),order='F')
#%% visualize correlation image
Cn = cm.local_correlations(Y)
pl.imshow(Cn,cmap='gray')   
#%% parameters of experiment
K=30 # number of neurons expected per patch
gSig=[7,7] # expected half size of neurons
merge_thresh=0.8 # merging threshold, max correlation allowed
p=2 #order of the autoregressive system
options = cnmf.utilities.CNMFSetParms(Y,n_processes,p=p,gSig=gSig,K=K,ssub=2,tsub=2,nb=1, normalize_init=True)
options['preprocess_params']['noise_method']='mean'
#%% PREPROCESS DATA AND INITIALIZE COMPONENTS
t1 = time()
Yr,sn,g,psx = cm.source_extraction.cnmf.pre_processing.preprocess_data(Yr,dview=dview,**options['preprocess_params'])
print((time() - t1))
#%%
t1 = time()
Atmp, Ctmp, b_in, f_in, center=cm.source_extraction.cnmf.initialization.initialize_components(Y, **options['init_params'])
print((time() - t1))
#%% Refine manually component by clicking on neurons 
refine_components=False
if refine_components:
    Ain,Cin = cm.source_extraction.cnmf.utilities.manually_refine_components(Y,options['init_params']['gSig'],coo_matrix(Atmp),Ctmp,Cn,thr=0.9)
else:
    Ain,Cin = Atmp, Ctmp
#%% plot estimated component
pl.figure()
crd = plot_contours(coo_matrix(Ain),Cn)  
pl.show()
#%% UPDATE SPATIAL COMPONENTS
#pl.close()
t1 = time()
A,b,Cin,f_in = cm.source_extraction.cnmf.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, dview=dview,**options['spatial_params'])
t_elSPATIAL = time() - t1
pl.figure()
crd = plot_contours(A,Cn)

#%% update_temporal_components
#pl.close()
t1 = time()
options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
C,A,b,f,S,bl,c1,neurons_sn,g,YrA = cm.source_extraction.cnmf.temporal.update_temporal_components(Yr,A,b,Cin,f_in,dview=dview,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
t_elTEMPORAL = time() - t1
print(t_elTEMPORAL) 
#%% merge components corresponding to the same neuron
t1 = time()
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cm.source_extraction.cnmf.merging.merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'], options['spatial_params'],dview=dview, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=merge_thresh, mx=50, fast_merge = True)
t_elMERGE = time() - t1
print(t_elMERGE)  


#%%
#plt.figure()
#crd = cm.source_extraction.cnmf.plot_contours(A_m,Cn,thr=0.9)
#%% refine spatial and temporal 
#pl.close()
t1 = time()
A2,b2,C2,f = cm.source_extraction.cnmf.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn,dview=dview, **options['spatial_params'])
options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
C2,A2,b2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cm.source_extraction.cnmf.temporal.update_temporal_components(Yr,A2,b2,C2,f,dview=dview, bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
print((time() - t1))

pl.figure()
crd = plot_contours(A2.tocsc()[:,:],Cn,thr=0.9)

#%%
#%%
final_frate = 10

Npeaks = 10
traces = C + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                                      N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

idx_components_r = np.where(r_values >= .85)[0]
idx_components_raw = np.where(fitness_raw < -40)[0]
idx_components_delta = np.where(fitness_delta < -40)[0]


#min_radius = gSig[0] - 2
#masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
#    A.tocsc(), min_radius, dims, num_std_threshold=1,
#    minCircularity=0.7, minInertiaRatio=0.2, minConvexity=.5)

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
#idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

print(' ***** ')
print((len(traces)))
print((len(idx_components)))
#print((len(idx_blobs)))



min_radius=gSig[0]-2
masks_ws,idx_blobs,idx_non_blobs=extract_binary_masks_blob(
A2.tocsc(), min_radius, dims, num_std_threshold=1, 
minCircularity= 0.6, minInertiaRatio = 0.2,minConvexity =.8)




idx_components=np.union1d(idx_components_r,idx_components_raw)
idx_components=np.union1d(idx_components,idx_components_delta)  
idx_blobs=np.intersect1d(idx_components,idx_blobs)   
idx_components_bad=np.setdiff1d(list(range(len(traces))),idx_components)

print(' ***** ')
print((len(traces)))
print((len(idx_components)))
print((len(idx_blobs)))
#%% visualize components
#pl.figure();
pl.subplot(1,3,1)
crd = plot_contours(A2.tocsc()[:,idx_components],Cn,thr=0.9)
pl.subplot(1,3,2)
crd = plot_contours(A2.tocsc()[:,idx_blobs],Cn,thr=0.9)
pl.subplot(1,3,3)
crd = plot_contours(A2.tocsc()[:,idx_components_bad],Cn,thr=0.9)
#%%
view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,idx_components]),C2[idx_components,:],b2,f2, dims[0],dims[1], YrA=YrA[idx_components,:],img=Cn)  
#%%
view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,idx_components_bad]),C2[idx_components_bad,:],b2,f2, dims[0],dims[1], YrA=YrA[idx_components_bad,:],img=Cn)  
#%% STOP CLUSTER
pl.close()
if not single_thread:    
    c.close()
    cm.stop_server()
