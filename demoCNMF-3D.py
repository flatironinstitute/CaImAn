
# coding: utf-8

# In[ ]:

# SKIP THIS IF YOU WANT TO USE THE NON WEB INTERFACE (can only be done when notebook run locally)
# %matplotlib inline


# In[1]:

#%%
try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
    print 1
except:
    print 'NOT IPYTHON'

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse
from scipy.sparse import coo_matrix
import tifffile
from time import time
import psutil
from scipy.ndimage.filters import gaussian_filter


# In[2]:

import bokeh.plotting as bpl
from bokeh.io import vform,hplot,vplot,gridplot
from bokeh.models import CustomJS, ColumnDataSource, Slider
from IPython.display import display, clear_output

bpl.output_notebook()


# In[3]:

# from scipy.sparse import spdiags, diags, coo_matrix
# import matplotlib.cm as cm
# import bokeh
# from bokeh.models import Range1d
   


# In[4]:

#%%
n_processes = np.maximum(np.int(psutil.cpu_count()*.75),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'
print "Stopping  cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server()


# In[5]:

def gen_data(p = 1, noise=.5, T=200, framerate=30, firerate=2., plot=False):
    if p==2:
        gamma = np.array([1.5,-.55])
    elif p==1:
        gamma = np.array([.9])
    else: raise
    dims = (30, 40, 50)  # size of image
    sig = (2, 2, 2)  # neurons size
    bkgrd = 10
    N = 20  # number of neurons
    np.random.seed(5)
    centers = np.asarray([[np.random.randint(5,x-5) 
                           for x in dims] for i in range(N)])
    Yr = np.zeros(dims + (T,), dtype=np.float32)
    trueSpikes = np.random.rand(N,T) < firerate / float(framerate)
    trueSpikes[:,0] = 0
    truth = trueSpikes.astype(np.float32)
    for i in range(2,T):
        if p ==2:
            truth[:,i] += gamma[0]*truth[:,i-1] + gamma[1]*truth[:,i-2] 
        else:
            truth[:,i] += gamma[0]*truth[:,i-1]
    for i in range(N):
        Yr[centers[i,0],centers[i,1],centers[i,2]] = truth[i]
    tmp=np.zeros(dims)
    tmp[15,20,25]=1.
    z=np.linalg.norm(gaussian_filter(tmp, sig).ravel())
    Yr = bkgrd + noise * np.random.randn(*(dims + (T,))) + 10 *gaussian_filter(Yr, sig + (0,)) /z
    d1,d2,d3,T=Yr.shape
    Yr=np.reshape(Yr,(d1*d2*d3,T),order='F').astype(np.float32)
 
    if plot:
        Y=np.reshape(Yr,(d1,d2,d3,T),order='F')
        Cn = cse.utilities.local_correlations(Y)
        plt.figure(figsize=(15,3))
        plt.plot(truth.T);
        plt.figure(figsize=(15,3))
        for c in centers:
            plt.plot(Y[c[0],c[1],c[2]]);

        plt.figure(figsize=(15,4))
        plt.subplot(131)
        plt.scatter(*centers.T[::-1],c='g')
        plt.imshow(Y.max(0).max(-1),cmap='hot');plt.title('Max.proj. x & t')
        plt.subplot(132)
        plt.scatter(*centers.T[[2,0,1]],c='g')
        plt.imshow(Y.max(1).max(-1),cmap='hot');plt.title('Max.proj. y & t')
        plt.subplot(133)
        plt.scatter(*centers.T[[1,0,2]],c='g')
        plt.imshow(Y.max(2).max(-1),cmap='hot');plt.title('Max.proj. z & t')
        plt.show()
        
    return Yr, truth, trueSpikes, centers, dims


# In[6]:

plt.close('all')
#%% SAVING TIFF FILE ON A SINGLE MEMORY MAPPABLE FILE
try:
    fname_new=cse.utilities.save_memmap(['movies/demoMovie3D.tif'],base_name='Yr')
except: #%% create 3d tiff file if not yet existent
    Yr, truth, trueSpikes, centers, dims = gen_data(p=2)
    # imagej takes data shapes up to 6 dimensions in TZCYXS order
    data = np.transpose(Yr.reshape(dims+(-1,1), order='F'), [3, 0, 4, 1, 2])
    t = tifffile.TiffWriter('movies/demoMovie3D.tif', imagej=True)
    t.save(data)
    t.close()
    fname_new=cse.utilities.save_memmap(['movies/demoMovie3D.tif'],base_name='Yr')


# In[8]:

Yr,dims,T=cse.utilities.load_memmap(fname_new)
Y=np.reshape(Yr,dims+(T,),order='F')
Cn = cse.utilities.local_correlations(Y)
plt.imshow(Cn.max(0) if len(Cn.shape)==3 else Cn, cmap='gray',vmin=np.percentile(Cn, 1), vmax=np.percentile(Cn, 99))    
plt.show()


# In[9]:

#%%
K=20 # number of neurons expected per patch
gSig=[2,2,2] # expected half size of neurons
merge_thresh=0.8 # merging threshold, max correlation allowed
p=2 #order of the autoregressive system
options = cse.utilities.CNMFSetParms(Y,n_processes,p=p,gSig=gSig,K=K)
cse.utilities.start_server()


# In[ ]:

#%% PREPROCESS DATA AND INITIALIZE COMPONENTS
t1 = time()
Yr,sn,g,psx = cse.pre_processing.preprocess_data(Yr,**options['preprocess_params'])
Atmp, Ctmp, b_in, f_in, center=cse.initialization.initialize_components(Y, **options['init_params'])                                                    
print time() - t1


# In[ ]:

options['init_params']['tsub']=10
options['init_params']['ssub']=2


# In[ ]:

refine_components=False
if refine_components:
    Ain,Cin = cse.utilities.manually_refine_components(Y,options['init_params']['gSig'],coo_matrix(Atmp),Ctmp,Cn,thr=0.9)
else:
    Ain,Cin = Atmp, Ctmp


# In[ ]:

plt.close('all')


# In[ ]:

p1=cse.nb_plot_contour(Cn.max(0),Ain.reshape(dims+(-1,),order='F').max(0).reshape((-1,K),order='F'),
                       dims[1],dims[2],thr=0.9,face_color=None, line_color='black',alpha=0.4,line_width=2)
bpl.show(p1)


# In[ ]:

#%% UPDATE SPATIAL COMPONENTS
t1 = time()
A,b,Cin = cse.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])
t_elSPATIAL = time() - t1
print t_elSPATIAL 
#clear_output(wait=True)
print('DONE!')


# In[ ]:

plt.figure(num=None, figsize=(9, 7), dpi=100, facecolor='w', edgecolor='k')
crd = cse.utilities.plot_contours(
    A.toarray().reshape(dims+(-1,),order='F').max(0).reshape((np.prod(dims[1:]),-1),order='F'),
                       Cn.max(0),thr=0.9)
plt.show()


# In[ ]:

plt.close('all')


# In[ ]:

p1=cse.nb_plot_contour(Cn.max(0),A.toarray().reshape(dims+(-1,),order='F').max(0).reshape((np.prod(dims[1:]),-1),order='F'),
                       dims[1],dims[2],thr=0.9,face_color=None, line_color='black',alpha=0.4,line_width=2)
bpl.show(p1)


# In[ ]:

plt.close()
t1 = time()
options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
C,f,S,bl,c1,neurons_sn,g,YrA = cse.temporal.update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
t_elTEMPORAL = time() - t1
print t_elTEMPORAL  
clear_output(wait=True)


# In[ ]:

#%% merge components corresponding to the same neuron
t1 = time()
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merging.merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'], options['spatial_params'], bl=bl, c1=c1, sn=neurons_sn, g=g, thr=0.8, mx=50, fast_merge = True)
t_elMERGE = time() - t1
print t_elMERGE  


# In[ ]:

#refine spatial and temporal components
t1 = time()
A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
print time() - t1
clear_output(wait=True)
print time() - t1 # 100 seconds
print('DONE!')


# In[ ]:

plt.figure(num=None, figsize=(9, 7), dpi=100, facecolor='w', edgecolor='k')
A_or, C_or, srt = cse.utilities.order_components(A2,C2)
crd = cse.utilities.plot_contours(coo_matrix(A_or.reshape(dims+(-1,),order='F').max(0).reshape((np.prod(dims[1:]), -1),order='F')),
                                  Cn.max(0),thr=0.9)
plt.show()


# In[ ]:

p2=cse.utilities.nb_plot_contour(Cn.max(0),A_or.reshape(dims+(-1,),order='F').max(0).reshape((dims[1]*dims[2], -1),order='F'),dims[1],dims[2],thr=0.9,face_color='purple', line_color='black',alpha=0.3,line_width=2)
bpl.show(p2)


# In[ ]:

# view patches per layer
traces_fluo=cse.utilities.nb_view_patches3d(Yr, A_or, C_or, b2, f2, dims, image_type='max',
                              max_projection=False,axis=0,thr = 0.9)


# In[ ]:

# view patches as maximum-projection
traces_fluo=cse.utilities.nb_view_patches3d(Yr, A_or, C_or, b2, f2, dims, image_type='mean',
                              max_projection=True,axis=0,thr = 0.9,denoised_color='red')


# In[ ]:

#%% STOP CLUSTER
cse.utilities.stop_server()


# In[ ]:



