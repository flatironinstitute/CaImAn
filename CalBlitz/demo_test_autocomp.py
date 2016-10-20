#%%
try:
    %load_ext autoreload
    %autoreload 2
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
import cv2
import scipy
from sklearn.decomposition import MiniBatchDictionaryLearning,PCA,NMF,IncrementalPCA
import h5py
from glob import glob
#%% CREATE PATCH OF DATA
import os
fnames=[]
for file in os.listdir("./"):
    if file.startswith("dendr") and file.endswith(".tif"):
        fnames.append(file)
fnames.sort()
print fnames  
#%%
Yr=cb.load(fnames[0],fr=30)    
#Yr=Yr[:,80:130,360:410]
pl.imshow(np.mean(Yr,0),cmap=pl.cm.gray)
        
#%%      
big_mov=[];
big_shifts=[]
for f in  fnames[:]:
    print f    
    Yr=cb.load(f,fr=30)        
    Yr=Yr[:,80:130,360:410]    
#    pl.imshow(np.mean(Yr,0))
    #Yr=Yr.resize(fx=1,fy=1,fz=1)
    Yr = np.transpose(Yr,(1,2,0)) 
    d1,d2,T=Yr.shape
    Yr=np.reshape(Yr,(d1*d2,T),order='F')
    print Yr.shape
#    np.save(fname[:-3]+'npy',np.asarray(Yr))
    big_mov.append(np.asarray(Yr))
    
#%% should motion correct here
    
#%%    
big_mov=np.concatenate(big_mov,axis=-1)
big_shifts=np.concatenate(big_shifts,axis=0)
#%%
np.save('Yr_DS_2.npy',big_mov)
np.savez('Yr_DS_2.npz',d1=d1,d2=d2)on correct  here


#%% SUE ANN PATCH

Yr1_tot=cb.load('dendritic_demo.tif',fr=30)    
Yr1=Yr1_tot
Yr2=Yr1.copy().bilateral_blur_2D(diameter=10,sigmaColor=10000,sigmaSpace=0)
Yr2,shifts,_,_=Yr2.motion_correct(remove_blanks=True)
Yr3=Yr1.copy().apply_shifts(shifts,remove_blanks=True)
Yr3.save('patch_sue.tif')
pl.imshow(np.mean(Yr3,0),cmap=pl.cm.gray)
#%% LOAD MOVIE AND MAKE DIMENSIONS COMPATIBLE WITH CNMF
reload=0
filename='resized_pf.tif'

#filename='patch_sue.tif'
#filename='patch.tif'
#filename='patch_2.tif'
#filename='PCsforPC.tif'
#filename='demoMovie.tif'
#filename='PCsforPC.tif'
#filename='demoMovie.tif'
#filename='selmanExample.tif'
t = tifffile.TiffFile(filename) 
Yr = t.asarray().astype(dtype=np.float32) 
#Yr=Yr[-3000:,:]
Yr = np.transpose(Yr,(1,2,0))
d1,d2,T=Yr.shape
Yr=np.reshape(Yr,(d1*d2,T),order='F')
#np.save('Y',Y)
np.save('Yr',Yr)
#Y=np.load('Y.npy',mmap_mode='r')
Yr=np.load('Yr.npy',mmap_mode='r')        
Y=np.reshape(Yr,(d1,d2,T),order='F')
Cn = cse.utilities.local_correlations(Y)
#n_pixels_per_process=d1*d2/n_processes # how to subdivide the work among processes

pl.imshow(Cn,cmap=pl.cm.gray)

#%% USING FILTERS TO INCREASE SNR
N=0#Y.shape[-1]
N1=30000

m=cb.movie(np.transpose(np.array(Y[:,:,N:N1]),[2,0,1]),fr=30)
# denoise using PCA
#m=m.IPCA_denoise(components=100,batch=10000)

# denoise using median filter
#m=cb.movie(scipy.ndimage.median_filter(m, size=(3,2,2), mode='nearest'),fr=30)

# denoise using bilateral filters 
#m=m.bilateral_blur_2D(diameter=10,sigmaColor=10000,sigmaSpace=0)

# denoise using percentile filter
#m=cb.movie(scipy.ndimage.percentile_filter(m, 90, size=(3,2,2), mode='nearest'),fr=30)

#denoise using gaussian filter: USE THIS!!!
m=cb.movie(scipy.ndimage.gaussian_filter(m, sigma=(.5,.5,.5), mode='nearest',truncate=2),fr=30)

# resize movie
m=m.resize(1,1,.2)
(m-np.mean(m)).play(gain=5.,magnification=2,fr=100)
#%%
m=m[:,10:60,27:495]
#%% ONLINE NMF USING CALBLITZ
#perc=8
#myfloat=np.float32
#m1= np.maximum(0,m-np.percentile(m,perc,axis=0))
#tm,sp=m1.online_NMF(n_components=15)
#pl.figure()
#for idx,mm in enumerate(sp):
#    pl.subplot(6,5,idx+1)
#    pl.imshow(mm,cmap=pl.cm.gray)
#%% PYTHON's BATCH NMF
remove_baseline=False
use_pixels_as_basis=True


    

# alpha is the L1 Norm regulatizer
#sue ann
#remove_baseline=True
#alpha= 10e2
#n_components=30

#demo dendritic
remove_baseline=True
alpha= 10e2
n_components=10
perc=20

# patch_1.tif  
#alpha= 20e2
#n_components=8
#remove_baseline=False
#perc=.00001 #PC 

# demoMovie.tif

#alpha= 50e2
#n_components=30
#perc=50
#remove_baseline=False
# PC
#alpha= 5e2
#n_components=50

# patch_2.tif
#alpha=5e1
#n_components=15



if remove_baseline:      
    m1= np.maximum(0,m-np.percentile(m,perc,axis=0))
else:
    m1=m


mdl = NMF(n_components=n_components,verbose=True,init='nndsvd',tol=1e-10,max_iter=500,shuffle=True,alpha=alpha,l1_ratio=1)


T,d1,d2=np.shape(m1)
d=d1*d2

if use_pixels_as_basis:
    yr=np.reshape(m1,[T,d],order='F')
else:
    yr=np.reshape(m1,[T,d],order='F').T


#yr=-scipy.linalg.toeplitz(np.concatenate([np.array([1,-.9]),np.zeros(398)])).dot(yr)
#yr=yr-np.min(yr)

if use_pixels_as_basis:
    C=mdl.fit_transform(yr)
    X=mdl.components_.T   
else:
    X=mdl.fit_transform(yr)
    C=mdl.components_.T   
    


#%%
pl.figure()
for idx,mm in enumerate(X.T):
    pl.subplot(10,1,idx+1)
    pl.imshow(np.reshape(mm,(d1,d2),order='F'),cmap=pl.cm.gray,vmin=np.percentile(mm,1),vmax=np.percentile(mm,98))
    pl.axis('off')    
#%%
bckr=[0,4]
pfs=[1,2,3,5,6,8]
pl.figure()
pl.subplot(2,1,1)    
pl.plot(C[:,bckr]+np.arange(0,80,40)[None,:])
pl.ylabel('a.u.')
pl.legend(list(bckr))
pl.axis('tight')
pl.subplot(2,1,2)
pl.plot(C[:,pfs]+np.arange(0,120,20)[None,:])
pl.xlabel('frames')
pl.ylabel('a.u.')
pl.axis('tight')
pl.legend(list([1,2,3,5,6,8]))
#%%
np.savez('res_auto_comp.npz',X=X,C=C,d1=d1,d2=d2,mdl=mdl)    
#%% NOW RELOAD ORIGINAL MOVIE AND APPLY NMF INITIALIZING WITH VALUES COMPUTED ON DOWNSAMPLED VERSION
if not use_pixels_as_basis:
    raise Exception('You cannot reuse spatial basis since you are using traces as basis')
    
m1=cb.movie(np.transpose(np.array(Y[:,:,N:N1]),[2,0,1]),fr=30)
m1=cb.movie(scipy.ndimage.gaussian_filter(m1, sigma=(1,1,0), mode='nearest',truncate=2),fr=30)
if remove_baseline:  
    m1= np.maximum(0,m1-np.percentile(m1,perc,axis=0))



T,d1,d2=np.shape(m1)
d=d1*d2
yr=np.reshape(m1,[T,d],order='F')

# FIT USING THE PREVIOUSLY COMPUTED FACTORIZATION
print ('Fitting....')
H=mdl.transform(yr)

# RUN AGAIN NMF INITIALIZING WITH THE COMPONENTS PREVIOUSLY COMPUTED
mdl1 = NMF(n_components=n_components,verbose=True,init='custom',tol=1e-10,max_iter=300,shuffle=True,alpha=alpha,l1_ratio=1)
traces= mdl1.fit_transform(yr,W=H,H=mdl.components_)

X1=mdl1.components_.T
pl.figure()
for idx,mm in enumerate(X1.T):
    pl.subplot(7,7,idx+1)
    pl.imshow(np.reshape(mm,(d1,d2),order='F'),cmap=pl.cm.gray,vmin=np.percentile(mm,1),vmax=np.percentile(mm,99))

pl.subplot(7,7,idx+2)
mu=np.mean(m1,0)
pl.imshow(mu,cmap=pl.cm.gray,vmin=np.percentile(mu,1),vmax=np.percentile(mu,99))
    
#%% PLAY DENOISED MOVIE
denoised=cb.movie(np.reshape(traces.dot(X1.T),[T,d1,d2],order='F'),fr=30)
denoised=cb.concatenate([denoised,cb.movie(np.transpose(np.array(Y[:,:,N:N1]),[2,0,1]),fr=30)],axis=2)
#denoised=cb.concatenate([denoised,m1],axis=1)

(denoised-np.mean(denoised,0)).play(gain=8.,magnification=8,backend='opencv',fr=30)
#%% PLAY DENOISED MOVIE ONLY UYSING SOME COMPONENTS
idx=np.array([0  ,1,2, 3,  4,5,6,7,8,9,10,11,12,13,14])
idx=np.array([0,12])
denoised=cb.movie(np.reshape(traces[:,idx].dot(X1[:,idx].T),[T,d1,d2],order='F'),fr=30)
(denoised-np.mean(denoised,0).min()).play(gain=5.,magnification=8,backend='opencv',fr=30)


#%% PLAY RESIDUAL MOVIE
residual=m1-denoised
(residual).play(gain=5.,magnification=4,backend='opencv',fr=100)

#%% plot components
if use_pixels_as_basis:
    pl.plot(traces)    


scipy.io.savemat('output_nmf.mat',{'traces':traces,'comps':X1})

#%% ONLINE NMF ########################################################################################################################
import spams
from PIL import Image
import time

N=0#Y.shape[-1]
N1=30000
use_pixels_as_basis=True
#remove_baseline=True # notice: online does not work without removing baseline!
#n_components=30

m1=cb.movie(np.transpose(np.array(Y[:,:,N:N1]),[2,0,1]),fr=30)
m1=cb.movie(scipy.ndimage.gaussian_filter(m1, sigma=(1,1,0), mode='nearest',truncate=2),fr=30)
if remove_baseline:  
    m1= np.maximum(0,m1-np.percentile(m1,perc,axis=0))

# resize movie
m=m.resize(1,1,.5)


myfloat=np.float32

T,d1,d2=np.shape(m1)
d=d1*d2

if use_pixels_as_basis:
    yr=np.reshape(m1,[T,d],order='F')
else:
    yr=np.reshape(m1,[T,d],order='F').T


X = np.asfortranarray(yr,dtype = myfloat)
########## FIRST EXPERIMENT ###########
tic = time.time()
# if you want to use NMF uncomment the following line
#(U,V) = spams.nmf(X,return_lasso= True,K = n_components,iter = -5)

# regularizer on space components
#spams.trainDL(X,return_model = True,D=np.asfortranarray(V.todense().T),posAlpha=True,posD=True,modeD=3,gamma1=.001,lambda1=3000,lambda2=0,mode=spams.spams_wrap.PENALTY,iter=-5)
if not use_pixels_as_basis:
#    (D,model)=spams.trainDL(X,return_model = True,K=n_components,posAlpha=True,posD=True,modeD=3,gamma1=.001,lambda1=2000,lambda2=0,mode=spams.spams_wrap.PENALTY,iter=-5)
#    U=model['B']
#    V=model['A']
    (U,V) = spams.nnsc(X,return_lasso=True,K=n_components,lambda1=1000,iter=-5)
else:
    (U,V) = spams.nnsc(X,return_lasso=True,K=n_components,lambda1=1000,iter=-5)

    
#(U,V) = spams.nnsc(np.asfortranarray(yr.T,dtype = myfloat),return_lasso=True,K=15,lambda1=100,iter=-5)
#model=dict()
#model['A']=np.asfortranarray(V.todense().T)
#model['B']=np.asfortranarray(U)
#model['iter']=100
#(D,model) = spams.trainDL(X,return_model = True,D=np.asfortranarray(V.todense().T),posAlpha=True,posD=True,modeD=3,gamma1=.001,lambda1=3000,lambda2=0,mode=spams.spams_wrap.PENALTY,iter=-5)


if use_pixels_as_basis:
    comp=V.todense()
else:
    comp=U.T    
#
#comp=V.T

tac = time.time()
t = tac - tic
pl.figure()
for idx,mm in enumerate(comp):
    pl.subplot(6,6,idx+1)
    pl.imshow(np.reshape(mm,(d1,d2),order='F'),cmap=pl.cm.gray)
#%% REALLY ONLINE

batch_size=5000;
K=30
lambda1=1000
iter_=1
A=[]
B=[]
total_iter=0
for count in range(1):
    print total_iter
    for idx in range(0,yr.shape[0],batch_size):
        total_iter+=batch_size
        fr=yr[idx:idx+batch_size]    
        
        if len(A) == 0:
#            (D,model) = spams.trainDL(X,K=K,return_model = True,posAlpha=True,posD=True,lambda1=lambda1,iter=1)
#            A=D
            (A,B) = spams.nnsc(np.asfortranarray(fr),return_lasso=True,K=K,lambda1=lambda1,iter=iter_)
#            (A,B) = spams.nmf(np.asfortranarray(fr),return_lasso=True,K=K,iter=iter_)
        else:
            model=dict()
            model['A']=np.asfortranarray(A)
            model['B']=np.asfortranarray(B.todense())
            model['iter']=count
            (A,B) = spams.nnsc(np.asfortranarray(fr),return_lasso=True,K=K,lambda1=lambda1,model=model,iter=iter_)
#            (A,B) = spams.nmf(np.asfortranarray(fr),return_lasso=True,K=K,model=model,iter=iter_)

if use_pixels_as_basis:
    comp=B.todense()
else:
    comp=A.T   


pl.figure()
for idx,mm in enumerate(comp):
    pl.subplot(6,6,idx+1)
    pl.imshow(np.reshape(mm,(d1,d2),order='F'),cmap=pl.cm.gray) 


#(U,model1) = spams.trainDL(X,return_model=True,K=15,lambda1=100)
#(U,V) = spams.nnsc(X,return_lasso=True,K=15,lambda1=100,model=model1)


#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************
#%% DO NOT LOOK AT THIS STUFF    ****************************************** STOP!!! ***********************************

#%%
m=cb.movie(np.transpose(np.array(Y[:,:,N:N1]),[2,0,1]),fr=30)
num_samp=np.round(m.shape[0])
traces=mdl.components_    
new_tr=np.zeros((traces.shape[0],num_samp))
for idx,tr in enumerate(traces):
    print idx
    new_tr[idx]=np.maximum(scipy.signal.resample(tr,num_samp),0)

mdl1 = NMF(n_components=15,verbose=True,init='custom',tol=1e-10,max_iter=200,shuffle=True,alpha=alpha,l1_ratio=1)


m1= np.maximum(0,m-np.percentile(m,perc,axis=0))[:]#[:,20:35,20:35].resize(1,1,.05)
T,d1,d2=np.shape(m1)
d=d1*d2
yr=np.reshape(m1,[T,d],order='F')
newX=mdl1.fit_transform(yr.T,W=X.copy(),H=new_tr)
pl.figure()
for idx,mm in enumerate(newX.T):
    pl.subplot(6,5,idx+1)
    pl.imshow(np.reshape(mm,(d1,d2),order='F'),cmap=pl.cm.jet)    
#%%
id_c=7
N=10
pl.plot(np.convolve(mdl.components_[id_c].T, np.ones((N,))/N, mode='valid'))


#%%
def mode(inputData, axis=None, dtype=None):

   """
   Robust estimator of the mode of a data set using the half-sample mode.
   
   .. versionadded: 1.0.3
   """
   import numpy
   if axis is not None:
      fnc = lambda x: mode(x, dtype=dtype)
      dataMode = numpy.apply_along_axis(fnc, axis, inputData)
   else:
      # Create the function that we can use for the half-sample mode
      def _hsm(data):
         if data.size == 1:
            return data[0]
         elif data.size == 2:
            return data.mean()
         elif data.size == 3:
            i1 = data[1] - data[0]
            i2 = data[2] - data[1]
            if i1 < i2:
               return data[:2].mean()
            elif i2 > i1:
               return data[1:].mean()
            else:
               return data[1]
         else:
#            wMin = data[-1] - data[0]
            wMin=np.inf
            N = data.size/2 + data.size%2 
            for i in xrange(0, N):
               w = data[i+N-1] - data[i] 
               if w < wMin:
                  wMin = w
                  j = i

            return _hsm(data[j:j+N])
            
      data = inputData.ravel()
      if type(data).__name__ == "MaskedArray":
         data = data.compressed()
      if dtype is not None:
         data = data.astype(dtype)
         
      # The data need to be sorted for this to work
      data = numpy.sort(data)
      
      # Find the mode
      dataMode = _hsm(data)
      
   return dataMode


def std(inputData, Zero=False, axis=None, dtype=None):
   """
   Robust estimator of the standard deviation of a data set.  
   
   Based on the robust_sigma function from the AstroIDL User's Library.
   
   .. versionchanged:: 1.0.3
      Added the 'axis' and 'dtype' keywords to make this function more
      compatible with numpy.std()
   """
   
   if axis is not None:
      fnc = lambda x: std(x, dtype=dtype)
      sigma = numpy.apply_along_axis(fnc, axis, inputData)
   else:
      data = inputData.ravel()
      if type(data).__name__ == "MaskedArray":
         data = data.compressed()
      if dtype is not None:
         data = data.astype(dtype)
         
      if Zero:
         data0 = 0.0
      else:
         data0 = numpy.median(data)
      maxAbsDev = numpy.median(numpy.abs(data-data0)) / 0.6745
      if maxAbsDev < __epsilon:
         maxAbsDev = (numpy.abs(data-data0)).mean() / 0.8000
      if maxAbsDev < __epsilon:
         sigma = 0.0
         return sigma
         
      u = (data-data0) / 6.0 / maxAbsDev
      u2 = u**2.0
      good = numpy.where( u2 <= 1.0 )
      good = good[0]
      if len(good) < 3:
         print "WARNING:  Distribution is too strange to compute standard deviation"
         sigma = -1.0
         return sigma
         
      numerator = ((data[good]-data0)**2.0 * (1.0-u2[good])**2.0).sum()
      nElements = (data.ravel()).shape[0]
      denominator = ((1.0-u2[good])*(1.0-5.0*u2[good])).sum()
      sigma = nElements*numerator / (denominator*(denominator-1.0))
      if sigma > 0:
         sigma = math.sqrt(sigma)
      else:
         sigma = 0.0
         
   return sigma
#%%
cmp_=np.reshape(X[:,id_c],(d1,d2), order='F')/(np.linalg.norm(X[:,id_c])**2)
vct=np.sum(m*cmp_,axis=(1,2))
pl.plot(vct-np.mean(vct))

#%%


Y_n=np.zeros(Y.shape)

N=1400
Y_n=Y_n[:,:,:N]
ym=np.median(Y,axis=-1)
clahe = cv2.createCLAHE(clipLimit=100., tileGridSize=(11,11))
ymm=np.uint8(255*(ym-np.min(ym))/(np.max(ym)-np.min(ym)))
ymm=clahe.apply(ymm)

y__=np.zeros(Y[:,:,0].shape,dtype=np.float32)
for fr in range(N):
     print fr
     y1=Y[:,:,fr].copy()
     y1 =   cv2.bilateralFilter(y1,5,10000,0)     
     y1=cv2.ximgproc.guidedFilter(ymm,y1,radius=1,eps=0)  
     Y_n[:,:,fr] =  y1

mn1=cb.movie(np.transpose(np.array(Y_n),[2,0,1]),fr=30)
mn2=cb.movie(np.transpose(np.array(Y[:,:,:N]),[2,0,1]),fr=30)
mn=cb.concatenate([mn1,mn2],axis=1)
(mn).play(gain=5.,magnification=4,backend='opencv',fr=100)
#%%
clahe = cv2.createCLAHE(clipLimit=100., tileGridSize=(11,11))
ymm=np.uint8(255*(ym-np.min(ym))/(np.max(ym)-np.min(ym)))
ymm=clahe.apply(ymm)
pl.imshow(ymm,cmap=pl.cm.gray)
#%%
pl.imshow(cv2.adaptiveThreshold(ymm,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2),cmap=pl.cm.gray)
pl.imshow(cv2.adaptiveThreshold(ymm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2),cmap=pl.cm.gray)
pl.imshow(cv2.threshold(ymm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[],cmap=pl.cm.gray)
#%%
def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result
    


spikes_mat=[]
tr=traces[:,10]
size_chunk=100
thresh=np.median(tr)+3*np.std(tr)
for idx in range(len(tr)-size_chunk-1):
    tr_tmp=tr[idx:idx+size_chunk]
    if tr_tmp[0]<thresh and tr_tmp[-1]<thresh:     
        spikes_mat.append(tr_tmp)


#%%
spikes_mat=np.vstack(spikes_mat)
#pl.imshow(spikes_mat,aspect='auto',interpolation='none')
alpha=10
mdl1 = NMF(n_components=50,verbose=True,tol=1e-10,max_iter=500,shuffle=True,alpha=alpha,l1_ratio=1)
X=mdl1.fit_transform(spikes_mat)
pl.plot(mdl1.components_.T)
#%%
from sklearn.cluster import KMeans
km=KMeans(50)

X=km.fit_transform(spikes_mat)
labels=km.labels_
pl.plot(km.cluster_centers_.T)
#%%

pl.plot(tr)
#%%
#ym=np.median(m,axis=0)
ym=m.bin_median()
#ym=np.std(m,axis=0)
#clahe = cv2.createCLAHE(clipLimit=100., tileGridSize=(11,11))
guide_filter=np.uint8(255*(ym-np.min(ym))/(np.max(ym)-np.min(ym)))
#guide_filter=clahe.apply(guide_filter)

mn2=m.copy()
mn1=m.copy()#.bilateral_blur_2D(diameter=0,sigmaColor=10000,sigmaSpace=0)     
mn1=mn1.guided_filter_blur_2D(guide_filter,radius=3, eps=0)   
   
mn=cb.concatenate([mn1,m],axis=1)
(mn-np.mean(mn)).play(gain=5.,magnification=4,backend='opencv',fr=300)
#%% FILTER TESTING FILTER
pl.imshow(np.mean(Y,axis=-1),cmap=pl.cm.gray)
#%% BILATERAL FILTER EXAMPLE
N=10000
#     
m=cb.movie(np.transpose(np.array(Y[:,:,:N]),[2,0,1]),fr=30)
m=m.resize(1,1,.1)


mn2=m.copy()
mn1=m.copy().bilateral_blur_2D(diameter=10,sigmaColor=10000,sigmaSpace=0)     

mn1,shifts,xcorrs, template=mn1.motion_correct()
mn2=mn2.apply_shifts(shifts)     
#mn1=cb.movie(np.transpose(np.array(Y_n),[2,0,1]),fr=30)
mn=cb.concatenate([mn1,m],axis=1)
(mn-np.mean(mn)).play(gain=2.,magnification=4,backend='opencv',fr=20)    
#%% EFTYCHIOS METHOD TO SEPARATE PIXELS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA,NMF
from sklearn.mixture import GMM

Ys=Yr
thresh_probability=0.5
num_psd_elms_high_freq=49;
cl_thr=0.8
#P.sn = sn(:);
#fprintf('  done \n');
psdx = np.sqrt(psx[:,3:]);
X = psdx[:,1:np.minimum(np.shape(psdx)[1],150)];
X = X-np.mean(X,axis=1)[:,np.newaxis]#     bsxfun(@minus,X,mean(X,2));     % center
#X = X/sn[:,np.newaxis]# 
X = X/np.percentile(X,90,axis=1)[:,np.newaxis]

#X = X/(+1e-5+np.std(X,axis=1)[:,np.newaxis])
#epsilon=1e-9
#X = X/(epsilon+np.linalg.norm(X,axis=1,ord=1)[:,np.newaxis])



pc=PCA(n_components=5)
cp=pc.fit_transform(X)

#nmf=NMF(n_components=2)
#nmr=nmf.fit_transform(X)

gmm=GMM(n_components=2)
Cx=gmm.fit_predict(cp)

L=gmm.predict_proba(cp)
Cx1=np.vstack([np.mean(X[Cx==0],0),np.mean(X[Cx==1],0)])

ind=np.argmin(np.mean(Cx1[:,-num_psd_elms_high_freq:],axis=1))
active_pixels = (L[:,ind]>thresh_probability)
active_pixels = L[:,ind]
pl.imshow(np.reshape((active_pixels),(d1,d2),order='F'))

#%%
ff=np.zeros(np.shape(A_or)[-1])
cl_thr=0.2
#ff = false(1,size(Am,2));
for i in range(np.shape(A_or)[-1]):
    a1 = A_or[:,i]
    a2 = A_or[:,i]*active_pixels
    if np.sum(a2**2) >= cl_thr**2*np.sum(a1**2):
        ff[i] = 1

id_set=1
cse.utilities.view_patches_bar(Yr,coo_matrix(A_or[:,ff==id_set]),C_or[ff==id_set,:],b2,f2, d1,d2, YrA=YrA[srt[ff==id_set],:])  




#km=KMeans(n_clusters=2)
#Cx=km.fit_transform(X)
#Cx=km.fit_transform(cp)
#Cx=km.cluster_centers_
#L=km.labels_
#ind=np.argmin(np.mean(Cx[:,-49:],axis=1))
#active_pixels = (L==ind)
#centroids = Cx;