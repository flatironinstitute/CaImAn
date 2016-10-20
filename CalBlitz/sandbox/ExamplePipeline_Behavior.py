# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015
@author: agiovann
"""
#%% add CalBlitz folder to python directory
path_to_CalBlitz_folder='/home/ubuntu/SOFTWARE/CalBlitz'
path_to_CalBlitz_folder='C:/Users/agiovann/Documents/SOFTWARE/CalBlitz/CalBlitz'

import sys
sys.path
sys.path.append(path_to_CalBlitz_folder)
#% add required packages
from XMovie import XMovie
import time
from pylab import plt
import numpy as np
from utils import matrixMontage
#% set basic ipython functionalities
try: 
    plt.ion()
    %load_ext autoreload
    %autoreload 2
except:
    print "Probably not a Ipython interactive environment" 


#%% define movie

#filename='k26_v1_176um_target_pursuit_001_005.tif'
#frameRate=.033;

#filename='20150522_1_1_001.tif'
       
filename='Z:/deverett/agchr2_03092015_01/a1_032115_a/a1_loc1/a1_loc1_02_cam1.npz'
frameRate=.064;

#data_type='fly'
#filename='20150522_1_1_001.tif'
#frameRate=.128;

#filename='agchr2_030915_01_040215_a_freestyle_01_cam1.avi'
#frameRate=0.0083;

#filename='img007.tif'
#frameRate=0.033;


#filename='img001.tif'
#filename='img002.tif'
#frameRate=0.033;
#%%
filename_py=filename[:-4]+'.npz'
filename_hdf5=filename[:-4]+'.hdf5'
filename_mc=filename[:-4]+'_mc.npz'
filename_analysis=filename[:-4]+'_analysis.npz'    
filename_traces=filename[:-4]+'_traces.npz'    

#%% load movie
m=XMovie(filename, frameRate=frameRate);

#%% example plot a frame
plt.imshow(m.mov[100],cmap=plt.cm.Greys_r)

#%% example play movie
m.playMovie(frate=.001,gain=10.0,magnification=1)

#%% example take first channel of two
channelId=1;
totalChannels=2;
m.makeSubMov(range(channelId-1,m.mov.shape[0],totalChannels))
#%%
np.savez(filename_py,mov=m.mov,frameRate=frameRate)
#%%
f = h5py.File(filename_hdf5, "w")
dset = f.create_dataset("mov",data=m.mov)
dset = f.create_dataset("frameRate",data=frameRate)
a=np.asarray(f['mov']) # or directly use f['mov']
#%%
m=XMovie(mat=np.load(filename_py)['mov'], frameRate=np.load(filename_py)['frameRate']); 
m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0)

#%%  prtition into two movies up and down

m.makeSubMov(range(1000))
#m.crop(crop_top=150,crop_bottom=150,crop_left=150,crop_right=150,crop_begin=0,crop_end=0)


#%% concatenate movies (it will add to the original movie)
# you have to create another movie new_mov=XMovie(...)
m.append(new_mov)

#%% motion correct run 3 times
# WHEN YOU RUN motion_correct YOUR ARE MODIFYING THE OBJECT!!!!

templates=[];
shifts=[];

max_shift_w=5;
max_shift_h=5;
num_iter=3; # numer of times motion correction is executed
template=None # here you can use your own template (best representation of the FOV)

for j in range(0,num_iter):
    template_used,shift=m.motion_correct(max_shift_w=max_shift_w,max_shift_h=max_shift_h,template=None,show_movie=False);
    templates.append(template_used)
    shift=np.asarray(shift)
    shifts.append(shift)

plt.plot(np.asarray(shifts).reshape((j+1)*shift.shape[0],shift.shape[1]))

#%% motion correct Cristina
initTime=time.time()

m=XMovie(mat=np.load(filename_py)['mov'], frameRate=np.load(filename_py)['frameRate']); 
m.makeSubMov(range(2000))
m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0)
m_tmp=m.copy()
_,shift=m_tmp.motion_correct(max_shift_w=60,max_shift_h=20,template=None,show_movie=False);
template=np.median(m_tmp.mov,axis=0)
m_tmp=m.copy()
_,shift=m_tmp.motion_correct(max_shift_w=60,max_shift_h=20,template=template,show_movie=False);
template=np.median(m_tmp.mov,axis=0)

m=XMovie(mat=np.load(filename_py)['mov'], frameRate=np.load(filename_py)['frameRate']); 
m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0)
_,shifts=m.motion_correct(max_shift_w=100,max_shift_h=20,template=template,show_movie=False);
    
print 'elapsed time:' + str(time.time()-initTime) 
#%%
matrixMontage(np.asarray(templates),cmap=plt.cm.gray,vmin=0,vmax=1000)
#%% apply shifts to original movie in order to minimize smoothing
if False:
    totalShifts=np.sum(np.asarray(shifts),axis=0)[:,0:2].tolist()
    m=XMovie(mat=np.load(filename_py)['mov'], frameRate=np.load(filename_py)['frameRate']); 
    m.applyShifstToMovie(totalShifts)
#%% apply shifts to another channel, you need to reload the original movie other channel (mov_other_channel)
if False:    
    # here reload the original imaging channel from movie
    totalShifts=np.sum(np.asarray(shifts),axis=0)[:,0:2].tolist()
    mov_other_channel.applyShifstToMovie(totalShifts)

#%% plot movie median
minBrightness=5;
maxBrightness=500;
plt.imshow(np.median(m.mov,axis=0),cmap=plt.cm.Greys_r,vmin=minBrightness,vmax=maxBrightness)

#%% save motion corrected movie inpython format along with the results. This takes some time now but will save  a lot later...
np.savez(filename_mc,mov=m.mov,frameRate=frameRate,templates=templates,shifts=shifts,max_shift=max_shift)

#%% RELOAD MOTION CORRECTED MOVIE
m=XMovie(mat=np.load(filename_mc)['mov'], frameRate=frameRate);    
max_shift=np.load(filename_mc)['max_shift']
shifts=np.load(filename_mc)['shifts']

#%% crop movie after motion correction. 
max_w= np.percentile(shifts[:,1],99)
min_w= np.percentile(shifts[:,1],1)
max_h= np.percentile(shifts[:,0],99)
min_h= np.percentile(shifts[:,0],1)
m.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)

#%% if you want to make a copy of the movie
if False:
    m_copy=m.copy()


#%% resize to increase SNR and have better convergence of segmentation algorithms
resizeMovie=True
if resizeMovie:
    fx=.5; # downsample a factor of four along x axis
    fy=.5;
    fz=.2; # downsample  a factor of 5 across time dimension
    m.resize(fx=fx,fy=fy,fz=fz)
else:
    fx,fy,fz=1,1,1

#%% compute delta f over f (DF/F)
initTime=time.time()print 'elapsed time:' + str(time.time()-initTime) 
m.computeDFF(secsWindow=10,quantilMin=50,subtract_minimum=True)
print 'elapsed time:' + str(time.time()-initTime) 

#%% compute subregions where to apply more efficiently facrtorization algorithms
fovs, mcoef, distanceMatrix=m.partition_FOV_KMeans(tradeoff_weight=.7,fx=.25,fy=.25,n_clusters=4,max_iter=500);
plt.imshow(fovs)

#%% create a denoised version of the movie, nice to visualize
if True:
    m2=m.copy()
    m2.IPCA_denoise(components = 100, batch = 1000)
    m2.playMovie(frate=.05,magnification=1,gain=2.0)
    
#%%

#%% compute spatial components via NMF
initTime=time.time()
space_spcomps,time_comps=m.NonnegativeMatrixFactorization(n_components=20,beta=1,tol=5e-7);
print 'elapsed time:' + str(time.time()-initTime) 
matrixMontage(np.asarray(space_spcomps),cmap=plt.cm.gray) # visualize components

#%% compute spatial components via ICA PCA
initTime=time.time()
spcomps=m.IPCA_stICA(components=20,mu=.5);
print 'elapsed time:' + str(time.time()-initTime) 
matrixMontage(spcomps,cmap=plt.cm.gray) # visualize components
 
#%% extract ROIs from spatial components 
#_masks,masks_grouped=m.extractROIsFromPCAICA(spcomps, numSTD=6, gaussiansigmax=2 , gaussiansigmay=2)
_masks,_=m.extractROIsFromPCAICA(spcomps, numSTD=10.0, gaussiansigmax=0 , gaussiansigmay=0)
matrixMontage(np.asarray(_masks),cmap=plt.cm.gray)

#%%  extract single ROIs from each mask
minPixels=5;
maxPixels=2500;
masks_tmp=[];
for mask in _masks:
    numPixels=np.sum(np.array(mask));        
    if (numPixels>minPixels and numPixels<maxPixels):
        print numPixels
        masks_tmp.append(mask>0)
        
masks_tmp=np.asarray(masks_tmp,dtype=np.float16)

# reshape dendrites if required(if the movie was resized)
if fx != 1 or fy !=1:
    mdend=XMovie(mat=np.asarray(masks_tmp,dtype=np.float32), frameRate=1);
    mdend.resize(fx=1/fx,fy=1/fy)
    all_masks=mdend.mov;
else:
    all_masks=masks_tmp              

all_masksForPlot=[kk*(ii+1)*1.0 for ii,kk in enumerate(all_masks)]
plt.imshow(np.max(np.asarray(all_masksForPlot,dtype=np.float16),axis=0))


#%% extract DF/F from orginal movie, needs to reload the motion corrected movie
m=XMovie(mat=np.load(filename_mc)['mov'], frameRate=frameRate);    
m.crop(max_shift,max_shift,max_shift,max_shift)    
minPercentileRemove=1;
# remove an estimation of what a Dark patch is, you should provide a better estimate
F0=np.percentile(m.mov,minPercentileRemove)
m.mov=m.mov-F0; 
traces, tracesDFF = m.extract_traces_from_masks(all_masks,type='DFF',window_sec=15,minQuantile=8)
plt.plot(tracesDFF)
#%%
plt.imshow(tracesDFF.T)


#%% save the results of the analysis in python format
np.savez(filename_analysis,all_masks=all_masks,spcomps=spcomps,fx=fx,fy=fy,fz=fz,traces=traces,tracesDFF=tracesDFF)

#%% save the results of the analysis and motion corrected movie in matlab format
import scipy.io as sio
sio.savemat(filename_analysis[:-4]+'.mat', {'all_masks':np.transpose(all_masks,(1,2,0)),'spcomps':np.transpose(spcomps,(1,2,0)),'traces':traces,'tracesDFF':tracesDFF})
m=XMovie(mat=np.load(filename_mc)['mov'], frameRate=frameRate);    
sio.savemat(filename_mc[:-4]+'.mat', {'mov':np.transpose(np.load(filename_mc)['mov'],(1,2,0)),'frameRate':frameRate,'shifts':shifts,'templates':np.transpose(templates,(1,2,0))})


