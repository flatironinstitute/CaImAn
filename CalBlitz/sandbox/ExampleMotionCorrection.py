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
import h5py
import calblitz as cb
import time
import pylab as pl
import numpy as np
#% set basic ipython functionalities
try: 
    pl.ion()
    %load_ext autoreload
    %autoreload 2
except:
    print "Probably not a Ipython interactive environment" 


#%% define movie

#filename='k26_v1_176um_target_pursuit_002_013.tif'
#frameRate=30;
#start_time=0;
#type_='Sue_Ann_AAV'

#filename='-431 562 27  face run 20X 3X022.tif'
#frameRate=8;
#start_time=0;
#type_='MLI'

#filename='-431 562 27  face  20X 3X021.tif'
#frameRate=8;
#start_time=0;
#type_='MLI'


#filename='k23_20150424_002_001.tif'
#frameRate=30;
#start_time=0;
#type_='Sue_Ann_AAV'

#filename='20150522_1_1_001.tif'
       
#filename='ac_001_001.tif'
#frameRate=30;

#data_type='fly'
#filename='20150522_1_1_001.tif'
#frameRate=7.8;

#filename='agchr2_030915_01_040215_a_freestyle_01_cam1.avi'
#frameRate=120;

#filename='img007.tif'
#frameRate=30;


#filename='img001.tif'
#filename='img002.tif'

filename='M_FLUO.tif'
frameRate=15.62;
start_time=0;
type_='granule'

#%%
filename_py=filename[:-4]+'.npz'
filename_hdf5=filename[:-4]+'.hdf5'
filename_mc=filename[:-4]+'_mc.npz'
filename_analysis=filename[:-4]+'_analysis.npz'    
filename_traces=filename[:-4]+'_traces.npz'    

#%% load movie
m=cb.load(filename, fr=frameRate,start_time=start_time);
if type_ is 'MLI':
    warn('Removing Channel 2')
    m=m[::2]
#%% load submovie
load_sub_movie=False
if load_sub_movie:
    m=cb.load(filename, fr=frameRate,start_time=start_time,subindices=range(0,1500,10));
    #% example plot a frame
    pl.imshow(m[100],cmap=pl.cm.Greys_r)
    
    #% example play movie
    m.play(fr=20,gain=1.0,magnification=1)

#%% save movie python format. Less efficient than hdf5 below
if False:
    m.save(filename_py)
#%% save movie hdf5 format. fastest
m.save(filename_hdf5)


#%%
if False:   
    m=cb.load(filename_py); 
#%%
m=cb.load(filename_hdf5); 
#m=m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0);
min_val_add=np.min(np.mean(m,axis=0))
#min_val_add=np.min(m)-1# movies must be strictly positive!!
m=m-min_val_add

#%% concatenate movies (it will add to the original movie)
# you have to create another movie new_mov=XMovie(...)
if False:
    conc_mov=cb.concatenate([m,m])
    
#%% quick and dirt
m=cb.load(filename_hdf5); 

m,shifts,xcorrs,template=m.motion_correct(max_shift_w=5,max_shift_h=5, num_frames_template=None, template = None,method='opencv')
    

#%% motion correct for template purpose. Just use a subset to compute the template


if type_ == 'granule':
    submov=m[::1,:]
    max_shift_h=5;
    max_shift_w=5;
else:
    submov=m[::5,:]
    max_shift_h=10;
    max_shift_w=10;
    
templ=np.nanmedian(submov,axis=0); # create template with portion of movie
shifts,xcorrs=submov.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=templ, method='opencv')  #
submov.apply_shifts(shifts,interpolation='cubic')
template=(np.nanmedian(submov,axis=0))
shifts,xcorrs=m.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=template, method='opencv')  #
m=m.apply_shifts(shifts,interpolation='cubic')
template=(np.median(m,axis=0))
#pl.subplot(1,2,1)
#pl.imshow(templ,cmap=pl.cm.gray)
#pl.subplot(1,2,2)
pl.imshow(template,cmap=pl.cm.gray)
#%% now use the good template to correct

max_shift_h=10;
max_shift_w=10;

m=cb.load(filename_hdf5);
min_val_add=np.float32(np.percentile(m,.0001))
m=m-min_val_add
shifts,xcorrs=m.extract_shifts(max_shift_w=max_shift_w, max_shift_h=max_shift_h, template=template, method='opencv')  #

if m.meta_data[0] is None:
    m.meta_data[0]=dict()
    
m.meta_data[0]['shifts']=shifts
m.meta_data[0]['xcorrs']=xcorrs
m.meta_data[0]['template']=template
m.meta_data[0]['min_val_add']=min_val_add


    



m.save(filename_hdf5)
#%% visualize shifts
pl.subplot(2,1,1)
pl.plot(shifts)
pl.subplot(2,1,2)
pl.plot(xcorrs) 
#%% reload and apply shifts
m=cb.load(filename_hdf5)
meta_data=m.meta_data[0];
shifts,min_val_add,xcorrs=[meta_data[x] for x in ['shifts','min_val_add','xcorrs']]
m=m.apply_shifts(shifts,interpolation='cubic')
#%% crop borders created by motion correction
if type_ == 'granule':
    m[np.nonzero(np.max(np.abs(shifts),axis=1)>=5)[0]]=np.mean(m,axis=0)
#    m[np.nonzero(np.max(np.abs(shifts),axis=1)>=5)[0]]=np.nan
    shifts=np.clip(shifts,-5,5)
    
max_h,max_w= np.max(shifts,axis=0)
min_h,min_w= np.min(shifts,axis=0)
m=m.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)

#%% save shifts and other just for safety
np.savez(filename_mc,template=template,shifts=shifts,xcorrs=xcorrs,max_shift_h=max_shift_h,max_shift_w=max_shift_w,min_val_add=min_val_add)

#%% play movie
m.play(fr=50,gain=10.0,magnification=.5)
#%% visualize average
meanMov=(np.mean(m,axis=0))
pl.imshow(meanMov,cmap=pl.cm.gray,vmax=200)

#%% motion correct run 3 times
# WHEN YOU RUN motion_correct YOUR ARE MODIFYING THE OBJECT!!!!
#
#templates=[];
#shifts=[];
#corrs=[]
#max_shift_w=5;
#max_shift_h=5;
#num_iter=3; # numer of times motion correction is executed
#template=None # here you can use your own template (best representation of the FOV)
#print np.min(m)
#for j in range(0,num_iter):
#    m,template_used,shift,corrs=m.motion_correct(max_shift_w=max_shift_w,max_shift_h=max_shift_h,template=None,show_movie=False);
#    templates.append(template_used)
#    shift=np.asarray(shift)
#    shifts.append(shift)
#    corrs.append(corrs)
#    
#pl.plot(np.asarray(shifts).reshape((j+1)*shift.shape[0],shift.shape[1]))

#%% motion correct Cristina
#if is_cristina:
#    initTime=time.time()
#    
#    m=XMovie(mat=np.load(filename_py)['mov'], frameRate=np.load(filename_py)['frameRate']); 
#    m.makeSubMov(range(2000))
#    m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0)
#    m_tmp=m.copy()
#    _,shift=m_tmp.motion_correct(max_shift_w=60,max_shift_h=20,template=None,show_movie=False);
#    template=np.median(m_tmp.mov,axis=0)
#    m_tmp=m.copy()
#    _,shift=m_tmp.motion_correct(max_shift_w=60,max_shift_h=20,template=template,show_movie=False);
#    template=np.median(m_tmp.mov,axis=0)
#    
#    m=XMovie(mat=np.load(filename_py)['mov'], frameRate=np.load(filename_py)['frameRate']); 
#    m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0)
#    _,shifts=m.motion_correct(max_shift_w=100,max_shift_h=20,template=template,show_movie=False);
#        
#    print 'elapsed time:' + str(time.time()-initTime) 
#    
#    
#    
#    cb.matrixMontage(np.asarray(templates),cmap=pl.cm.gray,vmin=0,vmax=1000)        

#%% 
medMov=np.median(mc,axis=0)

#%% plot movie median
minBrightness=0;
maxBrightness=250;
pl.imshow(medMov,cmap=pl.cm.Greys_r,vmin=minBrightness,vmax=maxBrightness)

#%% if you want to make a copy of the movie
if False:
    m_copy=m.copy()

#%%
mo=m.copy()


#%% resize to increase SNR and have better convergence of segmentation algorithms
resizeMovie=True
if resizeMovie:
    fx=.5; # downsample a factor of four along x axis
    fy=.5;
    fz=.2; # downsample  a factor of 5 across time dimension
    m=m.resize(fx=fx,fy=fy,fz=fz)
else:
    fx,fy,fz=1,1,1

#%% compute delta f over f (DF/F)
initTime=time.time()
m=m-np.min(m)+1;
m=m.computeDFF(secsWindow=10,quantilMin=50)
print 'elapsed time:' + str(time.time()-initTime) 

#%% denoise and local correlation
loc_corrs=m.local_correlations(eight_neighbours=True)
m=m.IPCA_denoise(components = 100, batch = 10000)
m=m*loc_corrs
#m=m*(loc_corrs>.5)

#%%
#pl.imshow(np.mean(mo,axis=0),cmap=pl.cm.gray)
pl.imshow(np.mean(m,axis=0))
pl.imshow(loc_corrs)



#%% compute subregions where to apply more efficiently facrtorization algorithms
fovs, mcoef, distanceMatrix=m.partition_FOV_KMeans(tradeoff_weight=.7,fx=.25,fy=.25,n_clusters=4,max_iter=500);
pl.imshow(fovs)

#%% create a denoised version of the movie, nice to visualize
if False:
    
    m.IPCA_denoise(components = 100, batch = 10000)
    
    
#%%
spcomps=m.IPCA_io(n_components=50, fun='logcosh', max_iter=1000, tol=1e-20)
spcomps=np.rollaxis(spcomps,2)
#%% compute spatial components via NMF
initTime=time.time()
space_spcomps,time_comps=(m-np.min(m)).NonnegativeMatrixFactorization(n_components=20,beta=1,tol=5e-7);
print 'elapsed time:' + str(time.time()-initTime) 
cb.matrixMontage(np.asarray(space_spcomps),cmap=pl.cm.gray) # visualize components

#%% compute spatial components via ICA PCA
from scipy.stats import mode
initTime=time.time()
spcomps=m.IPCA_stICA(components=40,mu=.5,batch=100000);
print 'elapsed time:' + str(time.time()-initTime) 
#cb.matrixMontage(spcomps,cmap=pl.cm.gray) # visualize components
 
#%% extract ROIs from spatial components 
#_masks,masks_grouped=m.extractROIsFromPCAICA(spcomps, numSTD=6, gaussiansigmax=2 , gaussiansigmay=2)
_masks,_=cb.extractROIsFromPCAICA(spcomps, numSTD=20.0, gaussiansigmax=1 , gaussiansigmay=1,thresh=.005)
#cb.matrixMontage(np.asarray(_masks),cmap=pl.cm.gray)

#%%  extract single ROIs from each mask
minPixels=8;
maxPixels=200;
masks_tmp=[];
for mask in _masks:
    numPixels=np.sum(np.array(mask));        
    if (numPixels>minPixels and numPixels<maxPixels):
        print numPixels
        masks_tmp.append(mask>0)
        
masks_tmp=np.asarray(masks_tmp,dtype=np.float16)
all_masksForPlot_tmp=[kk*(ii+1)*1.0 for ii,kk in enumerate(masks_tmp)]
len(all_masksForPlot_tmp)


#%%
# reshape dendrites if required(if the movie was resized)
if fx != 1 or fy !=1:
    mdend=cb.movie(np.array(masks_tmp,dtype=np.float32), fr=1);
    mdend=mdend.resize(fx=1/fx,fy=1/fy)
    all_masks=mdend;
else:
    all_masks=masks_tmp              

all_masksForPlot=[kk*(ii+1)*1.0 for ii,kk in enumerate(all_masks)]


#%%
mask_show=np.max(np.asarray(all_masksForPlot_tmp,dtype=np.float16),axis=0);
#loc_corrs=m.local_correlations(eight_neighbours=True)
#pl.subplot(2,1,1)
pl.imshow(loc_corrs,cmap=pl.cm.gray,vmin=0.5,vmax=1)
pl.imshow(mask_show,alpha=.3,vmin=0)

#%%


#pl.subplot(2,1,2)
#loc_corrs=mo.local_correlations(eight_neighbours=True)


#%% extract DF/F from orginal movie, needs to reload the motion corrected movie
m=cb.movie.load(filename_py); 
m=m.crop(crop_top=0,crop_bottom=1,crop_left=0,crop_right=0,crop_begin=0,crop_end=0);
shifts=np.load(filename_mc)['shifts']
totalShifts=np.sum(np.asarray(shifts),axis=0)[:,0:2].tolist()
m=m.apply_shifts(totalShifts) 
max_h,max_w= np.percentile(totalShifts,99,axis=0)
min_h,min_w= np.percentile(totalShifts,1,axis=0)
#max_h,max_w= np.max(totalShifts,axis=0)
#min_h,min_w= np.min(totalShifts,axis=0)
m=m.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)
 
minPercentileRemove=.1;
# remove an estimation of what a Dark patch is, you should provide a better estimate
F0=np.percentile(m,minPercentileRemove)
m=m-F0; 
traces = m.extract_traces_from_masks(all_masks)
#,type='DFF',window_sec=15,minQuantile=8
traces=traces.computeDFF(window_sec=1,minQuantile=8);
traces.plot()


#%%
pl.imshow(traces.T,aspect='auto',interpolation='none')


#%% save the results of the analysis in python format
traces.save('traces.npz')
np.savez(filename_analysis,all_masks=all_masks,spcomps=spcomps,fx=fx,fy=fy,fz=fz)

#%% save the results of the analysis and motion corrected movie in matlab format
import scipy.io as sio
sio.savemat(filename_analysis[:-4]+'.mat', {'all_masks':np.transpose(all_masks,(1,2,0)),'spcomps':np.transpose(spcomps,(1,2,0)),'traces':traces,'tracesDFF':tracesDFF})
m=XMovie(mat=np.load(filename_mc)['mov'], frameRate=frameRate);    
sio.savemat(filename_mc[:-4]+'.mat', {'mov':np.transpose(np.load(filename_mc)['mov'],(1,2,0)),'frameRate':frameRate,'shifts':shifts,'templates':np.transpose(templates,(1,2,0))})


