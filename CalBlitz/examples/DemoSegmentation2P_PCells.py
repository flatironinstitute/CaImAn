# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015
@author: agiovann
"""
#% add required packages
import h5py
import calblitz as cb
import time
import pylab as pl
import numpy as np
#% set basic ipython functionalities
pl.ion()
%load_ext autoreload
%autoreload 2


#%%
filename='movies/demoMovie_PC.tif'
frameRate=15.62;
start_time=0;
#%%
filename_py=filename[:-4]+'.npz'
filename_mc=filename[:-4]+'_mc.npz'
filename_analysis=filename[:-4]+'_analysis.npz'
#%% load and motion correct movie (see other Demo for more details)
m=cb.load(filename, fr=frameRate,start_time=start_time);
#%% automatic parameters motion correction
max_shift_h=10;
max_shift_w=10;
m,shifts,xcorrs,template=m.motion_correct(max_shift_w=max_shift_w,max_shift_h=max_shift_h, num_frames_template=None, template = None,method='opencv')

max_h,max_w= np.max(shifts,axis=0)
min_h,min_w= np.min(shifts,axis=0)
m=m.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)
#%%
m.save(filename_mc)
#%% play movie
print 'Playing movie, press q to stop...'
try:
    m.play(backend='opencv',fr=30,gain=2.,magnification=2)
except:
    m.play(fr=30,gain=2.0,magnification=1)

#%% compute delta f over f (DF/F)
initTime=time.time()
m=m-np.min(m)+1;
m,mbl=m.computeDFF(secsWindow=10,quantilMin=50)
print 'elapsed time:' + str(time.time()-initTime) 

#%% denoise and local correlation. this makes the movie look much better
if False:
    loc_corrs=m.local_correlations(eight_neighbours=True)
    m=m.IPCA_denoise(components = 100, batch = 100000)
    m=m*loc_corrs


#%% compute spatial components via ICA PCA
print 'Computing PCA + ICA...'
initTime=time.time()
spcomps=m.IPCA_stICA(componentsPCA=70,componentsICA = 50, mu=1, batch=1000000, algorithm='parallel', whiten=True, ICAfun='logcosh', fun_args=None, max_iter=2000, tol=1e-8, w_init=None, random_state=None);
print 'elapsed time:' + str(time.time()-initTime) 
cb.matrixMontage(spcomps,cmap=pl.cm.gray) # visualize components
 
#%% extract ROIs from spatial components 
#_masks,masks_grouped=m.extractROIsFromPCAICA(spcomps, numSTD=6, gaussiansigmax=2 , gaussiansigmay=2)
_masks,_=cb.extractROIsFromPCAICA(spcomps, numSTD=4.0, gaussiansigmax=.1 , gaussiansigmay=.2)
#cb.matrixMontage(np.asarray(_masks),cmap=pl.cm.gray)

#%%  extract single ROIs from each mask
minPixels=30;
maxPixels=4000;
masks_tmp=[];
for mask in _masks:
    numPixels=np.sum(np.array(mask));        
    if (numPixels>minPixels and numPixels<maxPixels):
#        print numPixels
        masks_tmp.append(mask>0)
        
masks_tmp=np.asarray(masks_tmp,dtype=np.float16)
all_masksForPlot_tmp=[kk*(ii+1)*1.0 for ii,kk in enumerate(masks_tmp)]
len(all_masksForPlot_tmp)


#%%
# reshape dendrites if required(if the movie was resized)
all_masks=masks_tmp              
all_masksForPlot=[kk*(ii+1)*1.0 for ii,kk in enumerate(all_masks)]


#%%
mask_show=np.max(np.asarray(all_masksForPlot_tmp,dtype=np.float16),axis=0);
loc_corrs=m.local_correlations(eight_neighbours=True)
#pl.subplot(2,1,1)
pl.imshow(loc_corrs,cmap=pl.cm.gray,vmin=0.5,vmax=1)
pl.imshow(mask_show>0,alpha=.3,vmin=0)
#%%
cb.matrixMontage(np.asarray(all_masksForPlot_tmp),cmap=pl.cm.gray)

#%% extract DF/F from orginal movie, needs to reload the motion corrected movie
m=cb.load(filename_mc)
# remove an estimation of what a Dark patch is, you should provide a better estimate
F0=np.min(np.min(m,axis=0))
m=m-F0; 
traces = m.extract_traces_from_masks(all_masks)
##,type='DFF',window_sec=15,minQuantile=8
tracesDFF=traces.computeDFF(window_sec=1,minQuantile=8)
tracesDFF.plot()
#%%
pl.imshow(traces.T,aspect='auto',interpolation='none')
#%% save the results of the analysis in python format
np.savez(filename_analysis,all_masks=all_masks,spcomps=spcomps,traces=traces,tracesDFF=tracesDFF,shifts=shifts)
#%% save the results of the analysis and motion corrected movie in matlab format
import scipy.io as sio
sio.savemat(filename_analysis[:-4]+'.mat', {'all_masks':np.transpose(all_masks,(1,2,0)),'spcomps':np.transpose(spcomps,(1,2,0)),'traces':traces,'tracesDFF':tracesDFF,'shifts':shifts})
sio.savemat(filename_mc[:-4]+'.mat', {'mov':np.transpose(m,(1,2,0)),'frameRate':frameRate,'shifts':shifts})
