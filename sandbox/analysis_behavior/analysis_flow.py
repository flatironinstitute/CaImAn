# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
from __future__ import print_function
from builtins import str
from builtins import range
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF

"""
try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import sys
import numpy as np
import psutil
import glob
import os
import scipy
from ipyparallel import Client
# mpl.use('Qt5Agg')
import pylab as pl
pl.ion()
#%
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.behavior import behavior
#%%
from scipy.sparse import coo_matrix
#%%
def get_nonzero_subarray(arr,mask):
    
    x, y = mask.nonzero()   
    
    return arr.toarray()[x.min():x.max()+1, y.min():y.max()+1]
#%%


#%% WHISKERS MASKS
is_wheel = False
mask_all = True
import os
import re
import scipy
gt_names = ['./AG051514-01/060914_C_-32 554 182_COND_C__2/points_trial1.mat',
'./gc-AG052014-02/062014_-219 -221 82_WHISK_COND__1/points_trial11.mat' ,
'./gc-AG052014-02/062014_-219 -221 82_WHISK_COND__1/points_trial1.mat' ,
'./gc-AG052014-02/062314_-213 -218 67_WHISK_COND_A__2/points_trial1.mat' ,
'./AG052014-01/063014_98 -639 144_COND_A__1/points_trial9.mat',
'./AG052014-01/070214_79 -645 131_COND_A_/points_trial99.mat']
expt_fold = [os.path.split(nm)[0].split('/')[1] for nm in gt_names]
expt_names = [os.path.split(nm)[0].split('/')[2] for nm in gt_names]
print(expt_names)
print (expt_fold)
#%%
for e in gt_names:
    print(e)    
    pts = scipy.io.loadmat(e)['points'][0][0]    
    whisk_sess = scipy.signal.savgol_filter(np.abs(np.diff(np.array(pts[8]).T,axis=0)),3,1)[2:]
    num_tr =str(re.findall('\d+',e[-8:-4])[0])
    mat_files = [os.path.join(os.path.split(e)[0],'trial'+num_tr+'.mat')]
    m = cm.load_movie_chain(mat_files[:1],fr=100)[2:]
    pl.figure()
    pl.imshow(m[0],cmap = 'gray')
    pl.plot(pts[0][10]/3,pts[8][10]/3,'r*')
    
    
    mask =behavior.select_roi(np.median(m[::100],0),1)[0]
    
    if mask_all:
        
        np.save(e[:-4]+'_mask_all.npy',mask)
            
    else:
        
        np.save(e[:-4]+'_mask_whisk.npy',mask)
    
    

#%% WHISKERS COMPONENTS
r_values = []
only_magnitude = False
n_components = 4
resize_fact = .5
alpha_spars = 0
#print(expt_names)
for e in gt_names:
    print(e)    
    pts = scipy.io.loadmat(e)['points'][0][0]    
    whisk_sess_pix = scipy.signal.savgol_filter(np.abs(np.diff(np.array(pts[8]).T,axis=0)),3,1)[2:]
    
    num_tr =str(re.findall('\d+',e[-8:-4])[0])
    mat_files = [os.path.join(os.path.split(e)[0],'trial'+num_tr+'.mat')]
    
    
    if mask_all:
        mask = coo_matrix(np.load(e[:-4]+'_mask_all.npy'))
        m = cm.load_movie_chain(mat_files[:1],fr=100)[2:].resize(resize_fact,resize_fact,1)
        mask = cm.movie(mask.toarray().astype(np.float32)[None,:,:]).resize(resize_fact,resize_fact,1)
        mask = coo_matrix(np.array(mask).squeeze())
    else:
        mask = coo_matrix(np.load(e[:-4]+'_mask_whisk.npy'))
#    pl.figure()
#    pl.imshow(m[0],cmap = 'gray')
#    pl.plot(pts[0][10]/3,pts[8][10]/3,'r*')
#    
#    
#    mask =behavior.select_roi(np.median(m[::100],0),1)[0]
    ms = [get_nonzero_subarray(mask.multiply(fr),mask) for fr in m]
    mask_new = get_nonzero_subarray(mask,mask) 
      
    ms = np.dstack(ms)
    ms = cm.movie(ms.transpose([2,0,1]))
    of_or = cm.behavior.behavior.compute_optical_flow(ms,do_show=False,polar_coord=False) 
    
    if only_magnitude:
        of = of_or
        spatial_filter_, time_trace_, norm_fact = cm.behavior.behavior.extract_components(np.sqrt(of[0]**2+of[1]**2),n_components=n_components,verbose = False,normalize_std=False,max_iter=1000)
    else:        
        of = of_or - np.min(of_or)
        if mask_all:
            spatial_filter_, time_trace_, norm_fact = cm.behavior.behavior.extract_components(of,n_components=n_components,verbose = True,normalize_std=False,max_iter=1000)
        else:
            spatial_filter_, time_trace_, norm_fact = cm.behavior.behavior.extract_components(of[:,:1000000,:,:],n_components=n_components,verbose = False,normalize_std=False,max_iter=1000)
    
    mags = []
    dircts = []
    ccs = []
    for ncmp in range(n_components):
        spatial_filter = spatial_filter_[ncmp]
        time_trace = time_trace_[ncmp]
        if only_magnitude:
            mag = scipy.signal.medfilt(time_trace,kernel_size=[1,1]).T
            mag =  scipy.signal.savgol_filter(mag.squeeze(),3,1)                
            dirct = None
        else:
            x,y = scipy.signal.medfilt(time_trace,kernel_size=[1,1]).T
            x =  scipy.signal.savgol_filter(x.squeeze(),3,1)                
            y =  scipy.signal.savgol_filter(y.squeeze(),3,1)
            mag,dirct = to_polar(x-cm.components_evaluation.mode_robust(x),y-cm.components_evaluation.mode_robust(y))
            dirct = scipy.signal.medfilt(dirct.squeeze(),kernel_size=1).T        
            


            
        
        
        spatial_mask = spatial_filter
        spatial_mask[spatial_mask<(np.max(spatial_mask[0])*.99)] = np.nan
        ofmk = of_or*spatial_mask[None,None,:,:]
        range_ = np.std(np.nanpercentile(np.sqrt(ofmk[0]**2+ofmk[1]**2),95,(1,2)))
        mag = (mag/np.std(mag))*range_
        mag = scipy.signal.medfilt(mag.squeeze(),kernel_size=1)
        dirct_orig = dirct.copy()
        if not only_magnitude:
            dirct[mag<1]=np.nan

        whisk_sess_pix_fl = whisk_sess_pix.copy()
    #    wheel_pix_sess_fl_[wheel_pix_sess_fl_< 2] = 2
    #    wheel_pix_sess_fl_= scipy.signal.medfilt(wheel_pix_sess_fl_*np.sign(wheel_pix_sess_fl_).squeeze(),kernel_size=3)
        pl.figure()
        pl.subplot(1,2,1)
        pl.imshow(spatial_filter)
        pl.subplot(1,2,2)
        pl.plot(mag*10,'-')
        pl.plot(whisk_sess_pix_fl)
        if not only_magnitude:
            pl.plot(dirct,'.-')
#        pl.plot(wheel_pix_sess,'c')
        
        cc = np.corrcoef(whisk_sess_pix_fl.squeeze(),mag[1:])
        r_values.append(cc[0,1])
        mags.append(mag)
        dircts.append(dirct_orig)
        ccs.append(cc[0,1])
    
    spatial_filter = spatial_filter_
    time_trace = time_trace_
    mag = mags
    dirct = dircts
    cc = ccs
    print(ccs)  
    
    if mask_all:
        np.savez(e[:-4]+'results_all.npy', mag = mag, dirct= dirct, whisk_sess_pix_fl = whisk_sess_pix_fl, spatial_filter = spatial_filter, time_trace = time_trace, mask = mask, cc = cc, of = of, of_or = of_or, pts = pts)
    else:
        np.savez(e[:-4]+'results_whisk.npy', mag = mag, dirct= dirct, whisk_sess_pix_fl = whisk_sess_pix_fl, spatial_filter = spatial_filter, time_trace = time_trace, mask = mask, cc = cc, of = of, of_or = of_or, pts = pts)

    
    
#%% WHISK PLOTS
resize_fact = .5
r_values=[]
for e in gt_names:
    print(e)
    
    fname = e[:-4]+'results_all.npy.npz'
   
    axlin = pl.subplot(4,2,2)
           
    with np.load(fname) as ld:
        
#        if ld['cc']>=0.9:
        idd = np.argmax(ld['cc'])
        for idd in range(len(ld['cc'])):
            print(ld['dirct'][idd])
            dirct = ld['dirct'][idd][2:]
            mag = ld['mag'][idd][2:]
            dirct[mag<.6*np.nanstd(mag)] = np.nan
            pl.subplot(4,2,1+idd*2)
            m = cm.load(os.path.join(os.path.split(fname)[0],'trial1.mat')).resize(resize_fact,resize_fact).squeeze()
            mask = ld['mask'].item().toarray() 

            min_x,min_y = np.min(np.where(mask),1)
#            max_x,max_y = np.max(np.where(mask),1)
            
            spfl = ld['spatial_filter'][idd]
            spfl = cm.movie(spfl[None,:,:]).squeeze()
            max_x,max_y = np.add( (min_x,min_y), np.shape(spfl) )
               
            mask[min_x:max_x,min_y:max_y] =  spfl
            mask[mask<np.nanpercentile(spfl,70)] = np.nan
#            spfl = ld['spatial_filter'][idd]
            pl.imshow(m[0],cmap = 'gray')
            pl.imshow(mask,alpha = .5)
            pl.axis('off')
            
           
            axelin = pl.subplot(4,2,2+idd*2,sharex = axlin)
            pl.plot(mag*10,'k')
            pl.plot(dirct,'r-',linewidth =  2)
            pts = ld['pts']    
            gt_pix = np.abs(scipy.signal.savgol_filter(np.diff(np.array(ld['pts'][8]).T,axis=0),3,1))[2:]        
            gt_pix_2 = scipy.signal.savgol_filter((np.diff(np.array(ld['pts'][8]).T,axis=0)),3,1)[2:]        
            print(np.corrcoef(gt_pix[1:],mag))
            pl.plot(gt_pix,'c')                
            pl.plot(gt_pix_2,'g')                
            r_values.append(ld['cc'][idd])
        break
#        pl.plot(gt_pix)
#        pl.title(str(ld['cc']))
#        print(np.corrcoef(whisk_pix,mag[2:741]))
[print(r) for r in r_values]
#%% WHEEL MASKS
is_wheel = True
#a = scipy.io.loadmat('AG051514-01.mat')
#expt_names = np.array([n[0][0] for n in a['exptNames']])
#idx_file =np.array([0,len(expt_names)-1,len(expt_names)-2])# all larger than 0.9
#a = scipy.io.loadmat('AG052014-01.mat')
#expt_names = np.array([n[0][0] for n in a['exptNames']])
#idx_file =  np.array([len(expt_names)-1,len(expt_names)-3,len(expt_names)-8]) # len(expt_names): 0.87 , -1: 0.1, -2: .87, -3, .57, -4 .33., -5:.22, -6:0, -7: .75
a = scipy.io.loadmat('AG051514-01.mat')
#a = scipy.io.loadmat('AG052014-01.mat')
#a = scipy.io.loadmat('gc-AG052014-02.mat')
expt_names = np.array([n[0][0] for n in a['exptNames']])
idx_file = np.array(range(len(expt_names))) # 1: 0.98,  len(expt_names) .18,  len(expt_names)-1:.47 (can be fixed?), -2:0,-3: 0.98,-4: 0.96, -5:0, -6:.87, -7:0.98; 

wheel_sessions = []
for sess in a['wheel_pix']:
    wheel_trials = []
    for tr1 in sess:        
        for tr2 in tr1:
            wheel_trials.append((tr2[0][0]))
    wheel_sessions.append(wheel_trials)            

wheel_sessions = np.array(wheel_sessions)
    
trials_id_sessions = []
for sess in a['trials']:
    trials_id_trials = []
    for tr1 in sess:        
        for tr2 in tr1:
            trials_id_trials.append((tr2[0]))
    trials_id_sessions.append(trials_id_trials) 

trials_id_sessions = np.array(trials_id_sessions)    
#%%    
mask_all = True
for e,session,trials_id in zip(expt_names[idx_file],wheel_sessions[idx_file],trials_id_sessions[idx_file]):
    print(e)
    mat_files = [os.path.join(e,'trial'+str(tr)+'.mat') for tr in trials_id]
    m = cm.load_movie_chain(mat_files[:5],fr=100)
    mask =behavior.select_roi(np.median(m[::100],0),1)[0]
    if mask_all:
        np.save(os.path.join(e,'mask_wheel_all.npy'),mask)
    else:
        np.save(os.path.join(e,'mask_wheel.npy'),mask)
        
        

        #%% WHEEL COMPONENTS  

chunk_start = 0
chunk_end = 10000  
r_values = []
only_magnitude = False
n_components = 5
resize_fact = .5
#print(expt_names)
for e,session,trials_id in zip(expt_names[idx_file],wheel_sessions[idx_file],trials_id_sessions[idx_file]):
    print(e)
    
    wheel_pix_sess = np.hstack(session)[chunk_start:chunk_end]    
    pl.figure()
    pl.plot(wheel_pix_sess)
    mat_files = [os.path.join(e,'trial'+str(tr)+'.mat') for tr in trials_id]
                 
                 
    m = cm.load_movie_chain(mat_files[:20],fr=100)[chunk_start:chunk_end]
    if mask_all:
        mask = coo_matrix(np.load(os.path.join(e,'mask_wheel_all.npy')))
        
        mask = cm.movie(mask.toarray().astype(np.float32)[None,:,:])
        mask = coo_matrix(np.array(mask).squeeze())
    else:
        mask = coo_matrix(np.load(os.path.join(e,'mask_wheel.npy')))
        
        
   
    ms = [get_nonzero_subarray(mask.multiply(fr),mask) for fr in m]
    ms = np.dstack(ms)
    ms = cm.movie(ms.transpose([2,0,1]))
    of_or = cm.behavior.behavior.compute_optical_flow(ms,do_show=False,polar_coord=False) 
    of_or = np.concatenate([cm.movie(of_or[0]).resize(resize_fact,resize_fact,1)[np.newaxis,:,:,:],cm.movie(of_or[1]).resize(resize_fact,resize_fact,1)[np.newaxis,:,:,:]],axis = 0)
    if only_magnitude:
        of = of_or
        spatial_filter_, time_trace_, norm_fact = cm.behavior.behavior.extract_components(np.sqrt(of[0]**2+of[1]**2),n_components=n_components,verbose = False,normalize_std=False,max_iter=1000)
    else:        
        of = of_or - np.min(of_or)
        if mask_all:
            spatial_filter_, time_trace_, norm_fact = cm.behavior.behavior.extract_components(of,n_components=n_components,verbose = True,normalize_std=False,max_iter=1000)
        else:
            spatial_filter_, time_trace_, norm_fact = cm.behavior.behavior.extract_components(of[:,:1000000,:,:],n_components=n_components,verbose = False,normalize_std=False,max_iter=1000)
    
    mags = []
    dircts = []
    ccs = []
    for ncmp in range(n_components):
        spatial_filter = spatial_filter_[ncmp]
        time_trace = time_trace_[ncmp]
        if only_magnitude:
            mag = scipy.signal.medfilt(time_trace,kernel_size=[1,1]).T
            mag =  scipy.signal.savgol_filter(mag.squeeze(),3,1)                
            dirct = None
        else:
            x,y = scipy.signal.medfilt(time_trace,kernel_size=[1,1]).T
            x =  scipy.signal.savgol_filter(x.squeeze(),3,1)                
            y =  scipy.signal.savgol_filter(y.squeeze(),3,1)
            mag,dirct = to_polar(x-cm.components_evaluation.mode_robust(x),y-cm.components_evaluation.mode_robust(y))
            dirct = scipy.signal.medfilt(dirct.squeeze(),kernel_size=1).T        
            


            
        
        
        spatial_mask = spatial_filter
        spatial_mask[spatial_mask<(np.max(spatial_mask[0])*.99)] = np.nan
        ofmk = of_or*spatial_mask[None,None,:,:]
        range_ = np.std(np.nanpercentile(np.sqrt(ofmk[0]**2+ofmk[1]**2),95,(1,2)))
        mag = (mag/np.std(mag))*range_
        mag = scipy.signal.medfilt(mag.squeeze(),kernel_size=1)
        dirct_orig = dirct.copy()
        if not only_magnitude:
            dirct[mag<1]=np.nan

        wheel_pix_sess_fl_ = wheel_pix_sess.copy()
    #    wheel_pix_sess_fl_[wheel_pix_sess_fl_< 2] = 2
    #    wheel_pix_sess_fl_= scipy.signal.medfilt(wheel_pix_sess_fl_*np.sign(wheel_pix_sess_fl_).squeeze(),kernel_size=3)
        wheel_pix_sess_fl_= scipy.signal.savgol_filter(wheel_pix_sess_fl_*np.sign(wheel_pix_sess_fl_).squeeze(),3,1)
        
        pl.plot(mag,'-')
        pl.plot(wheel_pix_sess_fl_)
        if not only_magnitude:
            pl.plot(dirct,'.-')
#        pl.plot(wheel_pix_sess,'c')
        
        cc = np.corrcoef(wheel_pix_sess_fl_,mag)
        r_values.append(cc[0,1])
        mags.append(mag)
        dircts.append(dirct_orig)
        ccs.append(cc[0,1])
    
    spatial_filter = spatial_filter_
    time_trace = time_trace_
    mag = mags
    dirct =dircts
    cc = ccs
    
    if not mask_all:
        np.savez(os.path.join(e,'results_wheel_2.npy'),mag = mag, dirct= dirct, wheel_pix_sess = wheel_pix_sess, spatial_filter = spatial_filter, time_trace = time_trace, mask = mask, cc = cc, of = of, of_or = of_or)
    else:
        np.savez(os.path.join(e,'results_wheel_all.npy'),mag = mag, dirct= dirct, wheel_pix_sess = wheel_pix_sess, spatial_filter = spatial_filter, time_trace = time_trace, mask = mask, cc = cc)

    print(ccs)
    

    
#%% WHEEL PLOT
r_values=[]
for e,session,trials_id in zip(expt_names[idx_file],wheel_sessions[idx_file],trials_id_sessions[idx_file]):
    print(e)
    if is_wheel:
        fname = os.path.join(e,'results_wheel.npy.npz')
    else:
        fname = os.path.join(e,'results_whisk.npy.npz')
    pl.figure()
    with np.load(fname) as ld:

#        if ld['cc']>=0.9:
     
        pl.plot(ld['mag'])
        pl.plot(ld['dirct'],'.-')
        pl.plot(ld['wheel_pix_sess'],'k')

        if is_wheel:
            gt_pix = ld['wheel_pix_sess']
        else:
            gt_pix = scipy.signal.savgol_filter(np.abs(np.diff(np.array(pts[0][0][8]).T,axis=0)),3,1)[2:741]
            
            
        r_values.append(ld['cc'])
    
    break    
#        pl.plot(gt_pix)
#        pl.title(str(ld['cc']))
#        print(np.corrcoef(whisk_pix,mag[2:741]))
[print(r) for r in r_values]
#%% Figure 2
r_values=[]
for e,session,trials_id in zip(expt_names[idx_file],wheel_sessions[idx_file],trials_id_sessions[idx_file]):
    print(e)
    if not mask_all:
        fname = os.path.join(e,'results_wheel.npy.npz')
    else:
        fname = os.path.join(e,'results_wheel_all.npy.npz')
    axlin = pl.subplot(5,2,2)
    with np.load(fname) as ld:

#        if ld['cc']>=0.9:
        cc = ld['cc']
        idd = np.argmax(cc)
        dirct = ld['dirct']
        mag = ld['mag']
        spatial_filter = ld['spatial_filter']
        
        print(cc)
        if type(cc) is not list:
            cc = [cc]
            dirct = [dirct]
            mag = [mag]
            spatial_filter = [spatial_filter]
            
        for idd in range(len(cc)):
#            pl.figure()
            
            print(dirct[idd])
            dirct = dirct[idd][2:]
            mag = mag[idd][2:]
            dirct[mag<.6*np.nanstd(mag)] = np.nan
            pl.subplot(5,2,1+idd*2)
            m = cm.load(os.path.join(os.path.split(fname)[0],'trial1.mat'))
            mask = ld['mask'].item().toarray()
            
            min_x,min_y = np.min(np.where(mask),1)
#            max_x,max_y = np.max(np.where(mask),1)
            
            spfl = spatial_filter[idd]
            spfl = cm.movie(spfl[None,:,:]).resize(1/resize_fact,1/resize_fact).squeeze()
#            spfl[spfl<np.nanmax(spfl)*.5] = np.nan
            max_x,max_y = np.add( (min_x,min_y), np.shape(spfl) )
               
            mask[min_x:max_x,min_y:max_y] =  spfl
            mask[mask<np.nanpercentile(spfl,95)] = np.nan
#            spfl = ld['spatial_filter'][idd]
            pl.imshow(m[0],cmap = 'gray')
            pl.imshow(mask,alpha = .5)
            pl.axis('off')
#            pl.imshow(ld['spatial_filter'][idd])
            axelin = pl.subplot(5,2,2+idd*2,sharex = axlin)
            pl.plot(mag,'k')
            pl.plot(dirct,'r-',linewidth = 2)
            
#            pl.plot(ld['wheel_pix_sess'],'k')
            r_values.append(cc[idd])
            pl.axis('tight')
#            pl.xlim([380,700])
        
            
        break
#        pl.plot(gt_pix)
#        pl.title(str(ld['cc']))
#        print(np.corrcoef(whisk_pix,mag[2:741]))
[print(r) for r in r_values]
#%% create_movie
of_or = cm.behavior.behavior.compute_optical_flow(m,do_show=False,polar_coord=False)
#%%
T,d1,d2 = np.shape(m)
X, Y = np.meshgrid(np.arange(d2,0,-10), np.arange(d1,0,-10))
X, Y = np.meshgrid(np.arange(0,d2,10), np.arange(0,d1,10))

for idx, fr in enumerate(m[600:]):
    pl.cla()
    pl.imshow(fr,cmap='gray')
    print(idx)
    pl.quiver(X,Y,of_or[0,600+idx,::10,::10],-of_or[1,600+idx,::10,::10],scale=1 / 0.01, color = 'r')
    pl.ginput(1)
    
#%%
m = cm.load_movie_chain(glob.glob('trial*.mat')[:5],fr=100)
#%%
mask = coo_matrix(behavior.select_roi(m.mean(0),1)[0])
#%
ms = [get_nonzero_subarray(mask.multiply(fr),mask) for fr in m]
ms = np.dstack(ms)
ms = cm.movie(ms.transpose([2,0,1]))
#%%
of_or = cm.behavior.behavior.compute_optical_flow(ms[:15000],do_show=False,polar_coord=False) 
#min1,min0 = np.min(of[1]),np.min(of[0])
#of[1] -= min1
#of[0] -= min0
of = of_or - np.min(of_or)
#%%
spatial_filter, time_trace, norm_fact = cm.behavior.behavior.extract_components(of[:,:15000,:,:],n_components=1,verbose = True,normalize_std=False,max_iter=400)
x,y = scipy.signal.medfilt(time_trace[0],kernel_size=[1,1]).T
spatial_mask = spatial_filter[0]
spatial_mask[spatial_mask<(np.max(spatial_mask[0])*.99)] = np.nan
ofmk = of_or*spatial_mask[None,None,:,:]
#%%
mag,dirct = to_polar(x-cm.components_evaluation.mode_robust(x),y-cm.components_evaluation.mode_robust(y))
range_ = np.std(np.nanpercentile(np.sqrt(ofmk[0]**2+ofmk[1]**2),95,(1,2)))
mag = (mag/np.std(mag))*range_
mag = scipy.signal.medfilt(mag.squeeze(),kernel_size=1)
dirct = scipy.signal.medfilt(dirct.squeeze(),kernel_size=1).T

dirct[mag<1]=np.nan
#mag[mag<.25]=np.nan


pl.plot(mag,'.-')
pl.plot(dirct,'.-')
#%%
spatial_filter, time_trace, norm_fact = cm.behavior.behavior.extract_components(of[:,:2000,:,:],n_components=1,verbose = True)
mag,dirct = scipy.signal.medfilt(time_trace[0],kernel_size=[1,1]).T
mag,dirct = time_trace[0].T

dirct[mag<.25]=np.nan
pl.plot(mag,'.-')
pl.plot(dirct,'.-')