# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:06:17 2016

@author: agiovann
"""
try:
    %load_ext autoreload
    %autoreload 2
    print 1
except:
    print 'NOT IPYTHON'

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
#plt.ion()

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
from glob import glob
import os
import scipy
from ipyparallel import Client
import calblitz as cb
from calblitz.granule_cells import utils_granule as gc
#%%
batch_mode= True
is_blob=True
#%%
# done '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627105123/'
#  errors:  '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160623161504/',
#base_folders=[
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627154015/',            
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160624105838/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160625132042/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160626175708/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627110747/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628100247/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160705103903/',

#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628162522/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160629123648/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160630120544/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160701113525/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160702152950/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160703173620/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160704130454/',
#              ]
#error:               '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711104450/', 
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712105933/',             
#base_folders=[
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160710134627/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160710193544/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711164154/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711212316/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712101950/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712173043/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160713100916/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160713171246/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714094320/',
#              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714143248/'
#              ]              
for base_folder in base_folders:
    img_descr=cb.utils.get_image_description_SI(glob(base_folder+'2016*.tif')[0])[0]
    f_rate=img_descr['scanimage.SI.hRoiManager.scanFrameRate']
    print f_rate    
    #%%
    fls=glob(os.path.join(base_folder,'2016*.tif'))
    fls.sort()     
    print fls 
    # verufy they are ordered 
    #%%
    triggers_img,trigger_names_img=gc.extract_triggers(fls,read_dictionaries=False)     
    np.savez(base_folder+'all_triggers.npz',triggers=triggers_img,trigger_names=trigger_names_img)   
    #%% get information from eyelid traces
    t_start=time()     
    camera_file=glob(os.path.join(base_folder,'*_cam2.h5'))
    assert len(camera_file)==1, 'there are none or two camera files'    
    res_bt=gc.get_behavior_traces(camera_file[0],t0=0,t1=8.0,freq=60,ISI=.25,draw_rois=False,plot_traces=False,mov_filt_1d=True,window_lp=5)   
    t_end=time()-t_start
    print t_end
    #%%
    np.savez(base_folder+'behavioral_traces.npz',**res_bt)
    #%%
    with np.load(base_folder+'behavioral_traces.npz') as ld:
        res_bt=dict(**ld)
    #%%
    pl.close()
    tm=res_bt['time']
    f_rate_bh=1/np.median(np.diff(tm))
    ISI=res_bt['trial_info'][0][3]-res_bt['trial_info'][0][2]
    eye_traces=np.array(res_bt['eyelid'])
    idx_CS_US=res_bt['idx_CS_US']
    idx_US=res_bt['idx_US']
    idx_CS=res_bt['idx_CS']
    
    idx_ALL=np.sort(np.hstack([idx_CS_US,idx_US,idx_CS]))
    eye_traces,amplitudes_at_US, trig_CRs=gc.process_eyelid_traces(eye_traces,tm,idx_CS_US,idx_US,idx_CS,thresh_CR=.15,time_CR_on=-.1,time_US_on=.05)
    
    idxCSUSCR = trig_CRs['idxCSUSCR']
    idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
    idxCSCR = trig_CRs['idxCSCR']
    idxCSNOCR = trig_CRs['idxCSNOCR']
    idxNOCR = trig_CRs['idxNOCR']
    idxCR = trig_CRs['idxCR']
    idxUS = trig_CRs['idxUS']
    idxCSCSUS=np.concatenate([idx_CS,idx_CS_US])
    
    
    pl.plot(tm,np.mean(eye_traces[idxCSUSCR],0))       
    pl.plot(tm,np.mean(eye_traces[idxCSUSNOCR],0))     
    pl.plot(tm,np.mean(eye_traces[idxCSCR],0))
    pl.plot(tm,np.mean(eye_traces[idxCSNOCR],0))    
    pl.plot(tm,np.mean(eye_traces[idx_US],0))
    pl.legend(['idxCSUSCR','idxCSUSNOCR','idxCSCR','idxCSNOCR','idxUS'])
    pl.xlabel('time to US (s)')
    pl.ylabel('eyelid closure')
    plt.axvspan(-ISI,ISI, color='g', alpha=0.2, lw=0)
    plt.axvspan(0,0.03, color='r', alpha=0.2, lw=0)
    
    pl.xlim([-.5,1])
    pl.savefig(base_folder+'behavioral_traces.pdf')
    #%%
    #pl.close()
    #bins=np.arange(0,1,.01)
    #pl.hist(amplitudes_at_US[idxCR],bins=bins)
    #pl.hist(amplitudes_at_US[idxNOCR],bins=bins)
    #pl.savefig(base_folder+'hist_behav.pdf')
    
    
    #%%
    pl.close() 
    f_results= glob(base_folder+'*results_analysis.npz')
    f_results.sort()
    for rs in f_results:
        print rs
    #%% load results and put them in lists
    A_s,C_s,YrA_s, Cn_s, b_s, f_s, shape =  gc.load_results(f_results)     
    B_s, lab_imgs, cm_s  = gc.threshold_components(A_s,shape, min_size=5,max_size=50,max_perc=.5)
    #%%
    if not batch_mode:
        for i,A_ in enumerate(B_s):
             sizes=np.array(A_.sum(0)).squeeze()
             pl.subplot(2,3,i+1)
             pl.imshow(np.reshape(A_.sum(1),shape,order='F'),cmap='gray',vmax=.5)
    #%% compute mask distances 
    if len(B_s)>1:
        max_dist=30
        D_s=gc.distance_masks(B_s,cm_s,max_dist)       
        np.savez(base_folder+'distance_masks.npz',D_s=D_s)
        #%%
        if not batch_mode:
            for ii,D in enumerate(D_s):
                pl.subplot(3,3,ii+1)
                pl.imshow(D,interpolation='None')
            
        #%% find matches
        matches,costs =  gc.find_matches(D_s, print_assignment=False)
        #%%
        neurons=gc.link_neurons(matches,costs,max_cost=0.6,min_FOV_present=None)
    else:
        neurons=[np.arange(B_s[0].shape[-1])]
    #%%
    np.savez(base_folder+'neurons_matching.npz',matches=matches,costs=costs,neurons=neurons,D_s=D_s)
    #%%
    re_load = False
    if re_load:
        import calblitz as cb
        from calblitz.granule_cells import utils_granule as gc
        from glob import glob
        import numpy as np
        import os
        import scipy 
        import pylab as pl
        import ca_source_extraction as cse
        
        if is_blob:
            with np.load(base_folder+'distance_masks.npz') as ld:
                D_s=ld['D_s']
            with np.load(base_folder+'neurons_matching.npz') as ld:
                locals().update(ld)
    
                
    
        with np.load(base_folder+'all_triggers.npz') as at:
            triggers_img=at['triggers']
            trigger_names_img=at['trigger_names'] 
            
        with np.load(base_folder+'behavioral_traces.npz') as ld: 
            res_bt = dict(**ld)
            tm=res_bt['time']
            f_rate_bh=1/np.median(np.diff(tm))
            ISI=res_bt['trial_info'][0][3]-res_bt['trial_info'][0][2]
            eye_traces=np.array(res_bt['eyelid'])
            idx_CS_US=res_bt['idx_CS_US']
            idx_US=res_bt['idx_US']
            idx_CS=res_bt['idx_CS']
            
            idx_ALL=np.sort(np.hstack([idx_CS_US,idx_US,idx_CS]))
            eye_traces,amplitudes_at_US, trig_CRs=gc.process_eyelid_traces(eye_traces,tm,idx_CS_US,idx_US,idx_CS,thresh_CR=.15,time_CR_on=-.1,time_US_on=.05)
            
            idxCSUSCR = trig_CRs['idxCSUSCR']
            idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
            idxCSCR = trig_CRs['idxCSCR']
            idxCSNOCR = trig_CRs['idxCSNOCR']
            idxNOCR = trig_CRs['idxNOCR']
            idxCR = trig_CRs['idxCR']
            idxUS = trig_CRs['idxUS']
            idxCSCSUS=np.concatenate([idx_CS,idx_CS_US])    
            
            
        f_results= glob(base_folder+'*results_analysis.npz')
        f_results.sort()
        for rs in f_results:
            print rs     
        print '*****'        
        A_s,C_s,YrA_s, Cn_s, b_s, f_s, shape =  gc.load_results(f_results) 
        if is_blob:
            remove_unconnected_components=True
        else:
            remove_unconnected_components=False
            
            neurons=[]
            for xx in A_s:
                neurons.append(np.arange(A_s[0].shape[-1]))
            
        B_s, lab_imgs, cm_s  = gc. threshold_components(A_s,shape, min_size=5,max_size=50,max_perc=.5,remove_unconnected_components=remove_unconnected_components)
    #%%
       
    row_cols=np.ceil(np.sqrt(len(A_s)))        
    for idx,B in enumerate(A_s):
         pl.subplot(row_cols,row_cols,idx+1)
         pl.imshow(np.reshape(B[:,neurons[idx]].sum(1),shape,order='F'))
    pl.savefig(base_folder+'neuron_matches.pdf')
         
    #%%
    if not batch_mode:  
        num_neurons=neurons[0].size
        for neuro in range(num_neurons):
            for idx,B in enumerate(A_s):
                 pl.subplot(row_cols,row_cols,idx+1)
                 pl.imshow(np.reshape(B[:,neurons[idx][neuro]].sum(1),shape,order='F'))
            pl.pause(.01)     
            for idx,B in enumerate(A_s):
                pl.subplot(row_cols,row_cols,idx+1)
                pl.cla()       
    
    #%%
    if 0:
        idx=0
        for  row, column in zip(matches[idx][0],matches[idx][1]):
            value = D_s[idx][row,column]
            if value < .5:
                pl.cla() 
                pl.imshow(np.reshape(B_s[idx][:,row].todense(),(512,512),order='F'),cmap='gray',interpolation='None')    
                pl.imshow(np.reshape(B_s[idx+1][:,column].todense(),(512,512),order='F'),alpha=.5,cmap='hot',interpolation='None')               
                if B_s[idx][:,row].T.dot(B_s[idx+1][:,column]).todense() == 0:
                    print 'Flaw'            
                pl.pause(.3)
    
    #%%
    tmpl_name=glob(base_folder+'*template_total.npz')[0]
    print tmpl_name
    with np.load(tmpl_name) as ld:
        mov_names_each=ld['movie_names']
    
    
    traces=[]
    traces_BL=[]
    traces_DFF=[]
    all_chunk_sizes=[]
    
    for idx, mov_names in enumerate(mov_names_each):
        idx=0
        A=A_s[idx][:,neurons[idx]]
    #    C=C_s[idx][neurons[idx]]
    #    YrA=YrA_s[idx][neurons[idx]]
        b=b_s[idx]
        f=f_s[idx]
        chunk_sizes=[]
        for mv in mov_names:
                base_name=os.path.splitext(os.path.split(mv)[-1])[0]
                with np.load(base_folder+base_name+'.npz') as ld:
                    TT=len(ld['shifts'])            
                chunk_sizes.append(TT)
    
                
        all_chunk_sizes.append(chunk_sizes)
    
        traces_,traces_DFF_,traces_BL_ = gc.generate_linked_traces(mov_names,chunk_sizes,A,b,f)
        traces=traces+traces_
        traces_DFF=traces_DFF+traces_DFF_
        traces_BL=traces_BL+traces_BL_
    
    #%%
    import pickle
    with open(base_folder+'traces.pk','w') as f: 
        pickle.dump(dict(traces=traces,traces_BL=traces_BL,traces_DFF=traces_DFF),f)   
    
    #%%
    if not batch_mode:
        with open(base_folder+'traces.pk','r') as f:    
            locals().update(pickle.load(f)   )
    #%%
    chunk_sizes=[]
    for idx,mvs in enumerate(mov_names_each):    
        print idx 
        for mv in mvs:
            base_name=os.path.splitext(os.path.split(mv)[-1])[0]
            with np.load(os.path.join(base_folder,base_name+'.npz')) as ld:
                TT=len(ld['shifts'])            
            chunk_sizes.append(TT)
            
            
    min_chunk=np.min(chunk_sizes)
    max_chunk=np.max(chunk_sizes)
    num_chunks=np.sum(chunk_sizes)
    #%%
    import copy
    Ftraces=copy.deepcopy(traces_DFF[:])
    
    #%%

    #%%
    interpolate=False
    CS_ALONE=0
    US_ALONE=   1
    CS_US=2
    
    samples_before=np.int(2.8*f_rate)
    samples_after=np.int(7.3*f_rate)-samples_before
    
    
    if interpolate:
        Ftraces_mat=np.zeros([len(chunk_sizes),len(traces[0]),max_chunk])
        abs_frames=np.arange(max_chunk)
    else:    
        Ftraces_mat=np.zeros([len(chunk_sizes),len(traces[0]),samples_after+samples_before])
        
    crs=idxCR
    nocrs=idxNOCR
    uss=idxUS
    
    triggers_img=np.array(triggers_img)
    
    idx_trig_CS=triggers_img[:][:,0]
    idx_trig_US=triggers_img[:][:,1]
    trial_type=triggers_img[:][:,2]
    length=triggers_img[:][:,-1]
    ISI=np.int(np.nanmedian(idx_trig_US)-np.nanmedian(idx_trig_CS))
    
    for idx,fr in enumerate(chunk_sizes):
    
        print idx
        
        if interpolate:
    
            if fr!=max_chunk:
        
                f1=scipy.interpolate.interp1d(np.arange(fr) , Ftraces[idx] ,axis=1, bounds_error=False, kind='linear')  
                Ftraces_mat[idx]=np.array(f1(abs_frames))
                
            else:
                
                Ftraces_mat[idx]=Ftraces[idx][:,trigs_US-samples_before]
        
        
        else:
    
            if trial_type[idx] == CS_ALONE:
                    Ftraces_mat[idx]=Ftraces[idx][:,np.int(idx_trig_CS[idx]+ISI-samples_before):np.int(idx_trig_CS[idx]+ISI+samples_after)]
            else:
                    Ftraces_mat[idx]=Ftraces[idx][:,np.int(idx_trig_US[idx]-samples_before):np.int(idx_trig_US[idx]+samples_after)]
    
    #%%
    wheel_traces, movement_at_CS, trigs_mov = gc.process_wheel_traces(np.array(res_bt['wheel']),tm,thresh_MOV_iqr=1000,time_CS_on=-.25,time_US_on=0)    
    print trigs_mov
    mn_idx_CS_US=np.intersect1d(idx_CS_US,trigs_mov['idxNO_MOV'])
    nm_idx_US=np.intersect1d(idx_US,trigs_mov['idxNO_MOV'])
    nm_idx_CS=np.intersect1d(idx_CS,trigs_mov['idxNO_MOV'])
    nm_idxCSUSCR = np.intersect1d(idxCSUSCR,trigs_mov['idxNO_MOV'])
    nm_idxCSUSNOCR = np.intersect1d(idxCSUSNOCR,trigs_mov['idxNO_MOV'])
    nm_idxCSCR = np.intersect1d(idxCSCR,trigs_mov['idxNO_MOV'])
    nm_idxCSNOCR = np.intersect1d(idxCSNOCR,trigs_mov['idxNO_MOV'])
    nm_idxNOCR = np.intersect1d(idxNOCR,trigs_mov['idxNO_MOV'])
    nm_idxCR = np.intersect1d(idxCR,trigs_mov['idxNO_MOV'])
    nm_idxUS = np.intersect1d(idxUS,trigs_mov['idxNO_MOV'])
    nm_idxCSCSUS=np.intersect1d(idxCSCSUS,trigs_mov['idxNO_MOV'])
    #%%
    threshold_responsiveness=0.1
    ftraces=Ftraces_mat.copy()       
    ftraces=ftraces-np.median(ftraces[:,:,:samples_before-ISI],axis=(2))[:,:,np.newaxis]   
    amplitudes_responses=np.mean(ftraces[:,:,samples_before+ISI-1:samples_before+ISI+1],-1)
    cell_responsiveness=np.median(amplitudes_responses[nm_idxCSCSUS],axis=0)
    fraction_responsive=len(np.where(cell_responsiveness>threshold_responsiveness)[0])*1./np.shape(ftraces)[1]
    print fraction_responsive
    ftraces=ftraces[:,cell_responsiveness>threshold_responsiveness,:]
    amplitudes_responses=np.mean(ftraces[:,:,samples_before+ISI-1:samples_before+ISI+1],-1)
    #%%
    np.savez('ftraces.npz',ftraces=ftraces,samples_before=samples_before,samples_after=samples_after,ISI=ISI)
    
    
    #%%pl.close()
    pl.close()
    t=np.arange(-samples_before,samples_after)/f_rate
    pl.plot(t,np.median(ftraces[nm_idxCR],axis=(0,1)),'-*')
    pl.plot(t,np.median(ftraces[nm_idxNOCR],axis=(0,1)),'-d')
    pl.plot(t,np.median(ftraces[nm_idxUS],axis=(0,1)),'-o')
    plt.axvspan((-ISI)/f_rate, 0, color='g', alpha=0.2, lw=0)
    plt.axvspan(0, 0.03, color='r', alpha=0.5, lw=0)
    pl.xlabel('Time to US (s)')
    pl.ylabel('DF/F')
    pl.xlim([-.5, 1])
    pl.legend(['CR+','CR-','US'])
    pl.savefig(base_folder+'eyelid_resp_by_trial.pdf')
    
    #%%
    if not batch_mode:
        pl.close()
        for cell in range(ftraces.shape[1]):   
        #    pl.cla()
            pl.subplot(11,10,cell+1)
            print cell
            tr_cr=np.median(ftraces[crs,cell,:],axis=(0))    
            tr_nocr=np.median(ftraces[nocrs,cell,:],axis=(0))    
            tr_us=np.median(ftraces[uss,cell,:],axis=(0))  
            pl.imshow(ftraces[np.concatenate([uss,nocrs,crs]),cell,:],aspect='auto',vmin=0,vmax=1)
            pl.xlim([samples_before-10,samples_before+10])
            pl.axis('off')
        #    pl.plot(tr_cr,'b')
        #    pl.plot(tr_nocr,'g')
        #    pl.plot(tr_us,'r')
        #    pl.legend(['CR+','CR-','US'])
        #    pl.pause(1)
    #%%
    import pandas
    
    bins=np.arange(-.1,.3,.05)
    n_bins=6
    dfs=[];
    dfs_random=[];
    x_name='ampl_eye'
    y_name='ampl_fl'
    for resps in amplitudes_responses.T:
        idx_order=np.arange(len(idxCSCSUS))    
        dfs.append(pandas.DataFrame(
            {y_name: resps[idxCSCSUS[idx_order]],
             x_name: amplitudes_at_US[idxCSCSUS]}))
             
        idx_order=np.random.permutation(idx_order)         
        dfs_random.append(pandas.DataFrame(
            {y_name: resps[idxCSCSUS[idx_order]],
             x_name: amplitudes_at_US[idxCSCSUS]}))
    
    
    r_s=[]
    r_ss=[]
    
    for df,dfr in zip(dfs,dfs_random): # random scramble
    
        if bins is None:
            [_,bins]=np.histogram(dfr.ampl_eye,n_bins)         
        groups = dfr.groupby(np.digitize(dfr.ampl_eye, bins))
        grouped_mean = groups.mean()
        grouped_sem = groups.sem()
        (r,p_val)=scipy.stats.pearsonr(grouped_mean.ampl_eye,grouped_mean.ampl_fl)
    #    r=np.corrcoef(grouped_mean.ampl_eye,grouped_mean.ampl_fl)[0,1]
        
        r_ss.append(r)
        
        if bins is None:
            [_,bins]=np.histogram(df.ampl_eye,n_bins)         
    
        groups = df.groupby(np.digitize(df.ampl_eye, bins))    
        grouped_mean = groups.mean()
        grouped_sem= groups.sem()    
        (r,p_val)=scipy.stats.pearsonr(grouped_mean.ampl_eye,grouped_mean.ampl_fl)
    #    r=np.corrcoef(grouped_mean.ampl_eye,grouped_mean.ampl_fl)[0,1]
        r_s.append(r)    
        if r_s[-1]>.86:
            pl.subplot(1,2,1)
            print 'found'
            pl.errorbar(grouped_mean.ampl_eye,grouped_mean.ampl_fl,grouped_sem.ampl_fl.as_matrix(),grouped_sem.ampl_eye.as_matrix(),fmt='.')
            pl.scatter(grouped_mean.ampl_eye,grouped_mean.ampl_fl,s=groups.apply(len).values*3)#
            pl.xlabel(x_name)
            pl.ylabel(y_name)
    
    mu_scr=np.mean(r_ss)
    
    std_scr=np.std(r_ss)
    [a,b]=np.histogram(r_s,20)
    
    pl.subplot(1,2,2)
    pl.plot(b[1:],scipy.signal.savgol_filter(a,3,1))  
    plt.axvspan(mu_scr-std_scr, mu_scr+std_scr, color='r', alpha=0.2, lw=0)
    pl.xlabel('correlation coefficients')
    pl.ylabel('bin counts')
    pl.savefig(base_folder+'correlations.pdf')
    
    
    
    #%%
    if not batch_mode:
        r_s=[]
        for resps in amplitudes_responses.T:
            r=np.corrcoef(amplitudes_at_US[idxCSCSUS],resps[idxCSCSUS])[0,1]    
        #    if r>.25: 
        #        pl.scatter(amplitudes_at_US[idxCSCSUS],resps[idxCSCSUS])
        #    bins=np.arange(-.3,1.5,.2)
        #    a,b=np.histogram(resps,bins)
        #    new_dat=[]
        #    for bb in a:
        #        
            r_s.append(r)
            pl.xlabel('Amplitudes CR')
            pl.ylabel('Amplitudes GC responses')
        
        pl.hist(r_s)    

#%%
##
##base_name='20160518133747_'
##cam1=base_name+'cam1.h5'
##cam2=base_name+'cam2.h5'
##meta_inf=base_name+'data.h5'
##
##mtot=[]
##eye_traces=[]
##tims=[]
##trial_info=[]
##
##with h5py.File(cam2) as f:
##
##    with h5py.File(meta_inf) as dt:
##
##        rois=np.asarray(dt['roi'],np.float32)
##        
##        trials = f.keys()
##        trials.sort(key=lambda(x): np.int(x.replace('trial_','')))
##        trials_idx=[np.int(x.replace('trial_',''))-1 for x in trials]
##
##        
##   
##        
##        for tr,idx_tr in zip(trials,trials_idx):
##            
##            print tr
##
##            trial=f[tr]  
##
##            mov=np.asarray(trial['mov'])        
##
##            if 0:
##
##                pl.imshow(np.mean(mov,0))
##                pts=pl.ginput(-1)
##                pts = np.asarray(pts, dtype=np.int32)
##                data = np.zeros(np.shape(mov)[1:], dtype=np.int32)
##        #        if CV_VERSION == 2:
##                #lt = cv2.CV_AA
##        #        elif CV_VERSION == 3:
##                lt = cv2.LINE_AA
##                cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)
##                rois[0]=data
###            eye_trace=np.mean(mov*rois[0],axis=(1,2))
###            mov_trace=np.mean((np.diff(np.asarray(mov,dtype=np.float32),axis=0)**2)*rois[1],axis=(1,2))
##            mov=np.transpose(mov,[0,2,1])
##            
##            mov=mov[:,:,::-1]
##
##            if  mov.shape[0]>0:
##                ts=np.array(trial['ts'])
##                if np.size(ts)>0:
##        #            print (ts[-1,0]-ts[0,0])
##                    new_ts=np.linspace(0,ts[-1,0]-ts[0,0],np.shape(mov)[0])
##                    
##                    print 1/np.mean(np.diff(new_ts))
##                    tims.append(new_ts)
##                    
##                mov=cb.movie(mov*rois[0][::-1].T,fr=1/np.mean(np.diff(new_ts)))
##                x_max,y_max=np.max(np.nonzero(np.max(mov,0)),1)
##                x_min,y_min=np.min(np.nonzero(np.max(mov,0)),1)
##                mov=mov[:,x_min:x_max,y_min:y_max]                                
##                mov=np.mean(mov, axis=(1,2))
##        
##                if mov.ndim == 3:
##                    window_hp=(177,1,1)
##                    window_lp=(7,1,1)
##                    bl=signal.medfilt(mov,window_hp)
##                    mov=signal.medfilt(mov-bl,window_lp)
##
##                else:
##                    window_hp=201
##                    window_lp=3
##                    bl=signal.medfilt(mov,window_hp)
###                    bl=cse.utilities.mode_robust(mov)
##                    mov=signal.medfilt(mov-bl,window_lp)
##
##                    
##                if mov.ndim == 3:
##                    eye_traces.append(np.mean(mov, axis=(1,2)))
##                else:
##                    eye_traces.append(mov)
##                    
##                mtot.append(mov)   
##                trial_info.append(dt['trials'][idx_tr,:])
##        #            cb.movie(mov,fr=1/np.mean(np.diff(new_ts)))
#
##%%
##%%
#sub_trig_img=downsample_triggers(triggers_img.copy(),fraction_downsample=.3)
##%%
#if num_frames_movie != triggers[-1,-1]:
#        raise Exception('Triggers values do not match!')
#        
##%% 
##fnames=[]
##sub_trig_names=trigger_names[39:95].copy()
##sub_trig=triggers[39:95].copy().T
##for a,b in zip(sub_trig_names,sub_trig):
##    fnames.append(a+'.hdf5')
##
##fraction_downsample=.333333333333333333333; # useful to downsample the movie across time. fraction_downsample=.1 measn downsampling by a factor of 10
##sub_trig[:2]=np.round(sub_trig[:2]*fraction_downsample)
##sub_trig[-1]=np.floor(sub_trig[-1]*fraction_downsample)
##sub_trig[-1]=np.cumsum(sub_trig[-1])
##fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample),remove_init=0,idx_xy=(slice(90,-10,None),slice(30,-120,None)))
###%%
##m=cb.load(fname_new,fr=30*fraction_downsample)
##T,d1,d2=np.shape(m)
##%%
##if T != sub_trig[-1,-1]:
##    raise Exception('Triggers values do not match!')
##%% how to take triggered aligned movie
#wvf=mmm.take(trg)
##%%
#newm=m.take(trg,axis=0)
#newm=newm.mean(axis=1)
##%%
#(newm-np.mean(newm,0)).play(backend='opencv',fr=3,gain=2.,magnification=1,do_loop=True)
##%%v
#Yr,d1,d2,T=cse.utilities.load_memmap(fname_new)
#d,T=np.shape(Yr)
#Y=np.reshape(Yr,(d1,d2,T),order='F') # 3D version of the movie 
#
##%%
#
#pl.plot(np.nanmedian(np.array(eye_traces).T,1))
#
##%%
#mov = np.concatenate(mtot,axis=0)           
#m1=cb.movie(mov,fr=1/np.mean(np.diff(new_ts)))
##x_max,y_max=np.max(np.nonzero(np.max(m,0)),1)
##x_min,y_min=np.min(np.nonzero(np.max(m,0)),1)
##m1=m[:,x_min:x_max,y_min:y_max]
##%% filters
#b, a = signal.butter(8, [.05, .5] ,'bandpass')
#pl.plot(np.mean(m1,(1,2))-80)
#pl.plot(signal.lfilter(b,a,np.mean(m1,(1,2))),linewidth=2)
##%%
#m1.play(backend='opencv',gain=1.,fr=f_rate,magnification=3)
##%% NMF
#comps, tim,_=cb.behavior.extract_components(np.maximum(0,m1-np.min(m1,0)),n_components=4,init='nndsvd',l1_ratio=1,alpha=0,max_iter=200,verbose=True)
#pl.plot(np.squeeze(np.array(tim)).T)
##%% ICA
#from sklearn.decomposition import FastICA
#fica=FastICA(n_components=3,whiten=True,max_iter=200,tol=1e-6)
#X=fica.fit_transform(np.reshape(m1,(m1.shape[0],m1.shape[1]*m1.shape[2]),order='F').T,)
#pl.plot(X)
##%%
#for count,c in enumerate(comps):
#    pl.subplot(2,3,count+1)
#    pl.imshow(c)
#    
##%%
#md=cse.utilities.mode_robust(m1,0)
#mm1=m1*(m1<md)
#rob_std=np.sum(mm1**2,0)/np.sum(mm1>0,0)
#rob_std[np.isnan(rob_std)]=0
#mm2=m1*(m1>(md+rob_std))
##%%
#            
#dt = h5py.File('20160423165229_data.h5')   
##sync for software
#np.array(dt['sync'])
#dt['sync'].attrs['keys']     
#dt['trials']
#dt['trials'].attrs
#dt['trials'].attrs['keys']
## you needs to apply here the sync on dt['sync'], like, 
#us_time_cam1=np.asarray(dt['trials'])[:,3] - np.array(dt['sync'])[1]
## main is used as the true time stamp, and you can adjust the value with respect to main sync value
#np.array(dt['sync']) # these are the values read on a unique clock from the three threads
##%%
#from skimage.external import tifffile
#
#tf=tifffile.TiffFile('20160423165229_00001_00001.tif')   
#imd=tf.pages[0].tags['image_description'].value
#for pag in tf.pages:
#    imd=pag.tags['image_description'].value
#    i2cd=si_parse(imd)['I2CData']
#    print (i2cd)
###%%
##with h5py.File('20160705103903_cam2.h5') as f1:
##    for k in f1.keys()[:1]:
##        m = np.array(f1[k]['mov'])
##        
##        
##pl.imshow(np.mean(m,0),cmap='gray')
###%%
##with h5py.File('20160705103903_data.h5') as f1:
##    print f1.keys()    
##    rois= np.array(f1['roi'])
###%%
##with h5py.File('20160705103903_cam2.h5') as f1:
##    for k in f1.keys()[:1]:
##        m = np.array(f1[k]['mov'])
##        
##        
##pl.imshow(np.mean(m,0),cmap='gray')
##pl.imshow(rois[0],alpha=.3)
##pl.imshow(rois[1],alpha=.3)
##       