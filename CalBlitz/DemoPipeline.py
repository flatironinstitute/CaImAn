# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""
#%
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
import calblitz as cb
from ipyparallel import Client
import os
import shutil
import glob
#%%
is_interactive=False
downsample_factor=.3
#%
backend='SLURM'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()*.75),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'
#%%

#folders=['20160711104450',
#'20160711164154','20160711212316',
#'20160710193544',

#'20160712105933',
#'20160712101950',
#'20160713100916',
#'20160714143248',
#'20160714094320',
#'20160710134627',
#'20160712173043']

#folders=[
#'20160713171246']   # it does not work partially. ValueError: I/O operation on closed file. After some CNMFs

    
#folders=[
#'20160628100247',
#'20160627154015',
#'20160628162522']


#folders=[
#'20160624105838',
#'20160626175708',
#'20160701113525',
#'20160702152950',
#'20160705103903',
#folders=[
#'20160627110747',
#'20160623161504',
#'20160625132042',
#'20160627105123',
#folders=[

#'20160629123648',
#'20160630120544',
#'20160703173620',
#'20160704130454',
#'20160705103903'
#]
#folders=[
#'',
#'',
#'',
#'',
#'','','']

errors=[]
for sess_folder in folders[:]:
    
    #% start cluster for efficient computation
    single_thread=False
    if single_thread:
        dview=None
    else:    
        try:
            c.close()
        except:
            print 'C was not existing, creating one'
        print "Stopping  cluster to avoid unnencessary use of memory...."
        sys.stdout.flush()  
        if backend == 'SLURM':
            try:
                cse.utilities.stop_server(is_slurm=True)
            except:
                print 'Nothing to stop'
            slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
            cse.utilities.start_server(slurm_script=slurm_script)
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            c = Client(ipython_dir=pdir, profile=profile)        
        else:
            
            cse.utilities.stop_server()
            cse.utilities.start_server()        
            c=Client()
    
        print 'Using '+ str(len(c)) + ' processes'
        dview=c[:len(c)]
    #%
    
    fnames=[]
    for file in glob.glob(sess_folder+"/*"):
        if file.startswith(sess_folder) and file.endswith(".tif"):
            fnames.append(os.path.abspath(file))
    fnames.sort()
    print fnames  
    idx_start=[i for i in xrange(len(fnames[0])) if fnames[0][i] != fnames[- 1][i]][0]
    base_name=fnames[0][:idx_start]
    
    #%
#        t1 = time()
#        file_res=cb.motion_correct_parallel(fnames,fr=30,template=None,margins_out=0,max_shift_w=45, max_shift_h=45,dview=dview,apply_smooth=True)
#        t2=time()-t1
#        print t2
    #%   
    all_movs=[]
    for f in  fnames:
        idx=f.find('.')
        with np.load(f[:idx+1]+'npz') as fl:
            print f
    #        pl.subplot(1,2,1)
    #        pl.imshow(fl['template'],cmap=pl.cm.gray)
    #        pl.subplot(1,2,2)
                 
            all_movs.append(fl['template'][np.newaxis,:,:])
    #        pl.plot(fl['shifts'])  
    #        pl.pause(.001)
    #        pl.cla()
    #%
    all_movs=cb.movie(np.concatenate(all_movs,axis=0),fr=1)        
    all_movs,shifts,_,_=all_movs.motion_correct(template=np.median(all_movs,axis=0))
    if is_interactive:
        all_movs.play(backend='opencv',gain=75.)
    #%
    all_movs=np.array(all_movs)
    #%
    num_movies_per_chunk=30        
    chunks=range(0,len(fnames),num_movies_per_chunk)
    chunks[-1]=len(fnames)
    print chunks
    movie_names=[]
    
    for idx in range(len(chunks)-1):
            print chunks[idx], chunks[idx+1]       
            movie_names.append(fnames[chunks[idx]:chunks[idx+1]])
    
    #%
    template_each=[];
    all_movs_each=[];
    movie_names=[]
    for idx in range(len(chunks)-1):
            print chunks[idx], chunks[idx+1]
            all_mov=all_movs[chunks[idx]:chunks[idx+1]]
            all_mov=cb.movie(all_mov,fr=1)
            all_mov,shifts,_,_=all_mov.motion_correct(template=np.median(all_mov,axis=0))
            template=np.median(all_mov,axis=0)
            all_movs_each.append(all_mov)
            template_each.append(template)
            movie_names.append(fnames[chunks[idx]:chunks[idx+1]])
            if is_interactive:
                pl.imshow(template,cmap=pl.cm.gray,vmax=100)
            
    np.savez(base_name+'-template_total.npz',template_each=template_each, all_movs_each=np.array(all_movs_each),movie_names=movie_names)        
    #%
    if is_interactive:
        for idx,mov in enumerate(all_movs_each):
            mov.play(backend='opencv',gain=50.,fr=100)
    #    mov.save(str(idx)+'sam_example.tif')
    #%
    
    #%
    file_res=[]
    for template,fn in zip(template_each,movie_names):
        print fn
        file_res.append(cb.motion_correct_parallel(fn,30*downsample_factor,dview= dview,template=template,margins_out=0,max_shift_w=35, max_shift_h=35,remove_blanks=False))
    #%
    if is_interactive:
        for f1 in  file_res:
            for f in f1:
                with np.load(f+'npz') as fl:
                    
                    pl.subplot(1,2,1)
                    pl.cla()
                    pl.imshow(fl['template'],cmap=pl.cm.gray)
                    pl.subplot(1,2,2)
                    pl.plot(fl['shifts'])       
                    pl.pause(0.001)
                    pl.cla()
            
    #%
    names_new_each=[]
    for mov_names_each in movie_names:   
        movie_names_hdf5=[]
        for mov_name in mov_names_each:
            movie_names_hdf5.append(mov_name[:-3]+'hdf5')
            #idx_x=slice(12,500,None)
            #idx_y=slice(12,500,None)
            #idx_xy=(idx_x,idx_y)
        idx_xy=None
        name_new=cse.utilities.save_memmap_each(movie_names_hdf5, dview=dview,base_name=None, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy )
        name_new.sort()    
        names_new_each.append(name_new)
        print name_new
    #%
    fnames_new_each=dview.map_sync(cse.utilities.save_memmap_join,names_new_each)
    #%
    if is_interactive:
        m=cb.load(fnames_new_each[-1],fr=5)
        m.play(backend='opencv',gain=50.,fr=30)

    #%
    rf=20# half-size of the patches in pixels. rf=25, patches are 50x50
    stride = 2 #amounpl.it of overlap between the patches in pixels    
    K=8 # number of neurons expected per patch
    gSig=[4,4] # expected half size of neurons
    merge_thresh=0.8 # merging threshold, max correlation allowed
    p=0 #order of the autoregressive system
    memory_fact=1; #unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
    save_results=True
    quality_threshold=-10
    num_samples_high_quality=4
    fudge_factor=0.96
    
    for fname_new  in fnames_new_each:
        print fname_new
        Yr,dims,T=cse.utilities.load_memmap(fname_new)
        d1,d2=dims
        Y=np.reshape(Yr,dims+(T,),order='F')
        Cn = cse.utilities.local_correlations(Y[:,:,:])
        n_processes=len(dview)
        options_patch = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=K,ssub=1,tsub=1,thr=merge_thresh)
        A_tot,C_tot,b,f,sn_tot, optional_outputs = cse.map_reduce.run_CNMF_patches(fname_new, (d1, d2, T), options_patch,rf=rf,stride = stride,
                                                                               dview=dview,memory_fact=memory_fact)
        print 'Number of components:' + str(A_tot.shape[-1])      
        if save_results:
            np.savez(fname_new[:-4]+'results_analysis_patch.npz',A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot,d1=d1,d2=d2,b=b,f=f)   
        
        
        options = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=A_tot.shape[-1],thr=merge_thresh)
        pix_proc=np.minimum(np.int((d1*d2)/n_processes/(T/2000.)),np.int((d1*d2)/n_processes)) # regulates the amount of memory used
        options['spatial_params']['n_pixels_per_process']=pix_proc
        options['temporal_params']['n_pixels_per_process']=pix_proc
        #% merge spatially overlaping and temporally correlated components      
        A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr,A_tot,[],np.array(C_tot),[],np.array(C_tot),[],options['temporal_params'],options['spatial_params'],dview=dview,thr=options['merging']['thr'],mx=np.Inf)     
        #% update temporal to get Y_r
        options['temporal_params']['p']=0
        options['temporal_params']['fudge_factor']=fudge_factor #change ifdenoised traces time constant is wrong
        options['temporal_params']['backend']='ipyparallel'
        C_m,f_m,S_m,bl_m,c1_m,neurons_sn_m,g2_m,YrA_m = cse.temporal.update_temporal_components(Yr,A_m,np.atleast_2d(b).T,C_m,f,dview=dview,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
        
        #%
        traces=C_m+YrA_m
        idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=num_samples_high_quality,robust_std=False)
        idx_components=idx_components[np.logical_and(True ,fitness < quality_threshold)]
        A_m=A_m[:,idx_components]
        C_m=C_m[idx_components,:]   
        
        #%
        print 'Number of components:' + str(A_m.shape[-1])  
        #% UPDATE SPATIAL OCMPONENTS
        t1 = time()
        A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn_tot,dview=dview, **options['spatial_params'])
        print time() - t1
        #% UPDATE TEMPORAL COMPONENTS
        options['temporal_params']['p']=p
        options['temporal_params']['fudge_factor']=fudge_factor #change ifdenoised traces time constant is wrong
        C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,dview=dview, bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
        #% Order components
        #A_or, C_or, srt = cse.utilities.order_components(A2,C2)
        
        #% order components according to a quality threshold and only select the ones wiht qualitylarger than quality_threshold. 
        
        traces=C2+YrA
        idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=num_samples_high_quality,robust_std=False)
        idx_components=idx_components[fitness<quality_threshold]
        print(idx_components.size*1./traces.shape[0])
        #% save analysis results in python and matlab format
        if save_results:
            np.savez(fname_new[:-4]+'results_analysis.npz',Cn=Cn,A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot, A2=A2.todense(),C2=C2,b2=b2,S2=S2,f2=f2,bl2=bl2,c12=c12, neurons_sn2=neurons_sn2, g21=g21,YrA=YrA,d1=d1,d2=d2,idx_components=idx_components, fitness=fitness, erfc=erfc)    
    
    #% stop server and remove log files
    cse.utilities.stop_server(is_slurm = (backend == 'SLURM')) 
    log_files=glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
#    except:
#        errors.append(sess_folder)