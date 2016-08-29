# example python script for loading neurofinder data
#
# for more info see:
#
# - http://neurofinder.codeneuro.org
# - https://github.com/codeneuro/neurofinder
#
# requires three python packages
#
# - numpy
# - scipy
# - matplotlib
#
#%%
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import ca_source_extraction as cse
import calblitz as cb
from scipy.misc import imread
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
import sys
import numpy as np
import ca_source_extraction as cse
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

#%%
folders_in=[
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.00.00.test/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.00.01.test/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.01.00.test/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.01.01.test/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.02.00.test/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.02.01.test/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.03.00.test/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.04.00.test/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.04.01.test/',
#%
#folders_in=[
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.00.07/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.00.00/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.01.00/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.01.01/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.02.00/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.02.01/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.03.00/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.04.00/',
'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.04.01/']

#%%
backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'
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
#%%
for folder_in in folders_in:
#    try:
        #%%'/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.00.00.test/']
        fname_mov=os.path.join(os.path.split(folder_in)[0], os.path.split(folder_in)[-1] + 'MOV.tif')
        print fname_mov
        files=sorted(glob(os.path.join(os.path.split(folder_in)[0],'images/*.tiff')))
        #%% LOAD MOVIE HERE USE YOUR METHOD, Movie is frames x dim2 x dim2
        m=cb.load_movie_chain(files,fr=30)
        m=m-np.min(m)
        m.save(fname_mov)        
            
        #%%
        downsample_factor = .3  
        base_name ='Yr'
        
        name_new=cse.utilities.save_memmap_each([fname_mov], dview=None,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=None )
        print name_new
        #%%
        fname_new=cse.utilities.save_memmap_join(name_new,base_name='Yr', n_chunks=6, dview=dview)
        
        #%%    
        #fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
        Yr,dims,T=cse.utilities.load_memmap(fname_new)
        d1,d2=dims
        Y=np.reshape(Yr,dims+(T,),order='F')
        #%%
        Cn = cse.utilities.local_correlations(Y[:,:,:3000])
        #pl.imshow(Cn,cmap='gray')  
        
        #%%
        rf=15 # half-size of the patches in pixels. rf=25, patches are 50x50
        stride = 2 #amounpl.it of overlap between the patches in pixels    
        K=5 # number of neurons expected per patch
        gSig=[5,5] # expected half size of neurons
        merge_thresh=0.8 # merging threshold, max correlation allowed
        p=2 #order of the autoregressive system
        memory_fact=1; #unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
        save_results=True
        #%% RUN ALGORITHM ON PATCHES
        options_patch = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=K,ssub=1,tsub=4,thr=merge_thresh)
        A_tot,C_tot,b,f,sn_tot, optional_outputs = cse.map_reduce.run_CNMF_patches(fname_new, (d1, d2, T), options_patch,rf=rf,stride = stride,
                                                                                dview=dview,memory_fact=memory_fact)
        print 'Number of components:' + str(A_tot.shape[-1])      
        
        #%%
        if save_results:
            np.savez('results_analysis_patch.npz',A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot,d1=d1,d2=d2,b=b,f=f)    
        #%% if you have many components this might take long!
        #pl.figure()
        #crd = cse.utilities.plot_contours(A_tot,Cn,thr=0.9)
        #%% set parameters for full field of view analysis
        options = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=A_tot.shape[-1],thr=merge_thresh)
        pix_proc=np.minimum(np.int((d1*d2)/n_processes/(T/2000.)),np.int((d1*d2)/n_processes)) # regulates the amount of memory used
        options['spatial_params']['n_pixels_per_process']=pix_proc
        options['temporal_params']['n_pixels_per_process']=pix_proc
        #%% merge spatially overlaping and temporally correlated components      
        A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr,A_tot,[],np.array(C_tot),[],np.array(C_tot),[],options['temporal_params'],options['spatial_params'],dview=dview,thr=options['merging']['thr'],mx=np.Inf)     
        #%% update temporal to get Y_r
        options['temporal_params']['p']=0
        options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
        options['temporal_params']['backend']='ipyparallel'
        C_m,f_m,S_m,bl_m,c1_m,neurons_sn_m,g2_m,YrA_m = cse.temporal.update_temporal_components(Yr,A_m,np.atleast_2d(b).T,C_m,f,dview=dview,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
        
        #%% get rid of evenrually noisy components. 
        # But check by visual inspection to have a feeling fot the threshold. Try to be loose, you will be able to get rid of more of them later!
        
        traces=C_m+YrA_m
        idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
        idx_components=idx_components[np.logical_and(True ,fitness < -5)]
        print(len(idx_components))
#        cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A_m.tocsc()[:,idx_components]),C_m[idx_components,:],b,f_m, d1,d2, YrA=YrA_m[idx_components,:]
#                        ,img=Cn)  
        #%%
        A_m=A_m[:,idx_components]
        C_m=C_m[idx_components,:]   
        
        #%% display components  DO NOT RUN IF YOU HAVE TOO MANY COMPONENTS
        #pl.figure()
        #crd = cse.utilities.plot_contours(A_m,Cn,thr=0.9)
        #%%
        print 'Number of components:' + str(A_m.shape[-1])  
        #%% UPDATE SPATIAL OCMPONENTS
        t1 = time()
        A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn_tot,dview=dview, **options['spatial_params'])
        print time() - t1
        #%% UPDATE TEMPORAL COMPONENTS
        options['temporal_params']['p']=p
        options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
        C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,dview=dview, bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
        #%% Order components
        #A_or, C_or, srt = cse.utilities.order_components(A2,C2)
        #%% stop server and remove log files
#        cse.utilities.stop_server(is_slurm = (backend == 'SLURM')) 
        log_files=glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)
        #%% order components according to a quality threshold and only select the ones wiht qualitylarger than quality_threshold. 
        quality_threshold=-0
        traces=C2+YrA
        idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
        idx_components=idx_components[fitness<quality_threshold]
        #_,_,idx_components=cse.utilities.order_components(A2,C2)
        print(idx_components.size*1./traces.shape[0])
        #%%
        #pl.figure();
        #crd = cse.utilities.plot_contours(A2.tocsc()[:,idx_components],Cn,thr=0.9)
        #%%
#        cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,idx_components]),C2[idx_components,:],b2,f2, d1,d2, YrA=YrA[idx_components,:])  
        #%% save analysis results in python and matlab format
        if save_results:
            np.savez(os.path.join(os.path.split(fname_new)[0],'results_analysis.npz'),Cn=Cn,A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot, A2=A2.todense(),C2=C2,b2=b2,S2=S2,f2=f2,bl2=bl2,c12=c12, neurons_sn2=neurons_sn2, g21=g21,YrA=YrA,d1=d1,d2=d2,idx_components=idx_components, fitness=fitness, erfc=erfc)    
        #    scipy.io.savemat('output_analysis_matlab.mat',{'A2':A2,'C2':C2 , 'YrA':YrA, 'S2': S2 ,'YrA': YrA, 'd1':d1,'d2':d2,'idx_components':idx_components, 'fitness':fitness })
        #%%
        ##%%
        #if save_results:
        #    %load_ext autoreload
        #    %autoreload 2
        #    import sys
        #    import numpy as np
        #    import ca_source_extraction as cse
        #    from scipy.sparse import coo_matrix
        #    import scipy
        #    import pylab as pl
        #    import calblitz as cb
        #    
        #    
        #    
        #    with np.load('results_analysis.npz')  as ld:
        #          locals().update(ld)
        #    
        #    fname_new=glob('Yr0*_.mmap')[0]
        #    
        #    Yr,(d1,d2),T=cse.utilities.load_memmap(fname_new)
        #    d,T=np.shape(Yr)
        #    Y=np.reshape(Yr,(d1,d2,T),order='F') # 3D version of the movie
        #    A2=scipy.sparse.coo_matrix(A2)
        #
        #    
        #    traces=C2+YrA
        #    idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
        #    #cse.utilities.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2,d1,d2,YrA = YrA[srt,:], secs=1)
        #    cse.utilities.view_patches_bar(Yr,A2.tocsc()[:,idx_components],C2[idx_components,:],b2,f2, d1,d2, YrA=YrA[idx_components,:])  
        #    dims=(d1,d2)
        #
        #
        #%% extract binary masks
        min_radius=gSig[0]
        masks_ws,pos_examples,neg_examples=cse.utilities.extract_binary_masks_blob(
        A2.tocsc()[:,:], min_radius, dims, num_std_threshold=1, 
        minCircularity= 0.5, minInertiaRatio = 0.2,minConvexity = .8)
        np.savez(os.path.join(os.path.split(fname_new)[0],'regions_CNMF.npz'),masks_ws=masks_ws,pos_examples=pos_examples,neg_examples=neg_examples)
#        #%%
#        pl.subplot(1,2,1)
        final_masks=np.array(masks_ws)[pos_examples]
#        pl.imshow(np.reshape(final_masks.max(0),dims,order='F'),vmax=1)
#        pl.imshow(Cn,cmap='gray',alpha=.8)
#
#        pl.title('Positive examples')
#        pl.subplot(1,2,2)
#        neg_examples_masks=np.array(masks_ws)[neg_examples]
#        pl.imshow(np.reshape(neg_examples_masks.max(0),dims,order='F'),vmax=1)
#        pl.imshow(Cn,cmap='gray',alpha=.8)
#
#        pl.title('Negative examples')
#        ##%%
#        #crd = cse.utilities.plot_contours(A2.tocsc()[:,pos_examples],Cn,thr=0.9)
#        ##%%
#        #masks_ben=cse.utilities.nf_read_roi_zip('neurofinder01.01_combined.zip',dims)
#        #masks_nf=cse.utilities.nf_load_masks('regions/regions.json',dims)
#        #
#        #%
#        # load the images
#        # show the outputs
#        plt.figure()
#        plt.subplot(2, 2, 1)
#        plt.imshow(final_masks.sum(axis=0), cmap='hot')
#        pl.imshow(Cn,cmap='gray',alpha=.8)
#        pl.title('CNMF')
#        plt.subplot(2, 2, 2)
#        plt.imshow(masks_nf.sum(axis=0), cmap='hot')
#        pl.imshow(Cn,cmap='gray',alpha=.8)
#        pl.title('NEUROFINDER')
#        plt.subplot(2, 2, 3)
#        plt.imshow(masks_ben.sum(0), cmap='hot')
#        pl.imshow(Cn,cmap='gray',alpha=.8)
#        pl.title('BEN')
#        plt.subplot(2, 2, 4)
#        pl.imshow(np.reshape(A2.tocsc()[:,:].sum(1),dims,order='F'),cmap='hot')
#        plt.imshow(Cn, alpha=.5,cmap='gray')
#        pl.title('A MATRIX')
        #plt.show()
        #%%
        regions_CNMF=cse.utilities.nf_masks_to_json( final_masks,os.path.join(os.path.split(fname_new)[0],'regions_CNMF.json'))
        
#    except:
#        np.save(os.path.join(os.path.split(folder_in)[0],'failure'),np.array(1))
#regions_BEN=cse.utilities.nf_masks_to_json( masks_ben,'regions_ben.json')
#%%
from neurofinder import load, centers, shapes
for folder_in_check in folders_in[-9:]:
    
    print folder_in_check
    a=load(os.path.join(folder_in_check,'regions_CNMF.json'))
    b=load(os.path.join(folder_in_check,'regions/regions.json'))
    #print match(a,b,threshold=5)
    re,pr=centers(a,b,threshold=5)
    incl,excl=shapes(a,b,threshold=5)
    fscore=2*(pr*re)/(pr+re)
    print 'Exclusion %.3f\nRecall %.3f\nCombined %.3f\nPrecision %.3f\nInclusion %.3f\n' % (excl,re,fscore,pr,incl)
#%%
from neurofinder import load, centers, shapes
results=[]
for folder_in_check in folders_in[:-9]:
    
    a=load(os.path.join(folder_in_check,'regions_CNMF.json'))  
    dset='.'.join(folder_in_check[:-1].split('.')[1:])
    print (dset)
    with np.load(os.path.join(folder_in_check,'regions_CNMF.npz')) as ld:
        masks_ws=ld['masks_ws']
        pos_examples=ld['pos_examples']
        neg_examples=ld['neg_examples']    
        regions_CNMF=cse.utilities.nf_masks_to_json( masks_ws[pos_examples],os.path.join(folder_in_check,'tmp.json'))
    dd=dict()
    dd['dataset']= dset   
    dd['regions']= regions_CNMF
    results.append(dd) 
#%%
import json
with open('results.json', 'w') as f:
  f.write(json.dumps(results))
#%% Inspect results
import cv2    
import numpy as np

for folder_in_check in folders_in:

    print folder_in_check

    with np.load(os.path.join(folder_in_check,'regions_CNMF.npz')) as ld:
        masks_ws=ld['masks_ws']
        pos_examples=ld['pos_examples']
        neg_examples=ld['neg_examples']    
    
    
    sizes=np.sum(masks_ws,axis=(1,2))
    pl.imshow(np.sum(masks_ws[sizes>(np.pi*(5**2))],0))
    M=[]
    for thresh in masks_ws:
        im2,contours,hierarchy = cv2.findContours(thresh.astype(np.uint8), 1, 2)
        cnt = contours[0]
        cnt = contours[0]
        mm=cv2.moments(cnt)
        M.append(np.atleast_2d(np.array(mm.values())))
     M1=np.concatenate(M,0)
     
#     for features in range(24):
#         pl.hist(M1[pos_examples,features],100,normed=True)   
#         pl.hist(M1[neg_examples,features],100,normed=True)  
#         pl.pause(1)
#         pl.cla()
    from sklearn import svm
    clf = svm.SVC()
    X=M1[np.hstack([pos_examples[:70],neg_examples[:70]])]
    
    y=np.hstack([np.zeros_like(pos_examples[:70]),np.ones_like(neg_examples[:70])])
    clf.fit(X, y)
    lbs=clf.predict(M1)
    pl.imshow(np.max(masks_ws[lbs==1],0))
    pl.imshow(np.max(masks_ws[pos_examples],0)*10,alpha=.5)

#    with np.load(os.path.join(folder_in_check,'results_analysis.npz'))  as ld:
#        Cn=ld['Cn']
#        dims=(ld['d1'],ld['d2'])
    
#    pl.subplot(1,2,1)
#    final_masks=np.array(masks_ws)[pos_examples]
#    pl.imshow(np.reshape(final_masks.max(0),dims,order='F'),vmax=1)
#    pl.imshow(Cn,cmap='gray',alpha=.8)
#    
#    pl.title('Positive examples')
#    pl.subplot(1,2,2)
#    neg_examples_masks=np.array(masks_ws)[neg_examples]
#    pl.imshow(np.reshape(neg_examples_masks.max(0),dims,order='F'),vmax=1)
#    pl.imshow(Cn,cmap='gray',alpha=.8)
#    
#    pl.title('Negative examples') 
#    pl.savefig(os.path.split(folder_in_check)[0] + '_result.pdf')
#    pl.pause(.1)
#    pl.close()
#    print os.path.join(folder_in_check,'regions_CNMF.json')
    regions_CNMF=cse.utilities.nf_masks_to_json( final_masks,os.path.join(folder_in_check,'regions_CNMF.json'))
    a=load(os.path.join(folder_in_check,'regions_CNMF.json'))
    b=load(os.path.join(folder_in_check,'regions/regions.json'))
    #print match(a,b,threshold=5)
    re,pr=centers(a,b,threshold=5)
    incl,excl=shapes(a,b,threshold=5)
    fscore=2*(pr*re)/(pr+re)
    print 'Exclusion %.3f\nRecall %.3f\nCombined %.3f\nPrecision %.3f\nInclusion %.3f\n' % (excl,re,fscore,pr,incl)
#%%   
for folder_in_check in folders_in[-2:-1]:

    print folder_in_check

    with np.load(os.path.join(folder_in_check,'results_analysis.npz')) as ld:  
        locals().update(ld)
   
    min_radius=gSig[0]
    masks_ws,pos_examples,neg_examples=cse.utilities.extract_binary_masks_blob(
    scipy.sparse.coo_matrix(A2), min_radius, (d1,d2), num_std_threshold=1, 
    minCircularity= 0.5, minInertiaRatio = 0.2,minConvexity = .8)
    np.savez(os.path.join(folder_in_check,'regions_CNMF.npz'),masks_ws=masks_ws,pos_examples=pos_examples,neg_examples=neg_examples)     
#from neurofinder import match,load,centers,shapes 
#a=load('regions/regions.json')
#b=load('regions_ben.json')
##print match(a,b,threshold=5)
#re,pr=centers(a,b,threshold=5)
#incl,excl=shapes(a,b,threshold=5)
#fscore=2*(pr*re)/(pr+re)
#print 'Exclusion %.3f\nRecall %.3f\nCombined %.3f\nPrecision %.3f\nInclusion %.3f\n' % (excl,re,fscore,pr,incl)
#    #%%
#    from neurofinder import match,load,centers,shapes
#    a=load(os.path.join(os.path.split(fname_new)[0],'regions/regions.json'))
#    b=load(os.path.join(os.path.split(fname_new)[0],'regions_CNMF.json'))
#    #print match(a,b,threshold=5)
#    re,pr=centers(a,b,threshold=5)
#    incl,excl=shapes(a,b,threshold=5)
#    fscore=2*(pr*re)/(pr+re)
#    print 'Exclusion %.3f\nRecall %.3f\nCombined %.3f\nPrecision %.3f\nInclusion %.3f\n' % (excl,re,fscore,pr,incl)
##%%
#a=load('regions_CNMF.json')
#b=load('regions_ben.json')
##print match(a,b,threshold=5)
#re,pr=centers(a,b,threshold=5)
#incl,excl=shapes(a,b,threshold=5)
#fscore=2*(pr*re)/(pr+re)
#print 'Exclusion %.3f\nRecall %.3f\nCombined %.3f\nPrecision %.3f\nInclusion %.3f\n' % (excl,re,fscore,pr,incl)
#

#%% ANDREA's JUNK 

#quality_threshold=-25
#traces=C2+YrA
#idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
#
#idx_components=np.where(fitness<quality_threshold)[0]
##_,_,idx_components=cse.utilities.order_components(A2,C2)
#print(idx_components.size*1./traces.shape[0])
##%%
#pl.subplot(1,2,1)
#final_masks=np.array(masks_ws)[np.union1d(pos_examples,idx_components)]
#pl.imshow(np.reshape(final_masks.mean(0),dims,order='F'),vmax=.001)
#pl.subplot(1,2,2)
#neg_examples_masks=np.array(masks_ws)[np.setdiff1d(neg_examples,idx_components)]
#pl.imshow(np.reshape(neg_examples_masks.mean(0),dims,order='F'),vmax=.001)
##%%
#params = cv2.SimpleBlobDetector_Params()
#params.blobColor=255
#params.minThreshold = max_fraction*255;
#params.maxThreshold = 255;
#params.thresholdStep= 10
#params.minArea = np.pi*((gSig[0]-1)**2)
#params.minCircularity= 0.2
#params.minInertiaRatio = 0.2
#params.filterByArea = False
#params.filterByCircularity = True
#params.filterByConvexity = True
#params.minConvexity = .2
#params.filterByInertia = True
#detector = cv2.SimpleBlobDetector_create(params)
#for m in neg_examples_masks:
#    m1=m.astype(np.uint8)*200    
#    m1=ndi.binary_fill_holes(m1)
#    keypoints = detector.detect(m1.astype(np.uint8)*200)
#    if len(keypoints)>0:
#        pl.cla()
#        pl.imshow(np.reshape(m,dims,order='F'),vmax=.001)
#        pl.pause(1)
#    else:
#        print 'skipped'
#
##pl.colorbar()
##%%
#
##%%
#masks_ben=ut.nf_read_roi_zip('neurofinder01.01_combined.zip',dims)
#regions2=cse.utilities.nf_masks_to_json( masks_ben,'masks_ben.json')
#pl.imshow(np.sum(masks_ben>0,0)>0)
#pl.pause(3)
#pl.imshow(5*np.sum(masks,0),alpha=.3)
#
##%%
## load the images
#masks=cse.utilities.nf_load_masks('regions/regions.json',np.shape(m)[1:])
## show the outputs
#plt.figure()
#plt.subplot(1, 2, 1)
#plt.imshow(m.sum(axis=0), cmap='gray')
#plt.subplot(1, 2, 2)
#plt.imshow(masks.sum(axis=0), cmap='gray')
#plt.show()
##%%
#fname='regions_2.json'
#regions2=cse.utilities.nf_masks_to_json(np.roll( masks,20,axis=1) ,fname)
##regions2=cse.utilities.nf_masks_to_json(masks,fname)
##%%  
