## -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:02:12 2016

@author: agiovann, adapted from motion correction algorithm from Selmaan Chettih
"""
#%%
import numpy as np
import pylab as pl
import cv2 
import h5py

#%%
def apply_shift_iteration(img,shift):
     sh_x_n,sh_y_n = shift
     w_i,h_i=img.shape
     M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])    
     return  cv2.warpAffine(img,M,(w_i,h_i),flags=cv2.INTER_CUBIC)
#%%
def apply_shift_online(movie_iterable,xy_shifts,save_base_name=None,order='F'):
    if len(movie_iterable) != len(xy_shifts):
        raise Exception('Number of shifts does not match movie length!')
    count=0
    new_mov=[]
    
    dims=(len(movie_iterable),)+movie_iterable[0].shape
    
    
    if save_base_name is not None:
        fname_tot = save_base_name + '_d1_' + str(dims[1]) + '_d2_' + str(dims[2]) + '_d3_' + str(
                1 if len(dims) == 3 else dims[3]) + '_order_' + str(order) + '_frames_' + str(dims[0]) + '_.mmap'
                
        big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                                shape=(np.prod(dims[1:]), dims[0]), order=order)
                                

    for page,shift in zip(movie_iterable,xy_shifts):  
         if count%100 == 0:
             print(count)
             
         if 'tifffile' in str(type(movie_iterable[0])):
             page=page.asarray()
                 
         img=np.array(page,dtype=np.float32)
         new_img = apply_shift_iteration(img,shift)
         if save_base_name is not None:
             big_mov[:,count] = np.reshape(img,np.prod(dims[1:]),order='F')
         else:
             new_mov.append(new_img)
         
         count+=1
    
    if save_base_name is not None:
        big_mov.flush()
        del big_mov    
        return fname_tot
    else:
        return np.array(new_mov) 
#%%
def motion_correct_online(movie_iterable,max_shift_w=25,max_shift_h=25,hfd5_name=None,init_frames_template=100,show_movie=False,bilateral_blur=False,template=None,min_count=1000):

     shifts=[];   # store the amount of shift in each frame
     xcorrs=[]; 
     
     if 'tifffile' in str(type(movie_iterable[0])):   
             init_mov=[m.asarray() for m in movie_iterable[:init_frames_template]]
     else:
             init_mov=movie_iterable[slice(0,init_frames_template,1)]
             
     if template is None:
         
         template=np.median(init_mov,0)    
         count=init_frames_template
     else:
         count=min_count
         
     #if np.percentile(template, 8) < - 0.1:
     to_remove=np.percentile(init_mov, 8)
     
     template=np.array(template-to_remove,dtype=np.float32)
          
#     buffer_=np.zeros((buffer_median,)+template.shape)            
     for page in movie_iterable:  
                  
         
         if 'tifffile' in str(type(movie_iterable[0])):
             page=page.asarray()
         
        
         img=np.array(page,dtype=np.float32)
         img=img-to_remove
         template_old=template
         new_img,template,shift,avg_corr = motion_correct_iteration(img,template,count,max_shift_w=max_shift_w,max_shift_h=max_shift_h,bilateral_blur=bilateral_blur)
         if count%100 == 0:
             print 'Relative change in template:' + str(np.sum(np.abs(template-template_old))/np.sum(np.abs(template)))
             print 'Iteration:'+ str(count)
             
         new_img = new_img + to_remove

         if show_movie:
            cv2.imshow('frame',new_img/200)
            cv2.waitKey(int(1./200*1000))
                  
         shifts.append(shift)
         xcorrs.append(avg_corr)
         count+=1
         
     template=template+to_remove
     return shifts,xcorrs,template      
#%%
def motion_correct_iteration(img,template,frame_num,max_shift_w=25,max_shift_h=25,bilateral_blur=False,diameter=10,sigmaColor=10000,sigmaSpace=0):

    h_i, w_i = template.shape
    ms_h = max_shift_h
    ms_w = max_shift_w
    
    if bilateral_blur:
        img=cv2.bilateralFilter(img,diameter,sigmaColor,sigmaSpace)    
    
    templ_crop=template[max_shift_h:h_i-max_shift_h,max_shift_w:w_i-max_shift_w].astype(np.float32)
    
    h,w = templ_crop.shape

    res = cv2.matchTemplate(img,templ_crop,cv2.TM_CCORR_NORMED)

    top_left = cv2.minMaxLoc(res)[3]
         
    avg_corr=np.mean(res)

    sh_y,sh_x = top_left

    if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
         # if max is internal, check for subpixel shift using gaussian
         # peak registration
         log_xm1_y = np.log(res[sh_x-1,sh_y]);
         log_xp1_y = np.log(res[sh_x+1,sh_y]);
         log_x_ym1 = np.log(res[sh_x,sh_y-1]);
         log_x_yp1 = np.log(res[sh_x,sh_y+1]);
         four_log_xy = 4*np.log(res[sh_x,sh_y]);

         sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
         sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
    else:
         sh_x_n = -(sh_x - ms_h)
         sh_y_n = -(sh_y - ms_w)

    M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
    
    new_img = cv2.warpAffine(img,M,(w_i,h_i),flags=cv2.INTER_CUBIC)
    
    new_templ=template*frame_num/(frame_num + 1) + 1./(frame_num + 1)*new_img     
    shift=[sh_x_n,sh_y_n]
    
    return new_img,new_templ,shift,avg_corr
#%%    
def bin_median(mat,window=10):

    ''' compute median of 3D array in along axis o by binning values
    Parameters
    ----------

    mat: ndarray
        input 3D matrix, time along first dimension
    
    window: int
        number of frames in a bin
        
        
    Returns
    -------
    img: 
        median image   
        
    '''
    T,d1,d2=np.shape(mat)
    num_windows=np.int(T/window)
    num_frames=num_windows*window
    img=np.median(np.mean(np.reshape(mat[:num_frames],(window,num_windows,d1,d2)),axis=0),axis=0)    
    return img
#%% with buffer
#    import skimage
#import cv2
#
#mean_online=0
#
#count=0
#bin_size=10
#count_part=0
#max_shift_w=25
#max_shift_h=25
#multicolor=False
#show_movie=False
#square_size=(64,64)
#fname='/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/images/k37_20160109_AM_150um_65mW_zoom2p2_00001_00001.tif'
#with skimage.external.tifffile.TiffFile(fname) as tf:
#    if multicolor:
#        n_frames_, h_i, w_i = (len(tf)/bin_size,)+tf[0].shape[:2]
#    else:
#        n_frames_, h_i, w_i = (len(tf)/bin_size,)+tf[0].shape
#    buffer_mean=np.zeros((bin_size,h_i,w_i)).astype(np.float32)    
#    means_partials=np.zeros((np.ceil(len(tf)/bin_size)+1,h_i,w_i)).astype(np.float32)  
#    
#
#    ms_w = max_shift_w
#    ms_h = max_shift_h
#    if multicolor:
#        template=np.median(tf.asarray(slice(0,100,1))[:,:,:,0],0)
#    else:
#        template=np.median(tf.asarray(slice(0,100,1)),0)
#        
#    to_remove=0
#    if np.percentile(template, 8) < - 0.1:
#        print('Pixels averages are too negative for template. Removing 1 percentile.')
#        to_remove=np.percentile(template,1)
#        template=template-to_remove
#        
#    means_partials[count_part]=template
#    
#    template=template[ms_h:h_i-ms_h,ms_w:w_i-ms_w].astype(np.float32)
#    h, w = template.shape      # template width and height
#    
#    
#    #% run algorithm, press q to stop it
#    shifts=[];   # store the amount of shift in each frame
#    xcorrs=[];   
#    for count,page in enumerate(tf):
#        
#        if count%bin_size==0 and count>0:
#            
#            print 'means_partials'
#            count_part+=1
#            means_partials[count_part]=np.mean(buffer_mean,0)
##            buffer_mean=np.zeros((bin_size,)+tf[0].shape).astype()
#            template=np.mean(means_partials[:count_part],0)[ms_h:h_i-ms_h,ms_w:w_i-ms_w]
#        if multicolor:
#            buffer_mean[count%bin_size]=page.asarray()[:,:,0]-to_remove
#        else:
#            buffer_mean[count%bin_size]=page.asarray()-to_remove
#                
#        res = cv2.matchTemplate(buffer_mean[count%bin_size],template,cv2.TM_CCORR_NORMED)
#        top_left = cv2.minMaxLoc(res)[3]
#             
#        avg_corr=np.mean(res);
#        sh_y,sh_x = top_left
#        bottom_right = (top_left[0] + w, top_left[1] + h)
#
#        if (0 < top_left[1] < 2 * ms_h-1) & (0 < top_left[0] < 2 * ms_w-1):
#             # if max is internal, check for subpixel shift using gaussian
#             # peak registration
#             log_xm1_y = np.log(res[sh_x-1,sh_y]);
#             log_xp1_y = np.log(res[sh_x+1,sh_y]);
#             log_x_ym1 = np.log(res[sh_x,sh_y-1]);
#             log_x_yp1 = np.log(res[sh_x,sh_y+1]);
#             four_log_xy = 4*np.log(res[sh_x,sh_y]);
#
#             sh_x_n = -(sh_x - ms_h + (log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
#             sh_y_n = -(sh_y - ms_w + (log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
#        else:
#             sh_x_n = -(sh_x - ms_h)
#             sh_y_n = -(sh_y - ms_w)
#
#        M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])
#        buffer_mean[count%bin_size]= cv2.warpAffine(buffer_mean[count%bin_size],M,(w_i,h_i),flags=cv2.INTER_LINEAR)
#        if show_movie:
#            cv2.imshow('frame',(buffer_mean[count%bin_size])*1./300)
#            cv2.waitKey(int(1./100*1000))
#        shifts.append([sh_x_n,sh_y_n])
#        xcorrs.append([avg_corr])
#        print count
#
##        mean_online=mean_online*count*1./(count + 1) + 1./(count + 1)*buffer_mean[count]
#
#        count+=1

#%% NON RIGID
#import scipy
#chone=cm.load('/Users/agiovann/Documents/MATLAB/Motion_Correction/M_FLUO_1.tif',fr=15)
#chone=chone[:,8:-8,8:-8]
#T=np.median(chone,axis=0)
#Nbasis = 8
#minIters = 5
#
##linear b-splines
#knots = np.linspace(1,np.shape(T)[0],Nbasis+1);
#knots = np.hstack([knots[0]-(knots[1]-knots[0]),knots,knots[-1]+(knots[-1]-knots[-2])]);
#
#weights=knots[:-2]
#order=len(knots)-len(weights)-1
#
#x=range(T.shape[0])
#
#B = np.zeros((len(x),len(weights)))
#for ii in range(len(knots)-order-1):
#    B[:,ii] = bin(this.knots,ii,this.order,x);
#end
#
#spl = fastBSpline(knots,knots(1:end-2))
#
#
#B = spl.getBasis((1:size(T,1))');
#Tnorm = T(:)-mean(T(:));
#Tnorm = Tnorm/sqrt(sum(Tnorm.^2));
#B = full(B);
#
#lambda = .0001*median(T(:))^2;
#theI = (eye(Nbasis+1)*lambda);
#
#Bi = B(:,1:end-1).*B(:,2:end);
#allBs = [B.^2,Bi];
#[xi,yi] = meshgrid(1:size(T,2),1:size(T,1)); 

#%%
#def doLucasKanade_singleFrame(T, I, B, allBs, xi, yi, theI, Tnorm, nBasis=4, minIters=5):
#
#    maxIters = 50
#    deltacorr = 0.0005
#    
#    _ , w = np.shape(T)
#    
#    #Find optimal image warp via Lucas Kanade    
#    c0 = mycorr(I(:), Tnorm);
#    
#    for ii = 1:maxIters
##        %Displaced template
##        Dx = repmat((B*dpx), 1, w);
##        Dy = repmat((B*dpy), 1, w);
#        
#        Id = interp2(I, xi, yi, 'linear', 0);
#                
#        %gradient
#        [dTx, dTy] = imgradientxy(Id, 'centraldifference');
#        dTx(:, [1, ]) = 0;
#        dTy([1, ], :) = 0;
#        
#        if ii > minIters
#            c = mycorr(Id(:), Tnorm);
#            if c - c0 < deltacorr && ii > 1
#                break;
#            
#            c0 = c;
#        
# 
#        del = T - Id;
# 
#        %special trick for g (easy)
#        gx = B'*sum(del.*dTx, 2);
#        gy = B'*sum(del.*dTy, 2);
# 
#        %special trick for H - harder
#        Hx = constructH(allBs'*sum(dTx.^2,2), nBasis+1) + theI;
#        Hy = constructH(allBs'*sum(dTy.^2,2), nBasis+1) + theI;
# 
#        dpx = Hx\gx;
#        dpy = Hy\gy;
#        
#    return [Id, dpx, dpy]
#
##         dpx = dpx + damping*dpx_;
##         dpy = dpy + damping*dpy_;
#    
#
#
#function thec = mycorr(A,B)
#    meanA = mean(A(:));
#    A = A(:) - meanA;
#    A = A / sqrt(sum(A.^2));
#    thec = A'*B;
#
# 
#function H2 = constructH(Hd,ns)
#%     H2d1 = Hd(1:ns)';
#%     H2d2 = [Hd(ns+1:);0]';
#%     H2d3 = [0;Hd(ns+1:)]';
#%     
#%     if isa(Hd, 'gpuArray')
#%         H2 = gpuArray.zeros(ns);
#%     else
#%         H2 = zeros(ns);
#%     
#%             
#%     H2((0:ns-1)*ns+(1:ns)) = H2d1;
#%     H2(((1:ns-1)*ns+(1:ns-1))) = H2d2(1:-1);
#%     H2(((0:ns-2)*ns+(1:ns-1))+1) = H2d3(2:);
#
#    if isa(Hd, 'gpuArray')
#        H2 = gpuArray.zeros(ns);
#    else
#        H2 = zeros(ns);
#    
#            
#    H2((0:ns-1)*ns+(1:ns)) = Hd(1:ns)';
#    H2(((1:ns-1)*ns+(1:ns-1))) = Hd(ns+1:)';
#    H2(((0:ns-2)*ns+(1:ns-1))+1) = Hd(ns+1:)';

#%%
#def motion_correct_parallel(file_names,fr,template=None,margins_out=0,max_shift_w=5, max_shift_h=5,remove_blanks=False,apply_smooth=False,dview=None,save_hdf5=True):
#    """motion correct many movies usingthe ipyparallel cluster
#    Parameters
#    ----------
#    file_names: list of strings
#        names of he files to be motion corrected
#    fr: double
#        fr parameters for calcblitz movie 
#    margins_out: int
#        number of pixels to remove from the borders    
#    
#    Return
#    ------
#    base file names of the motion corrected files
#    """
#    args_in=[];
#    for file_idx,f in enumerate(file_names):
#        if type(template) is list:
#            args_in.append((f,fr,margins_out,template[file_idx],max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5))
#        else:
#            args_in.append((f,fr,margins_out,template,max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5))
#        
#    try:
#        
#        if dview is not None:
##            if backend is 'SLURM':
##                if 'IPPPDIR' in os.environ and 'IPPPROFILE' in os.environ:
##                    pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
##                else:
##                    raise Exception('envirnomment variables not found, please source slurmAlloc.rc')
##        
##                c = Client(ipython_dir=pdir, profile=profile)
##                print 'Using '+ str(len(c)) + ' processes'
##            else:
##                c = Client()
#
#
#            file_res = dview.map_sync(process_movie_parallel, args_in)                         
#            dview.results.clear()       
#
#            
#
#        else:
#            
#            file_res = map(process_movie_parallel, args_in)        
#                 
#        
#        
#    except :   
#        
#        try:
#            if dview is not None:
#                
#                dview.results.clear()       
#
#        except UnboundLocalError as uberr:
#
#            print 'could not close client'
#
#        raise
#                                    
#    return file_res 

#%%
####
### Load and Motion Correct
#
## This first part just creates a cell array of filenames to load, may have
## to be changed if you're names are different than Alice's
#
##get all files names
##%%
#num_files = 2
#nameOffset = 0
#mouse_name = 'AG01'
#session_name = '010101'
#view_name = 'view1'
#slice_name = 'slice1'
#
#
#
#filenames=['M_FLUO_1.tif','M_FLUO_2.tif']
#fr=15.
#n_blocks_x=3
#n_blocks_y=3
#
#
#max_shift_rigid=10
#max_shift_aff=3
#
#fullfilename = filenames[0]
#chone=cm.load(fullfilename,fr=fr)
##chone=chone.crop(5,5,5,5)
#chone,_,_,_=chone.motion_correct(max_shift_rigid,max_shift_rigid,remove_blanks=True)
##info=imfinfo(fullfilename)
#Z,N,M=chone.shape
#
#
##scale movie for seamless intensities
##Clip bad image region
#meanlastframes=np.median(np.mean(chone[-400:],axis=(1,2)))
#meanfirstframes=np.median(np.mean(chone[:400],axis=(1,2)))
#chone=chone*(meanlastframes/meanfirstframes)    
#
##Construct Movie Segments
#
#xind = np.floor(np.linspace(0,M/2.,n_blocks_x))
#yind = np.floor(np.linspace(0,N/2.,n_blocks_y))
#segPos = []
#for x in range(len(xind)):
#    for y in range(len(yind)):
#        segPos.append([xind[x], yind[y],  np.floor(M/2.), np.floor(N/2.)])
#    
#
#nSeg = len(segPos)
#segPos=np.asarray(segPos,dtype=np.int)
##First order motion correction
##clear xshifts yshifts corThresh,
#xshifts=[]
#yshifts=[]
#new_chone=chone.copy().resize(1,1,.5).resize(1,1,2)
#
#for Seg in range(nSeg):
#    print(Seg)
#    tMov = new_chone[:,segPos[Seg,1]:segPos[Seg,1]+segPos[Seg,3],segPos[Seg,0]:segPos[Seg,0]+segPos[Seg,2]].copy()
#    print(tMov.shape)
#    print(segPos[Seg,:])
##        tBase = np.percentile(tMov,1)
##        tTop = np.percentile(tMov,99)
##        tMov = (tMov - tBase) / (tTop-tBase)
##        tMov[tMov<0] = 0 
##        tMov[tMov>1] = 1
#    tMov,xsh,xcorr,templ=tMov.motion_correct(max_shift_w=max_shift_aff, max_shift_h=max_shift_aff)
#    xsh=np.array(xsh)
#    xshifts.append(xsh[:,0])
#    yshifts.append(xsh[:,1])
##        [xshifts(Seg,:),yshifts(Seg,:)]=track_subpixel_wholeframe_motion_varythresh(...
##        tMov,median(tMov,3),5,0.9,100)
#
#xshifts=np.array(xshifts)
#yshifts=np.array(yshifts)
#    
#       
#pl.plot(np.array(xshifts).T,color='r')
#pl.plot(np.array(yshifts).T,color='g')
##%%
#from skimage import data
#from skimage import transform as tf
#
#usePos = range(nSeg)
#
#yoff = segPos[usePos,1] + np.floor(segPos[usePos,3]/2.)
#xoff = segPos[usePos,0] + np.floor(segPos[usePos,2]/2.)
#rpts = np.concatenate([xoff[:,np.newaxis], yoff[:,np.newaxis]],axis=1)
#
#chone=(chone-np.min(chone))/(np.max(chone)-np.min(chone))
#new_mov=[]
#for frame,img in enumerate(chone):
#    if frame250 ==0:
#        print(frame)
#    
#    xframe = xshifts[usePos,frame] + xoff
#    yframe = yshifts[usePos,frame] + yoff
#    fpts=np.concatenate([xframe[:,np.newaxis], yframe[:,np.newaxis]],axis=1)
#    tform = tf.ProjectiveTransform()
#    tform.estimate(rpts, fpts)
#    warped = tf.warp(img, tform, output_shape=img.shape)
#    new_mov.append(warped)
##    tform=fitgeotrans(fpts,rpts,'affine')
##    cated_movie(:,:,frame)=imwarp(cated_movie(:,:,frame),tform,'OutputView',R)   
#new_mov=cm.concatenate([chone,cm.movie(np.array(new_mov),fr=15)],axis=1)
##%%
#
##cated_movie(:,:,1) = cated_movie(:,:,2)
#
#blank=sum(cated_movie==0,3)
#xmin=find(median(blank,1)==0,1,'first'),
#xmax=find(median(blank,1)==0,1,'last'),
#ymin=find(median(blank,2)==0,1,'first'),
#ymax=find(median(blank,2)==0,1,'last'),
#cated_movie=cated_movie(ymin:ymax,xmin:xmax,:)
#save(sprintf('#s_#s_#s_#s',mouse_name,session_name,view_name,slice_name),'cated_movie','chone_mask','-v7.3')
#
### Calculate piecewise covariance, principle components, and feed to ICA algorithm
#
#M=size(cated_movie,1)
#N=size(cated_movie,2)
#Z=size(cated_movie,3)
#nPCs = 1e3
#
#cated_movie=reshape(cated_movie,M*N,size(cated_movie,3))
#pxmean = mean(cated_movie,2)
#cated_movie = cated_movie ./ repmat(pxmean/100,1,Z)
#
#display('------------Computing Principle Components-------------')
#[e,s,l]=pca(cated_movie,'NumComponents',nPCs)
#
#covtrace = double(sum(l))
#CovEvals = double(l(1:nPCs))
#mixedsig= double(e(:,1:nPCs))'
#mixedfilters = double(reshape(s(:,1:nPCs),M,N,nPCs))
#save(sprintf('#s_#s_#s_#s_PCs',mouse_name,session_name,view_name,slice_name),'mixedsig','mixedfilters','covtrace','CovEvals')
#clear e s l
#
#PCuse = 1:300
#mu=.2
#nIC = ceil(length(PCuse)/1)
#ica_A_guess = []
#termtol = 1e-7
#maxrounds = 1e3
#smwidth = 3
#thresh = 2
#arealims = [20 400]
#plotting = 1
#[ica_sig, ica_filters, ica_A, numiter] = CellsortICA(...
#     mixedsig,mixedfilters, CovEvals, PCuse, mu, nIC, ica_A_guess, termtol, maxrounds)
##CellsortChoosePCs(shiftdim(ica_filters,1),M,N)
#[ica_segments, segmentlabel, segcentroid] = CellsortSegmentation...
#    (ica_filters, smwidth, thresh, arealims, plotting)
#for i=1:size(ica_segments,1)
#    normSeg(:,:,i)=100*ica_segments(i,:,:)/norm(reshape(ica_segments(i,:,:),1,[]))
#
#normSeg = reshape(normSeg,M*N,[])
#SegTraces = normSeg' * cated_movie
#normSeg = reshape(normSeg,M,N,[])
#
#for i=1:size(ica_segments,1)
#    segSize(i) = squeeze(sum(sum(ica_segments(i,:,:)>0)))
#    segSkew(i) = skewness(reshape(ica_segments(i,:,:),1,[]))
#
#segSTD = sqrt(std(SegTraces,[],2)./mean(SegTraces,2))
#segSize = zscore(segSize)
#segSkew = zscore(segSkew)
#
#goodSeg = find(segSize-segSkew > 0)
#gTrace = SegTraces(goodSeg,:)
#save(sprintf('#s_#s_#s_#s_ICs',mouse_name,session_name,view_name,slice_name),...
#    'SegTraces','normSeg','ica_sig','ica_filters','ica_A','ica_segments','segmentlabel','segcentroid',...
#    'segSize','segSkew','segSTD','goodSeg','gTrace')
#figure,scatter(segSize,segSkew,50,segSTD,'filled')