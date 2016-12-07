## -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:02:12 2016

@author: agiovann, adapted from motion correction algorithm from Selmaan Chettih
"""
#%%
import numpy as np
import pylab as pl
import cv2 

import collections
import caiman as cm
import tifffile
import gc
import os
import time
#from numpy.fft import fftn,ifftn
#from accelerate.mkl.fftpack import fftn,ifftn
#from pyfftw.interfaces.numpy_fft import fftn, ifftn
#opencv = False

from cv2 import dft as fftn
from cv2 import idft as ifftn
opencv = True
from numpy.fft import ifftshift
from skimage.external.tifffile import TiffFile


#%%
def apply_shift_iteration(img,shift,border_nan=False):

     
     sh_x_n,sh_y_n = shift
     w_i,h_i=img.shape
     M = np.float32([[1,0,sh_y_n],[0,1,sh_x_n]])    
     img = cv2.warpAffine(img,M,(h_i,w_i),flags=cv2.INTER_CUBIC)
     if border_nan:  
         max_w,max_h,min_w,min_h=0,0,0,0
         max_h,max_w = np.ceil(np.maximum((max_h,max_w),shift)).astype(np.int)
         min_h,min_w = np.floor(np.minimum((min_h,min_w),shift)).astype(np.int)
         img[:max_h,:] = np.nan
         if min_h < 0:
             img[min_h:,:] = np.nan
         img[:,:max_w] = np.nan 
         if min_w < 0:
             img[:,min_w:] = np.nan
             
     return  img
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
             big_mov[:,count] = np.reshape(new_img,np.prod(dims[1:]),order='F')
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
  
def motion_correct_online_multifile(list_files,add_to_movie,order = 'C', **kwargs):
    
    
    kwargs['order'] = order
    all_names  = []   
    all_shifts = [] 
    all_xcorrs = []
    all_templates = []
    template = None
    kwargs_=kwargs.copy()
    kwargs_['order'] = order
    total_frames = 0
    for file_ in list_files:
        print 'Processing:' + file_
        kwargs_['template'] = template
        kwargs_['save_base_name'] = file_[:-4]
        tffl = tifffile.TiffFile(file_)
        shifts,xcorrs,template, fname_tot  =  motion_correct_online(tffl,add_to_movie,**kwargs_)
        all_names.append(fname_tot)
        all_shifts.append(shifts)
        all_xcorrs.append(xcorrs)
        all_templates.append(template)
        total_frames = total_frames + len(shifts)
    


    

    
    return all_names, all_shifts, all_xcorrs, all_templates
        
        
    
#%%
def motion_correct_online(movie_iterable,add_to_movie,max_shift_w=25,max_shift_h=25,save_base_name=None,order = 'C',\
                        init_frames_template=100, show_movie=False, bilateral_blur=False,template=None, min_count=1000,\
                        border_to_0=0, n_iter = 1, remove_blanks=False,show_template=False,return_mov=False, use_median_as_template = False):

     shifts=[];   # store the amount of shift in each frame
     xcorrs=[]; 
     
     if remove_blanks and n_iter==1:
         raise Exception('In order to remove blanks you need at least two iterations n_iter=2')
            
     
     if 'tifffile' in str(type(movie_iterable[0])):   
             if len(movie_iterable)==1:
                 print('******** WARNING ****** NEED TO LOAD IN MEMORY SINCE SHAPE OF PAGE IS THE FULL MOVIE')
                 movie_iterable = movie_iterable.asarray()
                 init_mov=movie_iterable[:init_frames_template]
                 
             else:
                 
                 init_mov=[m.asarray() for m in movie_iterable[:init_frames_template]]
     else:
             init_mov=movie_iterable[slice(0,init_frames_template,1)]
     
     
             
         
     dims=(len(movie_iterable),)+movie_iterable[0].shape 
     print "dimensions:" + str(dims)      
     
     
             
     
     if use_median_as_template:
        template = bin_median(movie_iterable)
          
     
     if template is None:        
         template = bin_median(init_mov)
         count=init_frames_template
         if np.percentile(template, 1) + add_to_movie < - 10:
             raise Exception('Movie too negative, You need to add a larger value to the movie (add_to_movie)')
         template=np.array(template + add_to_movie,dtype=np.float32)    
     else:
         if np.percentile(template, 1) < - 10:
             raise Exception('Movie too negative, You need to add a larger value to the movie (add_to_movie)')
         count=min_count
         
            
     
     min_mov = 0
     buffer_size_frames=100          
     buffer_size_template=100     
     buffer_frames=collections.deque(maxlen=buffer_size_frames)  
     buffer_templates=collections.deque(maxlen=buffer_size_template)  
     max_w,max_h,min_w,min_h=0,0,0,0
     
     big_mov = None
     if return_mov:
        mov=[]   
     else:
        mov = None
     
     
     for n in range(n_iter):  
         
         if n>0:
             count = init_frames_template
             
#         if (n>0) and  (n_iter == (n+1)):
#             buffer_templates = collections.deque(list(buffer_templates)[-10:])
#             
         if (save_base_name is not None) and (big_mov is None) and (n_iter == (n+1)):  

             if remove_blanks:
                 dims = (dims[0],dims[1]+min_h-max_h,dims[2]+min_w-max_w)
                 

             fname_tot = save_base_name + '_d1_' + str(dims[1]) + '_d2_' + str(dims[2]) + '_d3_' + str(1 if len(dims) == 3 else dims[3]) + '_order_' + str(order) + '_frames_' + str(dims[0]) + '_.mmap'
             big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32, shape=(np.prod(dims[1:]), dims[0]), order=order)       

         else:
             
             fname_tot = None  
         
         shifts_tmp = []
         xcorr_tmp = []
         for idx_frame,page in enumerate(movie_iterable):  
                               
             if 'tifffile' in str(type(movie_iterable[0])):
                 page=page.asarray()
                     
             img=np.array(page,dtype=np.float32)
             img=img+add_to_movie
             
             new_img,template_tmp,shift,avg_corr = motion_correct_iteration(img,template,count,max_shift_w=max_shift_w,max_shift_h=max_shift_h,bilateral_blur=bilateral_blur)
             
             max_h,max_w = np.ceil(np.maximum((max_h,max_w),shift)).astype(np.int)
             min_h,min_w = np.floor(np.minimum((min_h,min_w),shift)).astype(np.int)
             
             if count < (buffer_size_frames + init_frames_template):
                 template_old=template
                 template = template_tmp
             else:
                 template_old=template
             buffer_frames.append(new_img)
             
             if count%100 == 0:
                                  
                 if count >= (buffer_size_frames + init_frames_template):
                     buffer_templates.append(np.mean(buffer_frames,0))                     
                     template = np.median(buffer_templates,0)
                 
                 if show_template:
                     pl.cla()
                     pl.imshow(template,cmap='gray',vmin=250,vmax=350,interpolation='none')
                     pl.pause(.001)
                 
                 print 'Relative change in template:' + str(np.sum(np.abs(template-template_old))/np.sum(np.abs(template)))
                 print 'Iteration:'+ str(count)
                                                
             if border_to_0 > 0:
                new_img[:border_to_0,:]=min_mov
                new_img[:,:border_to_0]=min_mov
                new_img[:,-border_to_0:]=min_mov
                new_img[-border_to_0:,:]=min_mov
             
             shifts_tmp.append(shift)
             xcorr_tmp.append(avg_corr)
             
             if remove_blanks and n>0  and (n_iter == (n+1)):
                      
                      new_img = new_img[max_h:,:]
                      if min_h < 0:
                          new_img = new_img[:min_h,:]
                      new_img = new_img[:,max_w:] 
                      if min_w < 0:
                          new_img = new_img[:,:min_w]
                          
             if (save_base_name is not None) and (n_iter == (n+1)):
                          
                  
                 
                  big_mov[:,idx_frame] = np.reshape(new_img,np.prod(dims[1:]),order='F')
             
             if return_mov and (n_iter == (n+1)):
                 mov.append(new_img)
             
             if show_movie:
                cv2.imshow('frame',new_img/500)
                print shift
                if np.any(np.remainder(shift,1) == (0,0)):
                    1
#                    cv2.waitKey(int(1000))

                else:
                
                    cv2.waitKey(int(1./500*1000))
                      

             count+=1
         shifts.append(shifts_tmp)
         xcorrs.append(xcorr_tmp)
         
     if save_base_name is not None:
        print 'Flushing memory'
        big_mov.flush()
        del big_mov     
        gc.collect()
        
     

     return shifts,xcorrs,template, fname_tot, np.dstack(mov).transpose([2,0,1])     
#%%
def motion_correct_iteration(img,template,frame_num,max_shift_w=25,max_shift_h=25,bilateral_blur=False,diameter=10,sigmaColor=10000,sigmaSpace=0):

    h_i, w_i = template.shape
    ms_h = max_shift_h
    ms_w = max_shift_w
    
    if bilateral_blur:
        img=cv2.bilateralFilter(img,diameter,sigmaColor,sigmaSpace)    
#    if 1:
    templ_crop=template[max_shift_h:h_i-max_shift_h,max_shift_w:w_i-max_shift_w].astype(np.float32)
    
    h,w = templ_crop.shape

    res = cv2.matchTemplate(img,templ_crop,cv2.TM_CCORR_NORMED)
#    else:
#
#        templ_crop=template.astype(np.float32)
#        
#        h,w = templ_crop.shape
#        
#        img_pad =  np.pad(img,(max_shift_h,max_shift_w),mode='symmetric')
#        res = cv2.matchTemplate(img_pad,templ_crop,cv2.TM_CCORR_NORMED)

    top_left = cv2.minMaxLoc(res)[3]
         
#    avg_corr=np.max(cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED))
    avg_corr=np.max(res)

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
def process_movie_parallel(arg_in):


    fname,fr,margins_out,template,max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5=arg_in
    
    if template is not None:
        if type(template) is str:
            if os.path.exists(template):
                template=cm.load(template,fr=1)
            else:
                raise Exception('Path to template does not exist:'+template)                
#    with open(fname[:-4]+'.stout', "a") as log:
#        print fname
#        sys.stdout = log
        
    #    import pdb
    #    pdb.set_trace()
    type_input = str(type(fname)) 
    if 'movie' in type_input:        
        print type(fname)
        Yr=fname

    elif ('ndarray' in type_input):        
        Yr=cm.movie(np.array(fname,dtype=np.float32),fr=fr)
    elif 'str' in type_input: 
        Yr=cm.load(fname,fr=fr)
    else:
        raise Exception('Unkown input type:' + type_input)
   
    if Yr.ndim>1:

        print 'loaded'    

        if apply_smooth:

            print 'applying smoothing'

            Yr=Yr.bilateral_blur_2D(diameter=10,sigmaColor=10000,sigmaSpace=0)

#        bl_yr=np.float32(np.percentile(Yr,8))    

 #       Yr=Yr-bl_yr     # needed to remove baseline

        print 'Remove BL'

        if margins_out!=0:

            Yr=Yr[:,margins_out:-margins_out,margins_out:-margins_out] # borders create troubles

        print 'motion correcting'

        Yr,shifts,xcorrs,template=Yr.motion_correct(max_shift_w=max_shift_w, max_shift_h=max_shift_h,  method='opencv',template=template,remove_blanks=remove_blanks) 

  #      Yr = Yr + bl_yr           
        
        if ('movie' in type_input) or ('ndarray' in type_input):
            print 'Returning Values'
            return Yr, shifts, xcorrs, template

        else:     
            
            print 'median computing'        

            template=Yr.bin_median()

            print 'saving'  

            idx_dot=len(fname.split('.')[-1])

            if save_hdf5:

                Yr.save(fname[:-idx_dot]+'hdf5')        

            print 'saving 2'                 

            np.savez(fname[:-idx_dot]+'npz',shifts=shifts,xcorrs=xcorrs,template=template)

            print 'deleting'        

            del Yr

            print 'done!'
        
            return fname[:-idx_dot] 
        #sys.stdout = sys.__stdout__ 
    else:
        return None
           

#%%
def motion_correct_parallel(file_names,fr=10,template=None,margins_out=0,max_shift_w=5, max_shift_h=5,remove_blanks=False,apply_smooth=False,dview=None,save_hdf5=True):
    """motion correct many movies usingthe ipyparallel cluster
    Parameters
    ----------
    file_names: list of strings
        names of he files to be motion corrected
    fr: double
        fr parameters for calcblitz movie 
    margins_out: int
        number of pixels to remove from the borders    
    
    Return
    ------
    base file names of the motion corrected files
    """
    args_in=[];
    for file_idx,f in enumerate(file_names):
        if type(template) is list:
            args_in.append((f,fr,margins_out,template[file_idx],max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5))
        else:
            args_in.append((f,fr,margins_out,template,max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5))
        
    try:
        
        if dview is not None:
#            if backend is 'SLURM':
#                if 'IPPPDIR' in os.environ and 'IPPPROFILE' in os.environ:
#                    pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
#                else:
#                    raise Exception('envirnomment variables not found, please source slurmAlloc.rc')
#        
#                c = Client(ipython_dir=pdir, profile=profile)
#                print 'Using '+ str(len(c)) + ' processes'
#            else:
#                c = Client()


            file_res = dview.map_sync(process_movie_parallel, args_in)                         
            dview.results.clear()       

            

        else:
            
            file_res = map(process_movie_parallel, args_in)        
                 
        
        
    except :   
        
        try:
            if dview is not None:
                
                dview.results.clear()       

        except UnboundLocalError as uberr:

            print 'could not close client'

        raise
                                    
    return file_res 

#%%

def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : 2D ndarray
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : 2D ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(data.shape[1] / 2)).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(data.shape[0] / 2))
    )
    

    return row_kernel.dot(data).dot(col_kernel)


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be
        zero if images are non-negative).

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image
    target_amp : float
        The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() /\
        (src_amp * target_amp)
    return np.sqrt(np.abs(error))

#%%
def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb = None, shifts_ub = None, max_shifts = (10,10)):
    """
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters
    ----------
    src_image : ndarray
        Reference image.
    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier"
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        if opencv:
            src_freq_1 = fftn(src_image,flags=cv2.DFT_COMPLEX_OUTPUT+cv2.DFT_SCALE)
            src_freq  = src_freq_1[:,:,0]+1j*src_freq_1[:,:,1]
            src_freq   = np.array(src_freq, dtype=np.complex128, copy=False)            
            target_freq_1 = fftn(target_image,flags=cv2.DFT_COMPLEX_OUTPUT+cv2.DFT_SCALE)
            target_freq  = target_freq_1[:,:,0]+1j*target_freq_1[:,:,1]
            target_freq = np.array(target_freq , dtype=np.complex128, copy=False)
        else:
            src_image_cpx = np.array(src_image, dtype=np.complex128, copy=False)
            target_image_cpx = np.array(target_image, dtype=np.complex128, copy=False)
            src_freq = np.fft.fftn(src_image_cpx)
            target_freq = fftn(target_image_cpx)

    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if opencv:
        
        image_product_cv = np.dstack([np.real(image_product),np.imag(image_product)])
        cross_correlation = fftn(image_product_cv,flags=cv2.DFT_INVERSE+cv2.DFT_SCALE)
        cross_correlation = cross_correlation[:,:,0]+1j*cross_correlation[:,:,1]
    else:
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        cross_correlation = ifftn(image_product)
    
    
    # Locate maximum
    
    new_cross_corr  = np.abs(cross_correlation)


    if (shifts_lb is not None) or (shifts_ub is not None):
        
        if  (shifts_lb[0]<0) and (shifts_ub[0]>=0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0],:] = 0                                                                  
        else:
            new_cross_corr[:shifts_lb[0],:] = 0                
            new_cross_corr[shifts_ub[0]:,:] = 0    
            
        if  (shifts_lb[1]<0) and (shifts_ub[1]>=0):      
            new_cross_corr[:,shifts_ub[1]:shifts_lb[1]] = 0                                                      
        else:
            new_cross_corr[:,:shifts_lb[1]] = 0    
            new_cross_corr[:,shifts_ub[1]:] = 0    
            
            
    else:
    
        new_cross_corr[max_shifts[0]:-max_shifts[0],:] = 0   
        
        new_cross_corr[:,max_shifts[1]:-max_shifts[1]] = 0

#    pl.cla()
#    ranges = np.percentile(np.abs(cross_correlation),[1,99.99])
#    pl.subplot(1,2,1)
#    pl.imshow( np.abs(cross_correlation),interpolation = 'none',vmin = ranges[0],vmax = ranges[1])
#    pl.pause(.1)
#    pl.subplot(1,2,2)
#    pl.imshow(new_cross_corr,interpolation = 'none',vmin = ranges[0],vmax = ranges[1])
#    pl.pause(1)
    
    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        
        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
                              np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape),
                          dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts#, _compute_error(CCmax, src_amp, target_amp),\
        #_compute_phasediff(CCmax)
    
#%%
def sliding_window(image, overlaps, strides):
    ''' efficiently and lazily slides a window across the image
     Parameters
     ----------
     img:ndarray 2D
         image that needs to be slices
     windowSize: tuple 
         dimension of the patch
     strides: tuple
         stride in wach dimension    
         
     Returns:
     -------
     iterator containing five items
     dim_1, dim_2 coordinates in the patch grid 
     x, y: bottom border of the patch in the original matrix  
     patch: the patch
     '''
    windowSize = np.add(overlaps,strides)
    range_1 = range(0, image.shape[0]-windowSize[0], strides[0]) + [image.shape[0]-windowSize[0]]
    range_2 = range(0, image.shape[1]-windowSize[1], strides[1]) + [image.shape[1]-windowSize[1]]
    for dim_1, x in enumerate(range_1):
		for dim_2,y in enumerate(range_2):
			# yield the current window
			yield (dim_1, dim_2 , x, y, image[ x:x + windowSize[0],y:y + windowSize[1]])
#%%
def iqr(a):
    return np.percentile(a,75)-np.percentile(a,25)
#%%
def create_weight_matrix_for_blending(img, overlaps, strides):
   ''' create a matrix that is used to normalize the intersection of the stiched patches
   
   Parameters:
   -----------
   img: original image, ndarray
   shapes, overlaps, strides:  tuples
       shapes, overlaps and strides of the patches
   
   Returns:
   --------
   weight_mat: normalizing weight matrix    
   '''
   
   shapes = np.add(strides,overlaps)
   
   max_grid_1,max_grid_2 =  np.max(np.array([it[:2] for it in  sliding_window(img, overlaps, strides)]),0)

   for grid_1, grid_2 , x, y, _ in sliding_window(img, overlaps, strides):
       
       weight_mat = np.ones(shapes)  
       
       if grid_1 > 0:
           weight_mat[:overlaps[0],:] = np.linspace(0,1,overlaps[0])[:,None]
       if grid_1 < max_grid_1: 
           weight_mat[-overlaps[0]:,:] = np.linspace(1,0,overlaps[0])[:,None]
       if grid_2 > 0:
           weight_mat[:,:overlaps[1]] = weight_mat[:,:overlaps[1]]*np.linspace(0,1,overlaps[1])[None,:]
       if grid_2 < max_grid_2: 
           weight_mat[:,-overlaps[1]:] = weight_mat[:,-overlaps[1]:]*np.linspace(1,0,overlaps[1])[None,:]
       
       yield weight_mat
       
#%% 
def tile_and_correct(img,template, strides, overlaps,max_shifts, newoverlaps = None, newstrides = None, upsample_factor_grid=4,\
                upsample_factor_fft=10,show_movie=False,max_deviation_rigid=2,add_to_movie=0):
    
    ''' perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches
    
    Parameters:
    -----------
    img: ndaarray 2D
        image to correct

    template: ndarray
        reference image

    strides: tuple
        strides of the patches in which the FOV is subdivided

    overlaps: tuple
        amount of pixel overlaping between patches along each dimension

    max_shifts: tuple
        max shifts in x and y       

    newstrides:tuple
        strides between patches along each dimension when upsampling the vector fields
     
    newoverlaps:tuple
        amount of pixel overlaping between patches along each dimension when upsampling the vector fields
         
    upsample_factor_grid: int
        if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field
        
    upsample_factor_fft: int 
        resolution of fractional shifts 
    
    show_movie: boolean whether to visualize the original and corrected frame during motion correction
    
    max_deviation_rigid: int
        maximum deviation in shifts of each patch from the rigid shift (should not be large)    
    
    add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times
    

    
    '''    
    img = img.astype(np.float32)
    template = template.astype(np.float32)
    
    
    img = img + add_to_movie
    template = template + add_to_movie
    
    # compute rigid shifts    
    rigid_shts = register_translation(img,template,upsample_factor=upsample_factor_fft,max_shifts=max_shifts)

    
    
    # extract patches
    templates = [it[-1] for it in sliding_window(template,overlaps=overlaps,strides = strides)]
    xy_grid = [(it[0],it[1]) for it in sliding_window(template,overlaps=overlaps,strides = strides)]    
    num_tiles = np.prod(np.add(xy_grid[-1],1))    
    imgs = [it[-1] for it in sliding_window(img,overlaps=overlaps,strides = strides)]    
    dim_grid = tuple(np.add(xy_grid[-1],1))
    
    if max_deviation_rigid is not None:
        
        lb_shifts = np.ceil(np.subtract(rigid_shts,max_deviation_rigid)).astype(int)
        ub_shifts = np.floor(np.add(rigid_shts,max_deviation_rigid)).astype(int)

    else:

        lb_shifts = None
        ub_shifts = None
    
    #extract shifts for each patch        
    shfts = [register_translation(a,b,c, shifts_lb = lb_shifts, shifts_ub = ub_shifts, max_shifts = max_shifts) for a, b, c in zip (imgs,templates,[upsample_factor_fft]*num_tiles)]    
    
    # create a vector field
    shift_img_x = np.reshape(np.array(shfts)[:,0],dim_grid)  
    shift_img_y = np.reshape(np.array(shfts)[:,1],dim_grid)
    
    # create automatically upsample parameters if not passed
    if newoverlaps is None:
        newoverlaps = overlaps
    if newstrides is None:
        newstrides = tuple(np.round(np.divide(strides,upsample_factor_grid)).astype(np.int))
    
    newshapes = np.add(newstrides ,newoverlaps)
        
    
    
    imgs = [it[-1] for it in sliding_window(img,overlaps=newoverlaps,strides = newstrides)  ]
        
    xy_grid = [(it[0],it[1]) for it in sliding_window(img,overlaps=newoverlaps,strides = newstrides)]  

    start_step = [(it[2],it[3]) for it in sliding_window(img,overlaps=newoverlaps,strides = newstrides)]

    
    
    dim_new_grid = tuple(np.add(xy_grid[-1],1))

    shift_img_x = cv2.resize(shift_img_x,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
    shift_img_y = cv2.resize(shift_img_y,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
    num_tiles = np.prod(dim_new_grid)
   
   
    
#    shift_img_x[(np.abs(shift_img_x-rigid_shts[0])/iqr(shift_img_x-rigid_shts[0])/1.349)>max_sd_outlier] = np.median(shift_img_x)
#    shift_img_y[(np.abs(shift_img_y-rigid_shts[1])/iqr(shift_img_y-rigid_shts[1])/1.349)>max_sd_outlier] = np.median(shift_img_y)
#    
    
    total_shifts = [(-x,-y) for x,y in zip(shift_img_x.reshape(num_tiles),shift_img_y.reshape(num_tiles))]     

    imgs = [apply_shift_iteration(im,sh,border_nan=True) for im,sh in zip(imgs, total_shifts)]
    
    normalizer = np.zeros_like(img)*np.nan    
    new_img = np.zeros_like(img)*np.nan

    weight_matrix = create_weight_matrix_for_blending(img, newoverlaps, newstrides)

    for (x,y),(idx_0,idx_1),im,(sh_x,shy),weight_mat in zip(start_step,xy_grid,imgs,total_shifts,weight_matrix):

        prev_val_1 = normalizer[x:x + newshapes[0],y:y + newshapes[1]]        

        normalizer[x:x + newshapes[0],y:y + newshapes[1]] = np.nansum(np.dstack([~np.isnan(im)*1*weight_mat,prev_val_1]),-1)
        prev_val = new_img[x:x + newshapes[0],y:y + newshapes[1]]
        new_img[x:x + newshapes[0],y:y + newshapes[1]] = np.nansum(np.dstack([im*weight_mat,prev_val]),-1)

    new_img = new_img/normalizer
    if show_movie:
#        for xx,yy,(sh_x,sh_y) in zip(np.array(start_step)[:,0]+newshapes[0]/2,np.array(start_step)[:,1]+newshapes[1]/2,total_shifts):
#            new_img = cv2.arrowedLine(new_img,(xx,yy),(np.int(xx+sh_x*10),np.int(yy+sh_y*10)),5000,1)
            
        img = apply_shift_iteration(img,-rigid_shts,border_nan=True)
        img_show = new_img
        img_show = np.vstack([new_img,img])
        
        img_show = cv2.resize(img_show,None,fx=1,fy=1)
#        img_show = new_img
        
        cv2.imshow('frame',img_show/np.percentile(template,99))
        cv2.waitKey(int(1./500*1000))      
    
    else:
        cv2.destroyAllWindows()
#    xx,yy = np.array(start_step)[:,0]+newshapes[0]/2,np.array(start_step)[:,1]+newshapes[1]/2
#    pl.cla()
#    pl.imshow(new_img,vmin = 200, vmax = 500 ,cmap = 'gray',origin = 'lower')
#    pl.axis('off')
#    pl.quiver(yy,xx,shift_img_y, -shift_img_x, color = 'red')
#    pl.pause(.01)

          
    return new_img-add_to_movie, total_shifts,start_step,xy_grid    


