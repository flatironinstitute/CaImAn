# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:01:17 2015

@author: agiovann
"""

#%%
import cv2

import scipy.ndimage
import warnings
import numpy as np
from pylab import plt
from tempfile import NamedTemporaryFile
from IPython.display import HTML
import calblitz as cb
import numpy as np
from ipyparallel import Client
import os
import tifffile
#%%
def playMatrix(mov,gain=1.0,frate=.033):
    for frame in mov: 
        if gain!=1:
            cv2.imshow('frame',frame*gain)
        else:
            cv2.imshow('frame',frame)
            
        if cv2.waitKey(int(frate*1000)) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  
    cv2.destroyAllWindows()        
#%% montage
def matrixMontage(spcomps,*args, **kwargs):
    numcomps, width, height=spcomps.shape
    rowcols=int(np.ceil(np.sqrt(numcomps)));           
    for k,comp in enumerate(spcomps):        
        plt.subplot(rowcols,rowcols,k+1)       
        plt.imshow(comp,*args, **kwargs)                             
        plt.axis('off')         
        
#%% CVX OPT
#####    LOOK AT THIS! https://github.com/cvxgrp/cvxpy/blob/master/examples/qcqp.py
if False:
    from cvxopt import matrix, solvers
    A = matrix([ [ .3, -.4,  -.2,  -.4,  1.3 ],
                     [ .6, 1.2, -1.7,   .3,  -.3 ],
                     [-.3,  .0,   .6, -1.2, -2.0 ] ])
    b = matrix([ 1.5, .0, -1.2, -.7, .0])
    m, n = A.size
    I = matrix(0.0, (n,n))
    I[::n+1] = 1.0
    G = matrix([-I, matrix(0.0, (1,n)), I])
    h = matrix(n*[0.0] + [1.0] + n*[0.0])
    dims = {'l': n, 'q': [n+1], 's': []}
    x = solvers.coneqp(A.T*A, -A.T*b, G, h, dims)['x']
    print(x)    
    #%%
    from scipy.signal import lfilter
    dt=0.016;
    t=np.arange(0,10,dt)
    lambda_=dt*1;
    tau=.17;
    sigmaNoise=.1;
    tfilt=np.arange(0,4,dt);
    spikes=np.random.poisson(lam=lambda_,size=t.shape);
    print(np.sum(spikes))
    filtExp=np.exp(-tfilt/tau);
    simTraceCa=lfilter(filtExp,1,spikes);
    simTraceFluo=simTraceCa+np.random.normal(loc=0, scale=sigmaNoise,size=np.shape(simTraceCa));
    plt.plot(t,simTraceCa,'g')
    plt.plot(t,spikes,'r')
    plt.plot(t,simTraceFluo)           
    
      
    #%%
    #trtest=tracesDFF.D_5
    #simTraceFluo=trtest.Data';
    #dt=trtest.frameRate;
    #t=trtest.Time;
    #tau=.21;
    
    #%%
    gam=(1-dt/tau);
    numSamples=np.shape(simTraceFluo)[-1];
    G=np.diag(np.repeat(-gam,numSamples-1),-1) + np.diag(np.repeat(1,numSamples));
    A_2=- np.diag(np.repeat(1,numSamples));
    Aeq1=np.concatenate((G,A_2),axis=1);
    beq=np.hstack((simTraceFluo[0],np.zeros(numSamples-1)));
    A1=np.hstack((np.zeros(numSamples), -np.ones(numSamples)));
    
    
    #%%
    T = np.size(G,0);
    oness=np.ones((T));
    sqthr=np.sqrt(thr);
    #y(y<(mean(y(:)-3*std(y(:)))))=0;
#%%

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim,fps=20):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)
    

def display_animation(anim,fps=20):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim,fps=fps))    

def apply_function_per_movie_cluster(client_,function_name,args):
    """ use ipyparallel and SLURM to apply the same function on several datasets
    Parameters:
    -----------
    client_: pointer to ipyparallel client
        
    file_names: str
        list of file names
    function_name: handle
        name of function to apply

    args: list
        arguments to pass to function
    
    Returns:
    -------
    
    file_res: list 
        containing outputs
    """
    
    
    
    return file_res

def pre_preprocess_movie_labeling(dview, file_names, median_filter_size=(2,1,1), 
                                  resize_factors=[.2,.1666666666],diameter_bilateral_blur=4):
   def pre_process_handle(args):
#        import calblitz as cb 
        
        from scipy.ndimage import filters as ft
        import logging
        
        fil, resize_factors, diameter_bilateral_blur,median_filter_size=args
        
        name_log=fil[:-4]+ '_LOG'
        logger = logging.getLogger(name_log)
        hdlr = logging.FileHandler(name_log)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(logging.INFO)

        logger.info('START')
        logger.info(fil)
        mov=cb.load(fil,fr=30)
        logger.info('Read file')

        mov=mov.resize(1,1,resize_factors[0])
        logger.info('Resize')
        mov=mov.bilateral_blur_2D(diameter=diameter_bilateral_blur)
        logger.info('Bilateral')
        mov1=cb.movie(ft.median_filter(mov,median_filter_size),fr=30)
        logger.info('Median filter')
        #mov1=mov1-np.median(mov1,0)
        mov1=mov1.resize(1,1,resize_factors[1])
        logger.info('Resize 2')
        mov1=mov1-cb.utils.mode_robust(mov1,0)
        logger.info('Mode')
        mov=mov.resize(1,1,resize_factors[1])
        logger.info('Resize')
#        mov=mov-np.percentile(mov,1)
        
        mov.save(fil[:-4] + '_compress_.tif')
        logger.info('Save 1')
        mov1.save(fil[:-4] + '_BL_compress_.tif')
        logger.info('Save 2')
        return 1
        
   args=[]
   for name in file_names:
            args.append([name,resize_factors,diameter_bilateral_blur,median_filter_size])
            
   if dview is not None: 
       file_res = dview.map_sync(pre_process_handle, args)                         
       dview.results.clear()       
   else:
       file_res = map(pre_process_handle, args)     
        
   return file_res
    
#%%
def motion_correct_parallel(file_names,fr,template=None,margins_out=0,max_shift_w=5, max_shift_h=5,remove_blanks=False,apply_smooth=False,dview=None,save_hdf5=True):
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
def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    import numpy
    if axis is not None:
        fnc = lambda x: mode_robust(x, dtype=dtype)
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
                wMin = np.inf
                N = data.size / 2 + data.size % 2
                for i in xrange(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

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
#%%    
def process_movie_parallel(arg_in):
#    import calblitz
#    import calblitz.movies
    import calblitz as cb
    import numpy as np
    import sys
    

    

    fname,fr,margins_out,template,max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5=arg_in
    
    if template is not None:
        if type(template) is str:
            if os.path.exists(template):
                template=cb.load(template,fr=1)
            else:
                raise Exception('Path to template does not exist:'+template)                
#    with open(fname[:-4]+'.stout', "a") as log:
#        print fname
#        sys.stdout = log
        
    #    import pdb
    #    pdb.set_trace()
    
    if type(fname) is cb.movie or type(fname) is cb.movies.movie:
        print type(fname)
        Yr=fname

    else:        
        
        Yr=cb.load(fname,fr=fr)
        
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
        
        if type(fname) is cb.movie:

            return Yr 

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
def val_parse(v):
    # parse values from si tags into python objects if possible
    try:
        return eval(v)
    except:
        if v == 'true':
            return True
        elif v == 'false':
            return False
        elif v == 'NaN':
            return np.nan
        elif v == 'inf' or v == 'Inf':
            return np.inf

        else:
            return v

def si_parse(imd):

    # parse image_description field embedded by scanimage
    imd = imd.split('\n')
    imd = [i for i in imd if '=' in i]
    imd = [i.split('=') for i in imd]
    imd = [[ii.strip(' \r') for ii in i] for i in imd]
    imd = {i[0]:val_parse(i[1]) for i in imd}
    return imd

#%%
def get_image_description_SI(fname):
    """Given a tif file acquired with Scanimage it returns a dictionary containing the information in the image description field
    """
    image_descriptions=[]
    tf=tifffile.TiffFile(fname)
    for idx,pag in enumerate(tf.pages):
            if idx%1000==0:
                print(idx)
    #        i2cd=si_parse(pag.tags['image_description'].value)['I2CData']
            field=pag.tags['image_description'].value
            
            image_descriptions.append(si_parse(field))
    
    return image_descriptions