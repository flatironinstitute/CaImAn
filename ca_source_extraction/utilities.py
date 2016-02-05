# -*- coding: utf-8 -*-
"""A set of utilities, mostly for post-processing and visualization
Created on Sat Sep 12 15:52:53 2015

@author epnev
"""

import numpy as np
from scipy.sparse import spdiags, diags, coo_matrix
from matplotlib import pyplot as plt
from pylab import pause
import sys

try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.io import vform,hplot
    from bokeh.models import CustomJS, ColumnDataSource, Slider
    from bokeh.models import Range1d
except: 
    print "Bokeh could not be loaded. Either it is not installed or you are not running within a notebook"
    
import matplotlib as mpl
import matplotlib.cm as cm
import psutil
import subprocess
import time
import ipyparallel
from matplotlib.widgets import Slider
import ca_source_extraction
#%%


def CNMFSetParms(Y, K=30, gSig = [5,5], ssub = 1, tsub = 1, p = 2, **kwargs):
    """Dictionary for setting the CNMF parameters.
    Any parameter that is not set get a default value specified
    by the dictionary defaultopts
    """
    
    d1,d2,T=Y.shape
    n_processes = np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
    print 'using ' + str(n_processes) + ' processes'
    n_pixels_per_process=d1*d2/n_processes # how to subdivide the work among processes    
    
    options = dict()    
    options['preprocess_params']={ 'sn' : None,                  # noise level for each pixel
                                   'noise_range' : [0.25,0.5],   # range of normalized frequencies over which to average
                                   'noise_method':'logmexp',     # averaging method ('mean','median','logmexp')
                                   'n_processes':n_processes, 
                                   'n_pixels_per_process':n_pixels_per_process,   
                                   'compute_g':False,            # flag for estimating global time constant
                                   'p':p,                        # order of AR indicator dynamics     
                                   'lags':5,                     # number of autocovariance lags to be considered for time constant estimation
                                   'include_noise':False,        # flag for using noise values when estimating g
                                   'pixels':None                 # pixels to be excluded due to saturation
                                   }
    options['init_params'] = { 'K':K,                                          # number of components
                               'gSig':gSig,                                    # size of components (std of Gaussian)
                               'gSiz':list(np.array(gSig,dtype=int)*2+1),       # size of bounding box
                               'ssub':ssub,             # spatial downsampling factor
                               'tsub':tsub,             # temporal downsampling factor
                               'nIter':5,               # number of refinement iterations 
                               'kernel':None,           # user specified template for greedyROI
                               'maxIter':5              # number of HALS iterations
                               }
    options['spatial_params'] = {               
                        'd1':d1,                        # number of rows
                        'd2':d2,                        # number of columns
                        'method' : 'ellipse',           # method for determining footprint of spatial components ('ellipse' or 'dilate')                        
                        'dist':3,                       # expansion factor of ellipse
                        'n_processes':n_processes,      # number of process 
                        'n_pixels_per_process':n_pixels_per_process,    # number of pixels to be processed by eacg worker
                        'backend':'ipyparallel',
                        'memory_efficient':False
                    }
    options['temporal_params'] = {
                        'ITER':2,                   # block coordinate descent iterations
                        'method':'cvxpy',           # method for solving the constrained deconvolution problem ('cvx' or 'spgl1')
                        'p':p,                      # order of AR indicator dynamics
                        'n_processes':n_processes,
                        'backend':'ipyparallel',
                        'memory_efficient':False,                                
                        'bas_nonneg':True,          # flag for setting non-negative baseline (otherwise b >= min(y))
                        'noise_range':[.25,.5],     # range of normalized frequencies over which to average
                        'noise_method':'logmexp',   # averaging method ('mean','median','logmexp')
                        'lags':5,                   # number of autocovariance lags to be considered for time constant estimation
                        'fudge_factor':.98,         # bias correction factor (between 0 and 1, close to 1)
                        'verbosity':False
                    }

    return options


def local_correlations(Y,eight_neighbours=True, swap_dim = True):
     """Computes the correlation image for the input dataset Y
     
     Parameters
     -----------
     
     Y:   np.ndarray (3D)
          Input movie data in 3D format
     eight_neibhbours: Boolean
          Use 8 neighbors if true, and 4 if false (default = False)
     swap_dim: Boolean
          True indicates that time is listed in the last axis of Y (matlab format)
          and moves it in the front
     
     Returns
     --------
     
     rho: d1 x d2 matrix, cross-correlation with adjacent pixels
        
     """
     
     if swap_dim:
         Y = np.transpose(Y,tuple(np.hstack((Y.ndim-1,range(Y.ndim)[:-1]))))
    
     rho = np.zeros(np.shape(Y)[1:3])
     w_mov = (Y - np.mean(Y, axis = 0))/np.std(Y, axis = 0)
 
     rho_h = np.mean(np.multiply(w_mov[:,:-1,:], w_mov[:,1:,:]), axis = 0)
     rho_w = np.mean(np.multiply(w_mov[:,:,:-1], w_mov[:,:,1:,]), axis = 0)
     
     if True:
         rho_d1 = np.mean(np.multiply(w_mov[:,1:,:-1], w_mov[:,:-1,1:,]), axis = 0)
         rho_d2 = np.mean(np.multiply(w_mov[:,:-1,:-1], w_mov[:,1:,1:,]), axis = 0)


     rho[:-1,:] = rho[:-1,:] + rho_h
     rho[1:,:] = rho[1:,:] + rho_h
     rho[:,:-1] = rho[:,:-1] + rho_w
     rho[:,1:] = rho[:,1:] + rho_w
     
     if eight_neighbours:
         rho[:-1,:-1] = rho[:-1,:-1] + rho_d2
         rho[1:,1:] = rho[1:,1:] + rho_d1
         rho[1:,:-1] = rho[1:,:-1] + rho_d1
         rho[:-1,1:] = rho[:-1,1:] + rho_d2
          
     if eight_neighbours:
         neighbors = 8 * np.ones(np.shape(Y)[1:3])  
         neighbors[0,:] = neighbors[0,:] - 3;
         neighbors[-1,:] = neighbors[-1,:] - 3;
         neighbors[:,0] = neighbors[:,0] - 3;
         neighbors[:,-1] = neighbors[:,-1] - 3;
         neighbors[0,0] = neighbors[0,0] + 1;
         neighbors[-1,-1] = neighbors[-1,-1] + 1;
         neighbors[-1,0] = neighbors[-1,0] + 1;
         neighbors[0,-1] = neighbors[0,-1] + 1;
     else:
         neighbors = 4 * np.ones(np.shape(Y)[1:3]) 
         neighbors[0,:] = neighbors[0,:] - 1;
         neighbors[-1,:] = neighbors[-1,:] - 1;
         neighbors[:,0] = neighbors[:,0] - 1;
         neighbors[:,-1] = neighbors[:,-1] - 1;   

     rho = np.divide(rho, neighbors)

     return rho
     
def order_components(A,C):
     """Order components based on their maximum temporal value and size
     
     Parameters
     -----------
     A:   sparse matrix (d x K)
          spatial components
     C:   matrix or np.ndarray (K x T)
          temporal components
          
     A_or:  np.ndarray   
         ordered spatial components
     C_or:  np.ndarray  
         ordered temporal components
     srt:   np.ndarray  
         sorting mapping
     """
     A = np.array(A.todense())
     nA2 = np.sqrt(np.sum(A**2,axis=0))
     A = np.array(np.matrix(A)*diags(1/nA2,0))
     nA4 = np.sum(A**4,axis=0)**0.25
     C = np.array(diags(nA2,0)*np.matrix(C))
     mC = np.ndarray.max(np.array(C),axis=1)
     srt = np.argsort(nA4*mC)[::-1]
     A_or = A[:,srt]
     C_or = C[srt,:]
          
     return A_or, C_or, srt
     

def extract_DF_F(Y,A,C,i=None):
    """Extract DF/F values from spatial/temporal components and background
     
     Parameters
     -----------
     Y: np.ndarray
           input data (d x T)
     A: sparse matrix of np.ndarray 
           Set of spatial including spatial background (d x K)
     C: matrix
           Set of temporal components including background (K x T)
           
     Returns
     -----------
     C_df: matrix 
          temporal components in the DF/F domain
     Df:  np.ndarray
          vector with baseline values for each trace
    """
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.squeeze(np.array(A2.sum(axis=0)))
    A = A*diags(1/nA2,0)
    C = diags(nA2,0)*C         
     
    #if i is None:
    #    i = np.argmin(np.max(A,axis=0))

    Y = np.matrix(Y)         
    Yf =  A.transpose()*(Y - A*C) #+ A[:,i]*C[i,:])
    Df = np.median(np.array(Yf),axis=1)
    C_df = diags(1/Df,0)*C
             
    return C_df, Df


     
def com(A,d1,d2):
    """Calculation of the center of mass for spatial components
     Inputs:
     A:   np.ndarray 
          matrix of spatial components (d x K)
     d1:  int
          number of pixels in x-direction
     d2:  int
          number of pixels in y-direction
    
     Output:
     cm:  np.ndarray
          center of mass for spatial components (K x 2)
    """
    nr = np.shape(A)[-1]
    Coor=dict();
    Coor['x'] = np.kron(np.ones((d2,1)),np.expand_dims(range(d1),axis=1)); 
    Coor['y'] = np.kron(np.expand_dims(range(d2),axis=1),np.ones((d1,1)));
    cm = np.zeros((nr,2));        # vector for center of mass							   
    cm[:,0]=np.dot(Coor['x'].T,A)/A.sum(axis=0)
    cm[:,1]=np.dot(Coor['y'].T,A)/A.sum(axis=0)       
    
    return cm
    
    
def view_patches_bar(Yr,A,C,b,f,d1,d2,secs=1,img=None):
    """view spatial and temporal components (secs=0 interactive)
     
     Parameters
     -----------
     Yr:        np.ndarray 
            movie in format pixels (d) x frames (T)
     A:     sparse matrix
                matrix of spatial components (d x K)
     C:     np.ndarray
                matrix of temporal components (K x T)
     b:     np.ndarray
                spatial background (vector of length d)

     f:     np.ndarray
                temporal background (vector of length T)
     d1,d2: np/ndarray
                frame dimensions
     secs: float
                number of seconds in between component scrolling. secs=0 means interactive (click to scroll)
             
    """    

    plt.ion()
    nr,T = C.shape    
    nA2 = np.sum(np.array(A.todense())**2,axis=0)
    b = np.squeeze(b)
    f = np.squeeze(f)
    
    #Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr-b[:,np.newaxis]*f[np.newaxis] - A.dot(C))) + C)    
    Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr)-(A.T*np.matrix(b[:,np.newaxis]))*np.matrix(f[np.newaxis]) - (A.T.dot(A))*np.matrix(C)) + C)  
    A=A.todense()
    imgs=np.reshape(np.array(A),(d1,d2,nr),order='F')
    
    if img is None:
        img=np.mean(imgs[:,:,:-1],axis=-1)
#    Y_r = (Yr-b.dot(f)).T.dot(A.todense()).T/nA2[:,None]#-bl[:,None]
#    Y_r=[];
    bkgrnd=np.reshape(b,(d1,d2),order='F')
#    Atmp=A.copy()
#    Ctmp=C.copy()
#    for ii in range(C.shape[0]):
#        print ii
#        old_c=Ctmp[ii,:]
#        old_a=Atmp[:,ii]        
#        Atmp[:,ii]=0  
#        Ctmp[ii,:]=0
#        Y_r.append((Yr-b.dot(f)- Atmp.dot(Ctmp)).T.dot(A[:,ii]).T/nA2[ii])
#        Atmp[:,ii]=old_a  
#        Ctmp[ii,:]=old_c                
#    Y_r=np.asarray(Y_r)
    
#    fig = plt.subplots()
#    plt.subplots_adjust(left=0.05, bottom=0.2)
    fig=plt.figure(figsize=(20,20))

    axcomp =  plt.axes([0.05, 0.05, 0.9, 0.03])
    
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.4]) 
#    ax1.axis('off')
    ax3 = plt.axes([0.55, 0.55, 0.4, 0.4]) 
#    ax1.axis('off')
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])
#    axcolor = 'lightgoldenrodyellow'
#    axcomp = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    
    s_comp = Slider(axcomp, 'Component', 0, nr, valinit=0)
    vmax=np.percentile(img,98)    
    def update(val):
        i=np.int(np.round(s_comp.val))
        print 'Component:' + str(i)

    
        if i < nr:
            
            ax1.cla()
            imgtmp=imgs[:,:,i]
            ax1.imshow(imgtmp,interpolation='None',cmap=plt.cm.gray)
            ax1.set_title('Spatial component ' + str(i+1))
            ax1.axis('off')
            
            ax2.cla()
            ax2.plot(np.arange(T),np.squeeze(np.array(Y_r[i,:])),'c',linewidth=3) 
            ax2.plot(np.arange(T),np.squeeze(np.array(C[i,:])),'r',linewidth=2)             
            ax2.set_title('Temporal component ' + str(i+1)) 
            ax2.legend(labels = ['Filtered raw data','Inferred trace'])
            
            ax3.cla()
            ax3.imshow(img,interpolation='None',cmap=plt.cm.gray,vmax=vmax)
            imgtmp2=imgtmp.copy()    
            imgtmp2[imgtmp2==0]=np.nan
            ax3.imshow(imgtmp2,interpolation='None',alpha=0.5,cmap=plt.cm.hot)
        else:
            
            ax1.cla()
            ax1.imshow(bkgrnd,interpolation='None')
            ax1.set_title('Spatial background background')    
            
            ax2.cla()
            ax2.plot(np.arange(T),np.squeeze(np.array(f))) 
            ax2.set_title('Temporal background')  
     
    
    def arrow_key_image_control(event):
        
        if event.key == 'left':
            new_val=np.round(s_comp.val-1)
            if new_val<0:
                new_val=0
            s_comp.set_val(new_val)
                
        elif event.key == 'right':
            new_val=np.round(s_comp.val+1)    
            if new_val>nr:
                new_val=nr
            s_comp.set_val(new_val)
        else:
            pass    
    
    s_comp.on_changed(update)
    s_comp.set_val(0)
    id2 = fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()
        
    
    
def view_patches(Yr,A,C,b,f,d1,d2,YrA = None, secs=1):
    """view spatial and temporal components (secs=0 interactive)
     
     Parameters
     -----------
     Yr:        np.ndarray 
            movie in format pixels (d) x frames (T)
     A:     sparse matrix
                matrix of spatial components (d x K)
     C:     np.ndarray
                matrix of temporal components (K x T)
     b:     np.ndarray
                spatial background (vector of length d)

     f:     np.ndarray
                temporal background (vector of length T)
     d1,d2: np/ndarray
                frame dimensions
     secs: float
                number of seconds in between component scrolling. secs=0 means interactive (click to scroll)
             
    """    
    plt.ion()
    nr,T = C.shape    
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    A = A*spdiags(1/nA2,0,nr,nr)
    C = spdiags(nA2,0,nr,nr)*C
    #nA2 = np.sqrt(np.sum(np.array(A.todense())**2,axis=0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    #Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr-b[:,np.newaxis]*f[np.newaxis] - A.dot(C))) + C)    
    t1 = time.time()    
    #if YrA is None:
    #Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr)-(A.T*np.matrix(b[:,np.newaxis]))*np.matrix(f[np.newaxis]) - (A.T.dot(A))*np.matrix(C)) + C)
    Y_r = np.array(A.T*np.matrix(Yr)-(A.T*np.matrix(b[:,np.newaxis]))*np.matrix(f[np.newaxis]) - (A.T.dot(A))*np.matrix(C) + C)
    #else:
    #    Y_r = spdiags(1/nA2,0,nr,nr)*(YrA + C)
    #    Y_r2 = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr)-(A.T*np.matrix(b[:,np.newaxis]))*np.matrix(f[np.newaxis]) - (A.T.dot(A))*np.matrix(C)) + C)
    #    print np.sqrt(np.sum((Y_r-Y_r2)**2)/np.sum(Y_r**2))
    
    print time.time() - t1
    
    A=A.todense()
#    Y_r = (Yr-b.dot(f)).T.dot(A.todense()).T/nA2[:,None]#-bl[:,None]
#    Y_r=[];
#    
#    Atmp=A.copy()
#    Ctmp=C.copy()
#    for ii in range(C.shape[0]):
#        print ii
#        old_c=Ctmp[ii,:]
#        old_a=Atmp[:,ii]        
#        Atmp[:,ii]=0  
#        Ctmp[ii,:]=0
#        Y_r.append((Yr-b.dot(f)- Atmp.dot(Ctmp)).T.dot(A[:,ii]).T/nA2[ii])
#        Atmp[:,ii]=old_a  
#        Ctmp[ii,:]=old_c                
#    Y_r=np.asarray(Y_r)
    
    fig = plt.figure()
    thismanager = plt.get_current_fig_manager()
    thismanager.toolbar.pan()
    print('In order to scroll components you need to click on the plot')
    sys.stdout.flush()  
    for i in range(nr+1):
        if i < nr:
            ax1 = fig.add_subplot(2,1,1)
            plt.imshow(np.reshape(np.array(A[:,i]),(d1,d2),order='F'),interpolation='None')
            ax1.set_title('Spatial component ' + str(i+1))    
            ax2 = fig.add_subplot(2,1,2)
            plt.plot(np.arange(T),np.squeeze(np.array(Y_r[i,:])),'c',linewidth=3) 
            plt.plot(np.arange(T),np.squeeze(np.array(C[i,:])),'r',linewidth=2) 
            ax2.set_title('Temporal component ' + str(i+1)) 
            ax2.legend(labels = ['Filtered raw data','Inferred trace'])
            
            if secs>0:               
                plt.pause(secs) 
            else:
                plt.waitforbuttonpress()   
                
            fig.delaxes(ax2)
        else:
            ax1 = fig.add_subplot(2,1,1)
            plt.imshow(np.reshape(b,(d1,d2),order='F'),interpolation='None')
            ax1.set_title('Spatial background background')    
            ax2 = fig.add_subplot(2,1,2)
            plt.plot(np.arange(T),np.squeeze(np.array(f))) 
            ax2.set_title('Temporal background')      
            
def plot_contours(A,Cn,thr = 0.9, display_numbers = True, max_number = None,cmap=None, swap_dim=False,**kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates
     
     Parameters
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)
     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.995)
     display_number:     Boolean
               Display number of ROIs if checked (default True)
     max_number:    int
               Display the number for only the first max_number components (default None, display all numbers)
     cmap:     string
               User specifies the colormap (default None, default colormap)
               
     Returns
     --------
     Coor: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """
    from  scipy.sparse import issparse
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)
    

    if swap_dim:
        Cn=Cn.T
        print 'Swapping dim'
        
        
    d1,d2 = np.shape(Cn)
    d,nr = np.shape(A)       
    if max_number is None:
        max_number = nr
        
    x,y = np.mgrid[0:d1:1,0:d2:1]    
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
    ax = plt.gca()
    plt.imshow(Cn,interpolation=None,cmap=cmap)
    coordinates = []
    cm = com(A,d1,d2)
    for i in range(np.minimum(nr,max_number)):
        pars=dict(kwargs)
        indx = np.argsort(A[:,i],axis=None)[::-1]
        cumEn = np.cumsum(A[:,i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        if swap_dim:
            Bmat = np.reshape(Bvec,np.shape(Cn),order='C')
        else:
            Bmat = np.reshape(Bvec,np.shape(Cn),order='F')
            

        cs = plt.contour(y,x,Bmat,[thr])  
        # this fix is necessary for having disjoint figures and borders plotted correctly        
        p = cs.collections[0].get_paths()  
        v=np.atleast_2d([np.nan,np.nan]);        
        for pths in p: 
            vtx=pths.vertices
            num_close_coords=np.sum(np.isclose(vtx[0,:],vtx[-1,:]))
            if num_close_coords<2:
                if num_close_coords==0:                        
                    # case angle  
                    newpt=np.round(vtx[-1,:]/[d2,d1])*[d2,d1]
                    #import ipdb; ipdb.set_trace()
                    vtx=np.concatenate((vtx,newpt[np.newaxis,:]),axis=0)
                    
                else:
                    # case one is border    
                    vtx=np.concatenate((vtx,vtx[0,np.newaxis]),axis=0) 
                    #import ipdb; ipdb.set_trace()
                                                
            v=np.concatenate((v,vtx,np.atleast_2d([np.nan,np.nan])),axis=0) 
#        p = cs.collections[0].get_paths()[0]
#        v = p.vertices
            
        pars['CoM'] = np.squeeze(cm[i,:])
        pars['coordinates'] = v           
        pars['bbox'] = [np.floor(np.min(v[:,1])),np.ceil(np.max(v[:,1])),np.floor(np.min(v[:,0])),np.ceil(np.max(v[:,0]))]
        pars['neuron_id'] = i+1
        coordinates.append(pars)        
        
    if display_numbers:
        for i in range(np.minimum(nr,max_number)):
            if swap_dim:
                ax.text(cm[i,0],cm[i,1],str(i+1))
            else:
                ax.text(cm[i,1],cm[i,0],str(i+1))
            
    return coordinates


#def select_roi(img=None, n=0, ax=None, existing=None, mode='polygon', show_mode='mask', cmap=pl.cm.Greys_r, lasso_strictness=1):
#    """Select any number of regions of interest (ROI) in the movie.
#    
#    Parameters
#    ----------
#    img : np.ndarray
#        image over which to select roi
#    n : int
#        number of ROIs to select
#    ax : matplotlib.Axes
#        axes on which to show and select. If None, defaults to new, if 'current', defaults to current
#    existing : pyfluo.ROI
#        pre-existing rois to which to add selections
#    mode : 'polygon', 'lasso'
#        mode by which to select roi
#    show_mode : 'pts', 'mask'
#        mode by which to show existing rois
#    cmap : matplotlib.LinearSegmentedColormap
#        color map with which to display img
#    lasso_strictness : float
#        number from 0-inf, to do with tolerance for edge finding
#        
#    Returns
#    -------
#    ROI object
#    Notes
#    -----
#    Select points by clicking, and hit enter to finalize and ROI. Hit enter again to complete selection process.
#    author: BEN DEVERETT
#    """
#    if ax is None and img is None:
#        raise Exception('Image or axes must be supplied to select ROI.')
#
#    if ax == None:
#        fig = pl.figure()
#        ax = fig.add_subplot(111)
#    elif ax == 'current':
#        ax = pl.gca()
#    pl.sca(ax)
#    fig = ax.get_figure()
#
#    if img is not None:
#        shape = img.shape
#    elif ax is not None:
#        shape = [abs(np.diff(ax.get_ylim())), abs(np.diff(ax.get_xlim()))]
#

    


    
def manually_refine_components(Y,(dx,dy),A,C,Cn,thr = 0.9, display_numbers = True, max_number = None,cmap=None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates
     
     Parameters
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)
     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.995)
     display_number:     Boolean
               Display number of ROIs if checked (default True)
     max_number:    int
               Display the number for only the first max_number components (default None, display all numbers)
     cmap:     string
               User specifies the colormap (default None, default colormap)
      
               
     
     Returns
     --------
     Coor: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """
    from  scipy.sparse import issparse
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)
    

    
        
        
    d1,d2 = np.shape(Cn)
    d,nr = np.shape(A)       
    if max_number is None:
        max_number = nr
        
    x,y = np.mgrid[0:d1:1,0:d2:1]    
    
    
    plt.imshow(Cn,interpolation=None,cmap=cmap)
    coordinates = []
    cm = com(A,d1,d2)
    
    Bmat=np.zeros((np.minimum(nr,max_number),d1,d2))
    for i in range(np.minimum(nr,max_number)):
        pars=dict(kwargs)
        indx = np.argsort(A[:,i],axis=None)[::-1]
        cumEn = np.cumsum(A[:,i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn        
        Bmat[i] = np.reshape(Bvec,np.shape(Cn),order='F')
 
        
    
    

    T=np.shape(Y)[-1]

    plt.close()
    
    fig = plt.figure()
#    ax = fig.add_subplot(111)
    ax = plt.gca()
    ax.imshow(Cn,interpolation=None,cmap=cmap)
    for i in range(np.minimum(nr,max_number)):
        plt.contour(y,x,Bmat[i],[thr])
        
#    def on_pick(event):
#            artist = event.artist
#            xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
#            x, y = artist.get_xdata(), artist.get_ydata()
#            ind = event.ind
#            print 'Artist picked:', event.artist
#            print '{} vertices picked'.format(len(ind))
#            print 'Pick between vertices {} and {}'.format(min(ind), max(ind)+1)
#            print 'x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse)
#            print 'Data point:', x[ind[0]], y[ind[0]]
#            print
#        
#    fig, ax = plt.subplots()
#    
#    tolerance = 10 # points
#    ax.plot(range(10), 'ro-', picker=tolerance)
#    
#    fig.canvas.callbacks.connect('pick_event', on_pick)    
#    plt.show()    
    A3=np.reshape(A,(d1,d2,nr),order='F')
    while True:               
        
        pts = fig.ginput(1, timeout=0)    
        
        if pts != []:
            print pts
            xx,yy=np.round(pts[0]).astype(np.int);
            coords_y=np.array(range(yy-dy,yy+dy+1))
            coords_x=np.array(range(xx-dx,xx+dx+1))
            coords_y=coords_y[(coords_y>=0) & (coords_y<d1)]
            coords_x=coords_x[(coords_x>=0) & (coords_x<d2)]

            a3_tiny=A3[coords_y[0]:coords_y[-1]+1,coords_x[0]:coords_x[-1]+1,:]
            y3_tiny=Y[coords_y[0]:coords_y[-1]+1,coords_x[0]:coords_x[-1]+1,:]      
 
            dy_sz,dx_sz=np.shape(a3_tiny)[:-1]
            y2_tiny=np.reshape(y3_tiny,(dx_sz*dy_sz,T),order='F')
            a2_tiny=np.reshape(a3_tiny,(dx_sz*dy_sz,nr),order='F')
            y2_res=y2_tiny-a2_tiny.dot(C)
            y3_res=np.reshape(y2_res,(dy_sz,dx_sz,T),order='F')
#            plt.plot(xx,yy,'k*')                    
            
            a__, c__, center__, b_in__, f_in__=ca_source_extraction.initialization.greedyROI2d(y3_res, nr=1, gSig = [np.floor(dx_sz/2),np.floor(dy_sz/2)], gSiz = [dx_sz,dy_sz])
                      
            a_f = np.zeros((d,1))
            idxs=np.meshgrid(coords_y,coords_x)
            a_f[np.ravel_multi_index(idxs,(d1,d2),order='F').flatten()]=a__

            
            A=np.concatenate([A,a_f],axis=1)
            C=np.concatenate([C,c__],axis=0)      
            indx = np.argsort(a_f,axis=None)[::-1]
            cumEn = np.cumsum(a_f.flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            
            bmat = np.reshape(Bvec,np.shape(Cn),order='F')
            plt.contour(y,x,bmat,[thr])           
            pause(.1)
            
        elif pts == []:
            break
        if display_numbers:
            for i in range(np.minimum(nr,max_number)):            
                ax.text(cm[i,1],cm[i,0],str(i+1))
        nr+=1
        A3=np.reshape(A,(d1,d2,nr),order='F')
    
            
    return A,C

    
def update_order(A):
    '''Determines the update order of the temporal components given the spatial 
    components by creating a nest of random approximate vertex covers
     Input:
     A:    np.ndarray
          matrix of spatial components (d x K)
     
     Outputs:
     O:   list of sets
          list of subsets of components. The components of each subset can be updated in parallel
     lo:  list
          length of each subset
    Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''    
    K = np.shape(A)[-1]
    AA = A.T*A
    AA.setdiag(0)
    F = (AA)>0
    F = F.toarray()
    rem_ind = np.arange(K)
    O = []
    lo = []
    while len(rem_ind)>0:
        L = np.sort(app_vertex_cover(F[rem_ind,:][:,rem_ind]))
        if L.size:        
            ord_ind = set(rem_ind) - set(rem_ind[L])
            rem_ind = rem_ind[L]
        else:
            ord_ind = set(rem_ind)
            rem_ind = []
            
        O.append(ord_ind)
        lo.append(len(ord_ind))               
    
    return O[::-1],lo[::-1]
   
def app_vertex_cover(A):
    ''' Finds an approximate vertex cover for a symmetric graph with adjacency 
    matrix A.
     Input:
     A:    boolean 2d array (K x K)
          Adjacency matrix. A is boolean with diagonal set to 0
     
     Output:
     L:   A vertex cover of A
    Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''
    
    L = []
    while A.any():
        nz = np.nonzero(A)[0]          # find non-zero edges
        u = nz[np.random.randint(0,len(nz))]
        A[u,:] = False
        A[:,u] = False
        L.append(u)
    
    return np.asarray(L)    
#%%
def save_mat_in_chuncks(Yr,num_chunks,shape,mat_name='mat',axis=0): 
    """ save hdf5 matrix in chunks
    
    Parameters
    ----------
    file_name: str
        file_name of the hdf5 file to be chunked
    shape: tuples
        shape of the original chunked matrix
    idx: list
        indexes to slice matrix along axis
    mat_name: [optional] string
        name prefix for temporary files
    axis: int
        axis along which to slice the matrix   
    
    Returns:
    name of the saved file
            
    """
    
    Yr=np.array_split(Yr,num_chunks,axis=axis)  
    print "splitting array..."
    folder = tempfile.mkdtemp()  
    prev=0
    idxs=[]
    names=[];
    for mm in Yr:
        mm=np.array(mm)
        idxs.append(np.array(range(prev,prev+mm.shape[0])).T)
        new_name = os.path.join(folder, mat_name + '_'+str(prev) +'_'+str(len(idxs[-1])) ) 
        print "Saving " + new_name
        np.save(new_name,mm)
        names.append(new_name)        
        prev=prev+mm.shape[0]    
    
    return {'names':names,'idxs':idxs,'axis':axis,'shape':shape}   

def db_plot(*args,**kwargs):
    plt.plot(*args,**kwargs)
    plt.show()
    pause(1)
    
def nb_view_patches(Yr,A,C,b,f,d1,d2,image_neurons=None,thr = 0.99):
    
    colormap =cm.get_cmap("jet") #choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr,T = C.shape
    nA2 = np.sum(np.array(A)**2,axis=0)
    b = np.squeeze(b)
    f = np.squeeze(f)
    #Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr-b[:,np.newaxis]*f[np.newaxis] - A.dot(C))) + C)    
    Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr)-(A.T*np.matrix(b[:,np.newaxis]))*np.matrix(f[np.newaxis]) - (A.T.dot(A))*np.matrix(C)) + C)  

    
    bpl.output_notebook()
    x = np.arange(T)
    z=np.squeeze(np.array(Y_r[:,:].T))/100
    k=np.reshape(np.array(A),(d1,d2,A.shape[1]),order='F')
    if image_neurons is None:    
        image_neurons=np.sum(k,axis=2)  
    
    fig = plt.figure()
    coors = plot_contours(coo_matrix(A),image_neurons,thr = thr)
    plt.close()
#    cc=coors[0]['coordinates'];
    cc1=[cor['coordinates'][:,0] for cor in coors]
    cc2=[cor['coordinates'][:,1] for cor in coors]
    c1=cc1[0]
    c2=cc2[0]
    npoints=range(len(c1))
    
    source = ColumnDataSource(data=dict(x=x, y=z[:,0], z=z))    
    source2 = ColumnDataSource(data=dict(x=npoints,c1=c1,c2=c2,cc1=cc1,cc2=cc2))
    
    
    
    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    
    callback = CustomJS(args=dict(source=source,source2=source2), code="""
            var data = source.get('data');
            var f = cb_obj.get('value')
            x = data['x']
            y = data['y']
            z = data['z']
            
            for (i = 0; i < x.length; i++) {
                y[i] = z[i][f-1]             
            }
            
            var data2 = source2.get('data');
            c1 = data2['c1'];
            c2 = data2['c2'];
            cc1 = data2['cc1'];
            cc2 = data2['cc2'];
            
            for (i = 0; i < c1.length; i++) {            
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]            
            }
            source2.trigger('change');
            source.trigger('change');
            
        """)
    
    
    slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1, title="Neuron Number", callback=callback)
    xr = Range1d(start=0,end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0],end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr,plot_width=300, plot_height=300)
    plot1.image(image=[image_neurons[::-1,:]], x=0, y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1','c2',alpha=0.6, color='red',line_width=2,source=source2)
        
    
    layout = vform(slider, hplot(plot1,plot))
    
    bpl.show(layout)
    
    return Y_r

def nb_imshow(image,cmap='jet'):    
    colormap =cm.get_cmap(cmap) #choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    xr = Range1d(start=0,end=image.shape[1])
    yr = Range1d(start=image.shape[0],end=0)
    p = bpl.figure(x_range=xr, y_range=yr)
#    p = bpl.figure(x_range=[0,image.shape[1]], y_range=[0,image.shape[0]])
#    p.image(image=[image], x=0, y=0, dw=image.shape[1], dh=image.shape[0], palette=grayp)
    p.image(image=[image[::-1,:]], x=0, y=image.shape[0], dw=image.shape[1], dh=image.shape[0], palette=grayp)

    return p

def nb_plot_contour(image,A,d1,d2,thr=0.995,face_color=None, line_color='black',alpha=0.4,line_width=2,**kwargs):      
    p=nb_imshow(image,cmap='jet')   
    center = com(A,d1,d2)
    p.circle(center[:,1],center[:,0], size=10, color="black", fill_color=None, line_width=2, alpha=1)
    coors = plot_contours(coo_matrix(A),image,thr = thr)
    plt.close()
    #    cc=coors[0]['coordinates'];
    cc1=[np.clip(cor['coordinates'][:,0],0,d2) for cor in coors]
    cc2=[np.clip(cor['coordinates'][:,1],0,d1) for cor in coors]

    p.patches(cc1,cc2, alpha=.4, color=face_color,  line_color=line_color, line_width=2,**kwargs)
    return p


def start_server(ncpus):

    sys.stdout.write("Starting cluster...")
    sys.stdout.flush()

    subprocess.Popen(["ipcluster start -n {0}".format(ncpus)], shell=True)
    while True:
        try:
            c = ipyparallel.Client()
            if len(c) < ncpus:
                sys.stdout.write(".")
                sys.stdout.flush()
                raise ipyparallel.error.TimeoutError
            c.close()
            break
        except (IOError, ipyparallel.error.TimeoutError):
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(1)

    sys.stdout.write(" done\n")


def stop_server():

    sys.stdout.write("Stopping cluster...\n")
    sys.stdout.flush()
    
    proc=subprocess.Popen(["ipcluster stop"], shell=True,stderr=subprocess.PIPE)
    line_out=proc.stderr.readline()
    if 'CRITICAL' in line_out:
        sys.stdout.write("No cluster to stop...")
        sys.stdout.flush()
    elif 'Stopping' in line_out:       
        st=time.time()
        sys.stdout.write('Waiting for cluster to stop...')
        while (time.time() - st) < 4:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1)
    else:
        print '**** Unrecognized Syntax in ipcluster output, waiting for server to stop anyways ****'        
        
    
    sys.stdout.write(" done\n")
