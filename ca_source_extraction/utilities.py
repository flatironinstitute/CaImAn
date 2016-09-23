# -*- coding: utf-8 -*-
"""A set of utilities, mostly for post-processing and visualization
Created on Sat Sep 12 15:52:53 2015

@author epnev
"""

import numpy as np
from scipy.sparse import spdiags, diags, coo_matrix, csc, csc_matrix
from matplotlib import pyplot as plt
from pylab import pause
import sys
import os
try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.io import vform, hplot
    from bokeh.models import CustomJS, ColumnDataSource
    from bokeh.models import Range1d
except:
    print "Bokeh could not be loaded. Either it is not installed or you are not running within a notebook"

import matplotlib as mpl


import matplotlib.cm as cm
import subprocess
import time
import ipyparallel
from matplotlib.widgets import Slider
import ca_source_extraction
import shutil
import glob
import shlex
from skimage.external.tifffile import imread
import psutil
import json
from scipy import ndimage
import scipy

from skimage.filters import sobel
from skimage.morphology import watershed
from scipy import ndimage as ndi
#from skimage.feature import peak_local_max
import pylab as pl
import skimage
from scipy.sparse import issparse
from scipy.ndimage.measurements import center_of_mass
from ipyparallel import Client
from scipy.stats import norm
from skimage.draw import polygon
from warnings import warn
from scipy.optimize import linear_sum_assignment   

#%%


def CNMFSetParms(Y, n_processes, K=30, gSig=[5, 5], ssub=1, tsub=1, p=2, p_ssub=1, p_tsub=1, thr=0.8, method_init= 'greedy_roi', **kwargs):
    """Dictionary for setting the CNMF parameters.
    Any parameter that is not set get a default value specified
    by the dictionary default options
    """

    if type(Y) is tuple:
        dims, T = Y[:-1], Y[-1]
    else:
        dims, T = Y.shape[:-1], Y.shape[-1]

    print 'using ' + str(n_processes) + ' processes'
    n_pixels_per_process = np.prod(dims) / n_processes  # how to subdivide the work among processes

    options = dict()
    options['patch_params'] = {
        'ssub': p_ssub,             # spatial downsampling factor
        'tsub': p_tsub,              # temporal downsampling factor
        'only_init' : False
    }
    options['preprocess_params'] = {'sn': None,                  # noise level for each pixel
                                    # range of normalized frequencies over which to average
                                    'noise_range': [0.25, 0.5],
                                    # averaging method ('mean','median','logmexp')
                                    'noise_method': 'logmexp',
                                    'max_num_samples_fft': 3*1024,
                                    'n_pixels_per_process': n_pixels_per_process,
                                    'compute_g': False,            # flag for estimating global time constant
                                    'p': p,                        # order of AR indicator dynamics
                                    'lags': 5,                     # number of autocovariance lags to be considered for time constant estimation
                                    'include_noise': False,        # flag for using noise values when estimating g
                                    'pixels': None,                 # pixels to be excluded due to saturation
                                    }
    options['init_params'] = {'K': K,                                          # number of components
                              # size of components (std of Gaussian)
                              'gSig': gSig,
                              # size of bounding box
                              'gSiz': list(np.array(gSig, dtype=int) * 2 + 1),
                              'ssub': ssub,             # spatial downsampling factor
                              'tsub': tsub,             # temporal downsampling factor
                              'nIter': 5,               # number of refinement iterations
                              'kernel': None,           # user specified template for greedyROI
                              'maxIter': 5,              # number of HALS iterations
                              'method' : method_init,     # can be greedy_roi or sparse_nmf
                              'max_iter_snmf' : 500,
                              'alpha_snmf' : 10e2,
                              'sigma_smooth_snmf' : (.5,.5,.5),
                              'perc_baseline_snmf': 20
                              }
    options['spatial_params'] = {
        'dims': dims,                   # number of rows, columns [and depths]
        # method for determining footprint of spatial components ('ellipse' or 'dilate')
        'method': 'dilate',#'ellipse', 'dilate',
        'dist': 3,                       # expansion factor of ellipse
        'n_pixels_per_process': n_pixels_per_process,   # number of pixels to be processed by eacg worker
        'medw' : (3, 3),                                # window of median filter
        'thr_method' : 'nrg',                           #  Method of thresholding ('max' or 'nrg')
        'maxthr' : 0.1,                                 # Max threshold
        'nrgthr' : 0.9999,                              # Energy threshold
        'extract_cc' : True,                            # Flag to extract connected components (might want to turn to False for dendritic imaging)
        'se' : np.ones((3, 3), dtype=np.int),           # Morphological closing structuring element
        'ss' : np.ones((3, 3), dtype=np.int)            # Binary element for determining connectivity            
        }
    options['temporal_params'] = {
        'ITER': 2,                   # block coordinate descent iterations
        # method for solving the constrained deconvolution problem ('cvx' or 'cvxpy')
        'method': 'cvxpy',
        # if method cvxpy, primary and secondary (if problem unfeasible for approx
        # solution) solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
        'solvers': ['ECOS', 'SCS'],
        'p': p,                      # order of AR indicator dynamics
        'memory_efficient': False,
        # flag for setting non-negative baseline (otherwise b >= min(y))
        'bas_nonneg': True,
        # range of normalized frequencies over which to average
        'noise_range': [.25, .5],
        'noise_method': 'logmexp',   # averaging method ('mean','median','logmexp')
                        'lags': 5,                   # number of autocovariance lags to be considered for time constant estimation
                        # bias correction factor (between 0 and 1, close to 1)
                        'fudge_factor': .96,
                        'verbosity': False
    }
    options['merging'] = {
        'thr': thr,
    }
    return options

#%%


def load_memmap(filename):
    """ Load a memory mapped file created by the function save_memmap
    Parameters:
    -----------
        filename: str
            path of the file to be loaded

    Returns:
    --------
    Yr:
        memory mapped variable
    dims: tuple
        frame dimensions
    T: int
        number of frames

    """
    if os.path.splitext(filename)[1] == '.mmap':

        file_to_load = filename

        filename = os.path.split(filename)[-1]

        fpart = filename.split('_')[1:-1]

        d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]

        Yr = np.memmap(file_to_load, mode='r', shape=(
            d1 * d2 * d3, T), dtype=np.float32, order=order)

        return (Yr, (d1, d2 ), T) if d3 == 1 else (Yr, (d1, d2, d3), T)

    else:
        raise Exception('Not implemented consistently')
        Yr = np.load(filename, mmap_mode='r')
        return Yr, None, None

#%% 
def save_memmap_each(fnames, dview=None, base_name=None, resize_fact=(1, 1, 1), remove_init=0, idx_xy=None,xy_shifts=None):
    """
    Create several memory mapped files using parallel processing
    
    Parameters:
    -----------
    fnames: list of str
        list of path to the filenames
        
    dview: ipyparallel dview
        used to perform computation in parallel. If none it will be signle thread
        
    base_name str
        BaseName for the file to be creates. If not given the file itself is used

    resize_fact: tuple
        resampling factors for each dimension x,y,time. .1 = downsample 10X
        
    remove_init: int 
        number of samples to remove from the beginning of each chunk
        
    idx_xy: slice operator 
        used to perform slicing of the movie (to select a subportion of the movie)
    
    xy_shifts: list 
        x and y shifts computed by a motion correction algorithm to be applied before memory mapping
    
    Returns:
    --------
    fnames_tot: list
        paths to the created memory map files
    
    
    """
    order='C'
    pars=[]
    if xy_shifts is None:
        xy_shifts=[None]*len(fnames)
    
    if type(resize_fact)is not list:
        resize_fact=[resize_fact]*len(fnames)    
    
        
    for idx,f in enumerate(fnames):
        if base_name is not None:
            pars.append([f,base_name+str(idx),resize_fact[idx],remove_init,idx_xy,order,xy_shifts[idx]])
        else:
            pars.append([f,os.path.splitext(f)[0],resize_fact[idx],remove_init,idx_xy,order,xy_shifts[idx]])            
    
    if dview is not None:
        fnames_new=dview.map_sync(save_place_holder,pars)
    else:
        fnames_new=map(save_place_holder,pars)
    
    return fnames_new
#%%
def save_memmap_join(mmap_fnames,base_name=None, n_chunks=6, dview=None,async=False):
    
    tot_frames=0
    order='C'
    for f in mmap_fnames:
        Yr,dims,T=load_memmap(f)
        print f,T
        tot_frames+=T
        del Yr
       
    
    d=np.prod(dims)
    
    if base_name is None:        
        base_name = mmap_fnames[0]
        base_name = base_name[:base_name.find('_d1_')]+'-#-'+str(len(mmap_fnames)) 
        
    fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(1 if len(dims) == 2 else dims[2]) + '_order_' + str(order) + '_frames_' + str(tot_frames) + '_.mmap'
    fname_tot = os.path.join(os.path.split(mmap_fnames[0])[0],fname_tot)         
    
    print fname_tot
                
    big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,shape=(d, tot_frames), order='C')
    
    
    step=d/n_chunks
    pars=[]
    for ref in range(0,d-step+1,step):
        pars.append([fname_tot,d,tot_frames,mmap_fnames,ref,ref+step])
    # last batch should include the leftover pixels
    pars[-1][-1]=d

#    if ref+step+1<d:
#        print 'running on remaining pixels:' + str(ref+step-d)
#        pars.append([fname_tot,d,tot_frames,mmap_fnames,ref+step,d])
    
    if dview is not None:    
        res=dview.map_sync(save_portion, pars)
    else:
        res=map(save_portion, pars)
     

    np.savez(base_name+'.npz',mmap_fnames=mmap_fnames,fname_tot=fname_tot)
    
    print 'Deleting big mov'
    del big_mov
    
    return fname_tot

    
#%%
def save_portion(pars):
    
    big_mov,d,tot_frames,fnames,idx_start,idx_end=pars
    big_mov = np.memmap(big_mov, mode='r+', dtype=np.float32,shape=(d, tot_frames), order='C')
    Ttot=0
    Yr_tot=np.zeros((idx_end-idx_start,tot_frames))    
    print Yr_tot.shape
    for f in fnames:        
        print f
        Yr,dims,T=load_memmap(f)        
        print idx_start,idx_end
        Yr_tot[:,Ttot:Ttot+T]=np.array(Yr[idx_start:idx_end])
        Ttot=Ttot+T
        del Yr
    
    big_mov[idx_start:idx_end,:]=Yr_tot
    del Yr_tot
    print 'done'
    del big_mov    
    return Ttot
#%%
        
    
#%%
def save_place_holder(pars):
    """ To use map reduce
    """
    
    import ca_source_extraction as cse
    f,base_name,resize_fact,remove_init,idx_xy,order,xy_shifts=pars   
    return save_memmap([f],base_name=base_name,resize_fact=resize_fact,remove_init=remove_init, idx_xy=idx_xy, order=order,xy_shifts=xy_shifts)
#%%
#def save_memmap(filenames, base_name='Yr', resize_fact=(1, 1, 1), remove_init=0, idx_xy=None, order='F'):
#    """ Saves efficiently a list of tif files into a memory mappable file
#    Parameters
#    ----------
#        filenames: list
#            list of tif files
#        base_name: str
#            the base used to build the file name. IT MUST NOT CONTAIN "_"    
#        resize_fact: tuple
#            x,y, and z downampling factors (0.5 means downsampled by a factor 2) 
#        remove_init: int
#            number of frames to remove at the begining of each tif file (used for resonant scanning images if laser in rutned on trial by trial)
#        idx_xy: tuple size 2 [or 3 for 3D data]
#            for selecting slices of the original FOV, for instance idx_xy=(slice(150,350,None),slice(150,350,None))
#
#    Return
#    -------
#        fname_new: the name of the mapped file, the format is such that the name will contain the frame dimensions and the number of f
#
#    """
#    Ttot = 0
#    for idx, f in enumerate(filenames):
#        print f
#        if os.path.splitext(f)[-1] == '.hdf5':
#            import calblitz as cb
#            if idx_xy is None:
#                Yr = np.array(cb.load(f))[remove_init:]
#            elif len(idx_xy) == 2:
#                Yr = np.array(cb.load(f))[remove_init:, idx_xy[0], idx_xy[1]]
#            else:
#                Yr = np.array(cb.load(f))[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]
#        else:
#            if idx_xy is None:
#                Yr = imread(f)[remove_init:]
#            elif len(idx_xy) == 2:
#                Yr = imread(f)[remove_init:, idx_xy[0], idx_xy[1]]
#            else:
#                Yr = imread(f)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]
#
#        fx, fy, fz = resize_fact
#        if fx != 1 or fy != 1 or fz != 1:
#            try:
#                import calblitz as cb
#                Yr = cb.movie(Yr, fr=1)
#                Yr = Yr.resize(fx=fx, fy=fy, fz=fz)
#            except:
#                print('You need to install the CalBlitz package to resize the movie')
#                raise
#
#        T, dims = Yr.shape[0], Yr.shape[1:]
#        Yr = np.transpose(Yr, range(1, len(dims) + 1) + [0])
#        Yr = np.reshape(Yr, (np.prod(dims), T), order='F')
#
#        if idx == 0:
#            fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(
#                1 if len(dims) == 2 else dims[2]) + '_order_' + str(order)
#                                   
#            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
#                                shape=(np.prod(dims), T), order=order)
#        else:
#            big_mov = np.memmap(fname_tot, dtype=np.float32, mode='r+',
#                                shape=(np.prod(dims), Ttot + T), order=order)
#        #    np.save(fname[:-3]+'npy',np.asarray(Yr))
#
#        big_mov[:, Ttot:Ttot + T] = np.asarray(Yr, dtype=np.float32) + 1e-10
#        big_mov.flush()
#        Ttot = Ttot + T
#
#    fname_new = fname_tot + '_frames_' + str(Ttot) + '_.mmap'
#    os.rename(fname_tot, fname_new)
#    del big_mov
#
#    return fname_new
#%%
def save_memmap(filenames, base_name='Yr', resize_fact=(1, 1, 1), remove_init=0, idx_xy=None, order='F',xy_shifts=None):
    """ Saves efficiently a list of tif files into a memory mappable file
    Parameters
    ----------
        filenames: list
            list of tif files
        base_name: str
            the base used to build the file name. IT MUST NOT CONTAIN "_"    
        resize_fact: tuple
            x,y, and z downampling factors (0.5 means downsampled by a factor 2) 
        remove_init: int
            number of frames to remove at the begining of each tif file (used for resonant scanning images if laser in rutned on trial by trial)
        idx_xy: tuple size 2 [or 3 for 3D data]
            for selecting slices of the original FOV, for instance idx_xy=(slice(150,350,None),slice(150,350,None))
        order: string
            whether to save the file in 'C' or 'F' order     
        xy_shifts: list 
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping    

    Return
    -------
        fname_new: the name of the mapped file, the format is such that the name will contain the frame dimensions and the number of f

    """
    Ttot = 0
    for idx, f in enumerate(filenames):
        print f
        is_calblitz=False
#        if os.path.splitext(f)[-1] == '.hdf5':
        try: 
            
            import calblitz as cb  
            is_calblitz=True
            
        except:
            
            print('Calblitz not found, using imread from skimage')
            if xy_shifts is not None:
                raise Exception('Calblitz not installed, you cannot motion correct')
                
            if idx_xy is None:
                Yr = imread(f)[remove_init:]
            elif len(idx_xy) == 2:
                Yr = imread(f)[remove_init:, idx_xy[0], idx_xy[1]]
            else:
                Yr = imread(f)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]                  
        
        if is_calblitz:
            
            Yr=cb.load(f,fr=1)            
            if xy_shifts is not None:
                Yr=Yr.apply_shifts(xy_shifts,interpolation='cubic',remove_blanks=False)
            
            if idx_xy is None:
                Yr = np.array(Yr)[remove_init:]
            elif len(idx_xy) == 2:
                Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1]]
            else:
                Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]
    

        fx, fy, fz = resize_fact
        if fx != 1 or fy != 1 or fz != 1:
            if is_calblitz:
                Yr = cb.movie(Yr, fr=1)
                Yr = Yr.resize(fx=fx, fy=fy, fz=fz)
            else:
                raise Exception('You need to install the CalBlitz package to resize the movie')
                

        T, dims = Yr.shape[0], Yr.shape[1:]
        Yr = np.transpose(Yr, range(1, len(dims) + 1) + [0])
        Yr = np.reshape(Yr, (np.prod(dims), T), order='F')

        if idx == 0:
            fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(
                1 if len(dims) == 2 else dims[2]) + '_order_' + str(order)
            fname_tot = os.path.join(os.path.split(f)[0],fname_tot)         
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                                shape=(np.prod(dims), T), order=order)
        else:
            big_mov = np.memmap(fname_tot, dtype=np.float32, mode='r+',
                                shape=(np.prod(dims), Ttot + T), order=order)
        #    np.save(fname[:-3]+'npy',np.asarray(Yr))

        big_mov[:, Ttot:Ttot + T] = np.asarray(Yr, dtype=np.float32) + 1e-10
        big_mov.flush()
        del big_mov
        Ttot = Ttot + T

    fname_new = fname_tot + '_frames_' + str(Ttot) + '_.mmap'
    os.rename(fname_tot, fname_new)

    return fname_new
#%%


# #%%
# def save_memmap(filenames, base_name='Yr', resize_fact=(1, 1, 1), remove_init=0, idx_xy=None):
#     """ Saves efficiently a list of tif files into a memory mappable file
#     Parameters
#     ----------
#         filenames: list
#             list of tif files
#         base_name: str
#             the base used to build the file name. IT MUST NOT CONTAIN "_"
#         resize_fact: tuple
#             x,y, and z downampling factors (0.5 means downsampled by a factor 2)
#         remove_init: int
#             number of frames to remove at the begining of each tif file (used for resonant scanning images if laser in rutned on trial by trial)
#         idx_xy: tuple size 2
# for selecting slices of the original FOV, for instance
# idx_xy=(slice(150,350,None),slice(150,350,None))

#     Return
#     -------
# fname_new: the name of the mapped file, the format is such that the name
# will contain the frame dimensions and the number of f

#     """
#     order = 'F'
#     Ttot = 0
#     for idx, f in enumerate(filenames):
#         print f
#         if os.path.splitext(f)[-1] == '.hdf5':
#             import calblitz as cb
#             if idx_xy is None:
#                 Yr = np.array(cb.load(f))[remove_init:]
#             else:
#                 Yr = np.array(cb.load(f))[remove_init:, idx_xy[0], idx_xy[1]]
#         else:
#             if idx_xy is None:
#                 Yr = imread(f)[remove_init:]
#             else:
#                 Yr = imread(f)[remove_init:, idx_xy[0], idx_xy[1]]

#         fx, fy, fz = resize_fact
#         if fx != 1 or fy != 1 or fz != 1:
#             try:
#                 import calblitz as cb
#                 Yr = cb.movie(Yr, fr=1)
#                 Yr = Yr.resize(fx=fx, fy=fy, fz=fz)
#             except:
#                 print('You need to install the CalBlitz package to resize the movie')
#                 raise

#         [T, d1, d2] = Yr.shape
#         Yr = np.transpose(Yr, (1, 2, 0))
#         Yr = np.reshape(Yr, (d1 * d2, T), order=order)

#         if idx == 0:
#             fname_tot = base_name + '_d1_' + str(d1) + '_d2_' + str(d2) + '_order_' + str(order)
#             big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
#                                 shape=(d1 * d2, T), order=order)
#         else:
#             big_mov = np.memmap(fname_tot, dtype=np.float32, mode='r+',
#                                 shape=(d1 * d2, Ttot + T), order=order)
#         #    np.save(fname[:-3]+'npy',np.asarray(Yr))

#         big_mov[:, Ttot:Ttot + T] = np.asarray(Yr, dtype=np.float32) + 1e-10
#         big_mov.flush()
#         Ttot = Ttot + T

#     fname_new = fname_tot + '_frames_' + str(Ttot) + '_.mmap'
#     os.rename(fname_tot, fname_new)

#     return fname_new
# #%%


def local_correlations(Y, eight_neighbours=True, swap_dim=True):
    """Computes the correlation image for the input dataset Y

    Parameters
    -----------

    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format
    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively
    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front

    Returns
    --------

    rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, range(Y.ndim)[:-1]))))

    rho = np.zeros(np.shape(Y)[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    rho[:-1, :] = rho[:-1, :] + rho_h
    rho[1:, :] = rho[1:, :] + rho_h
    rho[:, :-1] = rho[:, :-1] + rho_w
    rho[:, 1:] = rho[:, 1:] + rho_w

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
        rho[:, :, :-1] = rho[:, :, :-1] + rho_d
        rho[:, :, 1:] = rho[:, :, 1:] + rho_d
        neighbors = 6 * np.ones(np.shape(Y)[1:])
        neighbors[0] = neighbors[0] - 1
        neighbors[-1] = neighbors[-1] - 1
        neighbors[:, 0] = neighbors[:, 0] - 1
        neighbors[:, -1] = neighbors[:, -1] - 1
        neighbors[:, :, 0] = neighbors[:, :, 0] - 1
        neighbors[:, :, -1] = neighbors[:, :, -1] - 1

    else:
        if eight_neighbours:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:, ]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:, ]), axis=0)
            rho[:-1, :-1] = rho[:-1, :-1] + rho_d2
            rho[1:, 1:] = rho[1:, 1:] + rho_d1
            rho[1:, :-1] = rho[1:, :-1] + rho_d1
            rho[:-1, 1:] = rho[:-1, 1:] + rho_d2

            neighbors = 8 * np.ones(np.shape(Y)[1:3])
            neighbors[0, :] = neighbors[0, :] - 3
            neighbors[-1, :] = neighbors[-1, :] - 3
            neighbors[:, 0] = neighbors[:, 0] - 3
            neighbors[:, -1] = neighbors[:, -1] - 3
            neighbors[0, 0] = neighbors[0, 0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1, 0] = neighbors[-1, 0] + 1
            neighbors[0, -1] = neighbors[0, -1] + 1
        else:
            neighbors = 4 * np.ones(np.shape(Y)[1:3])
            neighbors[0, :] = neighbors[0, :] - 1
            neighbors[-1, :] = neighbors[-1, :] - 1
            neighbors[:, 0] = neighbors[:, 0] - 1
            neighbors[:, -1] = neighbors[:, -1] - 1

    rho = np.divide(rho, neighbors)

    return rho


def order_components(A, C):
    """Order components based on their maximum temporal value and size

    Parameters
    -----------
    A:   sparse matrix (d x K)
         spatial components
    C:   matrix or np.ndarray (K x T)
         temporal components

    Returns
    -------
    A_or:  np.ndarray
        ordered spatial components
    C_or:  np.ndarray
        ordered temporal components
    srt:   np.ndarray
        sorting mapping

    """
    A = np.array(A.todense())
    nA2 = np.sqrt(np.sum(A**2, axis=0))
    K = len(nA2)
    A = np.array(np.matrix(A) * spdiags(1 / nA2, 0, K, K))
    nA4 = np.sum(A**4, axis=0)**0.25
    C = np.array(spdiags(nA2, 0, K, K) * np.matrix(C))
    mC = np.ndarray.max(np.array(C), axis=1)
    srt = np.argsort(nA4 * mC)[::-1]
    A_or = A[:, srt] * spdiags(nA2[srt], 0, K, K)
    C_or = spdiags(1. / nA2[srt], 0, K, K) * (C[srt, :])

    return A_or, C_or, srt


def extract_DF_F(Y, A, C, i=None):
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
    A = A * diags(1 / nA2, 0)
    C = diags(nA2, 0) * C

    # if i is None:
    #    i = np.argmin(np.max(A,axis=0))

    Y = np.matrix(Y)
    Yf = A.transpose() * (Y - A * C)  # + A[:,i]*C[i,:])
    Df = np.median(np.array(Yf), axis=1)
    C_df = diags(1 / Df, 0) * C

    return C_df, Df


def com(A, d1, d2):
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
    Coor = dict()
    Coor['x'] = np.kron(np.ones((d2, 1)), np.expand_dims(range(d1), axis=1))
    Coor['y'] = np.kron(np.expand_dims(range(d2), axis=1), np.ones((d1, 1)))
    cm = np.zeros((nr, 2))        # vector for center of mass
    cm[:, 0] = np.dot(Coor['x'].T, A) / A.sum(axis=0)
    cm[:, 1] = np.dot(Coor['y'].T, A) / A.sum(axis=0)

    return cm


def view_patches_bar(Yr, A, C, b, f, d1, d2, YrA=None, secs=1, img=None):
    """view spatial and temporal components interactively

     Parameters
     -----------
     Yr:    np.ndarray
            movie in format pixels (d) x frames (T)
     A:     sparse matrix
                matrix of spatial components (d x K)
     C:     np.ndarray
                matrix of temporal components (K x T)
     b:     np.ndarray
                spatial background (vector of length d)

     f:     np.ndarray
                temporal background (vector of length T)
     d1,d2: np.ndarray
                frame dimensions
     YrA:   np.ndarray
                 ROI filtered residual as it is given from update_temporal_components
                 If not given, then it is computed (K x T)

     img:   np.ndarray
                background image for contour plotting. Default is the image of all spatial components (d1 x d2)

    """

    plt.ion()
    nr, T = C.shape
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    #A = A*spdiags(1/nA2,0,nr,nr)
    #C = spdiags(nA2,0,nr,nr)*C
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(A.T * np.matrix(Yr) - (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(
            f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C) + C)
    else:
        Y_r = YrA + C

    A = A * spdiags(1 / nA2, 0, nr, nr)
    A = A.todense()
    imgs = np.reshape(np.array(A), (d1, d2, nr), order='F')
    if img is None:
        img = np.mean(imgs[:, :, :-1], axis=-1)

    bkgrnd = np.reshape(b, (d1, d2), order='F')
    fig = plt.figure(figsize=(10, 10))

    axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])

    ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
#    ax1.axis('off')
    ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
#    ax1.axis('off')
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])
#    axcolor = 'lightgoldenrodyellow'
#    axcomp = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

    s_comp = Slider(axcomp, 'Component', 0, nr, valinit=0)
    vmax = np.percentile(img, 98)

    def update(val):
        i = np.int(np.round(s_comp.val))
        print 'Component:' + str(i)

        if i < nr:

            ax1.cla()
            imgtmp = imgs[:, :, i]
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray)
            ax1.set_title('Spatial component ' + str(i + 1))
            ax1.axis('off')

            ax2.cla()
            ax2.plot(np.arange(T), np.squeeze(np.array(Y_r[i, :])), 'c', linewidth=3)
            ax2.plot(np.arange(T), np.squeeze(np.array(C[i, :])), 'r', linewidth=2)
            ax2.set_title('Temporal component ' + str(i + 1))
            ax2.legend(labels=['Filtered raw data', 'Inferred trace'])

            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None', alpha=0.5, cmap=plt.cm.hot)
        else:

            ax1.cla()
            ax1.imshow(bkgrnd, interpolation='None')
            ax1.set_title('Spatial background background')

            ax2.cla()
            ax2.plot(np.arange(T), np.squeeze(np.array(f)))
            ax2.set_title('Temporal background')

    def arrow_key_image_control(event):

        if event.key == 'left':
            new_val = np.round(s_comp.val - 1)
            if new_val < 0:
                new_val = 0
            s_comp.set_val(new_val)

        elif event.key == 'right':
            new_val = np.round(s_comp.val + 1)
            if new_val > nr:
                new_val = nr
            s_comp.set_val(new_val)
        else:
            pass

    s_comp.on_changed(update)
    s_comp.set_val(0)
    id2 = fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()


def view_patches(Yr, A, C, b, f, d1, d2, YrA=None, secs=1):
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
     d1,d2: np.ndarray
                frame dimensions
     YrA:   np.ndarray
                 ROI filtered residual as it is given from update_temporal_components
                 If not given, then it is computed (K x T)
     secs:  float
                number of seconds in between component scrolling. secs=0 means interactive (click to scroll)
     imgs:  np.ndarray
                background image for contour plotting. Default is the image of all spatial components (d1 x d2)

    """
    plt.ion()
    nr, T = C.shape
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    #A = A*spdiags(1/nA2,0,nr,nr)
    #C = spdiags(nA2,0,nr,nr)*C
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(A.T * np.matrix(Yr) - (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(
            f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C) + C)
    else:
        Y_r = YrA + C

    A = A.todense()

    fig = plt.figure()
    thismanager = plt.get_current_fig_manager()
    thismanager.toolbar.pan()
    print('In order to scroll components you need to click on the plot')
    sys.stdout.flush()
    for i in range(nr + 1):
        if i < nr:
            ax1 = fig.add_subplot(2, 1, 1)
            plt.imshow(np.reshape(np.array(A[:, i]) / nA2[i],
                                  (d1, d2), order='F'), interpolation='None')
            ax1.set_title('Spatial component ' + str(i + 1))
            ax2 = fig.add_subplot(2, 1, 2)
            plt.plot(np.arange(T), np.squeeze(np.array(Y_r[i, :])), 'c', linewidth=3)
            plt.plot(np.arange(T), np.squeeze(np.array(C[i, :])), 'r', linewidth=2)
            ax2.set_title('Temporal component ' + str(i + 1))
            ax2.legend(labels=['Filtered raw data', 'Inferred trace'])

            if secs > 0:
                plt.pause(secs)
            else:
                plt.waitforbuttonpress()

            fig.delaxes(ax2)
        else:
            ax1 = fig.add_subplot(2, 1, 1)
            plt.imshow(np.reshape(b, (d1, d2), order='F'), interpolation='None')
            ax1.set_title('Spatial background background')
            ax2 = fig.add_subplot(2, 1, 2)
            plt.plot(np.arange(T), np.squeeze(np.array(f)))
            ax2.set_title('Temporal background')


def plot_contours(A, Cn, thr=None, thr_method = 'max', maxthr = 0.2, nrgthr = 0.9, display_numbers=True, max_number=None, cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Parameters
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)
     thr_method: [optional] string
              Method of thresholding: 
                  'max' sets to zero pixels that have value less than a fraction of the max value
                  'nrg' keeps the pixels that contribute up to a specified fraction of the energy
     maxthr: [optional] scalar
                Threshold of max value
     nrgthr: [optional] scalar
                Threshold of energy
     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr
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
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print 'Swapping dim'

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    if max_number is None:
        max_number = nr

    if thr is not None:
        thr_method = 'nrg'
        nrgthr = thr
        warn("The way to call utilities.plot_contours has changed. Look at the definition for more details.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    ax = plt.gca()
    # Cn[np.isnan(Cn)]=0
    if vmax is None and vmin is None:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                   vmin=np.percentile(Cn[~np.isnan(Cn)], 1), vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                   vmin=vmin, vmax=vmax)

    coordinates = []
    cm = com(A, d1, d2)
    for i in range(np.minimum(nr, max_number)):
        pars = dict(kwargs)
        if thr_method == 'nrg':        
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = nrgthr
            
        else: # thr_method = 'max'
            if ~(thr_method == 'max'):
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:,i].flatten()
            Bvec /= np.max(Bvec)
            thr = maxthr                
            
        if swap_dim:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        else:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='F')
        cs = plt.contour(y, x, Bmat, [thr], colors=colors)
        # this fix is necessary for having disjoint figures and borders plotted correctly
        p = cs.collections[0].get_paths()
        v = np.atleast_2d([np.nan, np.nan])
        for pths in p:
            vtx = pths.vertices
            num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
            if num_close_coords < 2:
                if num_close_coords == 0:
                    # case angle
                    newpt = np.round(vtx[-1, :] / [d2, d1]) * [d2, d1]
                    #import ipdb; ipdb.set_trace()
                    vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                else:
                    # case one is border
                    vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                    #import ipdb; ipdb.set_trace()

            v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)
#        p = cs.collections[0].get_paths()[0]
#        v = p.vertices

        pars['CoM'] = np.squeeze(cm[i, :])
        pars['coordinates'] = v
        pars['bbox'] = [np.floor(np.min(v[:, 1])), np.ceil(np.max(v[:, 1])),
                        np.floor(np.min(v[:, 0])), np.ceil(np.max(v[:, 0]))]
        pars['neuron_id'] = i + 1
        coordinates.append(pars)

    if display_numbers:
        for i in range(np.minimum(nr, max_number)):
            if swap_dim:
                ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors)
            else:
                ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)

    return coordinates


def manually_refine_components(Y, (dx, dy), A, C, Cn, thr=0.9, display_numbers=True, max_number=None, cmap=None, **kwargs):
    """Plots contour of spatial components against a background image and allows to interactively add novel components by clicking with mouse

     Parameters
     -----------
     Y: ndarray
               movie in 2D
     (dx,dy): tuple
               dimensions of the square used to identify neurons (should be set to the galue of gsiz)
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
     A: np.ndarray
         matrix A os estimated  spatial component contributions
     C: np.ndarray
         array of estimated calcium traces

    """
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    if max_number is None:
        max_number = nr

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    plt.imshow(Cn, interpolation=None, cmap=cmap)
    coordinates = []
    cm = com(A, d1, d2)

    Bmat = np.zeros((np.minimum(nr, max_number), d1, d2))
    for i in range(np.minimum(nr, max_number)):
        pars = dict(kwargs)
        indx = np.argsort(A[:, i], axis=None)[::-1]
        cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        Bmat[i] = np.reshape(Bvec, np.shape(Cn), order='F')

    T = np.shape(Y)[-1]

    plt.close()

    fig = plt.figure()
#    ax = fig.add_subplot(111)
    ax = plt.gca()
    ax.imshow(Cn, interpolation=None, cmap=cmap,
              vmin=np.percentile(Cn[~np.isnan(Cn)], 1), vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    for i in range(np.minimum(nr, max_number)):
        plt.contour(y, x, Bmat[i], [thr])

    if display_numbers:
        for i in range(np.minimum(nr, max_number)):
            ax.text(cm[i, 1], cm[i, 0], str(i + 1))

    A3 = np.reshape(A, (d1, d2, nr), order='F')
    while True:

        pts = fig.ginput(1, timeout=0)

        if pts != []:
            print pts
            xx, yy = np.round(pts[0]).astype(np.int)
            coords_y = np.array(range(yy - dy, yy + dy + 1))
            coords_x = np.array(range(xx - dx, xx + dx + 1))
            coords_y = coords_y[(coords_y >= 0) & (coords_y < d1)]
            coords_x = coords_x[(coords_x >= 0) & (coords_x < d2)]
            a3_tiny = A3[coords_y[0]:coords_y[-1] + 1, coords_x[0]:coords_x[-1] + 1, :]
            y3_tiny = Y[coords_y[0]:coords_y[-1] + 1, coords_x[0]:coords_x[-1] + 1, :]
#            y3med = np.median(y3_tiny,axis=-1)
#            y3_tiny = y3_tiny - y3med[...,np.newaxis]
            #y3_tiny = y3_tiny-np.median(y3_tiny,axis=-1)

            dy_sz, dx_sz = np.shape(a3_tiny)[:-1]
            y2_tiny = np.reshape(y3_tiny, (dx_sz * dy_sz, T), order='F')
            a2_tiny = np.reshape(a3_tiny, (dx_sz * dy_sz, nr), order='F')
            y2_res = y2_tiny - a2_tiny.dot(C)
#            plt.plot(xx,yy,'k*')

            y3_res = np.reshape(y2_res, (dy_sz, dx_sz, T), order='F')
            a__, c__, center__, b_in__, f_in__ = ca_source_extraction.initialization.greedyROI2d(
                y3_res, nr=1, gSig=[np.floor(dx_sz / 2), np.floor(dy_sz / 2)], gSiz=[dx_sz, dy_sz])
#            a__ = model.fit_transform(np.maximum(y2_res,0));
#            c__ = model.components_;

            a_f = np.zeros((d, 1))
            idxs = np.meshgrid(coords_y, coords_x)
            a_f[np.ravel_multi_index(idxs, (d1, d2), order='F').flatten()] = a__

            A = np.concatenate([A, a_f], axis=1)
            C = np.concatenate([C, c__], axis=0)
            indx = np.argsort(a_f, axis=None)[::-1]
            cumEn = np.cumsum(a_f.flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            bmat = np.reshape(Bvec, np.shape(Cn), order='F')
            plt.contour(y, x, bmat, [thr])
            pause(.01)

        elif pts == []:
            break

        nr += 1
        A3 = np.reshape(A, (d1, d2, nr), order='F')

    return A, C


def update_order(A):
    '''Determines the update order of the temporal components given the spatial
    components by creating a nest of random approximate vertex covers
     Input:
     -------
     A:    np.ndarray
          matrix of spatial components (d x K)

     Outputs:
     ---------
     O:   list of sets
          list of subsets of components. The components of each subset can be updated in parallel
     lo:  list
          length of each subset

    Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''
    K = np.shape(A)[-1]
    AA = A.T * A
    AA.setdiag(0)
    F = (AA) > 0
    F = F.toarray()
    rem_ind = np.arange(K)
    O = []
    lo = []
    while len(rem_ind) > 0:
        L = np.sort(app_vertex_cover(F[rem_ind, :][:, rem_ind]))
        if L.size:
            ord_ind = set(rem_ind) - set(rem_ind[L])
            rem_ind = rem_ind[L]
        else:
            ord_ind = set(rem_ind)
            rem_ind = []

        O.append(ord_ind)
        lo.append(len(ord_ind))

    return O[::-1], lo[::-1]


def app_vertex_cover(A):
    ''' Finds an approximate vertex cover for a symmetric graph with adjacency matrix A.

     Parameters
     -----------
     A:    boolean 2d array (K x K)
          Adjacency matrix. A is boolean with diagonal set to 0

     Returns
     --------
     L:   A vertex cover of A
     Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''

    L = []
    while A.any():
        nz = np.nonzero(A)[0]          # find non-zero edges
        u = nz[np.random.randint(0, len(nz))]
        A[u, :] = False
        A[:, u] = False
        L.append(u)

    return np.asarray(L)
#%%


def save_mat_in_chuncks(Yr, num_chunks, shape, mat_name='mat', axis=0):
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

    Returns
    ---------
    name of the saved file

    """

    Yr = np.array_split(Yr, num_chunks, axis=axis)
    print "splitting array..."
    folder = tempfile.mkdtemp()
    prev = 0
    idxs = []
    names = []
    for mm in Yr:
        mm = np.array(mm)
        idxs.append(np.array(range(prev, prev + mm.shape[0])).T)
        new_name = os.path.join(folder, mat_name + '_' + str(prev) + '_' + str(len(idxs[-1])))
        print "Saving " + new_name
        np.save(new_name, mm)
        names.append(new_name)
        prev = prev + mm.shape[0]

    return {'names': names, 'idxs': idxs, 'axis': axis, 'shape': shape}


def db_plot(*args, **kwargs):
    # plot utility for debugging
    plt.plot(*args, **kwargs)
    plt.show()
    pause(1)


def nb_view_patches(Yr, A, C, b, f, d1, d2, image_neurons=None, thr=0.99, denoised_color=None):
    '''
    Interactive plotting utility for ipython notbook

    Parameters
    -----------
    Yr: np.ndarray
        movie
    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    d1,d2: floats
        dimensions of movie (x and y)

    image_neurons: np.ndarray
        image to be overlaid to neurons (for instance the average)

    thr: double
        threshold regulating the extent of the displayed patches

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    '''
    colormap = cm.get_cmap("jet")  # choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.sum(np.array(A)**2, axis=0)
    b = np.squeeze(b)
    f = np.squeeze(f)
    #Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr-b[:,np.newaxis]*f[np.newaxis] - A.dot(C))) + C)
    Y_r = np.array(spdiags(1 / nA2, 0, nr, nr) * (A.T * np.matrix(Yr) - (A.T *
                                                                         np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C)) + C)

    bpl.output_notebook()
    x = np.arange(T)
    z = np.squeeze(np.array(Y_r[:, :].T)) / 100
    k = np.reshape(np.array(A), (d1, d2, A.shape[1]), order='F')
    if image_neurons is None:
        image_neurons = np.nanmean(k, axis=2)

    fig = plt.figure()
    coors = plot_contours(coo_matrix(A), image_neurons, thr=thr)
    plt.close()
#    cc=coors[0]['coordinates'];
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]
    npoints = range(len(c1))

    source = ColumnDataSource(data=dict(x=x, y=z[:, 0], y2=C[0] / 100, z=z, z2=C.T / 100))
    source2 = ColumnDataSource(data=dict(x=npoints, c1=c1, c2=c2, cc1=cc1, cc2=cc2))

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1, line_alpha=0.6, color=denoised_color)

    callback = CustomJS(args=dict(source=source, source2=source2), code="""
            var data = source.get('data');
            var f = cb_obj.get('value')-1
            x = data['x']
            y = data['y']
            y2 = data['y2']

            for (i = 0; i < x.length; i++) {
                y[i] = data['z'][i][f]
                y2[i] = data['z2'][i][f]
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

    slider = bokeh.models.Slider(start=1, end=Y_r.shape[
                                 0], value=1, step=1, title="Neuron Number", callback=callback)
    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple', line_width=2, source=source2)

    layout = vform(slider, hplot(plot1, plot))

    bpl.show(layout)

    return Y_r


def get_contours3d(A, dims, thr=0.9):
    """Gets contour of spatial components and returns their coordinates

     Parameters
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
     dims: tuple of ints
               Spatial dimensions of movie (x, y, z)
     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.995)

     Returns
     --------
     Coor: list of coordinates with center of mass and
            contour plot coordinates (per layer) for each component
    """
    d, nr = np.shape(A)
    d1, d2, d3 = dims
    x, y = np.mgrid[0:d2:1, 0:d3:1]

    coordinates = []
    cm = np.asarray([center_of_mass(a.reshape(dims, order='F')) for a in A.T])
    for i in range(nr):
        pars = dict()
        indx = np.argsort(A[:, i], axis=None)[::-1]
        cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        Bmat = np.reshape(Bvec, dims, order='F')
        pars['coordinates'] = []
        for B in Bmat:
            cs = plt.contour(y, x, B, [thr])
            # this fix is necessary for having disjoint figures and borders plotted correctly
            p = cs.collections[0].get_paths()
            v = np.atleast_2d([np.nan, np.nan])
            for pths in p:
                vtx = pths.vertices
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = np.round(vtx[-1, :] / [d2, d1]) * [d2, d1]
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)

                v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

            pars['coordinates'] += [v]
        pars['CoM'] = np.squeeze(cm[i, :])
        pars['neuron_id'] = i + 1
        coordinates.append(pars)
    return coordinates


def nb_view_patches3d(Yr, A, C, b, f, dims, image_type='mean',
                      max_projection=False, axis=0, thr=0.99, denoised_color=None):
    '''
    Interactive plotting utility for ipython notbook

    Parameters
    -----------
    Yr: np.ndarray
        movie

    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    dims: tuple of ints
        dimensions of movie (x, y and z)

    image_type: 'mean', 'max' or 'corr'
        image to be overlaid to neurons (average, maximum or nearest neigbor correlation)

    max_projection: boolean
        plot max projection along specified axis if True, plot layers if False

    axis: int (0, 1 or 2)
        axis along which max projection is performed or layers are shown

    thr: double
        threshold regulating the extent of the displayed patches

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    '''
    d, T = Yr.shape
    order = range(4)
    order.insert(0, order.pop(axis))
    Yr = Yr.reshape(dims + (-1,), order='F').transpose(order).reshape((d, T), order='F')
    A = A.reshape(dims + (-1,), order='F').transpose(order).reshape((d, -1), order='F')
    dims = tuple(np.array(dims)[order[:3]])
    d1, d2, d3 = dims
    colormap = cm.get_cmap("jet")  # choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.sum(np.array(A)**2, axis=0)
    b = np.squeeze(b)
    f = np.squeeze(f)
    Y_r = np.array(spdiags(1 / nA2, 0, nr, nr) *
                   (A.T * np.matrix(Yr) - (A.T * np.matrix(b[:, np.newaxis])) *
                    np.matrix(f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C)) + C)

    bpl.output_notebook()
    x = np.arange(T)
    z = np.squeeze(np.array(Y_r[:, :].T)) / 100
    k = np.reshape(np.array(A), dims + (A.shape[1],), order='F')
    source = ColumnDataSource(data=dict(x=x, y=z[:, 0], y2=C[0] / 100, z=z, z2=C.T / 100))

    if max_projection:
        if image_type == 'corr':
            image_neurons = [(local_correlations(
                Yr.reshape(dims + (-1,), order='F'))[:, ::-1]).max(i)
                for i in range(3)]
        elif image_type in ['mean', 'max']:
            tmp = [({'mean': np.nanmean, 'max': np.nanmax}[image_type]
                    (k, axis=3)[:, ::-1]).max(i) for i in range(3)]
        else:
            raise ValueError("image_type must be 'mean', 'max' or 'corr'")

#         tmp = [np.nanmean(k.max(i), axis=2) for i in range(3)]
        image_neurons = np.nan * np.ones((int(1.05 * (d1 + d2)), int(1.05 * (d1 + d3))))
        image_neurons[:d2, -d3:] = tmp[0][::-1]
        image_neurons[:d2, :d1] = tmp[2].T[::-1]
        image_neurons[-d1:, -d3:] = tmp[1]
        offset1 = image_neurons.shape[1] - d3
        offset2 = image_neurons.shape[0] - d1
        coors = [plot_contours(coo_matrix(A.reshape(dims + (-1,), order='F').max(i)
                                          .reshape((np.prod(dims) / dims[i], -1), order='F')),
                               tmp[i], thr=thr) for i in range(3)]
        plt.close()
        cc1 = [[cor['coordinates'][:, 0] + offset1 for cor in coors[0]],
               [cor['coordinates'][:, 1] for cor in coors[2]],
               [cor['coordinates'][:, 0] + offset1 for cor in coors[1]]]
        cc2 = [[cor['coordinates'][:, 1] for cor in coors[0]],
               [cor['coordinates'][:, 0] for cor in coors[2]],
               [cor['coordinates'][:, 1] + offset2 for cor in coors[1]]]
        c1x = cc1[0][0]
        c2x = cc2[0][0]
        c1y = cc1[1][0]
        c2y = cc2[1][0]
        c1z = cc1[2][0]
        c2z = cc2[2][0]
        source2 = ColumnDataSource(data=dict(  # x=npointsx, y=npointsy, z=npointsz,
            c1x=c1x, c1y=c1y, c1z=c1z,
            c2x=c2x, c2y=c2y, c2z=c2z, cc1=cc1, cc2=cc2))
        callback = CustomJS(args=dict(source=source, source2=source2), code="""
                var data = source.get('data');
                var f = cb_obj.get('value')-1
                y = data['y']
                y2 = data['y2']
                for (i = 0; i < y.length; i++) {
                    y[i] = data['z'][i][f];
                    y2[i] = data['z2'][i][f];
                }

                var data2 = source2.get('data');
                c1x = data2['c1x'];
                c2x = data2['c2x'];
                c1y = data2['c1y'];
                c2y = data2['c2y'];
                c1z = data2['c1z'];
                c2z = data2['c2z'];
                cc1 = data2['cc1'];
                cc2 = data2['cc2'];
                for (i = 0; i < c1x.length; i++) {
                       c1x[i] = cc1[0][f][i]
                       c2x[i] = cc2[0][f][i]
                }
                for (i = 0; i < c1y.length; i++) {
                       c1y[i] = cc1[1][f][i]
                       c2y[i] = cc2[1][f][i]
                }
                for (i = 0; i < c1z.length; i++) {
                       c1z[i] = cc1[2][f][i]
                       c2z[i] = cc2[2][f][i]
                }
                source2.trigger('change');
                source.trigger('change');
            """)

    else:
        if image_type == 'corr':
            image_neurons = local_correlations(Yr.reshape(dims + (-1,), order='F'))[:, ::-1]
        elif image_type in ['mean', 'max']:
            image_neurons = {'mean': np.nanmean, 'max': np.nanmax}[image_type](k, axis=3)[:, ::-1]
        else:
            raise ValueError('image_type must be mean, max or corr')

        cmap = bokeh.models.mappers.LinearColorMapper([mpl.colors.rgb2hex(m)
                                                       for m in colormap(np.arange(colormap.N))])
        cmap.high = image_neurons.max()
        coors = get_contours3d(A, dims, thr=thr)
        plt.close()
        cc1 = [[l[:, 0] for l in n['coordinates']] for n in coors]
        cc2 = [[l[:, 1] for l in n['coordinates']] for n in coors]
        linit = int(round(coors[0]['CoM'][0]))  # pick initl layer in which first neuron lies
        c1 = cc1[0][linit]
        c2 = cc2[0][linit]
        source2 = ColumnDataSource(data=dict(c1=c1, c2=c2, cc1=cc1, cc2=cc2))
        x = range(d2)
        y = range(d3)
        source3 = ColumnDataSource(
            data=dict(im1=[image_neurons[linit]], im=image_neurons, xx=[x], yy=[y]))

        callback = CustomJS(args=dict(source=source, source2=source2), code="""
                var data = source.get('data');
                var f = slider_neuron.get('value')-1;
                var l = slider_layer.get('value')-1;
                y = data['y']
                y2 = data['y2']
                for (i = 0; i < y.length; i++) {
                    y[i] = data['z'][i][f];
                    y2[i] = data['z2'][i][f];
                }

                var data2 = source2.get('data');
                c1 = data2['c1'];
                c2 = data2['c2'];
                for (i = 0; i < c1.length; i++) {
                       c1[i] = data2['cc1'][f][l][i];
                       c2[i] = data2['cc2'][f][l][i];
                }
                source2.trigger('change');
                source.trigger('change');
            """)

        callback_layer = CustomJS(args=dict(source=source3, source2=source2), code="""
                var f = slider_neuron.get('value')-1;
                var l = slider_layer.get('value')-1;
                var im1 = source.get('data')['im1'][0];
                for (var i = 0; i < source.get('data')['xx'][0].length; i++) {
                    for (var j = 0; j < source.get('data')['yy'][0].length; j++){
                        im1[i][j] = source.get('data')['im'][l][i][j];
                    }
                }
                var data2 = source2.get('data');
                c1 = data2['c1'];
                c2 = data2['c2'];
                for (i = 0; i < c1.length; i++) {
                       c1[i] = data2['cc1'][f][l][i];
                       c2[i] = data2['cc2'][f][l][i];
                }
                source.trigger('change');
                source2.trigger('change');
            """)

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1, line_alpha=0.6, color=denoised_color)
    slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                 title="Neuron Number", callback=callback)
    xr = Range1d(start=0, end=image_neurons.shape[1] if max_projection else d3)
    yr = Range1d(start=image_neurons.shape[0] if max_projection else d2, end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

    if max_projection:
        plot1.image(image=[image_neurons[::-1, :]], x=0,
                    y=image_neurons.shape[0], dw=image_neurons.shape[1], dh=image_neurons.shape[0], palette=grayp)
        plot1.patch('c1x', 'c2x', alpha=0.6, color='purple', line_width=2, source=source2)
        plot1.patch('c1y', 'c2y', alpha=0.6, color='purple', line_width=2, source=source2)
        plot1.patch('c1z', 'c2z', alpha=0.6, color='purple', line_width=2, source=source2)
        layout = vform(slider, hplot(plot1, plot))
    else:
        slider_layer = bokeh.models.Slider(start=1, end=d1, value=linit + 1, step=1,
                                           title="Layer", callback=callback_layer)
        callback.args['slider_neuron'] = slider
        callback.args['slider_layer'] = slider_layer
        callback_layer.args['slider_neuron'] = slider
        callback_layer.args['slider_layer'] = slider_layer
        plot1.image(image='im1', x=[0], y=[d2], dw=[d3], dh=[d2],
                    color_mapper=cmap, source=source3)
        plot1.patch('c1', 'c2', alpha=0.6, color='purple', line_width=2, source=source2)
        layout = vform(slider, slider_layer, hplot(plot1, plot))

    bpl.show(layout)

    return Y_r


def nb_imshow(image, cmap='jet'):
    '''
    Interactive equivalent of imshow for ipython notebook
    '''
    colormap = cm.get_cmap(cmap)  # choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    xr = Range1d(start=0, end=image.shape[1])
    yr = Range1d(start=image.shape[0], end=0)
    p = bpl.figure(x_range=xr, y_range=yr)
#    p = bpl.figure(x_range=[0,image.shape[1]], y_range=[0,image.shape[0]])
#    p.image(image=[image], x=0, y=0, dw=image.shape[1], dh=image.shape[0], palette=grayp)
    p.image(image=[image[::-1, :]], x=0, y=image.shape[0],
            dw=image.shape[1], dh=image.shape[0], palette=grayp)

    return p


def nb_plot_contour(image, A, d1, d2, thr=0.995, face_color=None, line_color='black', alpha=0.4, line_width=2, **kwargs):
    '''Interactive Equivalent of plot_contours for ipython notebook

    Parameters
    -----------
    A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
    Image:  np.ndarray (2D)
               Background image (e.g. mean, correlation)
    d1,d2: floats
               dimensions os image
    thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.995)
    display_number:     Boolean
               Display number of ROIs if checked (default True)
    max_number:    int
               Display the number for only the first max_number components (default None, display all numbers)
    cmap:     string
               User specifies the colormap (default None, default colormap)

    '''
    p = nb_imshow(image, cmap='jet')
    center = com(A, d1, d2)
    p.circle(center[:, 1], center[:, 0], size=10, color="black",
             fill_color=None, line_width=2, alpha=1)
    coors = plot_contours(coo_matrix(A), image, thr=thr)
    plt.close()
    #    cc=coors[0]['coordinates'];
    cc1 = [np.clip(cor['coordinates'][:, 0], 0, d2) for cor in coors]
    cc2 = [np.clip(cor['coordinates'][:, 1], 0, d1) for cor in coors]

    p.patches(cc1, cc2, alpha=.4, color=face_color,  line_color=line_color, line_width=2, **kwargs)
    return p


def start_server(slurm_script=None, ipcluster="ipcluster"):
    '''
    programmatically start the ipyparallel server

    Parameters
    ----------
    ncpus: int
        number of processors
    ipcluster : str
        ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda2\\\\Scripts\\\\ipcluster.exe"
         Default: "ipcluster"    
    '''
    sys.stdout.write("Starting cluster...")
    sys.stdout.flush()
    ncpus=psutil.cpu_count()
    
    if slurm_script is None:
        if ipcluster == "ipcluster":
            p1 = subprocess.Popen("ipcluster start -n {0}".format(ncpus), shell=True, close_fds=(os.name != 'nt'))
        else:
            p1 = subprocess.Popen(shlex.split("{0} start -n {1}".format(ipcluster, ncpus)), shell=True, close_fds=(os.name != 'nt'))
#
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
                
    else:
        shell_source(slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        print 'Running on %d engines.' % (ne)
        c.close()
        sys.stdout.write(" done\n")
        

#%%


def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""


    pipe = subprocess.Popen(". %s; env" % script,  stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    env = dict()
    for line in output.splitlines():
        lsp = line.split("=", 1)
        if len(lsp) > 1:
            env[lsp[0]] = lsp[1]
#    env = dict((line.split("=", 1) for line in output.splitlines()))
    os.environ.update(env)
    pipe.stdout.close()
#%%


def stop_server(is_slurm=False, ipcluster='ipcluster',pdir=None,profile=None):
    '''
    programmatically stops the ipyparallel server
    Parameters
     ----------
     ipcluster : str
         ipcluster binary file name; requires 4 path separators on Windows
         Default: "ipcluster"
    '''
    sys.stdout.write("Stopping cluster...\n")
    sys.stdout.flush()

    if is_slurm:
        
        if pdir is None and profile is None:
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        print 'Shutting down %d engines.' % (ne)
        c.shutdown(hub=True)
        shutil.rmtree('profile_' + str(profile))
        try:
            shutil.rmtree('./log/')
        except:
            print 'creating log folder'

        files = glob.glob('*.log')
        os.mkdir('./log')

        for fl in files:
            shutil.move(fl, './log/')

    else:
        if ipcluster == "ipcluster":
            proc = subprocess.Popen("ipcluster stop", shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))
        else:
            proc = subprocess.Popen(shlex.split(ipcluster + " stop"),
                                    shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))

        line_out = proc.stderr.readline()
        if 'CRITICAL' in line_out:
            sys.stdout.write("No cluster to stop...")
            sys.stdout.flush()
        elif 'Stopping' in line_out:
            st = time.time()
            sys.stdout.write('Waiting for cluster to stop...')
            while (time.time() - st) < 4:
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)
        else:
            print line_out
            print '**** Unrecognized Syntax in ipcluster output, waiting for server to stop anyways ****'
        
        proc.stderr.close()
        
    sys.stdout.write(" done\n")

# def evaluate_components(A,Yr,psx):
#
#    #%% clustering components
#    Ys=Yr
#    psx = cse.pre_processing.get_noise_fft(Ys,get_spectrum=True);
#
#    #[sn,psx] = get_noise_fft(Ys,options);
#    #P.sn = sn(:);
#    #fprintf('  done \n');
#    psdx = np.sqrt(psx[:,3:]);
#    X = psdx[:,1:np.minimum(np.shape(psdx)[1],1500)];
#    #P.psdx = X;
#    X = X-np.mean(X,axis=1)[:,np.newaxis]#     bsxfun(@minus,X,mean(X,2));     % center
#    X = X/(+1e-5+np.std(X,axis=1)[:,np.newaxis])
#
#    from sklearn.cluster import KMeans
#    from sklearn.decomposition import PCA,NMF
#    from sklearn.mixture import GMM
#    pc=PCA(n_components=5)
#    nmf=NMF(n_components=2)
#    nmr=nmf.fit_transform(X)
#
#    cp=pc.fit_transform(X)
#    gmm=GMM(n_components=2)
#
#    Cx1=gmm.fit_predict(cp)
#
#    L=gmm.predict_proba(cp)
#
#    km=KMeans(n_clusters=2)
#    Cx=km.fit_transform(X)
#    Cx=km.fit_transform(cp)
#    Cx=km.cluster_centers_
#    L=km.labels_
#    ind=np.argmin(np.mean(Cx[:,-49:],axis=1))
#    active_pixels = (L==ind)
#    centroids = Cx;
#
#    ff = false(1,size(Am,2));
#    for i = 1:size(Am,2)
#        a1 = Am(:,i);
#        a2 = Am(:,i).*Pm.active_pixels(:);
#        if sum(a2.^2) >= cl_thr^2*sum(a1.^2)
#            ff(i) = true;
#        end
#    end
#%%


def evaluate_components(Y, traces, A, N=5, robust_std=False,thresh_overlap=.3, thresh_finess=-20,compute_r_values=True):
    """ Define a metric and order components according to the probabilty if some "exceptional events" (like a spike). Suvh probability is defined as the likeihood of observing the actual trace value over N samples given an estimated noise distribution. 
    The function first estimates the noise distribution by considering the dispersion around the mode. This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349. 
    Then, the probavility of having N consecutive eventsis estimated. This probability is used to order the components. 
    The algorithm also measures the reliability of the spatial mask by comparing the filters in A with the average of the movies over samples where exceptional events happen, after  removing (if possible)
    frames when neighboring neurons were active
    
    Parameters
    ----------
    Y: ndarray 
        movie x,y,t
    
    A: scipy sparse array
        spatial components    
        
    traces: ndarray
        Fluorescence traces 

    N: int
        N number of consecutive events

    thresh_overlap: float
        not used for the moment
    
    
    Returns
    -------
    idx_components: ndarray
        the components ordered according to the fitness

    fitness: ndarray


    erfc: ndarray
        probability at each time step of observing the N consequtive actual trace values given the distribution of noise
        
    r_values: list
        float values representing correlation between component and spatial mask obtained by averaging important points

    num_significant_samples: int
        quantify how many samples were used rto obtain the spatial mask by average
    
    
    """
   # import pdb
   # pdb.set_trace()
    d1,d2,T=np.shape(Y)
    dims=(d1,d2)
    md = mode_robust(traces, axis=1)

    ff1 = traces - md[:, None]
    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1**2, 1) / Ns)
#


    # compute z value
    z = (traces - md[:, None]) / (3 * sd_r[:, None])
    # probability of observing values larger or equal to z given notmal
    # distribution with mean md and std sd_r
    erf = 1 - norm.cdf(z)
    # use logarithm so that multiplication becomes sum
    erf = np.log(erf)
    filt = np.ones(N)
    # moving sum
    erfc = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=1, arr=erf)
    erfc = erfc [:,:T]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    ordered = np.argsort(fitness)

    idx_components = ordered  # [::-1]# selec only portion of components
    fitness = fitness[idx_components]
    erfc = erfc[idx_components]

    # compute the overlap between spatial and movie average across samples with significant events

    
    if compute_r_values:
        min_important_points=2
        
        r_values=[]
    
        num_significant_samples=[]
        #import pdb
        #pdb.set_trace()        
        AA = (A.T*A).toarray() 
        nA=np.sqrt(np.array(A.power(2).sum(0)))
        AA = AA/np.outer(nA,nA.T)      
        
        for er,idx_comp in zip(erfc,idx_components):
            
            a=np.array(A.tocsc()[:,idx_comp].todense()).squeeze()   
            px = np.nonzero(a)[0]       
            mn=np.min(er)
                                               
            overlaping_idx = np.where(AA[idx_comp,:]>thresh_overlap)
    #        overlaping_idx_corr = np.where(np.array([scipy.stats.pearsonr(a,np.array(aa.todense()).squeeze())[0] for aa in A.T.tocsc()[:,overlaping_idx]])>thresh_overlap)[0]    
    #        overlaping_idx=overlaping_idx[overlaping_idx_corr]
            idx_not_a=np.setdiff1d(overlaping_idx,idx_comp) 
            important_points=np.where(er<=np.minimum(thresh_finess,mn*0.9))[0]
                
            
            if len(important_points) > 0:
                
                if len(idx_not_a)>0: #identify points when neighbors are active
     
                   print('active neighbours:'+str(idx_not_a))   
                   activities_other_comps=erfc[idx_not_a].min(0)        
                   bad_points=np.where(activities_other_comps<-10)[0]
    
                   if len(np.setdiff1d(important_points,bad_points)) > min_important_points:        
    
                        important_points=np.setdiff1d(important_points,bad_points)
            
                   else:  
                       
                        print('Including active neighbours')        
                    
                important_points=np.minimum(important_points,T-1) 
                
                num_significant_samples.append(len(important_points))

                non_activity_points=np.where(er>-2)[0]
                
                img=np.mean(Y[:,:,important_points],-1)-np.mean(Y[:,:,non_activity_points],-1)
                
#                pl.imshow(img)
#                pl.pause(1)
            #    img = cse.utilities.local_correlations(Y[:,:,np.array(important_points)])
                r_values.append(scipy.stats.pearsonr(a[px],img.flatten(order='F')[px])[0])
           
            else:
                num_significant_samples.append(0)
                r_values.append(0)
                print('No significant samples')     
    else:
         num_significant_samples=None
         r_values=None
        
             
        

    return idx_components, fitness, erfc,r_values,num_significant_samples
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
#=======
#               return data[1]
#         else:
##            wMin = data[-1] - data[0]
#            wMin=np.inf
#            N = data.size/2 + data.size%2
#            for i in xrange(0, N):
#               w = data[i+N-1] - data[i]
#               if w < wMin:
#                  wMin = w
#                  j = i
#
#            return _hsm(data[j:j+N])
#
#      data = inputData.ravel()
#      if type(data).__name__ == "MaskedArray":
#         data = data.compressed()
#      if dtype is not None:
#         data = data.astype(dtype)
#
#      # The data need to be sorted for this to work
#      data = numpy.sort(data)
#
#      # Find the mode
#      dataMode = _hsm(data)
#
#   return dataMode
#>>>>>>> 9afa11dd3825fd3ac6298b5bcfb3869a55ce3c68

#%% threshold and remove spurious components    
def threshold_components(A_s,shape,min_size=5,max_size=np.inf,max_perc=.3):        
    """
    Threshold components output of a CNMF algorithm (A matrices)
    
    Parameters:
    ----------
    
    A_s: list 
        list of A matrice output from CNMF
    
    min_size: int
        min size of the component in pixels

    max_size: int
        max size of the component in pixels
        
    max_perc: float        
        fraction of the maximum of each component used to threshold 
        
        
    Returns:
    -------        
    
    B_s: list of the thresholded components
    
    lab_imgs: image representing the components in ndimage format

    cm_s: center of masses of each components
    """
    
    B_s=[]
    lab_imgs=[]
    
    cm_s=[]
    for A_ in A_s:
        print '*'
        max_comps=A_.max(0).todense().T
        tmp=[]
        cm=[]
        lim=np.zeros(shape)
        for idx,a in enumerate(A_.T):        
            #create mask by thresholding to 50% of the max
            print idx
            mask=np.reshape(a.todense()>(max_comps[idx]*max_perc),shape)        
            label_im, nb_labels = ndimage.label(mask)
            sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))            
            l_largest=(label_im==np.argmax(sizes))
            cm.append(scipy.ndimage.measurements.center_of_mass(l_largest,l_largest))
            lim[l_largest] = (idx+1)
    #       #remove connected components that are too small
            mask_size=np.logical_or(sizes<min_size,sizes>max_size)
            if np.sum(mask_size[1:])>1:
                print 'removing ' + str( np.sum(mask_size[1:])-1) + ' components'
            remove_pixel=mask_size[label_im]
            label_im[remove_pixel] = 0           

            label_im=(label_im>0)*1    
            
            tmp.append(label_im.flatten())
        
        
        cm_s.append(cm)    
        lab_imgs.append(lim)        
        B_s.append(csc.csc_matrix(np.array(tmp)).T)
    
    return B_s, lab_imgs, cm_s           

#%% compute mask distances
def distance_masks(M_s,cm_s,max_dist):
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order, with matrix i compared with matrix i+1
    
    Parameters
    ----------
    M_s: list of ndarrays
        The thresholded A matrices (masks) to compare, output of threshold_components
    
    cm_s: list of list of 2-ples
        the centroids of the components in each M_s
    
    max_dist: float
        maximum distance among centroids allowed between components. This corresponds to a distance at which two components are surely disjoined
    
    
    
    Returns:
    --------
    D_s: list of matrix distances
    
    """
    D_s=[]

    for M1,M2,cm1,cm2 in zip(M_s[:-1],M_s[1:],cm_s[:-1],cm_s[1:]):
        print 'New Pair **'
        M1= M1.copy()[:,:]
        M2= M2.copy()[:,:]
        d_1=np.shape(M1)[-1]
        d_2=np.shape(M2)[-1]
        D = np.ones((d_1,d_2));
        
        cm1=np.array(cm1)
        cm2=np.array(cm2)
        for i in range(d_1):
            if i%100==0:
                print i
            k=M1[:,np.repeat(i,d_2)]+M2
    #        h=M1[:,np.repeat(i,d_2)].copy()
    #        h.multiply(M2)
            for j  in range(d_2): 
    
                dist = np.linalg.norm(cm1[i]-cm2[j])
                if dist<max_dist:
                    union = k[:,j].sum()
    #                intersection = h[:,j].nnz
                    intersection= np.array(M1[:,i].T.dot(M2[:,j]).todense()).squeeze()
        ##            intersect= np.sum(np.logical_xor(M1[:,i],M2[:,j]))
        ##            union=np.sum(np.logical_or(M1[:,i],M2[:,j]))
                    if union  > 0:
                        D[i,j] = 1-1.*intersection/(union-intersection)
                    else:
#                        print 'empty component: setting distance to max'
                        D[i,j] = 1.
                        
                    if np.isnan(D[i,j]):
                        raise Exception('Nan value produced. Error in inputs')
                else:
                    D[i,j] = 1
        
        D_s.append(D)            
    return D_s   

#%% find matches

def find_matches(D_s, print_assignment=False):
    
    matches=[]
    costs=[]
    t_start=time.time()
    for ii,D in enumerate(D_s):
        DD=D.copy()    
        if np.sum(np.where(np.isnan(DD)))>0:
            raise Exception('Distance Matrix contains NaN, not allowed!')
        
       
    #    indexes = m.compute(DD)
#        indexes = linear_assignment(DD)
        indexes = linear_sum_assignment(DD)
        indexes2=[(ind1,ind2) for ind1,ind2 in zip(indexes[0],indexes[1])]
        matches.append(indexes)
        DD=D.copy()   
        total = []
        for row, column in indexes2:
            value = DD[row,column]
            if print_assignment:
                print '(%d, %d) -> %f' % (row, column, value)
            total.append(value)      
        print  'FOV: %d, shape: %d,%d total cost: %f' % (ii, DD.shape[0],DD.shape[1], np.sum(total))
        print time.time()-t_start
        costs.append(total)      
        
    return matches,costs
      
#%%
def link_neurons(matches,costs,max_cost=0.6,min_FOV_present=None):
    """
    Link neurons from different FOVs given matches and costs obtained from the hungarian algorithm
    
    Parameters
    ----------
    matches: lists of list of tuple
        output of the find_matches function
    
    costs: list of lists of scalars
        cost associated to each match in matches
        
    max_cost: float
        maximum allowed value of the 1- intersection over union metric    
    
    min_FOV_present: int
        number of FOVs that must consequently contain the neuron starting from 0. If none 
        the neuro must be present in each FOV
    Returns:
    --------
    neurons: list of arrays representing the indices of neurons in each FOV
    
    """
    if min_FOV_present is None:
        min_FOV_present=len(matches)
    
    neurons=[]
    num_neurons=0
#    Yr_tot=[]
    num_chunks=len(matches)+1
    for idx in range(len(matches[0][0])):
        neuron=[]
        neuron.append(idx)
#        Yr=YrA_s[0][idx]+C_s[0][idx]
        for match,cost,chk in zip(matches,costs,range(1,num_chunks)):
            rows,cols=match        
            m_neur=np.where(rows==neuron[-1])[0].squeeze()
            if m_neur.size > 0:                           
                if cost[m_neur]<=max_cost:
                    neuron.append(cols[m_neur])
#                    Yr=np.hstack([Yr,YrA_s[chk][idx]+C_s[chk][idx]])
                else:                
                    break
            else:
                break
        if len(neuron)>min_FOV_present:           
            num_neurons+=1        
            neurons.append(neuron)
#            Yr_tot.append(Yr)
            
    
    neurons=np.array(neurons).T
    print 'num_neurons:' + str(num_neurons)
#    Yr_tot=np.array(Yr_tot)
    return neurons
    
   
#%%
def nf_load_masks(file_name,dims):
     # load the regions (training data only)
    with open(file_name) as f:
        regions = json.load(f)
    
    def tomask(coords):
        mask = np.zeros(dims)
        mask[zip(*coords)] = 1
        return mask
    
    masks = np.array([tomask(s['coordinates']) for s in regions])
    return masks
#%%
def nf_masks_to_json(binary_masks,json_filename):
    """
    Take as input a tensor of binary mask and produces json format for neurofinder 
    
    Parameters:
    -----------
    binary_masks: 3d ndarray (components x dimension 1  x dimension 2)
    
    json_filename: str
    
    Returns:
    --------
    regions: list of dict
        regions in neurofinder format
        
    """
    regions=[]
    for m in binary_masks:
        coords = [[x,y] for x,y in zip(*np.where(m))]
        regions.append({"coordinates":coords})
    
    
    with open(json_filename, 'w') as f:
        f.write(json.dumps(regions))
        
    return regions            
    
#%%
# Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
# License: MIT

def nf_read_roi(fileobj):
    '''
    points = read_roi(fileobj)

    Read ImageJ's ROI format
    '''
# This is based on:
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html


    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256


    pos = [4]
    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != 'Iout':
        raise IOError('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used

    roi_type = get8()
    # Discard second Byte:
    get8()

    if not (0 <= roi_type < 11):
        print('roireader: ROI type %s not supported' % roi_type)

    if roi_type != 7:

        print('roireader: ROI type %s not supported (!= 7)' % roi_type)

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()

    x1 = getfloat() 
    y1 = getfloat() 
    x2 = getfloat() 
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise ValueError('roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)
    points[:,1] = [getc() for i in xrange(n_coordinates)]
    points[:,0] = [getc() for i in xrange(n_coordinates)]
    points[:,1] += left
    points[:,0] += top
    points -= 1
    
    
    return points
#%%
def nf_read_roi_zip(fname,dims):
    
    import zipfile
    with zipfile.ZipFile(fname) as zf:
        coords = [nf_read_roi(zf.open(n))
                    for n in zf.namelist()]
    
    def tomask(coords):

        mask = np.zeros(dims)
        coords=np.array(coords)
#        rr, cc = polygon(coords[:,0]+1, coords[:,1]+1)        
        rr, cc = polygon(coords[:,0]+1, coords[:,1]+1)        
        mask[rr,cc]=1
#        mask[zip(*coords)] = 1
        
        return mask
    
    masks = np.array([tomask(s-1) for s in coords])
    return masks

#%%    
def extract_binary_masks_blob(A,  neuron_radius,dims,num_std_threshold=1, minCircularity= 0.5, minInertiaRatio = 0.2,minConvexity = .8):
    """
    Function to extract masks from data. It will also perform a preliminary selectino of good masks based on criteria like shape and size
    
    Parameters:
    ----------
    A: scipy.sparse matris
        contains the components as outputed from the CNMF algorithm
        
    neuron_radius: float 
        neuronal radius employed in the CNMF settings (gSiz)
    
    num_std_threshold: int
        number of times above iqr/1.349 (std estimator) the median to be considered as threshold for the component
    
     
    
    minCircularity: float
        parameter from cv2.SimpleBlobDetector
    
    minInertiaRatio: float
        parameter from cv2.SimpleBlobDetector
    
    minConvexity: float
        parameter from cv2.SimpleBlobDetector
    
    Returns:
    --------
    masks: np.array
    pos_examples:
    neg_examples:
    
    """    
    import cv2

    
    params = cv2.SimpleBlobDetector_Params()
    params.minCircularity = minCircularity
    params.minInertiaRatio = minInertiaRatio 
    params.minConvexity = minConvexity    
    
    # Change thresholds
    params.blobColor=255
    
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep= 3
    
#    min_elevation_map=
#    max_elevation_map=
    
    params.minArea = np.pi*((neuron_radius*.75)**2)
    #params.maxArea = 4*np.pi*((gSig[0]-1)**2)
    
    
    
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = True
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    
    masks_ws=[]
    pos_examples=[] 
    neg_examples=[]


    for count,comp in enumerate(A.tocsc()[:].T):

        print count
        comp_d=np.array(comp.todense())
#        comp_d=comp_d*(comp_d>(np.max(comp_d)*max_fraction))
#        comp_d=comp_d*(comp_d>(np.max(comp_d)*0))        
        comp_orig=np.reshape(comp.todense(),dims,order='F')
        comp_orig=(comp_orig-np.min(comp_orig))/(np.max(comp_orig)-np.min(comp_orig))*255
        gray_image=np.reshape(comp_d,dims,order='F')
        gray_image=(gray_image-np.min(gray_image))/(np.max(gray_image)-np.min(gray_image))*255
        gray_image=gray_image.astype(np.uint8)    

        
        # segment using watershed
        markers = np.zeros_like(gray_image)
        elevation_map = sobel(gray_image)
        thr_1=np.percentile(gray_image[gray_image>0],50)
        iqr=np.diff(np.percentile(gray_image[gray_image>0],(25,75)))
        thr_2=thr_1 + num_std_threshold*iqr/1.35
        markers[gray_image < thr_1] = 1
        markers[gray_image > thr_2] = 2  
#        local_maxi = peak_local_max(elevation_map, num_peaks=1,indices=False)
#        markers = ndi.label(local_maxi)[0]
        edges = watershed(elevation_map, markers)-1
        # only keep largest object 
        label_objects, nb_labels = ndi.label(edges)
        sizes = np.bincount(label_objects.ravel())
#        pl.subplot(2,2,1)
#        pl.imshow(gray_image)
#        pl.subplot(2,2,2)
#        pl.imshow(elevation_map)        
#        pl.subplot(2,2,3)
#        pl.imshow(markers)
#        pl.subplot(2,2,4)
#        pl.imshow(edges)
#        pl.pause(4.)
        if len(sizes)>1:           
            idx_largest = np.argmax(sizes[1:])    
            edges=(label_objects==(1+idx_largest))
            edges=ndi.binary_fill_holes(edges)
        else:
            print 'empty component'
            edges=np.zeros_like(edges)
        
#        edges=skimage.morphology.convex_hull_image(edges)
        if 1:
            masks_ws.append(edges)
            keypoints = detector.detect((edges*200.).astype(np.uint8))
        else:            
            masks_ws.append(gray_image)
            keypoints = detector.detect(gray_image)
            
        if len(keypoints)>0:
    #        im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            pos_examples.append(count)

        else:
            
            neg_examples.append(count)

    return np.array(masks_ws),np.array(pos_examples),np.array(neg_examples)    
#%%
