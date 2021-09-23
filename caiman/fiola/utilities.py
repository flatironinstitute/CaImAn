#!/usr/bin/env python
"""
Relevant functions used in FIOLA
@author: @caichangjia
"""
import cv2
from matplotlib import path as mpl_path
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as nd
import scipy.sparse as spr
from sklearn.decomposition import NMF

import caiman as cm
from caiman.source_extraction.volpy.spikepursuit import denoise_spikes

def compute_std(data):
    """
    Compute standard deviation of the data based on the signal below the median

    Parameters
    ----------
    data : ndarary
        input data

    Returns
    -------
    std : float
        standard deviation
    """
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    return std

def non_symm_median_filter(t, filt_window):
    """
    median filter which is not symmetric

    Parameters
    ----------
    t : ndarray
        input signal
    filt_window : list
        filter size. First element in the list refers 
        to the number of past frames, second element in the 
        list refers to the number of future frames

    Returns
    -------
    m : ndarray
        output signal

    """
    m = t.copy()
    for i in range(len(t)):
        if (i > filt_window[0]) and (i < len(t) - filt_window[1]):
            m[i] = np.median(t[i - filt_window[0] : i + filt_window[1] + 1])
    return m

def normalize(data):
    """ Normalize the data
    Args: 
        data: ndarray
            input data
    
    Returns:
        data_norm: ndarray
            normalized data        
    """
    data = data - np.median(data)
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns))
    data_norm = data/std 
    return data_norm

def normalize_piecewise(data, step=5000):
    """ Normalize the data every step frames
    Args: 
        data: ndarray
            input data
        step: int
            normalize the data every step of frames separately    
    Returns:
        data_norm: ndarray
            normalized data        
    """
    data_norm = []
    for i in range(np.ceil(len(data)/step).astype(np.int16)):
        if (i + 1)*step > len(data):
            d = data[i * step:]
        else:
            d = data[i * step : (i + 1) * step]
        data_norm.append(normalize(d))
    data_norm = np.hstack(data_norm)
    return data_norm

def HALS4activity(Yr, A, C, iters=2):
    U = A.T.dot(Yr)
    V = A.T.dot(A) + np.finfo(A.dtype).eps
    for _ in range(iters):
        for m in range(len(U)):  # neurons and background
            C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                       V[m, m], 0, np.inf)
    return C

def hals_for_fiola(Y, A, C, b, f, bSiz=3, maxIter=5, semi_nmf=False, update_bg=True, use_spikes=False, hals_orig=False, fr=400):
    """ Hierarchical alternating least square method for solving NMF problem.
    The function is modified with more arguments compared to the original one.
    Y = A*C + b*f
    Args:
       Y:      d1 X d2 [X d3] X T, raw data.
           It will be reshaped to (d1*d2[*d3]) X T in this
           function
       A:      (d1*d2[*d3]) X K, initial value of spatial components
       C:      K X T, initial value of temporal components
       b:      (d1*d2[*d3]) X nb, initial value of background spatial component
       f:      nb X T, initial value of background temporal component
       bSiz:   int or tuple of int
        blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will
        be convolved with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0.
       maxIter: int,
           maximum iteration of iterating HALS.
       semi_nmf: bool,
           use semi-nmf (without nonnegative constraint on temporal traces) if True, 
           otherwise use nmf        
       update_bg: bool, 
           update background if True, otherwise background will be set zero
       use_spikes: bool, 
           if True the algorithm will detect spikes using VolPy offline 
           to optimize the spatial components A
       hals_orig: bool,
           if True the input matrix Y is from the original movie, otherwise the input matrix Y
           is thresholded at 0    
       fr: 
           frame rate of the movie           
           
    Returns:
        the updated A, C, b, f
    Authors:
        Johannes Friedrich, Andrea Giovannucci
    See Also:
        http://proceedings.mlr.press/v39/kimura14.pdf
    """
    # smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    nb = b.shape[1]  # number of background components
    if bSiz is not None:
        if isinstance(bSiz, (int, float)):
             bSiz = [bSiz] * len(dims)
        ind_A = nd.filters.uniform_filter(np.reshape(A,
                dims + (K,), order='F'), size=bSiz + [0])
        ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
    else:
        ind_A = A>1e-10
    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels
    def HALS4activity(Yr, A, C, iters=2, semi_nmf=False):
        U = A.T.dot(Yr)
        V = A.T.dot(A) + np.finfo(A.dtype).eps
        for _ in range(iters):
            for m in range(len(U)):  # neurons and background
                if semi_nmf:
                    print('use semi-nmf')
                    C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                               V[m, m], -np.inf, np.inf)
                else:
                    C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                               V[m, m], 0, np.inf)
        return C
    def HALS4shape(Yr, A, C, iters=2):
        U = C.dot(Yr.T)
        V = C.dot(C.T) + np.finfo(C.dtype).eps
        for _ in range(iters):
            for m in range(K):  # neurons
                ind_pixels = np.squeeze(ind_A[:, m].toarray())
                A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                           ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                            V[m, m]), 0, np.inf)
            for m in range(nb):  # background
                A[:, K + m] = np.clip(A[:, K + m] + ((U[K + m] - V[K + m].dot(A.T)) /
                                                     V[K + m, K + m]), 0, np.inf)
        return A
    Ab = np.c_[A, b]
    Cf = np.r_[C, f.reshape(nb, -1)]

    for thr_ in np.linspace(3.5,2.5,maxIter):    
        Cf = HALS4activity(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf, semi_nmf=semi_nmf)
        Cf_processed = Cf.copy()

        if not update_bg:
            Cf_processed[-nb:] = np.zeros(Cf_processed[-nb:].shape)

        if use_spikes:
            for i in range(Cf.shape[0]):
                if i < Cf.shape[0] - nb: 
                    if hals_orig:
                        bl = scipy.ndimage.percentile_filter(-Cf[i], 50, size=50)
                        tr = -Cf[i] - bl   
                        tr = tr-np.median(tr)
                        bl = bl+np.median(tr)
                    else:
                        bl = scipy.ndimage.percentile_filter(Cf[i], 50, size=50)
                        tr = Cf[i] - bl                    
                    
                    _, _, Cf_processed[i], _, _, _ = denoise_spikes(tr, window_length=3, clip=0, fr=fr,
                                      threshold=thr_, threshold_method='simple', do_plot=False)
                    
                    if hals_orig:
                        Cf_processed[i] = -Cf_processed[i] - bl    
                    else:
                        Cf_processed[i] = Cf_processed[i] + bl  
                    
        Cf = Cf_processed
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
        
    return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)

def nmf_sequential(y_seq, mask, seq, small_mask=True):
    """ Use rank-1 nmf to sequentially extract neurons' spatial filters.
    
    Parameters
    ----------
    y_seq : ndarray, T * (# of pixel)
        Movie after detrend. It should be in 2D format.
    mask : ndarray, T * d1 * d2
        Masks of neurons
    seq : ndarray
        order of rank-1 nmf on neurons.  
    small_mask : bool, optional
        Use small context region when doing rank-1 nmf. The default is True.

    Returns
    -------
    W : ndarray
        Temporal components of neurons.
    H : ndarray
        Spatial components of neuron.

    """
    W_tot = []
    H_tot = []    
    for i in seq:
        print(f'now processing neuron {i}')
        model = NMF(n_components=1, init='nndsvd', max_iter=100, verbose=False)
        y_temp, _ = select_masks(y_seq, (y_seq.shape[0], mask.shape[1], mask.shape[2]), mask=mask[i])
        if small_mask:
            mask_dilate = cv2.dilate(mask[i],np.ones((4,4),np.uint8),iterations = 1)
            x0 = np.where(mask_dilate>0)[0].min()
            x1 = np.where(mask_dilate>0)[0].max()
            y0 = np.where(mask_dilate>0)[1].min()
            y1 = np.where(mask_dilate>0)[1].max()
            context_region = np.zeros(mask_dilate.shape)
            context_region[x0:x1+1, y0:y1+1] = 1
            context_region = context_region.reshape([-1], order='F')
            y_temp_small = y_temp[:, context_region>0]
            W = model.fit_transform(np.maximum(y_temp_small,0))
            H_small = model.components_
            #plt.figure(); plt.imshow(H_small.reshape([x1-x0+1, y1-y0+1], order='F')); plt.colorbar(); plt.show()
            H = np.zeros((1, y_temp.shape[1]))
            H[:, context_region>0] = H_small
        else:
            W = model.fit_transform(np.maximum(y_temp,0))
            H = model.components_
        y_seq = y_seq - W@H
        W_tot.append(W)
        H_tot.append(H)
    H = np.vstack(H_tot)
    W = np.hstack(W_tot)
        
    return W, H

def select_masks(Y, shape, mask=None):
    """ Select a mask for nmf
    Args:
        Y: d1*d2 X T
            input data
        shape: tuple, (d1, d2)
            FOV size
        mask: ndarray with dimension d1 X d2
            if None, you will manually select contours of neuron
            if the mask is not None, it will be dilated
    Returns:
        updated mask and Y
    """
    
    if mask is None:
        m = cm.movie(Y).to_3D(shape=shape, order='F')
        frame = m[:10000].std(axis=0)
        plt.figure()
        plt.imshow(frame, cmap=mpl_cm.Greys_r)
        pts = []    
        while not len(pts):
            pts = plt.ginput(0)
            plt.close()
            path = mpl_path.Path(pts)
            mask = np.ones(np.shape(frame), dtype=bool)
            for ridx, row in enumerate(mask):
                for cidx, pt in enumerate(row):
                    if path.contains_point([cidx, ridx]):
                        mask[ridx, cidx] = False
        Y = cm.movie((1.0 - mask)*m).to_2D() 
    else:
        mask = cv2.dilate(mask,np.ones((4,4),np.uint8),iterations = 1)
        mask = (mask < 1)
        mask_2D = mask.reshape((mask.shape[0] * mask.shape[1]), order='F')
        Y = Y * (1.0 - mask_2D)
    # plt.figure();plt.plot(((m * (1.0 - mask)).mean(axis=(1, 2))))
    return Y, mask 
