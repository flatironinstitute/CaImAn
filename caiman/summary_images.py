# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:41:21 2016

@author: agiovann
"""
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
from scipy.ndimage.filters import correlate, convolve
import cv2
from caiman.source_extraction.cnmf.pre_processing import get_noise_fft, get_noise_fft_parallel
import pdb
#%%


def max_correlation_image(Y,bin_size = 1000, eight_neighbours = True, swap_dim = True):
    """Computes the max-correlation image for the input dataset Y  with bin_size

    Parameters
    -----------

    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format
    bin_size: scalar (integer)
         Length of bin_size (if last bin is smaller than bin_size < 2 bin_size is increased to impose uniform bins)
    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively
    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front

    Returns
    --------

    Cn: d1 x d2 [x d3] matrix, max correlation image

    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))
    
    T = Y.shape[0]
    if T <= bin_size:
        Cn_bins = local_correlations_fft(Y,eight_neighbours=eight_neighbours,swap_dim = False)
        return Cn_bins
    else:
        if T%bin_size < bin_size/2.:
            bin_size = T//(T//bin_size)
    
        n_bins = T//bin_size
        Cn_bins = np.zeros(((n_bins,)+Y.shape[1:]))
        for i in range(n_bins):
            Cn_bins[i] = local_correlations_fft(Y[i*bin_size:(i+1)*bin_size],eight_neighbours=eight_neighbours,swap_dim=False)
            print(i*bin_size)
    
        Cn = np.max(Cn_bins,axis=0)
        return Cn

def local_correlations_fft(Y, eight_neighbours=True, swap_dim=True):
    """Computes the correlation image for the input dataset Y  using a faster FFT based method

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

    Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

    """
    
    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))
    
    Y = Y.astype('float32')
    Y -= np.mean(Y,axis = 0)
    Y /= np.std(Y,axis = 0)
                        
    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3,3,3),dtype='float32')
            sz[1,1,1] = 0
        else:
            sz = np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,0,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]],dtype='float32')
    else:
        if eight_neighbours:
            sz = np.ones((3,3),dtype='float32')
            sz[1,1] = 0
        else:
            sz = np.array([[0,1,0],[1,0,1],[0,1,0]],dtype='float32')
        
    Yconv = convolve(Y,sz[np.newaxis,:],mode='constant')
    MASK = convolve(np.ones(Y.shape[1:],dtype='float32'),sz,mode='constant')
    Cn =  np.mean(Yconv*Y,axis=0)/MASK                       
        
    return Cn
            
    
    
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
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    rho = np.zeros(np.shape(Y)[1:])
    w_mov = old_div((Y - np.mean(Y, axis=0)), np.std(Y, axis=0))

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


def local_correlation(video_data, sz=None, d1=None, d2=None,
                      normalized=False, chunk_size=3000):
    """
    compute location correlations of the video data
    Args:
        video_data: T*d1*d2 3d array  or T*(d1*d2) matrix
        sz: method for computing location correlation {4, 8, [dmin, dmax]}
            4: use the 4 nearest neighbors
            8: use the 8 nearest neighbors
            [dmin, dmax]: use neighboring pixels with distances in [dmin, dmax]
        d1: row number
        d2: column number
        normalized: boolean
            if True: avoid the step of normalizing data
        chunk_size: integer
            divide long data into small chunks for faster running time

    Returns:
        d1*d2 matrix, the desired correlation image

    """
    total_frames = video_data.shape[0]
    if total_frames > chunk_size:
        # too many frames, compute correlation images in chunk mode
        n_chunk = np.floor(total_frames / chunk_size)

        cn = np.zeros(shape=(n_chunk, d1, d2))
        for idx in np.arange(n_chunk):
            cn[idx,] = local_correlation(
                video_data[chunk_size * idx + np.arange(
                    chunk_size),], sz, d1, d2, normalized)
        return np.max(cn, axis=0)

    # reshape data
    data = video_data.copy().astype('float32')

    if data.ndim == 2:
        data = data.reshape(total_frames, d1, d2)
    else:
        _, d1, d2 = data.shape

    # normalize data
    if not normalized:
        data -= np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = np.inf
        data /= data_std

    # construct a matrix indicating the locations of neighbors
    if (not sz) or (sz == 8):
        mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    elif sz == 4:
        mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    elif len(sz) == 2:
        sz = np.array(sz)
        temp = np.arange(-sz.max(), sz.max() + 1).reshape(2 * sz.max() + 1, 0)
        tmp_dist = np.sqrt(temp ** 2 + temp.transpose() ** 2)
        mask = (tmp_dist >= sz.min()) & (tmp_dist < sz.max())
    else:
        mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # compute local correlations
    data_filter = data.copy().astype('float32')
    for idx, img in enumerate(data_filter):
        data_filter[idx] = cv2.filter2D(img, -1, mask, borderType=0)

    return np.divide(np.mean(data_filter * data, axis=0), cv2.filter2D(
        np.ones(shape=(d1, d2)), -1, mask, borderType=1))


def correlation_pnr_filtered(data, gSig=4, gSiz=15, center_psf=True):
    """
    compute the correlation image and the peak-to-noise ratio (PNR) image

    Args:
        data: 2d or 3d numpy array.
            video data
        options: C-like struct variable
            it requires at least 5 fields: d1, d2, gSiz, gSig, thresh_init

    Returns:
        cn: d1*d2 matrix
            local correlation image
        pnr: d1*d2 matrix
            PNR image
        psf: gSiz*gSiz matrix
            kernel used for filtering data

    """

    # parameters
    sig = 3
    T, d1, d2 = data.shape
    data_raw = data.reshape(-1, d1, d2).astype('float32')

    # create a spatial filter for removing background
    psf = gen_filter_kernel(width=gSiz, sigma=gSig, center=center_psf)

    # filter data
    data_filtered = data_raw.copy()
    for idx, img in enumerate(data_filtered):
        data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)

    # compute peak-to-noise ratio
    data_filtered -= np.median(data_filtered, axis=0)
    data_max = np.max(data_filtered, axis=0)
    data_std = get_noise_fft(data_filtered.transpose())[0].transpose()
    # data_std = get_noise(data_filtered, method='diff2_med')
    pnr = np.divide(data_max, data_std)
    pnr[pnr < 0] = 0

    # remove small values
    tmp_data = data_filtered.copy() / data_std
    tmp_data[tmp_data < sig] = 0

    # compute correlation image
    cn = local_correlation(tmp_data, d1=d1, d2=d2)

    # return
    return cn, pnr, psf


def gen_filter_kernel(width=16, sigma=4, center=True):
    """
    create a gaussian kernel for spatially filtering the raw data

    Args:
        width: (float)
            width of the kernel
        sigma: (float)
            gaussian width of the kernel
        center:
            if True, subtract the mean of gaussian kernel

    Returns:
        psf: (2D numpy array, width x width)
            the desired kernel

    """
    rmax = (width - 1) / 2.0
    y, x = np.ogrid[-rmax:(rmax + 1), -rmax:(rmax + 1)]
    psf = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    psf = psf / psf.sum()
    if center:
        idx = (psf >= psf[0].max())
        psf[idx] -= psf[idx].mean()
        psf[~idx] = 0

    return psf
