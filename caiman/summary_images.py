# -*- coding: utf-8 -*-
""" functions that creates image from a video file

for plotting purposes mainly, reutrn correlation images ( local or max )

See Also:
------------

@url
.. image::
@author andrea giovannucci
"""
# \package caiman
# \version   1.0
# \copyright GNU General Public License v2.0
# \date Created on Thu Oct 20 11:41:21 2016


from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
from scipy.ndimage.filters import correlate, convolve
#%%


def max_correlation_image(Y,bin_size = 1000, eight_neighbours = True, swap_dim = True):
    """Computes the max-correlation image for the input dataset Y  with bin_size

    Parameters:
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

    Returns:
    --------

    Cn: d1 x d2 [x d3] matrix,
        max correlation image

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
            Cn_bins[i] = local_correlations_fft(Y[i*bin_size:(i+1)*bin_size],
                                                eight_neighbours=eight_neighbours,swap_dim=False)
            print(i*bin_size)
    
        Cn = np.max(Cn_bins,axis=0)
        return Cn

def local_correlations_fft(Y, eight_neighbours=True, swap_dim=True):
    """Computes the correlation image for the input dataset Y  using a faster FFT based method

    Parameters:
    -----------

    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format

    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively

    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front

    Returns:
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

    Parameters:
    -----------

    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format

    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively

    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front

    Returns:
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
