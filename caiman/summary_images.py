#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" functions that creates image from a video file

for plotting purposes mainly, return correlation images ( local or max )

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
import numpy as np
from scipy.ndimage.filters import convolve
import cv2
import itertools
from caiman.source_extraction.cnmf.pre_processing import get_noise_fft
#%%

def max_correlation_image(Y, bin_size=1000, eight_neighbours=True, swap_dim=True):
    """Computes the max-correlation image for the input dataset Y with bin_size

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
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    T = Y.shape[0]
    if T <= bin_size:
        Cn_bins = local_correlations_fft(
            Y, eight_neighbours=eight_neighbours, swap_dim=False)
        return Cn_bins
    else:
        if T % bin_size < bin_size / 2.:
            bin_size = T // (T // bin_size)

        n_bins = T // bin_size
        Cn_bins = np.zeros(((n_bins,) + Y.shape[1:]))
        for i in range(n_bins):
            Cn_bins[i] = local_correlations_fft(Y[i * bin_size:(i + 1) * bin_size],
                                                eight_neighbours=eight_neighbours, swap_dim=False)
            print(i * bin_size)

        Cn = np.max(Cn_bins, axis=0)
        return Cn


#%%
def local_correlations_fft(Y, eight_neighbours=True, swap_dim=True, opencv=True):
    """Computes the correlation image for the input dataset Y using a faster FFT based method

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

    opencv: Boolean
        If True process using open cv method

    Returns:
    --------

    Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

    """

    if swap_dim:
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    Y -= np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ystd[Ystd == 0] = np.inf
    Y /= Ystd

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype='float32')
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = Y.copy()
        for idx, img in enumerate(Yconv):
            Yconv[idx] = cv2.filter2D(img, -1, sz, borderType=0)
        MASK = cv2.filter2D(
            np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = convolve(
            np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')
    Cn = np.mean(Yconv * Y, axis=0) / MASK
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
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    rho = np.zeros(np.shape(Y)[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    rho[:-1, :] = rho[:-1, :] + rho_h
    rho[1:, :] = rho[1:, :] + rho_h
    rho[:, :-1] = rho[:, :-1] + rho_w
    rho[:, 1:] = rho[:, 1:] + rho_w

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(
            w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
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
            rho_d1 = np.mean(np.multiply(
                w_mov[:, 1:, :-1], w_mov[:, :-1, 1:, ]), axis=0)
            rho_d2 = np.mean(np.multiply(
                w_mov[:, :-1, :-1], w_mov[:, 1:, 1:, ]), axis=0)
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


def correlation_pnr(Y, gSig=None, center_psf=True, swap_dim=True):
    """
    compute the correlation image and the peak-to-noise ratio (PNR) image.
    If gSig is provided, then spatially filtered the video.

    Args:
        Y:  np.ndarray (3D or 4D).
            Input movie data in 3D or 4D format
        gSig:  scalar or vector.
            gaussian width. If gSig == None, no spatial filtering
        center_psf: Boolearn
            True indicates subtracting the mean of the filtering kernel
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
        cn: np.ndarray (2D or 3D).
            local correlation image of the spatially filtered (or not)
            data
        pnr: np.ndarray (2D or 3D).
            peak-to-noise ratios of all pixels/voxels

    """
    if swap_dim:
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    # parameters
    _, d1, d2 = Y.shape
    data_raw = Y.reshape(-1, d1, d2).astype('float32')

    # filter data
    data_filtered = data_raw.copy()
    if gSig:
        if not isinstance(gSig, list):
            gSig = [gSig, gSig]
        ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig])
        # create a spatial filter for removing background
        # psf = gen_filter_kernel(width=ksize, sigma=gSig, center=center_psf)

        if center_psf:
            for idx, img in enumerate(data_filtered):
                data_filtered[idx, ] = cv2.GaussianBlur(img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1) \
                    - cv2.boxFilter(img, ddepth=-1, ksize=ksize, borderType=1)
            # data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)
        else:
            for idx, img in enumerate(data_filtered):
                data_filtered[idx, ] = cv2.GaussianBlur(
                    img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1)

    # compute peak-to-noise ratio
    data_filtered -= np.mean(data_filtered, axis=0)
    data_max = np.max(data_filtered, axis=0)
    data_std = get_noise_fft(data_filtered.transpose())[0].transpose()
    # data_std = get_noise(data_filtered, method='diff2_med')
    pnr = np.divide(data_max, data_std)
    pnr[pnr < 0] = 0

    # remove small values
    tmp_data = data_filtered.copy() / data_std
    tmp_data[tmp_data < 3] = 0

    # compute correlation image
    # cn = local_correlation(tmp_data, d1=d1, d2=d2)
    cn = local_correlations_fft(tmp_data, swap_dim=False)

    return cn, pnr


def iter_chunk_array(arr, chunk_size):
    if ((arr.shape[0] // chunk_size)-1) > 0:
        for i in range((arr.shape[0] // chunk_size)-1):
            yield arr[chunk_size*i:chunk_size*(i+1)]
        yield arr[chunk_size*(i+1):]
    else:
        yield arr


def correlation_image_ecobost(mov, chunk_size=1000, dview=None):
    """ Compute correlation image as Erick. Removes the mean from each chunk
    before computing the correlation
    Params:
    -------
    mov: ndarray or list of str
        time x w x h

    chunk_size: int
        number of frames over which to compute the correlation (not working if
        passing list of string)

    """
    # MAP
    if type(mov) is list:
        if dview is not None:
            res = dview.map(map_corr, mov)
        else:
            res = map(map_corr, mov)

    else:
        scan = mov.astype(np.float32)
        num_frames = scan.shape[0]
        res = map(map_corr, iter_chunk_array(scan, chunk_size))

    sum_x, sum_sqx, sum_xy, num_frames = [np.sum(np.array(a), 0)
                                          for a in zip(*res)]
    # REDUCE
    # sum_x = chunk_sum # h x w
    # sum_sqx = chunk_sqsum # h x w
    # sum_xy = chunk_xysum # h x w x 8
    denom_factor = np.sqrt(num_frames * sum_sqx - sum_x ** 2)
    corrs = np.zeros(sum_xy.shape)
    for k in [0, 1, 2, 3]:
        rotated_corrs = np.rot90(corrs, k=k)
        rotated_sum_x = np.rot90(sum_x, k=k)
        rotated_dfactor = np.rot90(denom_factor, k=k)
        rotated_sum_xy = np.rot90(sum_xy, k=k)

        # Compute correlation
        rotated_corrs[1:, :, k] = (num_frames * rotated_sum_xy[1:, :, k] -
                                   rotated_sum_x[1:] * rotated_sum_x[:-1]) /\
                                  (rotated_dfactor[1:] * rotated_dfactor[:-1])
        rotated_corrs[1:, 1:, 4 + k] = (num_frames *rotated_sum_xy[1:, 1:, 4 + k]
                                       - rotated_sum_x[1:, 1:] * rotated_sum_x[:-1, : -1]) /\
                                       (rotated_dfactor[1:, 1:] * rotated_dfactor[:-1, :-1])

        # Return back to original orientation
        corrs = np.rot90(rotated_corrs, k=4 - k)
        sum_x = np.rot90(rotated_sum_x, k=4 - k)
        denom_factor = np.rot90(rotated_dfactor, k=4 - k)
        sum_xy = np.rot90(rotated_sum_xy, k=4 - k)

    correlation_image = np.sum(corrs, axis=-1)
    # edges
    norm_factor = 5 * np.ones(correlation_image.shape)
    # corners
    norm_factor[[0, -1, 0, -1], [0, -1, -1, 0]] = 3
    # center
    norm_factor[1:-1, 1:-1] = 8
    correlation_image /= norm_factor

    return correlation_image


def map_corr(scan):
    '''This part of the code is in a mapping function that's run over different
    movies in parallel
    '''
    import caiman as cm
    if type(scan) is str:
        scan = cm.load(scan)

    # h x w x num_frames
    chunk = np.array(scan).transpose([1, 2, 0])
    # Subtract overall brightness per frame
    chunk -= chunk.mean(axis=(0, 1))

    # Compute sum_x and sum_x^2
    chunk_sum = np.sum(chunk, axis=-1, dtype=float)
    chunk_sqsum = np.sum(chunk**2, axis=-1, dtype=float)

    # Compute sum_xy: Multiply each pixel by its eight neighbors
    chunk_xysum = np.zeros((chunk.shape[0], chunk.shape[1], 8))
    # amount of 90 degree rotations
    for k in [0, 1, 2, 3]:
        rotated_chunk = np.rot90(chunk, k=k)
        rotated_xysum = np.rot90(chunk_xysum, k=k)

        # Multiply each pixel by one above and by one above to the left
        rotated_xysum[1:, :, k] = np.sum(rotated_chunk[1:] * rotated_chunk[:-1],
                                         axis=-1, dtype=float)
        rotated_xysum[1:, 1:, 4 + k] = np.sum(rotated_chunk[1:, 1:] *
                                              rotated_chunk[:-1, :-1], axis=-1, dtype=float)

        # Return back to original orientation
        chunk = np.rot90(rotated_chunk, k=4 - k)
        chunk_xysum = np.rot90(rotated_xysum, k=4 - k)

    num_frames = chunk.shape[-1]

    return chunk_sum, chunk_sqsum, chunk_xysum, num_frames
