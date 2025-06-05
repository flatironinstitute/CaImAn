#!/usr/bin/env python

"""
functions that creates image from a video file

Primarily intended for plotting, returns correlation images ( local or max )
"""

import cv2
import logging
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure
from scipy.sparse import coo_matrix
from typing import Any, Optional

import caiman
import caiman.base.movies
from caiman.source_extraction.cnmf.pre_processing import get_noise_fft

def max_correlation_image(Y, bin_size: int = 1000, eight_neighbours: bool = True, swap_dim: bool = True) -> np.ndarray:
    """Computes the max-correlation image for the input dataset Y with bin_size

    Args:
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
        Cn: d1 x d2 [x d3] matrix,
            max correlation image
    """
    logger = logging.getLogger("caiman")

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    T = Y.shape[0]
    if T <= bin_size:
        Cn_bins = local_correlations_fft(Y, eight_neighbours=eight_neighbours, swap_dim=False)
        return Cn_bins
    else:
        if T % bin_size < bin_size / 2.:
            bin_size = T // (T // bin_size)

        n_bins = T // bin_size
        Cn_bins = np.zeros(((n_bins,) + Y.shape[1:]))
        for i in range(n_bins):
            Cn_bins[i] = local_correlations_fft(Y[i * bin_size:(i + 1) * bin_size],
                                                eight_neighbours=eight_neighbours,
                                                swap_dim=False)
            logger.debug(i * bin_size)

        Cn = np.max(Cn_bins, axis=0)
        return Cn

def local_correlations_fft(Y,
                           eight_neighbours: bool = True,
                           swap_dim: bool = True,
                           opencv: bool = True,
                           rolling_window=None) -> np.ndarray:
    """Computes the correlation image for the input dataset Y using a faster FFT based method

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format
    
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively
    
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front
    
        opencv: Boolean
            If True process using OpenCV method

        rolling_window: (undocumented)

    Returns:
        Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    if rolling_window is None:
        Y -= np.mean(Y, axis=0)
        Ystd = np.std(Y, axis=0)
        Ystd[Ystd == 0] = np.inf
        Y /= Ystd
    else:
        Ysum = np.cumsum(Y, axis=0)
        Yrm = (Ysum[rolling_window:] - Ysum[:-rolling_window]) / rolling_window
        Y[:rolling_window] -= Yrm[0]
        Y[rolling_window:] -= Yrm
        del Yrm, Ysum
        Ystd = np.cumsum(Y**2, axis=0)
        Yrst = np.sqrt((Ystd[rolling_window:] - Ystd[:-rolling_window]) / rolling_window)
        Yrst[Yrst == 0] = np.inf
        Y[:rolling_window] /= Yrst[0]
        Y[rolling_window:] /= Yrst
        del Ystd, Yrst

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            # yapf: disable
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                          dtype='float32')
            # yapf: enable
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = np.stack([cv2.filter2D(img, -1, sz, borderType=0) for img in Y])
        MASK = cv2.filter2D(np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = convolve(np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')

    YYconv = Yconv * Y
    del Y, Yconv
    if rolling_window is None:
        Cn = np.mean(YYconv, axis=0) / MASK
    else:
        YYconv_cs = np.cumsum(YYconv, axis=0)
        del YYconv
        YYconv_rm = (YYconv_cs[rolling_window:] - YYconv_cs[:-rolling_window]) / rolling_window
        del YYconv_cs
        Cn = YYconv_rm / MASK

    return Cn


def local_correlations_multicolor(Y, swap_dim: bool = True) -> np.ndarray:
    """Computes the correlation image with color depending on orientation

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format

        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
        rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """
    if Y.ndim == 4:
        raise Exception('Not Implemented')

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)
    rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:,]), axis=0)
    rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:,]), axis=0)

    return np.dstack([rho_h[:, 1:] / 2, rho_d1 / 2, rho_d2 / 2])


def local_correlations(Y, eight_neighbours: bool = True, swap_dim: bool = True, order_mean=1) -> np.ndarray:
    """Computes the correlation image for the input dataset Y

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format
    
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively
    
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

        order_mean: (undocumented)

    Returns:
        rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """
    print("Testing corr func")
    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    rho = np.zeros(Y.shape[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    # yapf: disable
    if order_mean == 0:
        rho = np.ones(Y.shape[1:])
        rho_h = rho_h
        rho_w = rho_w
        rho[:-1, :] = rho[:-1, :] * rho_h
        rho[1:,  :] = rho[1:,  :] * rho_h
        rho[:, :-1] = rho[:, :-1] * rho_w
        rho[:,  1:] = rho[:,  1:] * rho_w
    else:
        rho[:-1, :] = rho[:-1, :] + rho_h**(order_mean)
        rho[1:,  :] = rho[1:,  :] + rho_h**(order_mean)
        rho[:, :-1] = rho[:, :-1] + rho_w**(order_mean)
        rho[:,  1:] = rho[:,  1:] + rho_w**(order_mean)

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
        rho[:, :, :-1] = rho[:, :, :-1] + rho_d
        rho[:, :, 1:] = rho[:, :, 1:] + rho_d

        neighbors = 6 * np.ones(Y.shape[1:])
        neighbors[0]        = neighbors[0]        - 1
        neighbors[-1]       = neighbors[-1]       - 1
        neighbors[:,     0] = neighbors[:,     0] - 1
        neighbors[:,    -1] = neighbors[:,    -1] - 1
        neighbors[:,  :, 0] = neighbors[:,  :, 0] - 1
        neighbors[:, :, -1] = neighbors[:, :, -1] - 1

    else:
        if eight_neighbours:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:,]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:,]), axis=0)

            if order_mean == 0:
                rho_d1 = rho_d1
                rho_d2 = rho_d2
                rho[:-1, :-1] = rho[:-1, :-1] * rho_d2
                rho[1:,   1:] = rho[1:,   1:] * rho_d2
                rho[1:,  :-1] = rho[1:,  :-1] * rho_d1
                rho[:-1,  1:] = rho[:-1,  1:] * rho_d1
            else:
                rho[:-1, :-1] = rho[:-1, :-1] + rho_d2**(order_mean)
                rho[1:,   1:] = rho[1:,   1:] + rho_d2**(order_mean)
                rho[1:,  :-1] = rho[1:,  :-1] + rho_d1**(order_mean)
                rho[:-1,  1:] = rho[:-1,  1:] + rho_d1**(order_mean)

            neighbors = 8 * np.ones(Y.shape[1:3])
            neighbors[0,   :] = neighbors[0,   :] - 3
            neighbors[-1,  :] = neighbors[-1,  :] - 3
            neighbors[:,   0] = neighbors[:,   0] - 3
            neighbors[:,  -1] = neighbors[:,  -1] - 3
            neighbors[0,   0] = neighbors[0,   0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1,  0] = neighbors[-1,  0] + 1
            neighbors[0,  -1] = neighbors[0,  -1] + 1
        else:
            neighbors = 4 * np.ones(Y.shape[1:3])
            neighbors[0,  :]  = neighbors[0,  :] - 1
            neighbors[-1, :]  = neighbors[-1, :] - 1
            neighbors[:,  0]  = neighbors[:,  0] - 1
            neighbors[:, -1]  = neighbors[:, -1] - 1

    # yapf: enable
    if order_mean == 0:
        rho = np.power(rho, 1. / neighbors)
    else:
        rho = np.power(np.divide(rho, neighbors), 1 / order_mean)

    return rho


def correlation_pnr(Y, gSig=None, center_psf: bool = True, swap_dim: bool = True,
                    background_filter: str = 'disk') -> tuple[np.ndarray, np.ndarray]:
    """
    compute the correlation image and the peak-to-noise ratio (PNR) image.
    If gSig is provided, then spatially filtered the video.

    Args:
        Y:  np.ndarray (3D or 4D).
            Input movie data in 3D or 4D format
        gSig:  scalar or vector.
            gaussian width. If gSig == None, no spatial filtering
        center_psf: Boolean
            True indicates subtracting the mean of the filtering kernel
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front
        background_filter: str
            (undocumented)

    Returns:
        cn: np.ndarray (2D or 3D).
            local correlation image of the spatially filtered (or not)
            data
        pnr: np.ndarray (2D or 3D).
            peak-to-noise ratios of all pixels/voxels

    """
    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    # parameters
    _, d1, d2 = Y.shape
    data_raw = Y.reshape(-1, d1, d2).astype('float32')

    # filter data
    data_filtered = data_raw.copy()
    if gSig:
        if not isinstance(gSig, list):
            gSig = [gSig, gSig]
        ksize = tuple([int(2 * i) * 2 + 1 for i in gSig])

        if center_psf:
            if background_filter == 'box':
                for idx, img in enumerate(data_filtered):
                    data_filtered[idx, ] = cv2.GaussianBlur(
                        img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1) \
                        - cv2.boxFilter(img, ddepth=-1, ksize=ksize, borderType=1)
            else:
                psf = cv2.getGaussianKernel(ksize[0], gSig[0],
                                            cv2.CV_32F).dot(cv2.getGaussianKernel(ksize[1], gSig[1], cv2.CV_32F).T)
                ind_nonzero = psf >= psf[0].max()
                psf -= psf[ind_nonzero].mean()
                psf[~ind_nonzero] = 0
                for idx, img in enumerate(data_filtered):
                    data_filtered[idx,] = cv2.filter2D(img, -1, psf, borderType=1)

            # data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)
        else:
            for idx, img in enumerate(data_filtered):
                data_filtered[idx,] = cv2.GaussianBlur(img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1)

    # compute peak-to-noise ratio
    data_filtered -= data_filtered.mean(axis=0)
    data_max = np.max(data_filtered, axis=0)
    data_std = get_noise_fft(data_filtered.T, noise_method='mean')[0].T
    pnr = np.divide(data_max, data_std)
    pnr[pnr < 0] = 0

    # remove small values
    tmp_data = data_filtered.copy() / data_std
    tmp_data[tmp_data < 3] = 0

    # compute correlation image
    cn = local_correlations_fft(tmp_data, swap_dim=False)

    return cn, pnr


def iter_chunk_array(arr: np.array, chunk_size: int):
    if ((arr.shape[0] // chunk_size) - 1) > 0:
        for i in range((arr.shape[0] // chunk_size) - 1):
            yield arr[chunk_size * i:chunk_size * (i + 1)]
        yield arr[chunk_size * (i + 1):]
    else:
        yield arr


def correlation_image_ecobost(mov, chunk_size: int = 1000, dview=None):
    """ Compute correlation image as Erick. Removes the mean from each chunk
    before computing the correlation
    Args:
        mov: ndarray or list of str
            time x w x h

    chunk_size: int
        number of frames over which to compute the correlation (not working if
        passing list of string)
    """
    # MAP
    if isinstance(mov, list):
        if dview is not None:
            res = dview.map(map_corr, mov)
        else:
            res = map(map_corr, mov)

    else:
        scan = mov.astype(np.float32)
        num_frames = scan.shape[0]
        res = map(map_corr, iter_chunk_array(scan, chunk_size))

    sum_x, sum_sqx, sum_xy, num_frames = [np.sum(np.array(a), 0) for a in zip(*res)]
    denom_factor = np.sqrt(num_frames * sum_sqx - sum_x**2)
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
        rotated_corrs[1:, 1:, 4 + k] = (num_frames * rotated_sum_xy[1:, 1:, 4 + k]
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


def map_corr(scan) -> tuple[Any, Any, Any, int]:
    '''This part of the code is in a mapping function that's run over different
    movies in parallel
    '''
    # TODO: Tighten prototype above
    if isinstance(scan, str):
        scan = caiman.load(scan)

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
        rotated_xysum[1:, :, k] = np.sum(rotated_chunk[1:] * rotated_chunk[:-1], axis=-1, dtype=float)
        rotated_xysum[1:, 1:, 4 + k] = np.sum(rotated_chunk[1:, 1:] * rotated_chunk[:-1, :-1], axis=-1, dtype=float)

        # Return back to original orientation
        chunk = np.rot90(rotated_chunk, k=4 - k)
        chunk_xysum = np.rot90(rotated_xysum, k=4 - k)

    num_frames = chunk.shape[-1]

    return chunk_sum, chunk_sqsum, chunk_xysum, num_frames


def prepare_local_correlations(Y, swap_dim: bool = False,
                               eight_neighbours: bool = False) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Computes the correlation image and some statistics to update it online

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format

        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data
            Use 18 neighbors if true, and 6 if false for 4D data

    """
    # TODO: Tighten prototype above
    if swap_dim:
        Y = np.transpose(Y, (Y.ndim - 1,) + tuple(range(Y.ndim - 1)))

    T = len(Y)
    dims = Y.shape[1:]
    Yr = Y.T.reshape(-1, T)
    if Y.ndim == 4:
        d1, d2, d3 = dims
        sz = generate_binary_structure(3, 2 if eight_neighbours else 1)
        sz[1, 1, 1] = 0
    else:
        d1, d2 = dims
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='uint8')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='uint8')

    idx = [i - 1 for i in np.nonzero(sz)]

    def get_indices_of_neighbors(pixel):
        pixel = np.unravel_index(pixel, dims, order='F')
        x = pixel[0] + idx[0]
        y = pixel[1] + idx[1]
        if len(dims) == 3:
            z = pixel[2] + idx[2]
            inside = (x >= 0) * (x < d1) * (y >= 0) * (y < d2) * (z >= 0) * (z < d3)
            return np.ravel_multi_index((x[inside], y[inside], z[inside]), dims, order='F')
        else:
            inside = (x >= 0) * (x < d1) * (y >= 0) * (y < d2)
            return np.ravel_multi_index((x[inside], y[inside]), dims, order='F')

    N = [get_indices_of_neighbors(p) for p in range(np.prod(dims))]
    col_ind = np.concatenate(N)
    row_ind = np.concatenate([[i] * len(k) for i, k in enumerate(N)])
    num_neigbors = np.concatenate([[len(k)] * len(k) for k in N]).astype(Yr.dtype)

    first_moment = Yr.mean(1)
    second_moment = (Yr**2).mean(1)
    crosscorr = np.mean(Yr[row_ind] * Yr[col_ind], 1)
    sig = np.sqrt(second_moment - first_moment**2)

    M = coo_matrix(
        ((crosscorr - first_moment[row_ind] * first_moment[col_ind]) / (sig[row_ind] * sig[col_ind]) / num_neigbors,
         (row_ind, col_ind)),
        dtype=Yr.dtype)
    cn = M.dot(np.ones(M.shape[1], dtype=M.dtype)).reshape(dims, order='F')

    return first_moment, second_moment, crosscorr, col_ind, row_ind, num_neigbors, M, cn


def update_local_correlations(t,
                              frames,
                              first_moment,
                              second_moment,
                              crosscorr,
                              col_ind,
                              row_ind,
                              num_neigbors,
                              M,
                              del_frames=None) -> np.ndarray:
    """Updates sufficient statistics in place and returns correlation image"""
    dims = frames.shape[1:]
    stride = len(frames)
    if stride:
        frames = frames.reshape((stride, -1), order='F')
        if del_frames is None:
            tmp = 1 - float(stride) / t
            first_moment *= tmp
            second_moment *= tmp
            crosscorr *= tmp
        else:
            if stride > 10:
                del_frames = del_frames.reshape((stride, -1), order='F')
                first_moment -= del_frames.sum(0) / t
                second_moment -= (del_frames**2).sum(0) / t
                crosscorr -= np.sum(del_frames[:, row_ind] * del_frames[:, col_ind], 0) / t
            else:      # loop is faster
                for f in del_frames:
                    f = f.ravel(order='F')
                    first_moment -= f / t
                    second_moment -= (f**2) / t
                    crosscorr -= (f[row_ind] * f[col_ind]) / t
        if stride > 10:
            frames = frames.reshape((stride, -1), order='F')
            first_moment += frames.sum(0) / t
            second_moment += (frames**2).sum(0) / t
            crosscorr += np.sum(frames[:, row_ind] * frames[:, col_ind], 0) / t
        else:          # loop is faster
            for f in frames:
                f = f.ravel(order='F')
                first_moment += f / t
                second_moment += (f**2) / t
                crosscorr += (f[row_ind] * f[col_ind]) / t

    sig = np.sqrt(second_moment - first_moment**2)
    M.data = ((crosscorr - first_moment[row_ind] * first_moment[col_ind]) / (sig[row_ind] * sig[col_ind]) /
              num_neigbors)
    cn = M.dot(np.ones(M.shape[1], dtype=M.dtype)).reshape(dims, order='F')
    return cn


def local_correlations_movie(file_name,
                             tot_frames: Optional[int] = None,
                             fr: int = 30,
                             window: int = 30,
                             stride: int = 1,
                             swap_dim: bool = False,
                             eight_neighbours: bool = True,
                             mode: str = 'simple'):
    """
    Compute an online correlation image as moving average

    Args:
        Y:  string or np.ndarray (3D or 4D).
            Input movie filename or data
        tot_frames: int
            Number of frames considered
        fr: int
            Frame rate
        window: int
            Window length in frames
        stride: int
            Stride length in frames
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front
        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data
            Use 18 neighbors if true, and 6 if false for 4D data
        mode: 'simple', 'exponential', or 'cumulative'
            Mode of moving average

    Returns:
        corr_movie: caiman.movie (3D or 4D).
            local correlation movie

    """
    Y = caiman.load(file_name) if isinstance(file_name, str) else file_name
    Y = Y[..., :tot_frames] if swap_dim else Y[:tot_frames]
    first_moment, second_moment, crosscorr, col_ind, row_ind, num_neigbors, M, cn = \
        prepare_local_correlations(Y[..., :window] if swap_dim else Y[:window],
                                   swap_dim=swap_dim, eight_neighbours=eight_neighbours)
    if swap_dim:
        Y = np.transpose(Y, (Y.ndim - 1,) + tuple(range(Y.ndim - 1)))
    T = len(Y)
    dims = Y.shape[1:]
    corr_movie = np.zeros(((T - window) // stride + 1,) + dims, dtype=Y.dtype)
    corr_movie[0] = cn
    if mode == 'simple':
        for tt in range((T - window) // stride):
            corr_movie[tt + 1] = update_local_correlations(window, Y[tt * stride + window:(tt + 1) * stride + window],
                                                           first_moment, second_moment, crosscorr, col_ind, row_ind,
                                                           num_neigbors, M, cn, Y[tt * stride:(tt + 1) * stride]) # FIXME all params after M are invalid
    elif mode == 'exponential':
        for tt, frames in enumerate(Y[window:window + (T - window) // stride * stride].reshape((-1, stride) + dims)):
            corr_movie[tt + 1] = update_local_correlations(window, frames, first_moment, second_moment, crosscorr,
                                                           col_ind, row_ind, num_neigbors, M)
    elif mode == 'cumulative':
        for tt, frames in enumerate(Y[window:window + (T - window) // stride * stride].reshape((-1, stride) + dims)):
            corr_movie[tt + 1] = update_local_correlations(tt + window + 1, frames, first_moment, second_moment,
                                                           crosscorr, col_ind, row_ind, num_neigbors, M)
    else:
        raise Exception('mode of the moving average must be simple, exponential or cumulative')
    return caiman.movie(corr_movie, fr=fr)


def local_correlations_movie_offline(file_name,
                                     Tot_frames=None,
                                     fr: float = 10.,
                                     window: int = 100,
                                     stride: int = 100,
                                     swap_dim: bool = False,
                                     eight_neighbours: bool = True,
                                     order_mean: int = 1,
                                     ismulticolor: bool = False,
                                     dview=None,
                                     remove_baseline: bool = False,
                                     winSize_baseline: int = 50,
                                     quantil_min_baseline: float = 8,
                                     gaussian_blur: bool=False):
    """
    Efficient (parallel) computation of correlation image in shifting windows 
    with option for prior baseline removal

    Args:
        Y:  str
            path to movie file

        Tot_frames: int
            Number of total frames considered

        fr: int (100)
            Frame rate (optional)

        window: int (100)
            Window length in frames

        stride: int (30)
            Stride length in frames

        swap_dim: bool (False)
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front (default: False)

        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data
            Use 18 neighbors if true, and 6 if false for 4D data

        dview: map object
            Use it for parallel computation

        remove_baseline: bool (False)
            Flag for removing baseline prior to computation of CI

        winSize_baseline: int (50)
            Running window length for computing baseline

        quantile_min_baseline: float (8)
            Percentile used for baseline computations
            
        gaussian_blur: bool (False)
            Gaussian smooth the signal

    Returns:
        mm: caiman.movie (3D or 4D).
            local correlation movie

    """
    if Tot_frames is None:
        _, Tot_frames = caiman.base.movies.get_file_size(file_name)

    params:list = [[file_name, range(j, j + window), eight_neighbours, swap_dim,
                     order_mean, ismulticolor, remove_baseline, winSize_baseline,
                     quantil_min_baseline, gaussian_blur]
                    for j in range(0, Tot_frames - window, stride)]

    params.append([file_name, range(Tot_frames - window, Tot_frames), eight_neighbours, swap_dim,
                   order_mean, ismulticolor, remove_baseline, winSize_baseline,
                   quantil_min_baseline, gaussian_blur])

    if dview is None:
        parallel_result = list(map(local_correlations_movie_parallel, params))
    else:
        #TODO phrase better
        if 'multiprocessing' in str(type(dview)):
            parallel_result = dview.map_async(local_correlations_movie_parallel, params).get(4294967)
        else:
            parallel_result = dview.map_sync(local_correlations_movie_parallel, params)
            dview.results.clear()

    mm = caiman.movie(np.concatenate(parallel_result, axis=0), fr=fr/len(parallel_result))
    return mm


def local_correlations_movie_parallel(params:tuple) -> np.ndarray:
    mv_name, idx, eight_neighbours, swap_dim, order_mean, ismulticolor, remove_baseline, winSize_baseline, quantil_min_baseline, gaussian_blur = params
    mv = caiman.load(mv_name, subindices=idx, in_memory=True)
    if gaussian_blur:
        mv = mv.gaussian_blur_2D()

    if remove_baseline:
        mv.removeBL(quantilMin=quantil_min_baseline, windowSize=winSize_baseline, in_place=True)

    if ismulticolor:
        return local_correlations_multicolor(mv, swap_dim=swap_dim)[None, :, :].astype(np.float32)
    else:
        return local_correlations(mv, eight_neighbours=eight_neighbours, swap_dim=swap_dim,
                                  order_mean=order_mean)[None, :, :].astype(np.float32)
        
def mean_image(file_name,
                 Tot_frames=None,
                 fr: float = 10.,
                 window: int = 100,
                 dview=None):
    """
    Efficient (parallel) computation of mean image in chunks

    Args:
        Y:  str
            path to movie file

        Tot_frames: int
            Number of total frames considered

        fr: int (100)
            Frame rate (optional)

        window: int (100)
            Window length in frames

        dview: map object
            Use it for parallel computation
    
    Returns:
        mm: caiman.movie (2D).
            mean image

    """
    if Tot_frames is None:
        _, Tot_frames = caiman.base.movies.get_file_size(file_name)

    params:list = [[file_name, range(j * window, (j + 1) * window)]
                    for j in range(int(Tot_frames / window))]

    remain_frames = Tot_frames - int(Tot_frames / window) * window
    if remain_frames > 0:
        params.append([file_name, range(int(Tot_frames / window) * window, Tot_frames)])

    if dview is None:
        parallel_result = list(map(mean_image_parallel, params))
    else:
        if 'multiprocessing' in str(type(dview)):
            parallel_result = dview.map_async(mean_image_parallel, params).get(4294967)
        else:
            parallel_result = dview.map_sync(mean_image_parallel, params)
            dview.results.clear()

    mm = caiman.movie(np.concatenate(parallel_result, axis=0), fr=fr/len(parallel_result))
    if remain_frames > 0:
        mean_image = (mm[:-1].sum(axis=0) + (remain_frames / window) * mm[-1]) / (len(mm) - 1 + remain_frames / window)  
    else:
        mean_image = mm.mean(axis=0)
    return mean_image

def mean_image_parallel(params:tuple) -> np.ndarray:
    mv_name, idx = params
    mv = caiman.load(mv_name, subindices=idx, in_memory=True)
    return mv.mean(axis=0)[np.newaxis,:,:]
