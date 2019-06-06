#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A set of utilities, mostly for post-processing and visualization

We put arrays on disk as raw bytes, extending along the first dimension.
Alongside each array x we ensure the value x.dtype which stores the string
description of the array's dtype.

See Also:
------------

@url
.. image::
@author  epnev
"""
# \package caiman/dource_ectraction/cnmf
# \version   1.0
# \copyright GNU General Public License v2.0
# \date Created on Sat Sep 12 15:52:53 2015

from builtins import str
from builtins import range
from past.utils import old_div

import cv2
import h5py
import logging
import numpy as np
import os
import pylab as pl
import scipy
from scipy.sparse import spdiags, issparse, csc_matrix, csr_matrix
import scipy.ndimage.morphology as morph
from skimage.feature.peak import _get_high_intensity_peaks
import tifffile
from typing import List

from .initialization import greedyROI
from ...base.rois import com
from ...mmapping import parallel_dot_product, load_memmap
from ...utils.stats import df_percentile


def decimation_matrix(dims, sub):
    D = np.prod(dims)
    if sub == 2 and D <= 10000:  # faster for small matrices
        ind = np.arange(D) // 2 - \
            np.arange(dims[0], dims[0] + D) // (dims[0] * 2) * (dims[0] // 2) - \
            (dims[0] % 2) * (np.arange(D) % (2 * dims[0]) > dims[0]) * (np.arange(1, 1 + D) % 2)
    else:
        def create_decimation_matrix_bruteforce(dims, sub):
            dims_ds = tuple(1 + (np.array(dims) - 1) // sub)
            d_ds = np.prod(dims_ds)
            ds_matrix = np.eye(d_ds)
            ds_matrix = np.repeat(np.repeat(
                ds_matrix.reshape((d_ds,) + dims_ds, order='F'), sub, 1),
                sub, 2)[:, :dims[0], :dims[1]].reshape((d_ds, -1), order='F')
            ds_matrix /= ds_matrix.sum(1)[:, None]
            ds_matrix = csc_matrix(ds_matrix, dtype=np.float32)
            return ds_matrix
        tmp = create_decimation_matrix_bruteforce((dims[0], sub), sub).indices
        ind = np.concatenate([tmp] * (dims[1] // sub + 1))[:D] + \
            np.arange(D) // (dims[0] * sub) * ((dims[0] - 1) // sub + 1)
    data = 1. / np.unique(ind, return_counts=True)[1][ind]
    return csc_matrix((data, ind, np.arange(1 + D)), dtype=np.float32)


def peak_local_max(image, min_distance=1, threshold_abs=None,
                   threshold_rel=None, exclude_border=True, indices=True,
                   num_peaks=np.inf, footprint=None):
    """Find peaks in an image as coordinate list or boolean mask.

    Adapted from skimage to use opencv for speed.
    Replaced scipy.ndimage.maximum_filter by cv2.dilate.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    If peaks are flat (i.e. multiple adjacent pixels have identical
    intensities), the coordinates of all such pixels are returned.

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int, optional
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float, optional
        Minimum intensity of peaks, calculated as `max(image) * threshold_rel`.
    exclude_border : int, optional
        If nonzero, `exclude_border` excludes peaks from
        within `exclude_border`-pixels of the border of the image.
    indices : bool, optional
        If True, the output will be an array representing peak
        coordinates.  If False, the output will be a boolean array shaped as
        `image.shape` with peaks present at True elements.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance` (also for `exclude_border`).

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison of the dilated
    and original image, this function returns the coordinates or a mask of the
    peaks where the dilated image equals the original image.

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1.5,  0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 4],
           [3, 2]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_local_max(img2, exclude_border=0)
    array([[10, 10, 10]])

    """
    if type(exclude_border) == bool:
        exclude_border = min_distance if exclude_border else 0

    out = np.zeros_like(image, dtype=np.bool)

    if np.all(image == image.flat[0]):
        if indices is True:
            return np.empty((0, 2), np.int)
        else:
            return out

    # Non maximum filter
    if footprint is not None:
        # image_max = ndi.maximum_filter(image, footprint=footprint,
        #                                mode='constant')
        image_max = cv2.dilate(image, footprint=footprint, iterations=1)
    else:
        size = 2 * min_distance + 1
        # image_max = ndi.maximum_filter(image, size=size, mode='constant')
        image_max = cv2.dilate(image, cv2.getStructuringElement(
            cv2.MORPH_RECT, (size, size)), iterations=1)
    mask = image == image_max

    if exclude_border:
        # zero out the image borders
        for i in range(mask.ndim):
            mask = mask.swapaxes(0, i)
            remove = (footprint.shape[i] if footprint is not None
                      else 2 * exclude_border)
            mask[:remove // 2] = mask[-remove // 2:] = False
            mask = mask.swapaxes(0, i)

    # find top peak candidates above a threshold
    thresholds = []
    if threshold_abs is None:
        threshold_abs = image.min()
    thresholds.append(threshold_abs)
    if threshold_rel is not None:
        thresholds.append(threshold_rel * image.max())
    if thresholds:
        mask &= image > max(thresholds)

    # Select highest intensities (num_peaks)
    coordinates = _get_high_intensity_peaks(image, mask, num_peaks)

    if indices is True:
        return coordinates
    else:
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True
        return out


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if np.any(d1[o] != d2[o])}
    same = set(o for o in intersect_keys if np.all(d1[o] == d2[o]))
    return added, removed, modified, same


def computeDFF_traces(Yr, A, C, bl, quantileMin=8, frames_window=200):
    extract_DF_F(Yr, A, C, bl, quantileMin, frames_window)


def extract_DF_F(Yr, A, C, bl, quantileMin=8, frames_window=200, block_size=400, dview=None):
    """ Compute DFF function from cnmf output.

     Disclaimer: it might be memory inefficient

    Args:
        Yr: ndarray (2D)
            movie pixels X time

        A: scipy.sparse.coo_matrix
            spatial components (from cnmf cnm.A)

        C: ndarray
            temporal components (from cnmf cnm.C)

        bl: ndarray
            baseline for each component (from cnmf cnm.bl)

        quantile_min: float
            quantile minimum of the

        frames_window: int
            number of frames for running quantile

    Returns:
        Cdf:
            the computed Calcium acitivty to the derivative of f

    See Also:
        ..image::docs/img/onlycnmf.png
    """
    nA = np.array(np.sqrt(A.power(2).sum(0)).T)
    A = scipy.sparse.coo_matrix(A / nA.T)
    C = C * nA
    bl = (bl * nA.T).squeeze()
    nA = np.array(np.sqrt(A.power(2).sum(0)).T)

    T = C.shape[-1]
    if 'memmap' in str(type(Yr)):
        if block_size >= 500:
            print('Forcing single thread for memory issues')
            dview_res = None
        else:
            print('Using thread. If memory issues set block_size larger than 500')
            dview_res = dview

        AY = parallel_dot_product(Yr, A, dview=dview_res, block_size=block_size,
                                  transpose=True).T
    else:
        AY = A.T.dot(Yr)

    bas_val = bl[None, :]
    Bas = np.repeat(bas_val, T, 0).T
    AA = A.T.dot(A)
    AA.setdiag(0)
    Cf = (C - Bas) * (nA**2)
    C2 = AY - AA.dot(C)

    if frames_window is None or frames_window > T:
        Df = np.percentile(C2, quantileMin, axis=1)
        C_df = Cf / Df[:, None]

    else:
        Df = scipy.ndimage.percentile_filter(
            C2, quantileMin, (frames_window, 1))
        C_df = Cf / Df

    return C_df

def detrend_df_f(A, b, C, f, YrA=None, quantileMin=8, frames_window=500, 
                 flag_auto=True, use_fast=False, detrend_only=False):
    """ Compute DF/F signal without using the original data.
    In general much faster than extract_DF_F

    Args:
        A: scipy.sparse.csc_matrix
            spatial components (from cnmf cnm.A)

        b: ndarray
            spatial background components

        C: ndarray
            temporal components (from cnmf cnm.C)

        f: ndarray
            temporal background components

        YrA: ndarray
            residual signals

        quantile_min: float
            quantile used to estimate the baseline (values in [0,100])

        frames_window: int
            number of frames for computing running quantile

        flag_auto: bool
            flag for determining quantile automatically

        use_fast: bool
            flag for u´sing approximate fast percentile filtering

        detrend_only: bool (False)
            flag for only subtracting baseline and not normalizing by it.
            Used in 1p data processing where baseline fluorescence cannot be
            determined.

    Returns:
        F_df:
            the computed Calcium acitivty to the derivative of f
    """

    if C is None:
        logging.warning("There are no components for DF/F extraction!")
        return None
    
    if b is None or f is None:
        b = np.zeros((A.shape[0], 1))
        f = np.zeros((1, C.shape[1]))
        logging.warning("Background components not present. Results should" +
                        " not be interpreted as DF/F normalized but only" +
                        " as detrended.")
        detrend_only = True
    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)
    if 'array' not in str(type(b)):
        b = b.toarray()
    if 'array' not in str(type(C)):
        C = C.toarray()
    if 'array' not in str(type(f)):
        f = f.toarray()

    nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
    nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
    nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
    A = A * nA_inv_mat
    C = nA_mat * C
    if YrA is not None:
        YrA = nA_mat * YrA

    F = C + YrA if YrA is not None else C
    B = A.T.dot(b).dot(f)
    T = C.shape[-1]

    if flag_auto:
        data_prct, val = df_percentile(F[:, :frames_window], axis=1)
        if frames_window is None or frames_window > T:
            Fd = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(F, data_prct)])
            Df = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                           zip(B, data_prct)])
            if not detrend_only:
                F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
            else:
                F_df = F - Fd[:, None]
        else:
            if use_fast:
                Fd = np.stack([fast_prct_filt(f, level=prctileMin,
                                              frames_window=frames_window) for
                               f, prctileMin in zip(F, data_prct)])
                Df = np.stack([fast_prct_filt(f, level=prctileMin,
                                              frames_window=frames_window) for
                               f, prctileMin in zip(B, data_prct)])
            else:
                Fd = np.stack([scipy.ndimage.percentile_filter(
                    f, prctileMin, (frames_window)) for f, prctileMin in
                    zip(F, data_prct)])
                Df = np.stack([scipy.ndimage.percentile_filter(
                    f, prctileMin, (frames_window)) for f, prctileMin in
                    zip(B, data_prct)])
            if not detrend_only:
                F_df = (F - Fd) / (Df + Fd)
            else:
                F_df = F - Fd
    else:
        if frames_window is None or frames_window > T:
            Fd = np.percentile(F, quantileMin, axis=1)
            Df = np.percentile(B, quantileMin, axis=1)
            if not detrend_only:
                F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
            else:
                F_df = F - Fd[:, None]
        else:
            Fd = scipy.ndimage.percentile_filter(
                F, quantileMin, (frames_window, 1))
            Df = scipy.ndimage.percentile_filter(
                B, quantileMin, (frames_window, 1))
            if not detrend_only:
                F_df = (F - Fd) / (Df + Fd)
            else:
                F_df = F - Fd

    return F_df

def fast_prct_filt(input_data, level=8, frames_window=1000):
    """
    Fast approximate percentage filtering
    """

    data = np.atleast_2d(input_data).copy()
    T = np.shape(data)[-1]
    downsampfact = frames_window

    elm_missing = int(np.ceil(T * 1.0 / downsampfact)
                      * downsampfact - T)
    padbefore = int(np.floor(elm_missing / 2.))
    padafter = int(np.ceil(elm_missing / 2.))
    tr_tmp = np.pad(data.T, ((padbefore, padafter), (0, 0)), mode='reflect')
    numFramesNew, num_traces = np.shape(tr_tmp)
    #% compute baseline quickly

    tr_BL = np.reshape(tr_tmp, (downsampfact, int(numFramesNew / downsampfact),
                                num_traces), order='F')
    #import pdb
    #pdb.set_trace()
    tr_BL = np.percentile(tr_BL, level, axis=0)
    tr_BL = scipy.ndimage.zoom(np.array(tr_BL, dtype=np.float32),
                               [downsampfact, 1], order=3, mode='nearest',
                               cval=0.0, prefilter=True)

    if padafter == 0:
        data -= tr_BL.T
    else:
        data -= tr_BL[padbefore:-padafter].T

    return data.squeeze()
#%%
def detrend_df_f_auto(A, b, C, f, dims=None, YrA=None, use_annulus = True, 
                      dist1 = 7, dist2 = 5, frames_window=1000, 
                      use_fast = False):
    """
    Compute DF/F using an automated level of percentile filtering based on
    kernel density estimation.

    Args:
        A: scipy.sparse.csc_matrix
            spatial components (from cnmf cnm.A)

        b: ndarray
            spatial backgrounds

        C: ndarray
            temporal components (from cnmf cnm.C)

        f: ndarray
            temporal background components

        YrA: ndarray
            residual signals

        frames_window: int
            number of frames for running quantile

        use_fast: bool
            flag for using fast approximate percentile filtering

    Returns:
        F_df:
            the computed Calcium acitivty to the derivative of f
    """

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)
    if 'array' not in str(type(b)):
        b = b.toarray()
    if 'array' not in str(type(C)):
        C = C.toarray()
    if 'array' not in str(type(f)):
        f = f.toarray()

    nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
    nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
    nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
    A = A * nA_inv_mat
    C = nA_mat * C
    if YrA is not None:
        YrA = nA_mat * YrA

    F = C + YrA if YrA is not None else C
    K = A.shape[-1]
    A_ann = A.copy()

    if use_annulus:
        dist1 = 7
        dist2 = 5
        X, Y = np.meshgrid(np.arange(-dist1, dist1), np.arange(-dist1, dist1))
        R = np.sqrt(X**2+Y**2)
        R[R > dist1] = 0
        R[R < dist2] = 0
        R = R.astype('bool')

        for k in range(K):
            a = A[:, k].toarray().reshape(dims, order='F') > 0
            a2 = np.bitwise_xor(morph.binary_dilation(a, R), a)
            a2 = a2.astype(float).flatten(order='F')
            a2 /= np.sqrt(a2.sum())
            a2 = scipy.sparse.csc_matrix(a2)
            A_ann[:, k] = a2.T

    B = A_ann.T.dot(b).dot(f)
    T = C.shape[-1]

    data_prct, val = df_percentile(F[:, :frames_window], axis=1)

    if frames_window is None or frames_window > T:
        Fd = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                       zip(F, data_prct)])
        Df = np.stack([np.percentile(f, prctileMin) for f, prctileMin in
                       zip(B, data_prct)])
        F_df = (F - Fd[:, None]) / (Df[:, None] + Fd[:, None])
    else:
        if use_fast:
            Fd = np.stack([fast_prct_filt(f, level=prctileMin,
                                          frames_window=frames_window) for
                           f, prctileMin in zip(F, data_prct)])
            Df = np.stack([fast_prct_filt(f, level=prctileMin,
                                          frames_window=frames_window) for
                           f, prctileMin in zip(B, data_prct)])
        else:
            Fd = np.stack([scipy.ndimage.percentile_filter(
                f, prctileMin, (frames_window)) for f, prctileMin in
                zip(F, data_prct)])
            Df = np.stack([scipy.ndimage.percentile_filter(
                f, prctileMin, (frames_window)) for f, prctileMin in
                zip(B, data_prct)])
        F_df = (F - Fd) / (Df + Fd)

    return F_df

#%%


def manually_refine_components(Y, xxx_todo_changeme, A, C, Cn, thr=0.9, display_numbers=True,
                               max_number=None, cmap=None, **kwargs):
    """Plots contour of spatial components
     against a background image and allows to interactively add novel components by clicking with mouse

     Args:
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

     Returns:
         A: np.ndarray
             matrix A os estimated spatial component contributions

         C: np.ndarray
             array of estimated calcium traces
    """
    (dx, dy) = xxx_todo_changeme
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    if max_number is None:
        max_number = nr

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    pl.imshow(Cn, interpolation=None, cmap=cmap)
    cm = com(A, d1, d2)

    Bmat = np.zeros((np.minimum(nr, max_number), d1, d2))
    for i in range(np.minimum(nr, max_number)):
        indx = np.argsort(A[:, i], axis=None)[::-1]
        cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        Bmat[i] = np.reshape(Bvec, np.shape(Cn), order='F')

    T = np.shape(Y)[-1]

    pl.close()
    fig = pl.figure()
    ax = pl.gca()
    ax.imshow(Cn, interpolation=None, cmap=cmap,
              vmin=np.percentile(Cn[~np.isnan(Cn)], 1), vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    for i in range(np.minimum(nr, max_number)):
        pl.contour(y, x, Bmat[i], [thr])

    if display_numbers:
        for i in range(np.minimum(nr, max_number)):
            ax.text(cm[i, 1], cm[i, 0], str(i + 1))

    A3 = np.reshape(A, (d1, d2, nr), order='F')
    while True:
        pts = fig.ginput(1, timeout=0)

        if pts != []:
            print(pts)
            xx, yy = np.round(pts[0]).astype(np.int)
            coords_y = np.array(list(range(yy - dy, yy + dy + 1)))
            coords_x = np.array(list(range(xx - dx, xx + dx + 1)))
            coords_y = coords_y[(coords_y >= 0) & (coords_y < d1)]
            coords_x = coords_x[(coords_x >= 0) & (coords_x < d2)]
            a3_tiny = A3[coords_y[0]:coords_y[-1] +
                         1, coords_x[0]:coords_x[-1] + 1, :]
            y3_tiny = Y[coords_y[0]:coords_y[-1] +
                        1, coords_x[0]:coords_x[-1] + 1, :]

            dy_sz, dx_sz = np.shape(a3_tiny)[:-1]
            y2_tiny = np.reshape(y3_tiny, (dx_sz * dy_sz, T), order='F')
            a2_tiny = np.reshape(a3_tiny, (dx_sz * dy_sz, nr), order='F')
            y2_res = y2_tiny - a2_tiny.dot(C)

            y3_res = np.reshape(y2_res, (dy_sz, dx_sz, T), order='F')
            a__, c__, center__, b_in__, f_in__ = greedyROI(
                y3_res, nr=1, gSig=[np.floor(old_div(dx_sz, 2)), np.floor(old_div(dy_sz, 2))], gSiz=[dx_sz, dy_sz])

            a_f = np.zeros((d, 1))
            idxs = np.meshgrid(coords_y, coords_x)
            a_f[np.ravel_multi_index(
                idxs, (d1, d2), order='F').flatten()] = a__

            A = np.concatenate([A, a_f], axis=1)
            C = np.concatenate([C, c__], axis=0)
            indx = np.argsort(a_f, axis=None)[::-1]
            cumEn = np.cumsum(a_f.flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            bmat = np.reshape(Bvec, np.shape(Cn), order='F')
            pl.contour(y, x, bmat, [thr])
            pl.pause(.01)

        elif pts == []:
            break

        nr += 1
        A3 = np.reshape(A, (d1, d2, nr), order='F')

    return A, C

def app_vertex_cover(A):
    """ Finds an approximate vertex cover for a symmetric graph with adjacency matrix A.

    Args:
        A:  boolean 2d array (K x K)
            Adjacency matrix. A is boolean with diagonal set to 0

    Returns:
        L:   A vertex cover of A

    Authors:
    Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    """

    L = []
    while A.any():
        nz = np.nonzero(A)[0]          # find non-zero edges
        u = nz[np.random.randint(0, len(nz))]
        A[u, :] = False
        A[:, u] = False
        L.append(u)

    return np.asarray(L)


def update_order(A, new_a=None, prev_list=None, method='greedy'):
    '''Determines the update order of the temporal components given the spatial
    components by creating a nest of random approximate vertex covers

     Args:
         A:    np.ndarray
              matrix of spatial components (d x K)
         new_a: sparse array
              spatial component that is added, in order to efficiently update the orders in online scenarios
         prev_list: list of list
              orders from previous iteration, you need to pass if new_a is not None

     Returns:
         O:  list of sets
             list of subsets of components. The components of each subset can be updated in parallel
         lo: list
             length of each subset

    Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''
    K = np.shape(A)[-1]
    if new_a is None and prev_list is None:

        if method is 'greedy':
            prev_list, count_list = update_order_greedy(A, flag_AA=False)
        else:
            prev_list, count_list = update_order_random(A, flag_AA=False)
        return prev_list, count_list

    else:

        if new_a is None or prev_list is None:
            raise Exception(
                'In the online update order you need to provide both new_a and prev_list')

        counter = 0

        AA = A.T.dot(new_a)
        for group in prev_list:
            if AA[list(group)].sum() == 0:
                group.append(K)
                counter += 1
                break

        if counter == 0:
            if prev_list is not None:
                prev_list = list(prev_list)
                prev_list.append([K])

        count_list = [len(gr) for gr in prev_list]

        return prev_list, count_list

def order_components(A, C):
    """Order components based on their maximum temporal value and size

    Args:
        A:  sparse matrix (d x K)
            spatial components

        C:  matrix or np.ndarray (K x T)
            temporal components

    Returns:
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
    A = np.array(np.matrix(A) * spdiags(old_div(1, nA2), 0, K, K))
    nA4 = np.sum(A**4, axis=0)**0.25
    C = np.array(spdiags(nA2, 0, K, K) * np.matrix(C))
    mC = np.ndarray.max(np.array(C), axis=1)
    srt = np.argsort(nA4 * mC)[::-1]
    A_or = A[:, srt] * spdiags(nA2[srt], 0, K, K)
    C_or = spdiags(old_div(1., nA2[srt]), 0, K, K) * (C[srt, :])

    return A_or, C_or, srt

def update_order_random(A, flag_AA=True):
    """Determies the update order of temporal components using
    randomized partitions of non-overlapping components
    """

    K = np.shape(A)[-1]
    if flag_AA:
        AA = A.copy()
    else:
        AA = A.T.dot(A)

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


def update_order_greedy(A, flag_AA=True):
    """Determines the update order of the temporal components

    this, given the spatial components using a greedy method
    Basically we can update the components that are not overlapping, in parallel

    Args:
        A:  sparse crc matrix
            matrix of spatial components (d x K)
        OR:
            A.T.dot(A) matrix (d x d) if flag_AA = true

        flag_AA: boolean (default true)

     Returns:
         parllcomp:   list of sets
             list of subsets of components. The components of each subset can be updated in parallel

         len_parrllcomp:  list
             length of each subset

    Author:
        Eftychios A. Pnevmatikakis, Simons Foundation, 2017
    """
    K = np.shape(A)[-1]
    parllcomp:List = []
    for i in range(K):
        new_list = True
        for ls in parllcomp:
            if flag_AA:
                if A[i, ls].nnz == 0:
                    ls.append(i)
                    new_list = False
                    break
            else:
                if (A[:, i].T.dot(A[:, ls])).nnz == 0:
                    ls.append(i)
                    new_list = False
                    break

        if new_list:
            parllcomp.append([i])
    len_parrllcomp = [len(ls) for ls in parllcomp]
    return parllcomp, len_parrllcomp
#%%


def compute_residuals(Yr_mmap_file, A_, b_, C_, f_, dview=None, block_size=1000, num_blocks_per_run=5):
    '''compute residuals from memory mapped file and output of CNMF
        Args:
            A_,b_,C_,f_:
                from CNMF

            block_size: int
                number of pixels processed together

            num_blocks_per_run: int
                nnumber of parallel blocks processes

        Returns:
            YrA: ndarray
                residuals per neuron
    '''
    if not ('sparse' in str(type(A_))):
        A_ = scipy.sparse.coo_matrix(A_)

    Ab = scipy.sparse.hstack((A_, b_)).tocsc()

    Cf = np.vstack((C_, f_))

    nA = np.ravel(Ab.power(2).sum(axis=0))

    if 'mmap' in str(type(Yr_mmap_file)):
        YA = parallel_dot_product(Yr_mmap_file, Ab, dview=dview, block_size=block_size,
                                  transpose=True, num_blocks_per_run=num_blocks_per_run) * scipy.sparse.spdiags(old_div(1., nA), 0, Ab.shape[-1], Ab.shape[-1])
    else:
        YA = (Ab.T.dot(Yr_mmap_file)).T * \
            spdiags(old_div(1., nA), 0, Ab.shape[-1], Ab.shape[-1])

    AA = ((Ab.T.dot(Ab)) * scipy.sparse.spdiags(old_div(1., nA),
                                                0, Ab.shape[-1], Ab.shape[-1])).tocsr()

    return (YA - (AA.T.dot(Cf)).T)[:, :A_.shape[-1]].T

def normalize_AC(A, C, YrA, b, f, neurons_sn):
    """ Normalize to unit norm A and b
    Args:
        A,C,Yr,b,f:
            outputs of CNMF
    """
    if 'sparse' in str(type(A)):
        nA = np.ravel(np.sqrt(A.power(2).sum(0)))
    else:
        nA = np.ravel(np.sqrt((A**2).sum(0)))

    if A is not None:
        A /= nA

    if C is not None:
        C = np.array(C)
        C *= nA[:, None]

    if YrA is not None:
        YrA = np.array(YrA)
        YrA *= nA[:, None]

    if b is not None:
        if issparse(b):
            nB = np.ravel(np.sqrt(b.power(2).sum(0)))
            b = csc_matrix(b)
            for k, i in enumerate(b.indptr[:-1]):
                b.data[i:b.indptr[k + 1]] /= nB[k]
        else:
            nB = np.ravel(np.sqrt((b**2).sum(0)))
            b = np.atleast_2d(b)
            b /= nB
        if issparse(f):
            f = csr_matrix(f)
            for k, i in enumerate(f.indptr[:-1]):
                f.data[i:f.indptr[k + 1]] *= nB[k]
        else:
            f = np.atleast_2d(f)
            f *= nB[:, np.newaxis]

    if neurons_sn is not None:
        neurons_sn *= nA

    return csc_matrix(A), C, YrA, b, f, neurons_sn


def get_file_size(file_name, var_name_hdf5='mov'):
    """ Computes the dimensions of a file or a list of files without loading
    it/them in memory. An exception is thrown if the files have FOVs with
    different sizes
        Args:
            file_name: str or list
                locations of file(s) in memory

            var_name_hdf5: 'str'
                if loading from hdf5 name of the variable to load

        Returns:
            dims: list
                dimensions of FOV

            T: list
                number of timesteps in each file
    """
    if isinstance(file_name, str):
        if os.path.exists(file_name):
            _, extension = os.path.splitext(file_name)[:2]
            extension = extension.lower()
            if extension == '.tif' or extension == '.tiff':
                tffl = tifffile.TiffFile(file_name)
                siz = tffl.series[0].shape
                T, dims = siz[0], siz[1:]
            elif extension == '.avi':
                cap = cv2.VideoCapture(file_name)
                dims = [0, 0]
                try:
                    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    dims[0] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    dims[1] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                except():
                    print('Roll back to opencv 2')
                    T = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                    dims[0] = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                    dims[1] = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            elif extension == '.mmap':
                filename = os.path.split(file_name)[-1]
                Yr, dims, T = load_memmap(os.path.join(
                        os.path.split(file_name)[0], filename))
            elif extension in ('.h5', '.hdf5', '.nwb'):
                with h5py.File(file_name, "r") as f:
                    kk = list(f.keys())
                    if len(kk) == 1:
                        siz = f[kk[0]].shape
                    elif var_name_hdf5 in f:
                        if extension == '.nwb':
                            siz = f[var_name_hdf5]['data'].shape
                        else:
                            siz = f[var_name_hdf5].shape
                    else:
                        logging.error('The file does not contain a variable' +
                                      'named {0}'.format(var_name_hdf5))
                        raise Exception('Variable not found. Use one of the above')
                T, dims = siz[0], siz[1:]
            else:
                raise Exception('Unknown file type')
        else:
            raise Exception('File not found!')
    elif isinstance(file_name, list):
        if len(file_name) == 1:
            dims, T = get_file_size(file_name[0], var_name_hdf5=var_name_hdf5)
        else:
            dims, T = zip(*[get_file_size(fn, var_name_hdf5=var_name_hdf5)
                for fn in file_name])
            if len(list(set(dims))) > 1:
                raise Exception("Files have different FOV sizes.")
            else:
                dims = dims[0]
    else:
        raise Exception('Unknown input type')
    return dims, T
