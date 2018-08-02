#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A set of pre-processing operations in the input dataset:

1. Interpolation of missing data
2. Indentification of saturated pixels
3. Estimation of noise level for each imaged voxel
4. Estimation of global time constants

See Also:
------------

@authors: agiovann epnev
@image docs/img/greedyroi.png
"""
#\package caiman/source_extraction/cnmf
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Tue Jun 30 21:01:17 2015


from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import numpy as np
import scipy
from builtins import map
from builtins import range
from ...mmapping import load_memmap
from past.builtins import basestring
from past.utils import old_div

#%%


def interpolate_missing_data(Y):
    """
    Interpolate any missing data using nearest neighbor interpolation.
    Missing data is identified as entries with values NaN

    Parameters:
    ----------
    Y   np.ndarray (3D)
        movie, raw data in 3D format (d1 x d2 x T)

    Returns:
    ------
    Y   np.ndarray (3D)
        movie, data with interpolated entries (d1 x d2 x T)
    coor list
        list of interpolated coordinates

    Raise:
    ------
        Exception('The algorithm has not been tested with missing values (NaNs). Remove NaNs and rerun the algorithm.')
    """
    coor = []
    print('checking if missing data')
    if np.any(np.isnan(Y)):
        for idx, row in enumerate(Y):
            nans = np.where(np.isnan(row))[0]
            n_nans = np.where(~np.isnan(row))[0]
            coor.append((idx, nans))
            Y[idx, nans] = np.interp(nans, n_nans, row[n_nans])
        raise Exception(
            'The algorithm has not been tested with missing values (NaNs). Remove NaNs and rerun the algorithm.')

    return Y, coor

#%%


def find_unsaturated_pixels(Y, saturationValue=None, saturationThreshold=0.9, saturationTime=0.005):
    """Identifies the saturated pixels that are saturated and returns the ones that are not.
    A pixel is defined as saturated if its observed fluorescence is above
    saturationThreshold*saturationValue at least saturationTime fraction of the time.

    Inputs:
    ------
    Y: np.ndarray
        input movie data, either 2D or 3D with time in the last axis

    saturationValue: scalar (optional)
        Saturation level, default value the lowest power of 2 larger than max(Y)

    saturationThreshold: scalar between 0 and 1 (optional)
        Fraction of saturationValue above which the fluorescence is considered to
        be in the saturated region. Default value 0.9

    saturationTime: scalar between 0 and 1 (optional)
        Fraction of time that pixel needs to be in the saturated
        region to be considered saturated. Default: 0.005

    Output:
    ------
    normalPixels:   nd.array
        list of unsaturated pixels
    """
    if saturationValue == None:
        saturationValue = np.power(2, np.ceil(np.log2(np.max(Y)))) - 1

    Ysat = (Y >= saturationThreshold * saturationValue)
    pix = np.mean(Ysat, Y.ndim - 1).flatten('F') > saturationTime
    normalPixels = np.where(pix)

    return normalPixels


#%%
def get_noise_welch(Y, noise_range=[0.25, 0.5], noise_method='logmexp',
                    max_num_samples_fft=3072):
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Inputs:
    -------

    Y: np.ndarray

    Input movie data with time in the last axis

    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
        default: [0.25,0.5]

    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Output:
    ------
    sn: np.ndarray
        Noise level for each pixel
    """
    T = Y.shape[-1]
    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
                            Y[..., np.int(T // 2 - max_num_samples_fft / 3 / 2):
                            np.int(T // 2 + max_num_samples_fft / 3 / 2)],
                            Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = np.shape(Y)[-1]
    ff, Pxx = scipy.signal.welch(Y)
    Pxx = Pxx[..., (ff >= noise_range[0]) & (ff <= noise_range[1])]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx, -1) / 2),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx, -1) / 2),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx / 2), -1)))
    }[noise_method](Pxx)
    return sn


def get_noise_fft(Y, noise_range=[0.25, 0.5], noise_method='logmexp', max_num_samples_fft=3072,
                  opencv=True):
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Inputs:
    -------

    Y: np.ndarray

    Input movie data with time in the last axis

    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
        default: [0.25,0.5]

    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Output:
    ------
    sn: np.ndarray
        Noise level for each pixel
    """
    T = Y.shape[-1]
    # Y=np.array(Y,dtype=np.float64)

    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
                            Y[..., np.int(T // 2 - max_num_samples_fft / 3 / 2)
                                          :np.int(T // 2 + max_num_samples_fft / 3 / 2)],
                            Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = np.shape(Y)[-1]

    # we create a map of what is the noise on the FFT space
    ff = np.arange(0, 0.5 + 1. / T, 1. / T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1, ind2)
    # we compute the mean of the noise spectral density s
    if Y.ndim > 1:
        if opencv:
            import cv2
            try:
                cv2.setNumThreads(0)
            except:
                pass
            psdx = []
            for y in Y.reshape(-1, T):
                dft = cv2.dft(y, flags=cv2.DFT_COMPLEX_OUTPUT).squeeze()[
                    :len(ind)][ind]
                psdx.append(np.sum(1. / T * dft * dft, 1))
            psdx = np.reshape(psdx, Y.shape[:-1] + (-1,))
        else:
            xdft = np.fft.rfft(Y, axis=-1)
            xdft = xdft[..., ind[:xdft.shape[-1]]]
            psdx = 1. / T * abs(xdft)**2
        psdx *= 2
        sn = mean_psd(psdx, method=noise_method)

    else:
        xdft = np.fliplr(np.fft.rfft(Y))
        psdx = 1. / T * (xdft**2)
        psdx[1:] *= 2
        sn = mean_psd(psdx[ind[:psdx.shape[0]]], method=noise_method)

    return sn, psdx


def get_noise_fft_parallel(Y, n_pixels_per_process=100, dview=None, **kwargs):
    """parallel version of get_noise_fft.

    Parameters:
    -------
    Y: ndarray
        input movie (n_pixels x Time). Can be also memory mapped file.

    n_processes: [optional] int
        number of processes/threads to use concurrently

    n_pixels_per_process: [optional] int
        number of pixels to be simultaneously processed by each process

    backend: [optional] string
        the type of concurrency to be employed. only 'multithreading' for the moment

    **kwargs: [optional] dict
        all the parameters passed to get_noise_fft

    Returns:
    --------

    sn: ndarray(double)
        noise associated to each pixel
    """
    folder = tempfile.mkdtemp()

    # Pre-allocate a writeable shared memory map as a container for the
    # results of the parallel computation
    pixel_groups = list(
        range(0, Y.shape[0] - n_pixels_per_process + 1, n_pixels_per_process))

    if type(Y) is np.core.memmap:  # if input file is already memory mapped then find the filename
        Y_name = Y.filename

    else:
        if dview is not None:
            raise Exception('parallel processing requires memory mapped files')
        Y_name = Y

    argsin = [(Y_name, i, n_pixels_per_process, kwargs) for i in pixel_groups]
    pixels_remaining = Y.shape[0] % n_pixels_per_process
    if pixels_remaining > 0:
        argsin.append(
            (Y_name, Y.shape[0] - pixels_remaining, pixels_remaining, kwargs))

    if dview is None:
        print('Single Thread')
        results = list(map(fft_psd_multithreading, argsin))

    else:
        if 'multiprocessing' in str(type(dview)):
            results = dview.map_async(
                fft_psd_multithreading, argsin).get(4294967)
        else:
            ne = len(dview)
            print(('Running on %d engines.' % (ne)))
            if dview.client.profile == 'default':
                results = dview.map_sync(fft_psd_multithreading, argsin)

            else:
                print(('PROFILE:' + dview.client.profile))
                results = dview.map_sync(fft_psd_multithreading, argsin)

    _, _, psx_ = results[0]
    sn_s = np.zeros(Y.shape[0])
    psx_s = np.zeros((Y.shape[0], psx_.shape[-1]))
    for idx, sn, psx_ in results:
        sn_s[idx] = sn
        psx_s[idx, :] = psx_

    sn_s = np.array(sn_s)
    psx_s = np.array(psx_s)

    try:
        shutil.rmtree(folder)

    except:
        print(("Failed to delete: " + folder))
        raise

    return sn_s, psx_s
#%%


def fft_psd_parallel(Y, sn_s, i, num_pixels, **kwargs):
    """helper function to parallelize get_noise_fft

    Parameters:
    -----------
        Y: ndarray
                input movie (n_pixels x Time), can be also memory mapped file

    sn_s: ndarray (memory mapped)
        file where to store the results of computation.

    i: int
        pixel index start

    num_pixels: int
        number of pixel to select starting from i

    **kwargs: dict
        arguments to be passed to get_noise_fft


     Returns:
     -------
        idx: list
            list of the computed pixels

        res: ndarray(double)
            noise associated to each pixel
        psx: ndarray
            position of thoses pixels
    """
    idxs = list(range(i, i + num_pixels))
    res = get_noise_fft(Y[idxs], **kwargs)
    sn_s[idxs] = res
#%%


def fft_psd_multithreading(args):
    """helper function to parallelize get_noise_fft

    Parameters:
    -----------

    Y: ndarray
        input movie (n_pixels x Time), can be also memory mapped file

    sn_s: ndarray (memory mapped)
        file where to store the results of computation.

    i: int
        pixel index start

    num_pixels: int
        number of pixel to select starting from i

    **kwargs: dict
        arguments to be passed to get_noise_fft

    Returns:
    -------
        idx: list
            list of the computed pixels

        res: ndarray(double)
            noise associated to each pixel
        psx: ndarray
            position of thoses pixels


    """
    (Y, i, num_pixels, kwargs) = args
    if isinstance(Y, basestring):
        Y, _, _ = load_memmap(Y)

    idxs = list(range(i, i + num_pixels))
    print(len(idxs))
    res, psx = get_noise_fft(Y[idxs], **kwargs)

    return (idxs, res, psx)
#%%


def mean_psd(y, method='logmexp'):
    """
    Averaging the PSD

    Parameters:
    ----------

        y: np.ndarray
             PSD values

        method: string
            method of averaging the noise.
            Choices:
             'mean': Mean
             'median': Median
             'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
    -------
        mp: array
            mean psd
    """

    if method == 'mean':
        mp = np.sqrt(np.mean(old_div(y, 2), axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(old_div(y, 2), axis=-1))
    else:
        mp = np.log(old_div((y + 1e-10), 2))
        mp = np.mean(mp, axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)

    return mp


#%%

def estimate_time_constant(Y, sn, p=None, lags=5, include_noise=False, pixels=None):
    """
    Estimating global time constants for the dataset Y through the autocovariance function (optional).
    The function is no longer used in the standard setting of the algorithm since every trace has its own
    time constant.

    Inputs:
    -------
    Y: np.ndarray (2D)
        input movie data with time in the last axis

    p: positive integer
        order of AR process, default: 2

    lags: positive integer
        number of lags in the past to consider for determining time constants. Default 5

    include_noise: Boolean
        Flag to include pre-estimated noise value when determining time constants. Default: False

    pixels: np.ndarray
        Restrict estimation to these pixels (e.g., remove saturated pixels). Default: All pixels

    Output:
    -------
    g:  np.ndarray (p x 1)
        Discrete time constants
    """
    if p is None:
        raise Exception("You need to define p")
    if pixels is None:
        pixels = np.arange(old_div(np.size(Y), np.shape(Y)[-1]))

    from scipy.linalg import toeplitz
    npx = len(pixels)
    lags += p
    XC = np.zeros((npx, 2 * lags + 1))
    for j in range(npx):
        XC[j, :] = np.squeeze(axcov(np.squeeze(Y[pixels[j], :]), lags))

    gv = np.zeros(npx * lags)
    if not include_noise:
        XC = XC[:, np.arange(lags - 1, -1, -1)]
        lags -= p

    A = np.zeros((npx * lags, p))
    for i in range(npx):
        if not include_noise:
            A[i * lags + np.arange(lags), :] = toeplitz(np.squeeze(XC[i, np.arange(
                p - 1, p + lags - 1)]), np.squeeze(XC[i, np.arange(p - 1, -1, -1)]))
        else:
            A[i * lags + np.arange(lags), :] = toeplitz(np.squeeze(XC[i, lags + np.arange(
                lags)]), np.squeeze(XC[i, lags + np.arange(p)])) - (sn[i]**2) * np.eye(lags, p)
            gv[i * lags + np.arange(lags)] = np.squeeze(XC[i, lags + 1:])

    if not include_noise:
        gv = XC[:, p:].T
        gv = np.squeeze(np.reshape(gv, (np.size(gv), 1), order='F'))

    g = np.dot(np.linalg.pinv(A), gv)

    return g


def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag


    Parameters:
    ----------
    data : array
        Array containing fluorescence data

    maxlag : int
        Number of lags to use in autocovariance calculation

    Returns:
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    #xcov = xcov/np.concatenate([np.arange(T-maxlag,T+1),np.arange(T-1,T-maxlag-1,-1)])
    return np.real(old_div(xcov, T))


#%%
def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).

    Parameters:
    ----------
        value : int

    Returns:
    -------
        exponent : int
    """

    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent


def preprocess_data(Y, sn=None, dview=None, n_pixels_per_process=100, noise_range=[0.25, 0.5], noise_method='logmexp', compute_g=False, p=2, lags=5, include_noise=False, pixels=None, max_num_samples_fft=3000, check_nan=True):
    """
    Performs the pre-processing operations described above.

    Parameters:
    ----------
    Y: ndarray
        input movie (n_pixels x Time). Can be also memory mapped file.

    n_processes: [optional] int
        number of processes/threads to use concurrently

    n_pixels_per_process: [optional] int
        number of pixels to be simultaneously processed by each process

    p: positive integer
        order of AR process, default: 2

    lags: positive integer
        number of lags in the past to consider for determining time constants. Default 5

    include_noise: Boolean
        Flag to include pre-estimated noise value when determining time constants. Default: False

    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
        default: [0.25,0.5]

    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
    -------
        Y: ndarray
             movie preprocessed (n_pixels x Time). Can be also memory mapped file.

        g:  np.ndarray (p x 1)
            Discrete time constants

        psx: ndarray
            position of thoses pixels

        sn_s: ndarray (memory mapped)
            file where to store the results of computation.

    """
    if check_nan:
        Y, coor = interpolate_missing_data(Y)

    if sn is None:
        if dview is None:
            sn, psx = get_noise_fft(Y, noise_range=noise_range, noise_method=noise_method,
                                    max_num_samples_fft=max_num_samples_fft)
        else:
            sn, psx = get_noise_fft_parallel(Y, n_pixels_per_process=n_pixels_per_process, dview=dview,
                                             noise_range=noise_range, noise_method=noise_method,
                                             max_num_samples_fft=max_num_samples_fft)
    else:
        psx = None

    if compute_g:
        g = estimate_time_constant(Y, sn, p=p, lags=lags,
                                   include_noise=include_noise, pixels=pixels)
    else:
        g = None

    # psx  # no need to keep psx in memory as long a we don't use it elsewhere
    return Y, sn, g, None
