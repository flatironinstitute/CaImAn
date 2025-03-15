#!/usr/bin/env python

""" Initialize the component for the CNMF

contain a list of functions to initialize the neurons and the corresponding traces with
different set of methods like ICA PCA, greedy roi
"""

import cv2
import logging
from math import sqrt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from multiprocessing import current_process
import numpy as np
import scipy
import scipy.ndimage as nd
from scipy.ndimage import center_of_mass, correlate
import scipy.sparse as spr
from skimage.morphology import disk
from sklearn.decomposition import NMF, FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import randomized_svd, squared_norm, randomized_range_finder
import sys
import warnings

import caiman
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
from caiman.source_extraction.cnmf.pre_processing import get_noise_fft, get_noise_welch
from caiman.source_extraction.cnmf.spatial import circular_constraint, connectivity_constraint
from caiman.utils.stats import pd_solve, compressive_nmf
from caiman.utils.utils import parmap

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    profile
except:
    def profile(a): return a

#FIXME review this and find a better way to do it
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def resize(Y, size, interpolation=cv2.INTER_LINEAR):
    """faster and 3D compatible version of skimage.transform.resize"""
    if Y.ndim == 2:
        return cv2.resize(Y, tuple(size[::-1]), interpolation=interpolation)
    elif Y.ndim == 3:
        if np.isfortran(Y):
            return (cv2.resize(np.array(
                [cv2.resize(y, size[:2], interpolation=interpolation) for y in Y.T]).T
                .reshape((-1, Y.shape[-1]), order='F'),
                (size[-1], np.prod(size[:2])), interpolation=interpolation).reshape(size, order='F'))
        else:
            return np.array([cv2.resize(y, size[:0:-1], interpolation=interpolation) for y in
                             cv2.resize(Y.reshape((len(Y), -1), order='F'),
                                        (np.prod(Y.shape[1:]), size[0]), interpolation=interpolation)
                             .reshape((size[0],) + Y.shape[1:], order='F')])
    else:  # TODO deal with ndim=4
        raise NotImplementedError

def decimate_last_axis(y, sub):
    q = y.shape[-1] // sub
    r = y.shape[-1] % sub
    Y_ds = np.zeros(y.shape[:-1] + (q + (r > 0),), dtype=y.dtype)
    Y_ds[..., :q] = y[..., :q * sub].reshape(y.shape[:-1] + (-1, sub)).mean(-1)
    if r > 0:
        Y_ds[..., -1] = y[..., -r:].mean(-1)
    return Y_ds


def downscale(Y, ds, opencv=False):
    """downscaling without zero padding
    faster version of skimage.transform._warps.block_reduce(Y, ds, np.nanmean, np.nan)"""
    logger = logging.getLogger("caiman")

    d = Y.ndim
    if opencv and (d in [2, 3]):
        if d == 2:
            Y = Y[..., None]
            ds = tuple(ds) + (1,)
        else:
            Y_ds = caiman.base.movies.movie(Y.transpose(2, 0, 1)).resize(fx=1. / ds[0], fy=1. / ds[1], fz=1. / ds[2],
                                                      interpolation=cv2.INTER_AREA).transpose(1, 2, 0)
        logger.info('Downscaling using OpenCV')
    else:
        if d > 3:
            # raise NotImplementedError
            # slower and more memory intensive version using skimage
            from skimage.transform._warps import block_reduce
            return block_reduce(Y, ds, np.nanmean, np.nan)
        elif d == 1:
            return decimate_last_axis(Y, ds)
        elif d == 2:
            Y = Y[..., None]
            ds = tuple(ds) + (1,)

        if d == 3 and Y.shape[-1] > 1 and ds[0] == ds[1]:
            ds_mat = caiman.source_extraction.cnmf.utilities.decimation_matrix(Y.shape[:2], ds[0])
            Y_ds = ds_mat.dot(Y.reshape((-1, Y.shape[-1]), order='F')).reshape(
                (1 + (Y.shape[0] - 1) // ds[0], 1 + (Y.shape[1] - 1) // ds[0], -1), order='F')
            if ds[2] > 1:
                Y_ds = decimate_last_axis(Y_ds, ds[2])
        else:
            q = np.array(Y.shape) // np.array(ds)
            r = np.array(Y.shape) % np.array(ds)
            s = q * np.array(ds)
            Y_ds = np.zeros(q + (r > 0), dtype=Y.dtype)
            Y_ds[:q[0], :q[1], :q[2]] = (Y[:s[0], :s[1], :s[2]]
                                         .reshape(q[0], ds[0], q[1], ds[1], q[2], ds[2])
                                         .mean(1).mean(2).mean(3))
            if r[0]:
                Y_ds[-1, :q[1], :q[2]] = (Y[-r[0]:, :s[1], :s[2]]
                                          .reshape(r[0], q[1], ds[1], q[2], ds[2])
                                          .mean(0).mean(1).mean(2))
                if r[1]:
                    Y_ds[-1, -1, :q[2]] = (Y[-r[0]:, -r[1]:, :s[2]]
                                           .reshape(r[0], r[1], q[2], ds[2])
                                           .mean(0).mean(0).mean(1))
                    if r[2]:
                        Y_ds[-1, -1, -1] = Y[-r[0]:, -r[1]:, -r[2]:].mean()
                if r[2]:
                    Y_ds[-1, :q[1], -1] = (Y[-r[0]:, :s[1]:, -r[2]:]
                                           .reshape(r[0], q[1], ds[1], r[2])
                                           .mean(0).mean(1).mean(1))
            if r[1]:
                Y_ds[:q[0], -1, :q[2]] = (Y[:s[0], -r[1]:, :s[2]]
                                          .reshape(q[0], ds[0], r[1], q[2], ds[2])
                                          .mean(1).mean(1).mean(2))
                if r[2]:
                    Y_ds[:q[0], -1, -1] = (Y[:s[0]:, -r[1]:, -r[2]:]
                                           .reshape(q[0], ds[0], r[1], r[2])
                                           .mean(1).mean(1).mean(1))
            if r[2]:
                Y_ds[:q[0], :q[1], -1] = (Y[:s[0], :s[1], -r[2]:]
                                          .reshape(q[0], ds[0], q[1], ds[1], r[2])
                                          .mean(1).mean(2).mean(2))
    return Y_ds if d == 3 else Y_ds[:, :, 0]

def initialize_components(Y, K=30, gSig=[5, 5], gSiz=None, ssub=1, tsub=1, nIter=5, maxIter=5, nb=1,
                          kernel=None, use_hals=True, normalize_init=True, img=None, method_init='greedy_roi',
                          max_iter_snmf=500, alpha_snmf=0.5, sigma_smooth_snmf=(.5, .5, .5),
                          perc_baseline_snmf=20, options_local_NMF=None, rolling_sum=False,
                          rolling_length=100, sn=None, options_total=None,
                          min_corr=0.8, min_pnr=10, seed_method='auto', ring_size_factor=1.5,
                          center_psf=False, ssub_B=2, init_iter=2, remove_baseline = True,
                          SC_kernel='heat', SC_sigma=1, SC_thr=0, SC_normalize=True, SC_use_NN=False,
                          SC_nnn=20, lambda_gnmf=1, snmf_l1_ratio:float=0.0):
    """
    Initialize components. This function initializes the spatial footprints, temporal components,
    and background which are then further refined by the CNMF iterations. There are four
    different initialization methods depending on the data you're processing:

    greedy_roi: GreedyROI method used in standard 2p processing (default)
    corr_pnr:   GreedyCorr method used for processing 1p data
    sparse_nmf: Sparse NMF method suitable for dendritic/axonal imaging
    graph_nmf:  Graph NMF method also suitable for dendritic/axonal imaging

    The GreedyROI method by default is not using the RollingGreedyROI method. This can
    be changed through the binary flag 'rolling_sum'.

    All the methods can be used for volumetric data except 'corr_pnr' which is only
    available for 2D data.

    It is also by default followed by hierarchical alternative least squares (HALS) NMF.
    Optional use of spatio-temporal downsampling to boost speed.

    Args:
        Y: np.ndarray
            d1 x d2 [x d3] x T movie, raw data.

        K: [optional] int
            number of neurons to extract (default value: 30). Maximal number for method 'corr_pnr'.

        gSig: [optional] list,tuple
            standard deviation of neuron size along x and y [and z] (default value: (5,5).

        gSiz: [optional] list,tuple
            half width of bounding box used for components during initialization (default 2*gSig + 1).

        nIter: [optional] int
            number of iterations for shape tuning (default 5).

        maxIter: [optional] int
            number of iterations for HALS algorithm (default 5).

        ssub: [optional] int
            spatial downsampling factor recommended for large datasets (default 1, no downsampling).

        tsub: [optional] int
            temporal downsampling factor recommended for long datasets (default 1, no downsampling).

        kernel: [optional] np.ndarray
            User specified kernel for greedyROI
            (default None, greedy ROI searches for Gaussian shaped neurons)

        use_hals: [optional] bool
            Whether to refine components with the hals method

        normalize_init: [optional] bool
            Whether to normalize_init data before running the initialization

        img: optional [np 2d array]
            Image with which to normalize. If not present use the mean + offset

        method_init: {'greedy_roi', 'corr_pnr', 'sparse_nmf', 'graph_nmf', 'pca_ica'}
            Initialization method (default: 'greedy_roi')

        max_iter_snmf: int
            Maximum number of sparse NMF iterations

        alpha_snmf: scalar
            Sparsity penalty

        rolling_sum: boolean
            Detect new components based on a rolling sum of pixel activity (default: False)

        rolling_length: int
            Length of rolling window (default: 100)

        center_psf: Boolean
            True indicates centering the filtering kernel for background
            removal. This is useful for data with large background
            fluctuations.

        min_corr: float
            minimum local correlation coefficients for selecting a seed pixel.

        min_pnr: float
            minimum peak-to-noise ratio for selecting a seed pixel.

        seed_method: str {'auto', 'manual', 'semi'}
            methods for choosing seed pixels
            'semi' detects K components automatically and allows to add more manually
            if running as notebook 'semi' and 'manual' require a backend that does not
            inline figures, e.g. %matplotlib tk

        ring_size_factor: float
            it's the ratio between the ring radius and neuron diameters.

        nb: integer
            number of background components for approximating the background using NMF model

        sn: ndarray
            per pixel noise

        options_total: dict
            the option dictionary

        ssub_B: int, optional
            downsampling factor for 1-photon imaging background computation

        init_iter: int, optional
            number of iterations for 1-photon imaging initialization

        snmf_l1_ratio: float
            Used only by sparse NMF, passed to NMF call

    Returns:
        Ain: np.ndarray
            (d1 * d2 [ * d3]) x K , spatial filter of each neuron.

        Cin: np.ndarray
            T x K , calcium activity of each neuron.

        center: np.ndarray
            K x 2 [or 3] , inferred center of each neuron.

        bin: np.ndarray
            (d1 * d2 [ * d3]) x nb, initialization of spatial background.

        fin: np.ndarray
            nb x T matrix, initialization of temporal background

    """
    logger = logging.getLogger("caiman")
    method = method_init

    if gSiz is None:
        gSiz = 2 * (np.asarray(gSig) + .5).astype(int) + 1

    d, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    # rescale according to downsampling factor
    gSig = np.asarray(gSig, dtype=float) / ssub
    gSiz = np.round(np.asarray(gSiz) / ssub).astype(int)

    if normalize_init:
        logger.info('Variance Normalization')
        if img is None:
            img = np.mean(Y, axis=-1)
            img += np.median(img)
            img += np.finfo(np.float32).eps

        Y = Y / np.reshape(img, d + (-1,), order='F')
        alpha_snmf /= np.mean(img) # normalize alpha for sparse nmf
    else:
        Y = np.array(Y)

    # spatial downsampling

    if ssub != 1 or tsub != 1:

        if method == 'corr_pnr':
            logger.info("Spatial/Temporal downsampling 1-photon")
            # this increments the performance against ground truth and solves border problems
            Y_ds = downscale(Y, tuple([ssub] * len(d) + [tsub]), opencv=False)
        else:
            logger.info("Spatial/Temporal downsampling 2-photon")
            # this increments the performance against ground truth and solves border problems
            Y_ds = downscale(Y, tuple([ssub] * len(d) + [tsub]), opencv=True)
    else:
        Y_ds = Y

    ds = Y_ds.shape[:-1]
    if nb > min(np.prod(ds), Y_ds.shape[-1]):
        nb = -1

    logger.info('Roi Initialization...')
    if method == 'greedy_roi':
        Ain, Cin, _, b_in, f_in = greedyROI(
            Y_ds, nr=K, gSig=gSig, gSiz=gSiz, nIter=nIter, kernel=kernel, nb=nb,
            rolling_sum=rolling_sum, rolling_length=rolling_length, seed_method=seed_method)

        if use_hals:
            logger.info('Refining Components using HALS NMF iterations')
            Ain, Cin, b_in, f_in = hals(
                Y_ds, Ain, Cin, b_in, f_in, maxIter=maxIter)
    elif method == 'corr_pnr':
        Ain, Cin, _, b_in, f_in, extra_1p = greedyROI_corr(
            Y, Y_ds, max_number=K, gSiz=gSiz[0], gSig=gSig[0], min_corr=min_corr, min_pnr=min_pnr,
            ring_size_factor=ring_size_factor, center_psf=center_psf, options=options_total,
            sn=sn, nb=nb, ssub=ssub, ssub_B=ssub_B, init_iter=init_iter, seed_method=seed_method)

    elif method == 'sparse_nmf':
        Ain, Cin, _, b_in, f_in = sparseNMF(
            Y_ds, nr=K, nb=nb, max_iter_snmf=max_iter_snmf, alpha=alpha_snmf,
            sigma_smooth=sigma_smooth_snmf, remove_baseline=remove_baseline, perc_baseline=perc_baseline_snmf, l1_ratio=snmf_l1_ratio)

    elif method == 'compressed_nmf':
        Ain, Cin, _, b_in, f_in = compressedNMF(
            Y_ds, nr=K, nb=nb, max_iter_snmf=max_iter_snmf,
            sigma_smooth=sigma_smooth_snmf, remove_baseline=remove_baseline, perc_baseline=perc_baseline_snmf)

    elif method == 'graph_nmf':
        Ain, Cin, _, b_in, f_in = graphNMF(
            Y_ds, nr=K, nb=nb, max_iter_snmf=max_iter_snmf, lambda_gnmf=lambda_gnmf,
            sigma_smooth=sigma_smooth_snmf, remove_baseline=remove_baseline,
            perc_baseline=perc_baseline_snmf, SC_kernel=SC_kernel,
            SC_sigma=SC_sigma, SC_use_NN=SC_use_NN, SC_nnn=SC_nnn,
            SC_normalize=SC_normalize, SC_thr=SC_thr)

    elif method == 'pca_ica':
        Ain, Cin, _, b_in, f_in = ICA_PCA(
            Y_ds, nr=K, sigma_smooth=sigma_smooth_snmf, truncate=2, fun='logcosh', tol=1e-10,
            max_iter=max_iter_snmf, remove_baseline=True, perc_baseline=perc_baseline_snmf, nb=nb)

    else:
        print(method)
        raise Exception("Unsupported initialization method")

    K = np.shape(Ain)[-1]

    if Ain.size > 0 and not center_psf and ssub != 1:

        Ain = np.reshape(Ain, ds + (K,), order='F')

        if len(ds) == 2:
            Ain = resize(Ain, d + (K,))

        else:  # resize only deals with 2D images, hence apply resize twice
            Ain = np.reshape([resize(a, d[1:] + (K,))
                              for a in Ain], (ds[0], d[1] * d[2], K), order='F')
            Ain = resize(Ain, (d[0], d[1] * d[2], K))

        Ain = np.reshape(Ain, (np.prod(d), K), order='F')

    sparse_b = spr.issparse(b_in)
    if (nb > 0 or nb == -1) and (ssub != 1 or tsub != 1):
        b_in = np.reshape(b_in.toarray() if sparse_b else b_in, ds + (-1,), order='F')

        if len(ds) == 2:
            b_in = resize(b_in, d + (b_in.shape[-1],))
        else:
            b_in = np.reshape([resize(b, d[1:] + (b_in.shape[-1],))
                               for b in b_in], (ds[0], d[1] * d[2], -1), order='F')
            b_in = resize(b_in, (d[0], d[1] * d[2], b_in.shape[-1]))

        b_in = np.reshape(b_in, (np.prod(d), -1), order='F')
        if sparse_b:
            b_in = spr.csc_matrix(b_in)

        f_in = resize(np.atleast_2d(f_in), [b_in.shape[-1], T])

    elif nb == 0:
        b_in = np.empty((np.prod(d), 0), dtype=np.float32, order='F')
        f_in = np.empty((0, T), dtype=np.float32)

    if Ain.size > 0:
        Cin = resize(Cin, [K, T])
        center = np.asarray(
            [center_of_mass(a.reshape(d, order='F')) for a in Ain.T])
    else:
        Cin = np.empty((K, T), dtype=np.float32)
        center = []

    if normalize_init:
        if Ain.size > 0:
            Ain = Ain * np.reshape(img, (np.prod(d), -1), order='F')
        if sparse_b:
            b_in = spr.diags(img.ravel(order='F')).dot(b_in)
        else:
            b_in = b_in * np.reshape(img, (np.prod(d), -1), order='F')
    if method == 'corr_pnr' and ring_size_factor is not None:
        return scipy.sparse.csc_matrix(Ain), Cin, b_in, f_in, center, extra_1p
    else:
        return scipy.sparse.csc_matrix(Ain), Cin, b_in, f_in, center

def ICA_PCA(Y_ds, nr, sigma_smooth=(.5, .5, .5), truncate=2, fun='logcosh',
            max_iter=1000, tol=1e-10, remove_baseline=True, perc_baseline=20, nb=1):
    """ Initialization using ICA and PCA. DOES NOT WORK WELL WORK IN PROGRESS"

    Args:
        Y_ds
        nr
        sigma_smooth
        truncate
        fun
        max_iter
        tol
        remove_baseline
        perc_baseline
        nb
    """

    print("not a function to use in the moment ICA PCA \n")
    m = scipy.ndimage.gaussian_filter(np.transpose(
        Y_ds, [2, 0, 1]), sigma=sigma_smooth, mode='nearest', truncate=truncate)
    if remove_baseline:
        bl = np.percentile(m, perc_baseline, axis=0)
        m1 = np.maximum(0, m - bl)
    else:
        bl = np.zeros(m.shape[1:])
        m1 = m
    pca_comp = nr

    T, d1, d2 = np.shape(m1)
    d = d1 * d2
    yr = np.reshape(m1, [T, d], order='F')

    [U, S, V] = scipy.sparse.linalg.svds(yr, pca_comp)
    S = np.diag(S)
    whiteningMatrix = np.dot(scipy.linalg.inv(S), U.T)
    whitesig = np.dot(whiteningMatrix, yr)
    f_ica = FastICA(whiten=False, fun=fun, max_iter=max_iter, tol=tol)
    S_ = f_ica.fit_transform(whitesig.T)
    A_in = f_ica.mixing_
    A_in = np.dot(A_in, whitesig)

    masks = np.reshape(A_in.T, (d1, d2, pca_comp),
                       order='F').transpose([2, 0, 1])

    masks = np.array(caiman.base.rois.extractROIsFromPCAICA(masks)[0])

    if masks.size > 0:

        C_in = caiman.base.movies.movie(
            m1).extract_traces_from_masks(np.array(masks)).T
        A_in = np.reshape(masks, [-1, d1 * d2], order='F').T

    else:

        A_in = np.zeros([d1 * d2, pca_comp])
        C_in = np.zeros([pca_comp, T])

    m1 = yr.T - A_in.dot(C_in) + np.maximum(0, bl.flatten())[:, np.newaxis]

    model = NMF(n_components=nb, init='random', random_state=0)

    b_in = model.fit_transform(np.maximum(m1, 0)).astype(np.float32)
    f_in = model.components_.astype(np.float32)

    center = caiman.base.rois.com(A_in, d1, d2)

    return A_in, C_in, center, b_in, f_in

def sparseNMF(Y_ds, nr, max_iter_snmf=200, alpha=0.5, sigma_smooth=(.5, .5, .5),
              remove_baseline=True, perc_baseline=20, nb=1, truncate=2, l1_ratio=0.0):
    """
    Initialization using sparse NMF

    Args:
        Y_ds: nd.array or movie (x, y, T [,z])
            data

        nr: int
            number of components

        max_iter_snm: int
            number of iterations

        alpha_snmf:
            sparsity regularizer (alpha_W)

        sigma_smooth_snmf:
            smoothing along z,x, and y (.5,.5,.5)

        perc_baseline_snmf:
            percentile to remove from movie before NMF

        nb: int
            Number of background components

        l1_ratio: float
            Parameter to NMF call

    Returns:
        A: np.array
            2d array of size (# of pixels) x nr with the spatial components.
            Each column is ordered columnwise (matlab format, order='F')

        C: np.array
            2d array of size nr X T with the temporal components

        center: np.array
            2d array of size nr x 2 [ or 3] with the components centroids
    """
    logger = logging.getLogger("caiman")

    m = scipy.ndimage.gaussian_filter(np.transpose(
        Y_ds, np.roll(np.arange(Y_ds.ndim), 1)), sigma=sigma_smooth,
        mode='nearest', truncate=truncate)
    if remove_baseline:
        logger.info('Removing baseline')
        bl = np.percentile(m, perc_baseline, axis=0)
        m1 = np.maximum(0, m - bl)
    else:
        logger.info('Not removing baseline')
        bl = np.zeros(m.shape[1:])
        m1 = m

    T, dims = m1.shape[0], m1.shape[1:]
    d = np.prod(dims)
    yr = np.reshape(m1, [T, d], order='F')

    logger.info(f"Running SparseNMF with alpha_W={alpha} and l1_ratio={l1_ratio}")
    mdl = NMF(n_components=nr, 
              verbose=False, 
              init='nndsvd', 
              tol=1e-10,
              max_iter=max_iter_snmf, 
              shuffle=False, 
              alpha_W=alpha, 
              l1_ratio=l1_ratio)
    C = mdl.fit_transform(yr).T
    A = mdl.components_.T
    A_in = A
    C_in = C

    m1 = yr.T - A_in.dot(C_in) + np.maximum(0, bl.flatten())[:, np.newaxis]
    model = NMF(n_components=nb, init='random',
                random_state=0, max_iter=max_iter_snmf)
    b_in = model.fit_transform(np.maximum(m1, 0)).astype(np.float32)
    f_in = model.components_.astype(np.float32)
    center = caiman.base.rois.com(A_in, *dims)

    return A_in, C_in, center, b_in, f_in


def compressedNMF(Y_ds, nr, r_ov=10, max_iter_snmf=500,
                  sigma_smooth=(.5, .5, .5), remove_baseline=False,
                  perc_baseline=20, nb=1, truncate=2, tol=1e-3):
    logger = logging.getLogger("caiman")
    m = scipy.ndimage.gaussian_filter(np.transpose(
            Y_ds, np.roll(np.arange(Y_ds.ndim), 1)), sigma=sigma_smooth,
            mode='nearest', truncate=truncate)
    if remove_baseline:
        logger.info('REMOVING BASELINE')
        bl = np.percentile(m, perc_baseline, axis=0)
        m = np.maximum(0, m - bl)
    else:
        logger.info('NOT REMOVING BASELINE')
        bl = np.zeros(m.shape[1:])

    T, dims = m.shape[0], m.shape[1:]
    d = np.prod(dims)
    yr = np.reshape(m, [T, d], order='F')
#    L = randomized_range_finder(yr, nr + r_ov, 3)
#    R = randomized_range_finder(yr.T, nr + r_ov, 3)
#    Yt = L.T.dot(yr).dot(R)
#    c_in, a_in = compressive_nmf(Yt, L, R.T, nr)
#    C_in = L.dot(c_in)
#    A_in = a_in.dot(R.T)
#    A_in = A_in.T
#    C_in = C_in.T
    A, C, USV = nnsvd_init(yr, nr, r_ov=r_ov)
    W_r = np.random.randn(d, nr + r_ov)
    W_l = np.random.randn(T, nr + r_ov)
    US = USV[0]*USV[1]
    YYt = US.dot(USV[2].dot(USV[2].T)).dot(US.T)
#    YYt = yr.dot(yr.T)

    B = YYt.dot(YYt.dot(US.dot(USV[2].dot(W_r))))
    PC, _ = np.linalg.qr(B)

    B = USV[2].T.dot(US.T.dot(YYt.dot(YYt.dot(W_l))))
    PA, _ = np.linalg.qr(B)
#    mdl = NMF(n_components=nr, verbose=False, init='nndsvd', tol=1e-10,
#              max_iter=1)
#    C = mdl.fit_transform(yr).T
#    A = mdl.components_.T

    yrPA = yr.dot(PA)
    yrPC = PC.T.dot(yr)
    for it in range(max_iter_snmf):

        C__ = C.copy()
        A__ = A.copy()
        C_ = C.dot(PC)
        A_ = PA.T.dot(A)

        C = C*(yrPA.dot(A_)/(C.T.dot(A_.T.dot(A_))+np.finfo(C.dtype).eps)).T
        A = A*(yrPC.T.dot(C_.T))/(A.dot(C_.dot(C_.T)) +  np.finfo(C.dtype).eps)
        nA = np.sqrt((A**2).sum(0))
        A /= nA
        C *= nA[:, np.newaxis]
        if (np.linalg.norm(C - C__)/np.linalg.norm(C__) < tol) & (np.linalg.norm(A - A__)/np.linalg.norm(A__) < tol):
            logger.info(f'Graph NMF converged after {it + 1} iterations')
            break
    A_in = A
    C_in = C

    m1 = yr.T - A_in.dot(C_in) + np.maximum(0, bl.flatten(order='F'))[:, np.newaxis]
    model = NMF(n_components=nb, init='random',
                random_state=0, max_iter=max_iter_snmf)
    b_in = model.fit_transform(np.maximum(m1, 0)).astype(np.float32)
    f_in = model.components_.astype(np.float32)
    center = caiman.base.rois.com(A_in, *dims)

    return A_in, C_in, center, b_in, f_in


def graphNMF(Y_ds, nr, max_iter_snmf=500, lambda_gnmf=1,
             sigma_smooth=(.5, .5, .5), remove_baseline=True,
             perc_baseline=20, nb=1, truncate=2, tol=1e-3, SC_kernel='heat',
             SC_normalize=True, SC_thr=0, SC_sigma=1, SC_use_NN=False,
             SC_nnn=20):
    logger = logging.getLogger("caiman")
    m = scipy.ndimage.gaussian_filter(np.transpose(
    Y_ds, np.roll(np.arange(Y_ds.ndim), 1)), sigma=sigma_smooth,
    mode='nearest', truncate=truncate)
    if remove_baseline:
        logger.info('REMOVING BASELINE')
        bl = np.percentile(m, perc_baseline, axis=0)
        m1 = np.maximum(0, m - bl)
    else:
        logger.info('NOT REMOVING BASELINE')
        bl = np.zeros(m.shape[1:])
        m1 = m
    T, dims = m1.shape[0], m1.shape[1:]
    d = np.prod(dims)
    yr = np.reshape(m1, [T, d], order='F')
    mdl = NMF(n_components=nr, verbose=False, init='nndsvd', tol=1e-10,
              max_iter=5)
    C = mdl.fit_transform(yr).T
    A = mdl.components_.T
    W = caiman.source_extraction.cnmf.utilities.fast_graph_Laplacian_patches(
            [np.reshape(m, [T, d], order='F').T, [], SC_kernel, SC_sigma, SC_thr,
             SC_nnn, SC_normalize, SC_use_NN])
    D = scipy.sparse.spdiags(W.sum(0), 0, W.shape[0], W.shape[0])
    for it in range(max_iter_snmf):
        C_ = C.copy()
        A_ = A.copy()
        C = C*(yr.dot(A)/(C.T.dot(A.T.dot(A))+np.finfo(C.dtype).eps)).T
        A = A*(yr.T.dot(C.T) + lambda_gnmf*(W.dot(A)))/(A.dot(C.dot(C.T)) + lambda_gnmf*D.dot(A) + np.finfo(C.dtype).eps)
        nA = np.sqrt((A**2).sum(0))
        A /= nA
        C *= nA[:, np.newaxis]
        if (np.linalg.norm(C - C_)/np.linalg.norm(C_) < tol) & (np.linalg.norm(A - A_)/np.linalg.norm(A_) < tol):
            logger.info(f'Graph NMF converged after {it + 1} iterations')
            break
    A_in = A
    C_in = C

    m1 = yr.T - A_in.dot(C_in) + np.maximum(0, bl.flatten(order='F'))[:, np.newaxis]
    model = NMF(n_components=nb, init='random',
                random_state=0, max_iter=max_iter_snmf)
    b_in = model.fit_transform(np.maximum(m1, 0)).astype(np.float32)
    f_in = model.components_.astype(np.float32)
    center = caiman.base.rois.com(A_in, *dims)

    return A_in, C_in, center, b_in, f_in

def greedyROI(Y, nr=30, gSig=[5, 5], gSiz=[11, 11], nIter=5, kernel=None, nb=1,
              rolling_sum=False, rolling_length=100, seed_method='auto'):
    """
    Greedy initialization of spatial and temporal components using spatial Gaussian filtering

    Args:
        Y: np.array
            3d or 4d array of fluorescence data with time appearing in the last axis.

        nr: int
            number of components to be found

        gSig: scalar or list of integers
            standard deviation of Gaussian kernel along each axis

        gSiz: scalar or list of integers
            size of spatial component

        nIter: int
            number of iterations when refining estimates

        kernel: np.ndarray
            User specified kernel to be used, if present, instead of Gaussian (default None)

        nb: int
            Number of background components

        rolling_max: boolean
            Detect new components based on a rolling sum of pixel activity (default: True)

        rolling_length: int
            Length of rolling window (default: 100)

        seed_method: str {'auto', 'manual', 'semi'}
            methods for choosing seed pixels
            'semi' detects nr components automatically and allows to add more manually
            if running as notebook 'semi' and 'manual' require a backend that does not
            inline figures, e.g. %matplotlib tk

    Returns:
        A: np.array
            2d array of size (# of pixels) x nr with the spatial components. Each column is
            ordered columnwise (matlab format, order='F')

        C: np.array
            2d array of size nr X T with the temporal components

        center: np.array
            2d array of size nr x 2 [ or 3] with the components centroids

    Author:
        Eftychios A. Pnevmatikakis and Andrea Giovannucci based on a matlab implementation by Yuanjun Gao
            Simons Foundation, 2015

    See Also:
        http://www.cell.com/neuron/pdf/S0896-6273(15)01084-3.pdf


    """
    logger = logging.getLogger("caiman")
    logger.info("Greedy initialization of spatial and temporal components using spatial Gaussian filtering")
    d = np.shape(Y)
    Y[np.isnan(Y)] = 0
    med = np.median(Y, axis=-1)
    Y = Y - med[..., np.newaxis]
    gHalf = np.array(gSiz) // 2
    gSiz = 2 * gHalf + 1
    # we initialize every values to zero
    if seed_method.lower() == 'manual':
        nr = 0
    A = np.zeros((np.prod(d[0:-1]), nr), dtype=np.float32)
    C = np.zeros((nr, d[-1]), dtype=np.float32)
    center = np.zeros((nr, Y.ndim - 1), dtype='uint16')

    rho = imblur(Y, sig=gSig, siz=gSiz, nDimBlur=Y.ndim - 1, kernel=kernel)
    if rolling_sum:
        logger.info('Using rolling sum for initialization (RollingGreedyROI)')
        rolling_filter = np.ones(
            (rolling_length), dtype=np.float32) / rolling_length
        rho_s = scipy.signal.lfilter(rolling_filter, 1., rho**2)
        v = np.amax(rho_s, axis=-1)
    else:
        logger.info('Using total sum for initialization (GreedyROI)')
        v = np.sum(rho**2, axis=-1)

    if seed_method.lower() != 'manual':
        for k in range(nr):
            # we take the highest value of the blurred total image and we define it as
            # the center of the neuron
            ind = np.argmax(v)
            ij = np.unravel_index(ind, d[0:-1])
            for c, i in enumerate(ij):
                center[k, c] = i

            # we define a squared size around it
            ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                     for c in range(len(ij))]
            # we create an array of it (fl like) and compute the trace like the pixel ij through time
            dataTemp = np.array(
                Y[tuple([slice(*a) for a in ijSig])].copy(), dtype=np.float32)
            traceTemp = np.array(np.squeeze(rho[ij]), dtype=np.float32)

            coef, score = finetune(dataTemp, traceTemp, nIter=nIter)
            C[k, :] = np.squeeze(score)
            dataSig = coef[..., np.newaxis] * \
                score.reshape([1] * (Y.ndim - 1) + [-1])
            xySig = np.meshgrid(*[np.arange(s[0], s[1])
                                  for s in ijSig], indexing='xy')
            arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze()
                            for s in xySig], dtype=int)
            indices = np.ravel_multi_index(arr, d[0:-1], order='F')

            A[indices, k] = np.reshape(
                coef, (1, np.size(coef)), order='C').squeeze()
            Y[tuple([slice(*a) for a in ijSig])] -= dataSig.copy()
            if k < nr - 1 or seed_method.lower() != 'auto':
                Mod = [[np.maximum(ij[c] - 2 * gHalf[c], 0),
                        np.minimum(ij[c] + 2 * gHalf[c] + 1, d[c])] for c in range(len(ij))]
                ModLen = [m[1] - m[0] for m in Mod]
                Lag = [ijSig[c] - Mod[c][0] for c in range(len(ij))]
                dataTemp = np.zeros(ModLen)
                dataTemp[tuple([slice(*a) for a in Lag])] = coef
                dataTemp = imblur(dataTemp[..., np.newaxis],
                                  sig=gSig, siz=gSiz, kernel=kernel)
                temp = dataTemp * score.reshape([1] * (Y.ndim - 1) + [-1])
                rho[tuple([slice(*a) for a in Mod])] -= temp.copy()
                if rolling_sum:
                    rho_filt = scipy.signal.lfilter(
                        rolling_filter, 1., rho[tuple([slice(*a) for a in Mod])]**2)
                    v[tuple([slice(*a) for a in Mod])] = np.amax(rho_filt, axis=-1)
                else:
                    v[tuple([slice(*a) for a in Mod])] = \
                        np.sum(rho[tuple([slice(*a) for a in Mod])]**2, axis=-1)
        center = center.tolist()
    else:
        center = []

    if seed_method.lower() in ('manual', 'semi'):
        # manually pick seed pixels
        while True:
            fig = plt.figure(figsize=(13, 12))
            ax = plt.axes([.04, .04, .95, .18])
            sc_all = []
            sc_select = []
            plt.axes([0, .25, 1, .7])
            sc_all.append(plt.scatter([], [], color='g'))
            sc_select.append(plt.scatter([], [], color='r'))
            plt.imshow(v, interpolation=None, vmin=np.percentile(v[~np.isnan(v)], 1),
                       vmax=np.percentile(v[~np.isnan(v)], 99), cmap='gray')
            if len(center):
                plt.scatter(*np.transpose(center)[::-1], c='b')
            plt.axis('off')
            plt.suptitle(
                'Click to add component. Click again on it to remove it. Press any key to update figure. Add more components, or press any key again when done.')
            centers = []

            def key_press(event):
                plt.close(fig)

            def onclick(event):
                new_center = int(round(event.xdata)), int(round(event.ydata))
                if new_center in centers:
                    centers.remove(new_center)
                else:
                    centers.append(new_center)
                print(centers)
                ax.clear()
                if len(centers): 
                    ax.plot(Y[centers[-1][1], centers[-1][0]], c='r')
                    for sc in sc_all:
                        sc.set_offsets(centers)
                    for sc in sc_select:
                        sc.set_offsets(centers[-1:])
                else:
                    for sc in sc_all:
                        sc.set_offsets(np.zeros((0,2)))
                    for sc in sc_select:
                        sc.set_offsets(np.zeros((0,2)))                    
                plt.draw()

            cid = fig.canvas.mpl_connect('key_press_event', key_press)
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show(block=True)

            if centers == []:
                break
            centers = np.array(centers)[:,::-1].tolist()
            center += centers
            
            # we initialize every values to zero
            A_ = np.zeros((np.prod(d[0:-1]), len(centers)), dtype=np.float32)
            C_ = np.zeros((len(centers), d[-1]), dtype=np.float32)
            for k, ij in enumerate(centers):
                # we define a squared size around it
                ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                         for c in range(len(ij))]
                # we create an array of it (fl like) and compute the trace like the pixel ij through time
                dataTemp = np.array(
                    Y[tuple([slice(*a) for a in ijSig])].copy(), dtype=np.float32)
                traceTemp = np.array(np.squeeze(rho[tuple(ij)]), dtype=np.float32)

                coef, score = finetune(dataTemp, traceTemp, nIter=nIter)
                C_[k, :] = np.squeeze(score)
                dataSig = coef[..., np.newaxis] * \
                    score.reshape([1] * (Y.ndim - 1) + [-1])
                xySig = np.meshgrid(*[np.arange(s[0], s[1])
                                      for s in ijSig], indexing='xy')
                arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze()
                                for s in xySig], dtype=int)
                indices = np.ravel_multi_index(arr, d[0:-1], order='F')

                A_[indices, k] = np.reshape(
                    coef, (1, np.size(coef)), order='C').squeeze()
                Y[tuple([slice(*a) for a in ijSig])] -= dataSig.copy()
                
                Mod = [[np.maximum(ij[c] - 2 * gHalf[c], 0),
                        np.minimum(ij[c] + 2 * gHalf[c] + 1, d[c])] for c in range(len(ij))]
                ModLen = [m[1] - m[0] for m in Mod]
                Lag = [ijSig[c] - Mod[c][0] for c in range(len(ij))]
                dataTemp = np.zeros(ModLen)
                dataTemp[tuple([slice(*a) for a in Lag])] = coef
                dataTemp = imblur(dataTemp[..., np.newaxis],
                                  sig=gSig, siz=gSiz, kernel=kernel)
                temp = dataTemp * score.reshape([1] * (Y.ndim - 1) + [-1])
                rho[tuple([slice(*a) for a in Mod])] -= temp.copy()
                if rolling_sum:
                    rho_filt = scipy.signal.lfilter(
                        rolling_filter, 1., rho[tuple([slice(*a) for a in Mod])]**2)
                    v[tuple([slice(*a) for a in Mod])] = np.amax(rho_filt, axis=-1)
                else:
                    v[tuple([slice(*a) for a in Mod])] = \
                        np.sum(rho[tuple([slice(*a) for a in Mod])]**2, axis=-1)
            A = np.concatenate([A, A_], 1)
            C = np.concatenate([C, C_])

    res = np.reshape(Y, (np.prod(d[0:-1]), d[-1]),
                     order='F') + med.flatten(order='F')[:, None]
    if nb > 0:
        model = NMF(n_components=nb, init='nndsvdar')
        b_in = model.fit_transform(np.maximum(res, 0)).astype(np.float32)
        f_in = model.components_.astype(np.float32)
    else:
        b_in = np.empty((A.shape[0], 0), dtype=np.float32, order='F')
        f_in = np.empty((0, C.shape[1]), dtype=np.float32)
    return A, C, np.array(center, dtype='uint16'), b_in, f_in

def finetune(Y, cin, nIter=5):
    """compute a initialized version of A and C

    Args:
        Y:  D1*d2*T*K patches

        c: array T*K
            the initial calcium traces

        nIter: int
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
    a: array (d1,D2) the computed A as l2(Y*C)/Y*C

    c: array(T) C as the sum of As on x*y axis
"""
    debug_ = False
    if debug_:
        import os
        f = open('_LOG_1_' + str(os.getpid()), 'w+')
        f.write('Y:' + str(np.mean(Y)) + '\n')
        f.write('cin:' + str(np.mean(cin)) + '\n')
        f.close()

    # we compute the multiplication of patches per traces ( non negatively )
    for _ in range(nIter):
        a = np.maximum(np.dot(Y, cin), 0)
        a = a / np.sqrt(np.sum(a**2) + np.finfo(np.float32).eps)  # compute the l2/a
        # c as the variation of those patches
        cin = np.sum(Y * a[..., np.newaxis], tuple(np.arange(Y.ndim - 1)))

    return a, cin

def imblur(Y, sig=5, siz=11, nDimBlur=None, kernel=None, opencv=True):
    """
    Spatial filtering with a Gaussian or user defined kernel

    The parameters are specified in GreedyROI

    Args:
        Y: np.ndarray
            d1 x d2 [x d3] x T movie, raw data.

        sig: [optional] list,tuple
            half size of neurons

        siz: [optional] list,tuple
            size of filter kernel (default 2*sig + 1).

        nDimBlur: [optional]
            if you want to specify the number of dimension

        kernel: [optional]
            if you want to specify a kernel

        opencv: [optional]
            if you want to process to the blur using OpenCV method

    Returns:
        the blurred image
    """
    # TODO: document (jerem)
    if kernel is None:
        if nDimBlur is None:
            nDimBlur = Y.ndim - 1
        else:
            nDimBlur = np.min((Y.ndim, nDimBlur))

        if np.isscalar(sig):
            sig = sig * np.ones(nDimBlur)

        if np.isscalar(siz):
            siz = siz * np.ones(nDimBlur)

        X = Y.copy()
        if opencv and nDimBlur == 2:
            if X.ndim > 2:
                # if we are on a video we repeat for each frame
                for frame in range(X.shape[-1]):
                    if sys.version_info >= (3, 0):
                        X[:, :, frame] = cv2.GaussianBlur(X[:, :, frame], tuple(
                            siz), sig[0], None, sig[1], cv2.BORDER_CONSTANT)
                    else:
                        X[:, :, frame] = cv2.GaussianBlur(X[:, :, frame], tuple(siz), sig[
                                                          0], sig[1], cv2.BORDER_CONSTANT, 0)

            else:
                if sys.version_info >= (3, 0):
                    X = cv2.GaussianBlur(
                        X, tuple(siz), sig[0], None, sig[1], cv2.BORDER_CONSTANT)
                else:
                    X = cv2.GaussianBlur(
                        X, tuple(siz), sig[0], sig[1], cv2.BORDER_CONSTANT, 0)
        else:
            for i in range(nDimBlur):
                h = np.exp(-np.arange(-np.floor(siz[i] / 2),
                                       np.floor(siz[i] / 2) + 1)**2 / (2 * sig[i]**2))
                h /= np.sqrt(h.dot(h))
                shape = [1] * len(Y.shape)
                shape[i] = -1
                X = correlate(X, h.reshape(shape), mode='constant')

    else:
        X = correlate(Y, kernel[..., np.newaxis], mode='constant')
        # for t in range(np.shape(Y)[-1]):
        #    X[:,:,t] = correlate(Y[:,:,t],kernel,mode='constant', cval=0.0)

    return X

def hals(Y, A, C, b, f, bSiz=3, maxIter=5):
    """ Hierarchical alternating least square method for solving NMF problem

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

       maxIter: maximum iteration of iterating HALS.

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

    ind_A = spr.csc_matrix(ind_A)  # indicator of nonzero pixels

    def HALS4activity(Yr, A, C, iters=2):
        U = A.T.dot(Yr)
        V = A.T.dot(A) + np.finfo(A.dtype).eps
        for _ in range(iters):
            for m in range(len(U)):  # neurons and background
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
    Cf = np.r_[C, f]
    for _ in range(maxIter):
        Cf = HALS4activity(np.reshape(
            Y, (np.prod(dims), T), order='F'), Ab, Cf)
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)

    if nb == 0:
        return Ab, Cf, b, f
    else:
        return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)


@profile
def greedyROI_corr(Y, Y_ds, max_number=None, gSiz=None, gSig=None, center_psf=True,
                   min_corr=None, min_pnr=None, seed_method='auto',
                   min_pixel=3, bd=0, thresh_init=2, ring_size_factor=None, nb=1, options=None,
                   sn=None, save_video=False, video_name='initialization.mp4', ssub=1,
                   ssub_B=2, init_iter=2):
    """
    initialize neurons based on pixels' local correlations and peak-to-noise ratios.

    Args:
        *** see init_neurons_corr_pnr for descriptions of following input arguments ***
        data:
        max_number:
        gSiz:
        gSig:
        center_psf:
        min_corr:
        min_pnr:
        seed_method:
        min_pixel:
        bd:
        thresh_init:
        swap_dim:
        save_video:
        video_name:
        *** see init_neurons_corr_pnr for descriptions of above input arguments ***

        ring_size_factor: float
            it's the ratio between the ring radius and neuron diameters.
        ring_model: Boolean
            True indicates using ring model to estimate the background
            components.
        nb: integer
            number of background components for approximating the background using NMF model
            for nb=0 the exact background of the ringmodel (b0 and W) is returned
            for nb=-1 the full rank background B is returned
            for nb<-1 no background is returned
        ssub_B: int, optional
            downsampling factor for 1-photon imaging background computation
        init_iter: int, optional
            number of iterations for 1-photon imaging initialization
    """
    logger = logging.getLogger("caiman")
    if min_corr is None or min_pnr is None:
        raise Exception(
            'Either min_corr or min_pnr are None. Both of them must be real numbers.')

    logger.info('One photon initialization (GreedyCorr)')
    o = options['temporal'].copy()
    o['s_min'] = None
    if o['p'] > 1:
        o['p'] = 1
    A, C, _, _, center = init_neurons_corr_pnr(
        Y_ds, max_number=max_number, gSiz=gSiz, gSig=gSig,
        center_psf=center_psf, min_corr=min_corr,
        min_pnr=min_pnr * np.sqrt(np.size(Y) / np.size(Y_ds)),
        seed_method=seed_method, deconvolve_options=o,
        min_pixel=min_pixel, bd=bd, thresh_init=thresh_init,
        swap_dim=True, save_video=save_video, video_name=video_name)

    dims = Y.shape[:2]
    T = Y.shape[-1]
    d1, d2, total_frames = Y_ds.shape
    tsub = int(round(float(T) / total_frames))
    B = Y_ds.reshape((-1, total_frames), order='F') - A.dot(C)

    if ring_size_factor is not None:
        # background according to ringmodel
        logger.info('Computing ring model background')
        W, b0 = compute_W(Y_ds.reshape((-1, total_frames), order='F'),
                          A, C, (d1, d2), ring_size_factor * gSiz, ssub=ssub_B)

        def compute_B(b0, W, B):  # actually computes -B to efficiently compute Y-B in place
            if ssub_B == 1:
                B = -b0[:, None] - W.dot(B - b0[:, None])  # "-B"
            else:
                B = -b0[:, None] - (np.repeat(np.repeat(W.dot(
                    downscale(B.reshape((d1, d2, B.shape[-1]), order='F'),
                              (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F') -
                    downscale(b0.reshape((d1, d2), order='F'),
                              (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
                    .reshape(((d1 - 1) // ssub_B + 1, (d2 - 1) // ssub_B + 1, -1), order='F'),
                    ssub_B, 0), ssub_B, 1)[:d1, :d2].reshape((-1, B.shape[-1]), order='F'))  # "-B"
            return B

        B = compute_B(b0, W, B)  # "-B"
        B += Y_ds.reshape((-1, total_frames), order='F')  # "Y-B"

        logger.info('Updating spatial components')
        A, _, C, _ = caiman.source_extraction.cnmf.spatial.update_spatial_components(
            B, C=C, f=np.zeros((0, total_frames), np.float32), A_in=A,
            sn=np.sqrt(downscale((sn**2).reshape(dims, order='F'),
                                 tuple([ssub] * len(dims))).ravel() / tsub) / ssub,
            b_in=np.zeros((d1 * d2, 0), np.float32),
            dview=None, dims=(d1, d2), **options['spatial'])
        logger.info('Updating temporal components')
        C, A = caiman.source_extraction.cnmf.temporal.update_temporal_components(
            B, spr.csc_matrix(A, dtype=np.float32),
            np.zeros((d1 * d2, 0), np.float32),
            C, np.zeros((0, total_frames), np.float32),
            dview=None, bl=None, c1=None, sn=None, g=None, **o)[:2]

        # find more neurons in residual
        # print('Compute Residuals')
        for i in range(init_iter - 1):
            if max_number is not None:
                max_number -= A.shape[-1]
            if max_number != 0:
                if i == init_iter-2 and seed_method.lower()[:4] == 'semi':
                    seed_method, min_corr, min_pnr = 'manual', 0, 0
                logger.info('Searching for more neurons in the residual')
                A_R, C_R, _, _, center_R = init_neurons_corr_pnr(
                    (B - A.dot(C)).reshape(Y_ds.shape, order='F'),
                    max_number=max_number, gSiz=gSiz, gSig=gSig,
                    center_psf=center_psf, min_corr=min_corr, min_pnr=min_pnr,
                    seed_method=seed_method, deconvolve_options=o,
                    min_pixel=min_pixel, bd=bd, thresh_init=thresh_init,
                    swap_dim=True, save_video=save_video, video_name=video_name)
                A = spr.coo_matrix(np.concatenate((A.toarray(), A_R), 1))
                C = np.concatenate((C, C_R), 0)

        # 1st iteration on decimated data
        logger.info('Merging components')
        A, C = caiman.source_extraction.cnmf.merging.merge_components(
            B, A, [], C, None, [], C, [], o, options['spatial'],
            dview=None, thr=options['merging']['merge_thr'], mx=np.inf, fast_merge=True)[:2]
        A = A.astype(np.float32)
        C = C.astype(np.float32)
        logger.info('Updating spatial components')
        A, _, C, _ = caiman.source_extraction.cnmf.spatial.update_spatial_components(
            B, C=C, f=np.zeros((0, total_frames), np.float32), A_in=A,
            sn=np.sqrt(downscale((sn**2).reshape(dims, order='F'),
                                 tuple([ssub] * len(dims))).ravel() / tsub) / ssub,
            b_in=np.zeros((d1 * d2, 0), np.float32),
            dview=None, dims=(d1, d2), **options['spatial'])
        A = A.astype(np.float32)
        logger.info('Updating temporal components')
        C, A = caiman.source_extraction.cnmf.temporal.update_temporal_components(
            B, spr.csc_matrix(A),
            np.zeros((d1 * d2, 0), np.float32),
            C, np.zeros((0, total_frames), np.float32),
            dview=None, bl=None, c1=None, sn=None, g=None, **o)[:2]

        logger.info('Recomputing background')
        # background according to ringmodel
        W, b0 = compute_W(Y_ds.reshape((-1, total_frames), order='F'),
                          A, C, (d1, d2), ring_size_factor * gSiz, ssub=ssub_B)

        # 2nd iteration on non-decimated data
        K = C.shape[0]
        if T > total_frames:
            C = np.repeat(C, tsub, 1)[:, :T]
            Ys = (Y if ssub == 1 else downscale(
                Y, (ssub, ssub, 1))).reshape((-1, T), order='F')
            # N.B: upsampling B in space is fine, but upsampling in time doesn't work well,
            # cause the error in upsampled background can be of similar size as neural signal
            B = Ys - A.dot(C)
        else:
            B = Y_ds.reshape((-1, T), order='F') - A.dot(C)
        B = compute_B(b0, W, B)  # "-B"
        if nb > 0 or nb == -1:
            B0 = -B
        if ssub > 1:
            B = np.reshape(B, (d1, d2, -1), order='F')
            B = (np.repeat(np.repeat(B, ssub, 0), ssub, 1)[:dims[0], :dims[1]]
                 .reshape((-1, T), order='F'))
            A = A.toarray().reshape((d1, d2, K), order='F')
            A = spr.csc_matrix(np.repeat(np.repeat(A, ssub, 0), ssub, 1)[:dims[0], :dims[1]]
                               .reshape((np.prod(dims), K), order='F'))
        B += Y.reshape((-1, T), order='F')  # "Y-B"

        logger.info('Merging components')
        A, C = caiman.source_extraction.cnmf.merging.merge_components(
            B, A, [], C, None, [], C, [], o, options['spatial'],
            dview=None, thr=options['merging']['merge_thr'], mx=np.inf, fast_merge=True)[:2]
        A = A.astype(np.float32)
        C = C.astype(np.float32)
        logger.info('Updating spatial components')
        options['spatial']['se'] = np.ones((1,) * len((d1, d2)), dtype=np.uint8)
        A, _, C, _ = caiman.source_extraction.cnmf.spatial.update_spatial_components(
            B, C=C, f=np.zeros((0, T), np.float32), A_in=A, sn=sn,
            b_in=np.zeros((np.prod(dims), 0), np.float32),
            dview=None, dims=dims, **options['spatial'])
        logger.info('Updating temporal components')
        C, A, b__, f__, S, bl, c1, neurons_sn, g1, YrA, lam__ = \
            caiman.source_extraction.cnmf.temporal.update_temporal_components(
                B, spr.csc_matrix(A, dtype=np.float32),
                np.zeros((np.prod(dims), 0), np.float32), C, np.zeros((0, T), np.float32),
                dview=None, bl=None, c1=None, sn=None, g=None, **options['temporal'])

        A = A.toarray()
        if nb > 0 or nb == -1:
            B = B0

    use_NMF = True
    if nb == -1:
        logger.info('Returning full background')
        b_in = spr.eye(len(B), dtype='float32')
        f_in = B
    elif nb > 0:
        logger.info(f'Estimate low rank background (rank = {nb})')
        print(nb)
        if use_NMF:
            model = NMF(n_components=nb, init='nndsvdar')
            b_in = model.fit_transform(np.maximum(B, 0))
            # f_in = model.components_.squeeze()
            f_in = np.linalg.lstsq(b_in, B)[0]
        else:
            b_in, s_in, f_in = spr.linalg.svds(B, k=nb)
            f_in *= s_in[:, np.newaxis]
    else:
        b_in = np.empty((A.shape[0], 0), order='F')
        f_in = np.empty((0, T))
        if nb == 0:
            logger.info('Returning background as b0 and W')
            return (A, C, center.T, b_in.astype(np.float32), f_in.astype(np.float32),
                    (S.astype(np.float32), bl, c1, neurons_sn, g1, YrA, lam__,
                     W, b0))
        else:
            logger.info("Not returning background")
    return (A, C, center.T, b_in.astype(np.float32), f_in.astype(np.float32),
            None if ring_size_factor is None else
            (S.astype(np.float32), bl, c1, neurons_sn, g1, YrA, lam__))


@profile
def init_neurons_corr_pnr(data, max_number=None, gSiz=15, gSig=None,
                          center_psf=True, min_corr=0.8, min_pnr=10,
                          seed_method='auto', deconvolve_options=None,
                          min_pixel=3, bd=1, thresh_init=2, swap_dim=True,
                          save_video=False, video_name='initialization.mp4',
                          background_filter='disk'):
    """
    using greedy method to initialize neurons by selecting pixels with large
    local correlation and large peak-to-noise ratio
    Args:
        data: np.ndarray (3D)
            the data used for initializing neurons. its dimension can be
            d1*d2*T or T*d1*d2. If it's the latter, swap_dim should be
            False; otherwise, True.
        max_number: integer
            maximum number of neurons to be detected. If None, then the
            algorithm will stop when all pixels are below the thresholds.
        gSiz: float
            average diameter of a neuron
        gSig: float number or a vector with two elements.
            gaussian width of the gaussian kernel used for spatial filtering.
        center_psf: Boolean
            True indicates centering the filtering kernel for background
            removal. This is useful for data with large background
            fluctuations.
        min_corr: float
            minimum local correlation coefficients for selecting a seed pixel.
        min_pnr: float
            minimum peak-to-noise ratio for selecting a seed pixel.
        seed_method: str {'auto', 'manual'} or list/array/tuple of seed pixels
            methods for choosing seed pixels
            if running as notebook 'manual' requires a backend that does not
            inline figures, e.g. %matplotlib tk
        deconvolve_options: dict
            all options for deconvolving temporal traces.
        min_pixel: integer
            minimum number of nonzero pixels for one neuron.
        bd: integer
            pixels that are bd pixels away from the boundary will be ignored for initializing neurons.
        thresh_init: float
            pixel values smaller than thresh_init*noise will be set as 0
            when computing the local correlation image.
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab
            format)
        save_video: Boolean
            save the initialization procedure if it's True
        video_name: str
            name of the video to be saved.

    Returns:
        A: np.ndarray (d1*d2*T)
            spatial components of all neurons
        C: np.ndarray (K*T)
            nonnegative and denoised temporal components of all neurons
        C_raw: np.ndarray (K*T)
            raw calcium traces of all neurons
        S: np.ndarray (K*T)
            deconvolved calcium traces of all neurons
        center: np.ndarray
            center locations of all neurons
    """
    logger = logging.getLogger("caiman")

    if swap_dim:
        d1, d2, total_frames = data.shape
        data_raw = np.transpose(data, [2, 0, 1])
    else:
        total_frames, d1, d2 = data.shape
        data_raw = data

    data_filtered = data_raw.copy()
    if gSig:
        # spatially filter data
        if not isinstance(gSig, list):
            gSig = [gSig, gSig]
        ksize = tuple([int(2 * i) * 2 + 1 for i in gSig])
        # create a spatial filter for removing background

        if center_psf:
            if background_filter == 'box':
                for idx, img in enumerate(data_filtered):
                    data_filtered[idx, ] = cv2.GaussianBlur(
                        img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1) \
                        - cv2.boxFilter(img, ddepth=-1, ksize=ksize, borderType=1)
            else:
                psf = cv2.getGaussianKernel(ksize[0], gSig[0], cv2.CV_32F).dot(
                    cv2.getGaussianKernel(ksize[1], gSig[1], cv2.CV_32F).T)
                ind_nonzero = psf >= psf[0].max()
                psf -= psf[ind_nonzero].mean()
                psf[~ind_nonzero] = 0
                for idx, img in enumerate(data_filtered):
                    data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)
        else:
            for idx, img in enumerate(data_filtered):
                data_filtered[idx, ] = cv2.GaussianBlur(img, ksize=ksize, sigmaX=gSig[0],
                                                        sigmaY=gSig[1], borderType=1)

    # compute peak-to-noise ratio
    data_filtered -= data_filtered.mean(axis=0)
    data_max = np.max(data_filtered, axis=0)
    noise_pixel = get_noise_fft(data_filtered.T, noise_method='mean')[0].T
    pnr = np.divide(data_max, noise_pixel)

    # remove small values and only keep pixels with large fluorescence signals
    tmp_data = np.copy(data_filtered)
    tmp_data[tmp_data < thresh_init * noise_pixel] = 0
    # compute correlation image
    cn = caiman.summary_images.local_correlations_fft(tmp_data, swap_dim=False)
    del(tmp_data)
#    cn[np.isnan(cn)] = 0  # remove abnormal pixels

    # make required copy here, after memory intensive computation of cn
    data_raw = data_raw.copy()

    # screen seed pixels as neuron centers
    v_search = cn * pnr
    v_search[(cn < min_corr) | (pnr < min_pnr)] = 0
    ind_search = (v_search <= 0)  # indicate whether the pixel has
    # been searched before. pixels with low correlations or low PNRs are
    # ignored directly. ind_search[i]=0 means the i-th pixel is still under
    # consideration of being a seed pixel

    # pixels near the boundaries are ignored because of artifacts
    ind_bd = np.zeros(shape=(d1, d2)).astype(
        bool)  # indicate boundary pixels
    if bd > 0:
        ind_bd[:bd, :] = True
        ind_bd[-bd:, :] = True
        ind_bd[:, :bd] = True
        ind_bd[:, -bd:] = True

    ind_search[ind_bd] = 1

    # creating variables for storing the results
    if not max_number:
        # maximum number of neurons
        max_number = np.int32((ind_search.size - ind_search.sum()) / 5)
    Ain = np.zeros(shape=(max_number, d1, d2),
                   dtype=np.float32)  # neuron shapes
    Cin = np.zeros(shape=(max_number, total_frames),
                   dtype=np.float32)  # de-noised traces
    Sin = np.zeros(shape=(max_number, total_frames),
                   dtype=np.float32)  # spiking # activity
    Cin_raw = np.zeros(shape=(max_number, total_frames),
                       dtype=np.float32)  # raw traces
    center = np.zeros(shape=(2, max_number))  # neuron centers

    num_neurons = 0  # number of initialized neurons
    continue_searching = max_number > 0
    min_v_search = min_corr * min_pnr
    [ii, jj] = np.meshgrid(range(d2), range(d1))
    pixel_v = ((ii * 10 + jj) * 1e-5).astype(np.float32)

    if save_video:
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Initialization procedure', artist='CaImAn',
                        comment='CaImAn is cool!')
        writer = FFMpegWriter(fps=2, metadata=metadata)
        # visualize the initialization procedure.
        fig = plt.figure(figsize=(12, 8), facecolor=(0.9, 0.9, 0.9))
        # with writer.saving(fig, "initialization.mp4", 150):
        writer.setup(fig, video_name, 150)

        ax_cn = plt.subplot2grid((2, 3), (0, 0))
        ax_cn.imshow(cn)
        ax_cn.set_title('Correlation')
        ax_cn.set_axis_off()

        ax_pnr_cn = plt.subplot2grid((2, 3), (0, 1))
        ax_pnr_cn.imshow(cn * pnr)
        ax_pnr_cn.set_title('Correlation*PNR')
        ax_pnr_cn.set_axis_off()

        ax_cn_box = plt.subplot2grid((2, 3), (0, 2))
        ax_cn_box.imshow(cn)
        ax_cn_box.set_xlim([54, 63])
        ax_cn_box.set_ylim([54, 63])
        ax_cn_box.set_title('Correlation')
        ax_cn_box.set_axis_off()

        ax_traces = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        ax_traces.set_title('Activity at the seed pixel')

        writer.grab_frame()

    all_centers = []
    while continue_searching:
        tmp_kernel = np.ones(shape=tuple([int(round(gSiz / 4.))] * 2))
        if not isinstance(seed_method, str):
            rsub_max, csub_max = np.transpose(np.round(seed_method).astype(int))
            v_max = cv2.dilate(v_search, tmp_kernel)
            local_max = v_max[rsub_max, csub_max]
            ind_local_max = local_max.argsort()[::-1]
            continue_searching = False 

        elif seed_method.lower() == 'manual':
            # manually pick seed pixels
            fig = plt.figure(figsize=(14,6))
            ax = plt.axes([.03, .05, .96, .22])
            sc_all = []
            sc_select = []
            for i in range(3):
                plt.axes([.01+.34*i, .3, .3, .61])
                sc_all.append(plt.scatter([],[], color='g'))
                sc_select.append(plt.scatter([],[], color='r'))
                title = ('corr*pnr', 'correlation (corr)', 'peak-noise-ratio (pnr)')[i]
                img = (v_search, cn, pnr)[i]
                plt.imshow(img, interpolation=None, vmin=np.percentile(img[~np.isnan(img)], 1),
                           vmax=np.percentile(img[~np.isnan(img)], 99), cmap='gray')
                if len(all_centers):
                    plt.scatter(*np.transpose(all_centers), c='b')
                plt.axis('off')
                plt.title(title)
            plt.suptitle('Click to add component. Click again on it to remove it. Press any key to update figure. Add more components, or press any key again when done.')
            centers = []

            def key_press(event):
                plt.close(fig)

            def onclick(event):
                new_center = int(round(event.xdata)), int(round(event.ydata))
                if new_center in centers:
                    centers.remove(new_center)
                else:
                    centers.append(new_center)
                print(centers)
                ax.clear()
                if len(centers):
                    ax.plot(data_filtered[:, centers[-1][1], centers[-1][0]], c='r')
                    for sc in sc_all:
                        sc.set_offsets(centers)
                    for sc in sc_select:
                        sc.set_offsets(centers[-1:])
                else:
                    for sc in sc_all:
                        sc.set_offsets(np.zeros((0,2)))
                    for sc in sc_select:
                        sc.set_offsets(np.zeros((0,2)))      
                plt.draw()

            cid = fig.canvas.mpl_connect('key_press_event', key_press)
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show(block=True)

            if centers == []:
                break
            all_centers += centers
            csub_max, rsub_max = np.transpose(centers)
            v_max = cv2.dilate(v_search, tmp_kernel)
            local_max = v_max[rsub_max, csub_max]
            ind_local_max = local_max.argsort()[::-1]

        else:
            # local maximum, for identifying seed pixels in following steps
            v_search[(cn < min_corr) | (pnr < min_pnr)] = 0
            # add an extra value to avoid repeated seed pixels within one ROI.
            v_search = cv2.medianBlur(v_search, 3) + pixel_v
            v_search[ind_search] = 0
            v_max = cv2.dilate(v_search, tmp_kernel)

            # automatically select seed pixels as the local maximums
            v_max[(v_search != v_max) | (v_search < min_v_search)] = 0
            v_max[ind_search] = 0
            [rsub_max, csub_max] = v_max.nonzero()  # subscript of seed pixels
            local_max = v_max[rsub_max, csub_max]
            n_seeds = len(local_max)  # number of candidates
            if n_seeds == 0:
                # no more candidates for seed pixels
                break
            else:
                # order seed pixels according to their corr * pnr values
                ind_local_max = local_max.argsort()[::-1]
            img_vmax = np.median(local_max)

        # try to initialization neurons given all seed pixels
        for ith_seed, idx in enumerate(ind_local_max):
            r = rsub_max[idx]
            c = csub_max[idx]
            ind_search[r, c] = True  # this pixel won't be searched
            if v_search[r, c] < min_v_search:
                # skip this pixel if it's not sufficient for being a seed pixel
                continue

            # roughly check whether this is a good seed pixel
            # y0 = data_filtered[:, r, c]
            # if np.max(y0) < thresh_init * noise_pixel[r, c]:
            #     v_search[r, c] = 0
            #     continue
            y0 = np.diff(data_filtered[:, r, c])
            if y0.max() < 3 * y0.std():
                v_search[r, c] = 0
                continue

            # if Ain[:, r, c].sum() > 0 and np.max([scipy.stats.pearsonr(y0, cc)[0]
            #                                       for cc in Cin_raw[Ain[:, r, c] > 0]]) > .7:
            #     v_search[r, c] = 0
            #     continue

            # crop a small box for estimation of ai and ci
            r_min = max(0, r - gSiz)
            r_max = min(d1, r + gSiz + 1)
            c_min = max(0, c - gSiz)
            c_max = min(d2, c + gSiz + 1)
            nr = r_max - r_min
            nc = c_max - c_min
            patch_dims = (nr, nc)  # patch dimension
            data_raw_box = \
                data_raw[:, r_min:r_max, c_min:c_max].reshape(-1, nr * nc)
            data_filtered_box = \
                data_filtered[:, r_min:r_max, c_min:c_max].reshape(-1, nr * nc)
            # index of the seed pixel in the cropped box
            ind_ctr = np.ravel_multi_index((r - r_min, c - c_min),
                                           dims=(nr, nc))

            # neighbouring pixels to update after initializing one neuron
            r2_min = max(0, r - 2 * gSiz)
            r2_max = min(d1, r + 2 * gSiz + 1)
            c2_min = max(0, c - 2 * gSiz)
            c2_max = min(d2, c + 2 * gSiz + 1)

            if save_video:
                ax_pnr_cn.cla()
                ax_pnr_cn.imshow(v_search, vmin=0, vmax=img_vmax)
                ax_pnr_cn.set_title('Neuron %d' % (num_neurons + 1))
                ax_pnr_cn.set_axis_off()
                ax_pnr_cn.plot(csub_max[ind_local_max[ith_seed:]], rsub_max[
                    ind_local_max[ith_seed:]], '.r', ms=5)
                ax_pnr_cn.plot(c, r, 'or', markerfacecolor='red')

                ax_cn_box.imshow(cn[r_min:r_max, c_min:c_max], vmin=0, vmax=1)
                ax_cn_box.set_title('Correlation')

                ax_traces.cla()
                ax_traces.plot(y0)
                ax_traces.set_title('The fluo. trace at the seed pixel')

                writer.grab_frame()

            [ai, ci_raw, ind_success] = extract_ac(data_filtered_box,
                                                   data_raw_box, ind_ctr, patch_dims)

            if (not ind_success) or (np.sum(ai > 0) < min_pixel):
                # bad initialization. discard and continue
                continue
            else:
                # cheers! good initialization.
                center[:, num_neurons] = [c, r]
                Ain[num_neurons, r_min:r_max, c_min:c_max] = ai
                Cin_raw[num_neurons] = ci_raw.squeeze()
                if deconvolve_options['p']:
                    # deconvolution
                    ci, baseline, c1, _, _, si, _ = \
                        constrained_foopsi(ci_raw, **deconvolve_options)
                    if ci.sum() == 0:
                        continue
                    Cin[num_neurons] = ci
                    Sin[num_neurons] = si
                else:
                    # no deconvolution
                    ci = ci_raw.copy()
                    ci[ci < 0] = 0
                    if ci.sum() == 0:
                        continue
                    Cin[num_neurons] = ci.squeeze()

                if save_video:
                    # mark the seed pixel on the correlation image
                    ax_cn.plot(c, r, '.r')

                    ax_cn_box.cla()
                    ax_cn_box.imshow(ai)
                    ax_cn_box.set_title('Spatial component')

                    ax_traces.cla()
                    ax_traces.plot(ci_raw)
                    ax_traces.plot(ci, 'r')
                    ax_traces.set_title('Temporal component')

                    writer.grab_frame()

                # avoid searching nearby pixels
                ind_search[r_min:r_max, c_min:c_max] += (ai > ai.max() / 2)

                # remove the spatial-temporal activity of the initialized
                # and update correlation image & PNR image
                # update the raw data
                data_raw[:, r_min:r_max, c_min:c_max] -= \
                    ai[np.newaxis, ...] * ci[..., np.newaxis, np.newaxis]

                if gSig:
                    # spatially filter the neuron shape
                    tmp_img = Ain[num_neurons, r2_min:r2_max, c2_min:c2_max]
                    if center_psf:
                        if background_filter == 'box':
                            ai_filtered = cv2.GaussianBlur(tmp_img, ksize=ksize, sigmaX=gSig[0],
                                                           sigmaY=gSig[1], borderType=1) \
                                - cv2.boxFilter(tmp_img, ddepth=-1, ksize=ksize, borderType=1)
                        else:
                            ai_filtered = cv2.filter2D(tmp_img, -1, psf, borderType=1)
                    else:
                        ai_filtered = cv2.GaussianBlur(tmp_img, ksize=ksize, sigmaX=gSig[0],
                                                       sigmaY=gSig[1], borderType=1)
                    # update the filtered data
                    data_filtered[:, r2_min:r2_max, c2_min:c2_max] -= \
                        ai_filtered[np.newaxis, ...] * ci[..., np.newaxis, np.newaxis]
                    data_filtered_box = data_filtered[:, r2_min:r2_max, c2_min:c2_max].copy()
                else:
                    data_filtered_box = data_raw[:, r2_min:r2_max, c2_min:c2_max].copy()

                # update PNR image
                # data_filtered_box -= data_filtered_box.mean(axis=0)
                max_box = np.max(data_filtered_box, axis=0)
                noise_box = noise_pixel[r2_min:r2_max, c2_min:c2_max]
                pnr_box = np.divide(max_box, noise_box)
                pnr[r2_min:r2_max, c2_min:c2_max] = pnr_box
                pnr_box[pnr_box < min_pnr] = 0

                # update correlation image
                data_filtered_box[data_filtered_box <
                                  thresh_init * noise_box] = 0
                cn_box = caiman.summary_images.local_correlations_fft(
                    data_filtered_box, swap_dim=False)
                cn_box[np.isnan(cn_box) | (cn_box < 0)] = 0
                cn[r_min:r_max, c_min:c_max] = cn_box[
                    (r_min - r2_min):(r_max - r2_min), (c_min - c2_min):(c_max - c2_min)]
                cn_box[cn_box < min_corr] = 0
                cn_box = cn[r2_min:r2_max, c2_min:c2_max]

                # update v_search
                v_search[r2_min:r2_max, c2_min:c2_max] = cn_box * pnr_box
                v_search[ind_search] = 0
                # avoid searching nearby pixels
                # v_search[r_min:r_max, c_min:c_max] *= (ai < np.max(ai) / 2.)

                # increase the number of detected neurons
                num_neurons += 1  #
                if num_neurons == max_number:
                    continue_searching = False
                    break
                else:
                    if num_neurons % 100 == 1:
                        logger.info(f'{num_neurons - 1} neurons have been initialized')

    logger.info(f'In total, {num_neurons} neurons were initialized.')
    # A = np.reshape(Ain[:num_neurons], (-1, d1 * d2)).transpose()
    A = np.reshape(Ain[:num_neurons], (-1, d1 * d2), order='F').transpose()
    C = Cin[:num_neurons]
    C_raw = Cin_raw[:num_neurons]
    S = Sin[:num_neurons]
    center = center[:, :num_neurons]

    if save_video:
        plt.close()
        writer.finish()

    return A, C, C_raw, S, center


@profile
def extract_ac(data_filtered, data_raw, ind_ctr, patch_dims):
    # parameters
    min_corr_neuron = 0.9  # 7
    max_corr_bg = 0.3
    data_filtered = data_filtered.copy()

    # compute the temporal correlation between each pixel and the seed pixel
    data_filtered -= data_filtered.mean(axis=0)  # data centering
    tmp_std = np.sqrt(np.sum(data_filtered ** 2, axis=0))  # data
    # normalization
    tmp_std[tmp_std == 0] = 1
    data_filtered /= tmp_std
    y0 = data_filtered[:, ind_ctr]  # fluorescence trace at the center
    tmp_corr = np.dot(y0.reshape(1, -1), data_filtered)  # corr. coeff. with y0
    # pixels in the central area of neuron
    ind_neuron = (tmp_corr > min_corr_neuron).squeeze()
    # pixels outside of neuron's ROI
    ind_bg = (tmp_corr < max_corr_bg).squeeze()

    # extract temporal activity
    ci = np.mean(data_filtered[:, ind_neuron], axis=1)
    # initialize temporal activity of the neural
    if ci.dot(ci) == 0:  # avoid empty results
        return None, None, False

    # roughly estimate the background fluctuation
    y_bg = np.median(data_raw[:, ind_bg], axis=1).reshape(-1, 1)\
        if np.any(ind_bg) else np.ones((len(ci), 1), np.float32)
    # extract spatial components
    X = np.concatenate([ci.reshape(-1, 1), y_bg, np.ones(y_bg.shape, np.float32)], 1)
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, data_raw)
    try:
        #ai = np.linalg.inv(XX).dot(Xy)[0]
        # ai = np.linalg.solve(XX, Xy)[0]
        ai = pd_solve(XX, Xy)[0]
    except:
        ai = scipy.linalg.lstsq(XX, Xy)[0][0]
    ai = ai.reshape(patch_dims)
    ai[ai < 0] = 0

    # post-process neuron shape
    ai = circular_constraint(ai)
    ai = connectivity_constraint(ai)

    # remove baseline
    # ci -= np.median(ci)
    sn = get_noise_welch(ci)
    y_diff = np.concatenate([[-1], np.diff(ci)])
    b = np.median(ci[(y_diff >= 0) * (y_diff < sn)])
    ci -= b
    if np.isnan(ci.sum()):
        return None, None, False

    # return results
    return ai, ci, True


@profile
def compute_W(Y, A, C, dims, radius, data_fits_in_memory=True, ssub=1, tsub=1, parallel=False):
    """compute background according to ring model
    solves the problem
        min_{W,b0} ||X-W*X|| with X = Y - A*C - b0*1'
    subject to
        W(i,j) = 0 for each pixel j that is not in ring around pixel i
    Problem parallelizes over pixels i
    Fluctuating background activity is W*X, constant baselines b0.

    Args:
        Y: np.ndarray (2D or 3D)
            movie, raw data in 2D or 3D (pixels x time).
        A: np.ndarray or sparse matrix
            spatial footprint of each neuron.
        C: np.ndarray
            calcium activity of each neuron.
        dims: tuple
            x, y[, z] movie dimensions
        radius: int
            radius of ring
        data_fits_in_memory: [optional] bool
            If true, use faster but more memory consuming computation
        ssub: int
            spatial downscale factor
        tsub: int
            temporal downscale factor
        parallel: bool
            If true, use multiprocessing to process pixels in parallel

    Returns:
        W: scipy.sparse.csr_matrix (pixels x pixels)
            estimate of weight matrix for fluctuating background
        b0: np.ndarray (pixels,)
            estimate of constant background baselines
    """

    if current_process().name != 'MainProcess':
        # no parallelization over pixels if already processing patches in parallel
        parallel = False

    T = Y.shape[1]
    d1 = (dims[0] - 1) // ssub + 1
    d2 = (dims[1] - 1) // ssub + 1

    radius = int(round(radius / float(ssub)))
    ring = disk(radius + 1)
    ring[1:-1, 1:-1] -= disk(radius)
    ringidx = [i - radius - 1 for i in np.nonzero(ring)]

    def get_indices_of_pixels_on_ring(pixel):
        x = pixel % d1 + ringidx[0]
        y = pixel // d1 + ringidx[1]
        inside = (x >= 0) * (x < d1) * (y >= 0) * (y < d2)
        return x[inside] + y[inside] * d1

    b0 = np.array(Y.mean(1)) - A.dot(C.mean(1))

    if ssub > 1:
        ds_mat = caiman.source_extraction.cnmf.utilities.decimation_matrix(dims, ssub)
        ds = lambda x: ds_mat.dot(x)
    else:
        ds = lambda x: x

    if data_fits_in_memory:
        if ssub == 1 and tsub == 1:
            X = Y - A.dot(C) - b0[:, None]
        else:
            X = decimate_last_axis(ds(Y), tsub) - \
                (ds(A).dot(decimate_last_axis(C, tsub)) if A.size > 0 else 0) - \
                ds(b0).reshape((-1, 1), order='F')

        def process_pixel(p):
            index = get_indices_of_pixels_on_ring(p)
            B = X[index]
            tmp = np.array(B.dot(B.T))
            tmp[np.diag_indices(len(tmp))] += np.trace(tmp) * 1e-5
            tmp2 = X[p]
            data = pd_solve(tmp, B.dot(tmp2))
            return index, data
    else:

        def process_pixel(p):
            index = get_indices_of_pixels_on_ring(p)
            if ssub == 1 and tsub == 1:
                B = Y[index] - A[index].dot(C) - b0[index, None]
            else:
                B = decimate_last_axis(ds(Y), tsub)[index] - \
                    (ds(A)[index].dot(decimate_last_axis(C, tsub)) if A.size > 0 else 0) - \
                    ds(b0).reshape((-1, 1), order='F')[index]
            tmp = np.array(B.dot(B.T))
            tmp[np.diag_indices(len(tmp))] += np.trace(tmp) * 1e-5
            if ssub == 1 and tsub == 1:
                tmp2 = Y[p] - A[p].dot(C).ravel() - b0[p]
            else:
                tmp2 = decimate_last_axis(ds(Y), tsub)[p] - \
                    (ds(A)[p].dot(decimate_last_axis(C, tsub)) if A.size > 0 else 0) - \
                    ds(b0).reshape((-1, 1), order='F')[p]
            data = pd_solve(tmp, B.dot(tmp2))
            return index, data

    Q = list((parmap if parallel else map)(process_pixel, range(d1 * d2)))
    indices, data = np.array(Q, dtype=object).T
    indptr = np.concatenate([[0], np.cumsum(list(map(len, indices)))])
    indices = np.concatenate(indices)
    data = np.concatenate(data)
    return spr.csr_matrix((data, indices, indptr), dtype='float32'), b0.astype(np.float32)

def nnsvd_init(X, n_components, r_ov=10, eps=1e-6, random_state=42):
    # NNDSVD initialization from scikit learn package (modified)
    U, S, V = randomized_svd(X, n_components + r_ov, random_state=random_state)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    C = W.T
    A = H.T
    return A[:, 1:n_components], C[:n_components], (U, S, V) #

def norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    """
    return sqrt(squared_norm(x))
