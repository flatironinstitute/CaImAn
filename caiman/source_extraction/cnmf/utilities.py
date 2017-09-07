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
#\package caiman/dource_ectraction/cnmf
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Sat Sep 12 15:52:53 2015

from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
from scipy.sparse import diags, spdiags, issparse
from .initialization import greedyROI
from ...base.rois import com
import pylab as pl
import psutil
import scipy
from ...mmapping import parallel_dot_product


#%%
def CNMFSetParms(Y, n_processes, K=30, gSig=[5, 5], gSiz = None, ssub=2, tsub=2, p=2, p_ssub=2, p_tsub=2,
                 thr=0.8, method_init='greedy_roi', nb=1, nb_patch=1, n_pixels_per_process=None, block_size=None,
                 check_nan=True, normalize_init=True, options_local_NMF=None, remove_very_bad_comps=False,
                 alpha_snmf=10e2, update_background_components=True, 
                 low_rank_background=True, min_corr = .85, min_pnr = 20, deconvolve_options_init = None,
                 ring_size_factor = 1.5, center_psf = True):
    """Dictionary for setting the CNMF parameters.

    Any parameter that is not set get a default value specified
    by the dictionary default options

    PRE-PROCESS PARAMS#############
    sn: None,
        noise level for each pixel

    noise_range: [0.25, 0.5]
             range of normalized frequencies over which to average

    noise_method': 'mean'
             averaging method ('mean','median','logmexp')

    max_num_samples_fft': 3*1024

    n_pixels_per_process: 1000

    compute_g': False
        flag for estimating global time constant

    p : 2
         order of AR indicator dynamics

    lags: 5
        number of autocovariance lags to be considered for time constant estimation

    include_noise: False
            flag for using noise values when estimating g

    pixels: None
         pixels to be excluded due to saturation

    check_nan: True

    INIT PARAMS###############

    K:     30
        number of components

    gSig: [5, 5]
          size of bounding box

    gSiz: [int(round((x * 2) + 1)) for x in gSig],

    ssub:   2
        spatial downsampling factor

    tsub:   2
        temporal downsampling factor

    nIter: 5
        number of refinement iterations

    kernel: None
        user specified template for greedyROI

    maxIter: 5
        number of HALS iterations

    method: method_init
        can be greedy_roi or sparse_nmf, local_NMF

    max_iter_snmf : 500

    alpha_snmf: 10e2

    sigma_smooth_snmf : (.5,.5,.5)

    perc_baseline_snmf: 20

    nb:  1
        number of background components

    normalize_init:
        whether to pixelwise equalize the movies during initialization

    options_local_NMF:
        dictionary with parameters to pass to local_NMF initializaer

    SPATIAL PARAMS##########

        dims: dims
            number of rows, columns [and depths]

        method: 'dilate','ellipse', 'dilate'
            method for determining footprint of spatial components ('ellipse' or 'dilate')

        dist: 3
            expansion factor of ellipse
        n_pixels_per_process: n_pixels_per_process
            number of pixels to be processed by eacg worker

        medw: (3,)*len(dims)
            window of median filter
        thr_method: 'nrg'
           Method of thresholding ('max' or 'nrg')

        maxthr: 0.1
            Max threshold

        nrgthr: 0.9999
            Energy threshold


        extract_cc: True
            Flag to extract connected components (might want to turn to False for dendritic imaging)

        se: np.ones((3,)*len(dims), dtype=np.uint8)
             Morphological closing structuring element

        ss: np.ones((3,)*len(dims), dtype=np.uint8)
            Binary element for determining connectivity


        update_background_components:bool
            whether to update the background components in the spatial phase

        low_rank_background:bool
            whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
            (to be used with one background per patch) 

        method_ls:'lasso_lars'
            'nnls_L0'. Nonnegative least square with L0 penalty
            'lasso_lars' lasso lars function from scikit learn
            'lasso_lars_old' lasso lars from old implementation, will be deprecated

        TEMPORAL PARAMS###########

        ITER: 2
            block coordinate descent iterations

        method:'oasis', 'cvxpy',  'oasis'
            method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
            if method cvxpy, primary and secondary (if problem unfeasible for approx solution)

        solvers: ['ECOS', 'SCS']
             solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'

        p:
            order of AR indicator dynamics

        memory_efficient: False

        bas_nonneg: True
            flag for setting non-negative baseline (otherwise b >= min(y))

        noise_range: [.25, .5]
            range of normalized frequencies over which to average

        noise_method: 'mean'
            averaging method ('mean','median','logmexp')

        lags: 5,
            number of autocovariance lags to be considered for time constant estimation

        fudge_factor: .96
            bias correction factor (between 0 and 1, close to 1)

        nb

        verbosity: False

        block_size : block_size
            number of pixels to process at the same time for dot product. Make it smaller if memory problems
    """

    if type(Y) is tuple:
        dims, T = Y[:-1], Y[-1]
    else:
        dims, T = Y.shape[:-1], Y.shape[-1]

    # print(('using ' + str(n_processes) + ' processes'))
    # if n_pixels_per_process is None:
    #     avail_memory_per_process = np.array(psutil.virtual_memory()[1])/2.**30/n_processes
    #     mem_per_pix = 3.6977678498329843e-09
    #     n_pixels_per_process = np.int(avail_memory_per_process/8./mem_per_pix/T)
    #     n_pixels_per_process = np.int(np.minimum(n_pixels_per_process,np.prod(dims) // n_processes))

    # if block_size is None:
    #     block_size = n_pixels_per_process

    # print(('using ' + str(n_pixels_per_process) + ' pixels per process'))
    # print(('using ' + str(block_size) + ' block_size'))

    options = dict()
    options['patch_params'] = {
        'ssub': p_ssub,             # spatial downsampling factor
        'tsub': p_tsub,              # temporal downsampling factor
        'only_init': True,
        'skip_refinement': False,
        'remove_very_bad_comps': remove_very_bad_comps,
        'nb': nb_patch
    }

    options['preprocess_params'] = {'sn': None,                  # noise level for each pixel
                                    # range of normalized frequencies over which to average
                                    'noise_range': [0.25, 0.5],
                                    # averaging method ('mean','median','logmexp')
                                    'noise_method': 'mean',
                                    'max_num_samples_fft': 3 * 1024,
                                    'n_pixels_per_process': n_pixels_per_process,
                                    'compute_g': False,            # flag for estimating global time constant
                                    'p': p,                        # order of AR indicator dynamics
                                    # number of autocovariance lags to be considered for time
                                    # constant estimation
                                    'lags': 5,
                                    'include_noise': False,        # flag for using noise values when estimating g
                                    'pixels': None,
                                    # pixels to be excluded due to saturation
                                    'check_nan': check_nan

                                    }
    
    gSig = gSig if gSig is not None else [-1, -1]
    

    options['init_params'] = {'K': K,                  # number of components
                              'gSig': gSig,                               # size of bounding box
                              'gSiz': [np.int(np.round((x * 2) + 1)) for x in gSig] if gSiz is  None else gSiz,
                              'ssub': ssub,             # spatial downsampling factor
                              'tsub': tsub,             # temporal downsampling factor
                              'nIter': 5,               # number of refinement iterations
                              'kernel': None,           # user specified template for greedyROI
                              'maxIter': 5,              # number of HALS iterations
                              'method': method_init,     # can be greedy_roi or sparse_nmf, local_NMF
                              'max_iter_snmf': 500,
                              'alpha_snmf': alpha_snmf,
                              'sigma_smooth_snmf': (.5, .5, .5),
                              'perc_baseline_snmf': 20,
                              'nb': nb,                 # number of background components
                              # whether to pixelwise equalize the movies during initialization
                              'normalize_init': normalize_init,
                              # dictionary with parameters to pass to local_NMF initializaer
                              'options_local_NMF': options_local_NMF,
                              'min_corr': min_corr,
                              'min_pnr' : min_pnr,
                              'deconvolve_options_init' : deconvolve_options_init,
                              'ring_size_factor': ring_size_factor,
                              'center_psf' : center_psf,                              
                              }

    options['spatial_params'] = {
        'dims': dims,                   # number of rows, columns [and depths]
        # method for determining footprint of spatial components ('ellipse' or 'dilate')
        'method': 'dilate',  # 'ellipse', 'dilate',
        'dist': 3,                       # expansion factor of ellipse
        'n_pixels_per_process': n_pixels_per_process,   # number of pixels to be processed by eacg worker
        'medw': (3,) * len(dims),                                # window of median filter
        'thr_method': 'nrg',  # Method of thresholding ('max' or 'nrg')
        'maxthr': 0.1,                                 # Max threshold
        'nrgthr': 0.9999,                              # Energy threshold
        # Flag to extract connected components (might want to turn to False for dendritic imaging)
        'extract_cc': True,
        # Morphological closing structuring element
        'se': np.ones((3,) * len(dims), dtype=np.uint8),
        # Binary element for determining connectivity
        'ss': np.ones((3,) * len(dims), dtype=np.uint8),
        'nb': nb,                                      # number of background components
        'method_ls': 'lasso_lars',               # 'nnls_L0'. Nonnegative least square with L0 penalty
        #'lasso_lars' lasso lars function from scikit learn
        #'lasso_lars_old' lasso lars from old implementation, will be deprecated
        # whether to update the background components in the spatial phase
        'update_background_components': update_background_components,
        # whether to update the using a low rank approximation. In the False case
        # all the nonzero elements of the background components are updated using
        # hals
        'low_rank_background': low_rank_background
        #(to be used with one background per patch)
    }
    options['temporal_params'] = {
        'ITER': 2,                   # block coordinate descent iterations
        # method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
        'method': 'oasis',  # 'cvxpy', # 'oasis'
        # if method cvxpy, primary and secondary (if problem unfeasible for approx
        # solution) solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
        'solvers': ['ECOS', 'SCS'],
        'p': p,                      # order of AR indicator dynamics
        'memory_efficient': False,
        # flag for setting non-negative baseline (otherwise b >= min(y))
        'bas_nonneg': True,
        # range of normalized frequencies over which to average
        'noise_range': [.25, .5],
        'noise_method': 'mean',   # averaging method ('mean','median','logmexp')
        'lags': 5,                   # number of autocovariance lags to be considered for time constant estimation
        'fudge_factor': .96,         # bias correction factor (between 0 and 1, close to 1)
        'nb': nb,                   # number of background components
        'verbosity': False,
        # number of pixels to process at the same time for dot product. Make it
        # smaller if memory problems
        'block_size': block_size
    }
    options['merging'] = {
        'thr': thr,
    }
    return options


#%%
def computeDFF_traces(Yr, A, C, bl, quantileMin=8, frames_window=200):
    extract_DF_F(Yr, A, C,  bl, quantileMin, frames_window)


#%%
def extract_DF_F(Yr, A, C,  bl, quantileMin=8, frames_window=200, block_size=400, dview=None):
    """ Compute DFF function from cnmf output.

     Disclaimer: it might be memory inefficient

    Parameters:
    -----------
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

    frammes_window: int
        number of frames for running quantile

    Returns:
    -------

    Cdf:
        the computed Calcium acitivty to the derivative of f

    See Also:
    -------

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
        Df = scipy.ndimage.percentile_filter(C2, quantileMin, (frames_window, 1))
        C_df = Cf / Df

    return C_df


#%%
def manually_refine_components(Y, xxx_todo_changeme, A, C, Cn, thr=0.9, display_numbers=True,
                               max_number=None, cmap=None, **kwargs):
    """Plots contour of spatial components

     against a background image and allows to interactively add novel components by clicking with mouse

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
            a3_tiny = A3[coords_y[0]:coords_y[-1] + 1, coords_x[0]:coords_x[-1] + 1, :]
            y3_tiny = Y[coords_y[0]:coords_y[-1] + 1, coords_x[0]:coords_x[-1] + 1, :]

            dy_sz, dx_sz = np.shape(a3_tiny)[:-1]
            y2_tiny = np.reshape(y3_tiny, (dx_sz * dy_sz, T), order='F')
            a2_tiny = np.reshape(a3_tiny, (dx_sz * dy_sz, nr), order='F')
            y2_res = y2_tiny - a2_tiny.dot(C)

            y3_res = np.reshape(y2_res, (dy_sz, dx_sz, T), order='F')
            a__, c__, center__, b_in__, f_in__ = greedyROI(
                y3_res, nr=1, gSig=[np.floor(old_div(dx_sz, 2)), np.floor(old_div(dy_sz, 2))], gSiz=[dx_sz, dy_sz])

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
            pl.contour(y, x, bmat, [thr])
            pl.pause(.01)

        elif pts == []:
            break

        nr += 1
        A3 = np.reshape(A, (d1, d2, nr), order='F')

    return A, C


#%%
def app_vertex_cover(A):
    """ Finds an approximate vertex cover for a symmetric graph with adjacency matrix A.

     Parameters:
     -----------
     A:    boolean 2d array (K x K)
          Adjacency matrix. A is boolean with diagonal set to 0

     Returns:
     --------
     L:   A vertex cover of A

     @authors by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    """

    L = []
    while A.any():
        nz = np.nonzero(A)[0]          # find non-zero edges
        u = nz[np.random.randint(0, len(nz))]
        A[u, :] = False
        A[:, u] = False
        L.append(u)

    return np.asarray(L)


def update_order(A):
    """Determines the update order of the temporal components

    this, given the spatial components, by creating a nest of random approximate vertex covers
    Basically we can update the components that are not overlapping, in parallel

     Input:
     -------
     A:    np.ndarray
          matrix of spatial components (d x K)

     Outputs:
     ---------
     parllcomp:   list of sets
          list of subsets of components. The components of each subset can be updated in parallel

     len_parrllcomp:  list
          length of each subset

    @authors: Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    """
    K = np.shape(A)[-1]
    AA = A.T * A
    AA.setdiag(0)
    F = (AA) > 0
    F = F.toarray()
    rem_ind = np.arange(K)
    parllcomp = []
    len_parrllcomp = []
    while len(rem_ind) > 0:
        L = np.sort(app_vertex_cover(F[rem_ind, :][:, rem_ind]))
        if L.size:
            ord_ind = set(rem_ind) - set(rem_ind[L])
            rem_ind = rem_ind[L]
        else:
            ord_ind = set(rem_ind)
            rem_ind = []

        parllcomp.append(ord_ind)
        len_parrllcomp.append(len(ord_ind))

    return parllcomp[::-1], len_parrllcomp[::-1]


def update_order_greedy(A, flag_AA=True):
    """Determines the update order of the temporal components

    this, given the spatial components using a greedy method
    Basically we can update the components that are not overlapping, in parallel

    Input:
     -------
     A:       sparse crc matrix
              matrix of spatial components (d x K)
     OR
              A.T.dot(A) matrix (d x d) if flag_AA = true

     flag_AA: boolean (default true)

     Outputs:
     ---------
     parllcomp:   list of sets
          list of subsets of components. The components of each subset can be updated in parallel

     len_parrllcomp:  list
          length of each subset

    @author: Eftychios A. Pnevmatikakis, Simons Foundation, 2017
    """
    K = np.shape(A)[-1]
    parllcomp = []
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
def order_components(A, C):
    """Order components based on their maximum temporal value and size

    Parameters:
    -----------
    A:   sparse matrix (d x K)
         spatial components

    C:   matrix or np.ndarray (K x T)
         temporal components

    Returns:
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
    A = np.array(np.matrix(A) * spdiags(old_div(1, nA2), 0, K, K))
    nA4 = np.sum(A**4, axis=0)**0.25
    C = np.array(spdiags(nA2, 0, K, K) * np.matrix(C))
    mC = np.ndarray.max(np.array(C), axis=1)
    srt = np.argsort(nA4 * mC)[::-1]
    A_or = A[:, srt] * spdiags(nA2[srt], 0, K, K)
    C_or = spdiags(old_div(1., nA2[srt]), 0, K, K) * (C[srt, :])

    return A_or, C_or, srt


#%%
# def save_mat_in_chuncks(Yr, num_chunks, shape, mat_name='mat', axis=0):
#    """ save hdf5 matrix in chunks
#
#    Parameters
#    ----------
#    file_name: str
#        file_name of the hdf5 file to be chunked
#    shape: tuples
#        shape of the original chunked matrix
#    idx: list
#        indexes to slice matrix along axis
#    mat_name: [optional] string
#        name prefix for temporary files
#    axis: int
#        axis along which to slice the matrix
#
#    Returns
#    ---------
#    name of the saved file
#
#    """
#
#    Yr = np.array_split(Yr, num_chunks, axis=axis)
#    print "splitting array..."
#    folder = tempfile.mkdtemp()
#    prev = 0
#    idxs = []
#    names = []
#    for mm in Yr:
#        mm = np.array(mm)
#        idxs.append(np.array(range(prev, prev + mm.shape[0])).T)
#        new_name = os.path.join(folder, mat_name + '_' + str(prev) + '_' + str(len(idxs[-1])))
#        print "Saving " + new_name
#        np.save(new_name, mm)
#        names.append(new_name)
#        prev = prev + mm.shape[0]
#
#    return {'names': names, 'idxs': idxs, 'axis': axis, 'shape': shape}


# def evaluate_components(A,Yr,psx):
#
#    #%% clustering components
#    Ys=Yr
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


# def extract_DF_F(Y, A, C, i=None):
#    """Extract DF/F values from spatial/temporal components and background
#
#     Parameters
#     -----------
#     Y: np.ndarray
#           input data (d x T)
#     A: sparse matrix of np.ndarray
#           Set of spatial including spatial background (d x K)
#     C: matrix
#           Set of temporal components including background (K x T)
#
#     Returns
#     -----------
#     C_df: matrix
#          temporal components in the DF/F domain
#     Df:  np.ndarray
#          vector with baseline values for each trace
#    """
#    A2 = A.copy()
#    A2.data **= 2
#    nA2 = np.squeeze(np.array(A2.sum(axis=0)))
#    A = A * diags(old_div(1, nA2), 0)
#    C = diags(nA2, 0) * C
#
#    # if i is None:
#    #    i = np.argmin(np.max(A,axis=0))
#
#    Y = np.matrix(Y)
#    Yf = A.transpose() * (Y - A * C)  # + A[:,i]*C[i,:])
#    Df = np.median(np.array(Yf), axis=1)
#    C_df = diags(old_div(1, Df), 0) * C
#
#    return C_df, Df
