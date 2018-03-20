#!/usr/bin/env python

"""
Created on Wed Aug 05 20:38:27 2015

# -*- coding: utf-8 -*-
@author: agiovann
"""

from __future__ import division
from __future__ import print_function
# noinspection PyCompatibility
from past.builtins import basestring
from builtins import zip
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags
from scipy.linalg import eig
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
from scipy.ndimage import label, binary_dilation
from sklearn.decomposition import NMF
from warnings import warn
import numpy as np
import scipy
import time
import tempfile
import os
import shutil
from ...mmapping import load_memmap, parallel_dot_product
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_closing
from scipy.ndimage.measurements import label


def basis_denoising(y, c, boh, sn, id2_, px):
    if np.size(c) > 0:
        _, _, a, _, _ = lars_regression_noise(y, c, 1, sn)
    else:
        return (None, None, None)
    return a, px, id2_
#%% update_spatial_components (in parallel)


def update_spatial_components(Y, C=None, f=None, A_in=None, sn=None, dims=None, min_size=3, max_size=8, dist=3, normalize_yyt_one=True,
                              method='ellipse', expandCore=None, dview=None, n_pixels_per_process=128,
                              medw=(3, 3), thr_method='nrg', maxthr=0.1, nrgthr=0.9999, extract_cc=True, b_in=None,
                              se=np.ones((3, 3), dtype=np.int), ss=np.ones((3, 3), dtype=np.int), nb=1,
                              method_ls='lasso_lars', update_background_components=True, low_rank_background=True, block_size=1000, num_blocks_per_run=20):
    """update spatial footprints and background through Basis Pursuit Denoising

    for each pixel i solve the problem
        [A(i,:),b(i)] = argmin sum(A(i,:))
    subject to
        || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

    for each pixel the search is limited to a few spatial components

    Parameters:
    ----------
    Y: np.ndarray (2D or 3D)
        movie, raw data in 2D or 3D (pixels x time).

    C: np.ndarray
        calcium activity of each neuron.

    f: np.ndarray
        temporal profile  of background activity.

    A_in: np.ndarray
        spatial profile of background activity. If A_in is boolean then it defines the spatial support of A.
        Otherwise it is used to determine it through determine_search_location

    b_in: np.ndarray
        you can pass background as input, especially in the case of one background per patch, since it will update using hals

    dims: [optional] tuple
        x, y[, z] movie dimensions

    min_size: [optional] int

    max_size: [optional] int

    dist: [optional] int

    sn: [optional] float
        noise associated with each pixel if known

    backend [optional] str
        'ipyparallel', 'single_thread'
        single_thread:no parallelization. It can be used with small datasets.
        ipyparallel: uses ipython clusters and then send jobs to each of them
        SLURM: use the slurm scheduler

    n_pixels_per_process: [optional] int
        number of pixels to be processed by each thread

    method: [optional] string
        method used to expand the search for pixels 'ellipse' or 'dilate'

    expandCore: [optional]  scipy.ndimage.morphology
        if method is dilate this represents the kernel used for expansion

    dview: view on ipyparallel client
            you need to create an ipyparallel client and pass a view on the processors (client = Client(), dview=client[:])

    medw, thr_method, maxthr, nrgthr, extract_cc, se, ss: [optional]
        Parameters for components post-processing. Refer to spatial.threshold_components for more details

    nb: [optional] int
        Number of background components

    method_ls:
        method to perform the regression for the basis pursuit denoising.
             'nnls_L0'. Nonnegative least square with L0 penalty
             'lasso_lars' lasso lars function from scikit learn
             'lasso_lars_old' lasso lars from old implementation, will be deprecated

        normalize_yyt_one: bool
            wheter to norrmalize the C and A matrices so that diag(C*C.T) are ones

    update_background_components:bool
        whether to update the background components in the spatial phase

    low_rank_background:bool
        whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
        (to be used with one background per patch)


    Returns:
    --------
    A: np.ndarray
         new estimate of spatial footprints

    b: np.ndarray
        new estimate of spatial background

    C: np.ndarray
         temporal components (updated only when spatial components are completely removed)

    f: np.ndarray
        same as f_in except if empty component deleted.

    Raise:
    -------
    Exception('You need to define the input dimensions')

    Exception('Dimension of Matrix Y must be pixels x time')

    Exception('Dimension of Matrix C must be neurons x time')

    Exception('Dimension of Matrix f must be background comps x time ')

    Exception('Either A or C need to be determined')

    Exception('Dimension of Matrix A must be pixels x neurons ')

    Exception('You need to provide estimate of C and f')

    Exception('Not implemented consistently')

    Exception("Failed to delete: " + folder)
    """
    print('Initializing update of Spatial Components')

    if expandCore is None:
        expandCore = iterate_structure(
            generate_binary_structure(2, 1), 2).astype(int)

    if dims is None:
        raise Exception('You need to define the input dimensions')

    # shape transformation and tests
    Y, A_in, C, f, n_pixels_per_process, rank_f, d, T = test(
        Y, A_in, C, f, n_pixels_per_process, nb)

    start_time = time.time()
    print('computing the distance indicators')
    # we compute the indicator from distance indicator
    ind2_, nr, C, f, b_, A_in = computing_indicator(
        Y, A_in, b_in, C, f, nb, method, dims, min_size, max_size, dist, expandCore, dview)
    if normalize_yyt_one and C is not None:
        C = np.array(C)
        nr_C = np.shape(C)[0]
        d_ = scipy.sparse.lil_matrix((nr_C, nr_C))
        d_.setdiag(np.sqrt(np.sum(C ** 2, 1)))
        A_in = A_in * d_
        C = old_div(C, np.sqrt(np.sum(C ** 2, 1)[:, np.newaxis]))

    if b_in is None:
        b_in = b_

    print('memmaping')
    # we create a memory map file if not already the case, we send Cf, a
    # matrix that include background components
    C_name, Y_name, folder = creatememmap(Y, np.vstack((C, f)), dview)

    # we create a pixel group array (chunks for the cnmf)for the parrallelization of the process
    print('Updating Spatial Components using lasso lars')
    cct = np.diag(C.dot(C.T))
    pixel_groups = []
    for i in range(0, np.prod(dims) - n_pixels_per_process + 1, n_pixels_per_process):
        pixel_groups.append([Y_name, C_name, sn, ind2_, list(
            range(i, i + n_pixels_per_process)), method_ls, cct, ])
    if i < np.prod(dims):
        pixel_groups.append([Y_name, C_name, sn, ind2_, list(
            range(i, np.prod(dims))), method_ls, cct])
    A_ = np.zeros((d, nr + np.size(f, 0)))  # init A_
    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            parallel_result = dview.map_async(
                regression_ipyparallel, pixel_groups).get(4294967)
        else:
            parallel_result = dview.map_sync(
                regression_ipyparallel, pixel_groups)
            dview.results.clear()
    else:
        parallel_result = list(map(regression_ipyparallel, pixel_groups))

    for chunk in parallel_result:
        for pars in chunk:
            px, idxs_, a = pars
            A_[px, idxs_] = a

    print("thresholding components")
    A_ = threshold_components(A_, dims, dview=dview, medw=medw, thr_method=thr_method,
                              maxthr=maxthr, nrgthr=nrgthr, extract_cc=extract_cc, se=se, ss=ss)

    ff = np.where(np.sum(A_, axis=0) == 0)  # remove empty components
    if np.size(ff) > 0:
        ff = ff[0]
        print('eliminating {} empty spatial components'.format(len(ff)))
        A_ = np.delete(A_, list(ff[ff < nr]), 1)
        C = np.delete(C, list(ff[ff < nr]), 0)
        nr = nr - len(ff[ff < nr])
        if low_rank_background:
            background_ff = list(filter(lambda i: i >= nb, ff - nr))
            f = np.delete(f, background_ff, 0)
        else:
            background_ff = list(filter(lambda i: i >= 0, ff - nr))
            f = np.delete(f, background_ff, 0)
            b_in = np.delete(b_in, background_ff, 1)

    A_ = A_[:, :nr]
    A_ = coo_matrix(A_)

    print("Computing residuals")
    if 'memmap' in str(type(Y)):
        Y_resf = parallel_dot_product(Y, f.T, dview=dview, block_size=block_size, num_blocks_per_run=num_blocks_per_run) - \
            A_.dot(coo_matrix(C[:nr, :]).dot(f.T))
    else:
        # Y*f' - A*(C*f')
        Y_resf = np.dot(Y, f.T) - A_.dot(coo_matrix(C[:nr, :]).dot(f.T))

    if update_background_components:

        if b_in is None:
            # update baseline based on residual
            b = np.fmax(Y_resf.dot(np.linalg.inv(f.dot(f.T))), 0)
        else:
            ind_b = [np.where(_b)[0] for _b in b_in.T]
            b = HALS4shape_bckgrnd(Y_resf, b_in, f, ind_b)

    else:
        if b_in is None:
            raise Exception(
                'If you set the update_background_components to True you have to pass them as input to update_spatial')
        # try:
        #    b = np.delete(b_in, background_ff, 0)
        # except NameError:
        b = b_in

    print(("--- %s seconds ---" % (time.time() - start_time)))
    try:  # clean up
        # remove temporary file created
        print("Removing tempfiles created")
        shutil.rmtree(folder)
    except:
        raise Exception("Failed to delete: " + folder)

    return A_, b, C, f


#%%
def HALS4shape_bckgrnd(Y_resf, B, F, ind_B, iters=5):
    K = B.shape[-1]
    U = Y_resf.T
    V = F.dot(F.T)
    for _ in range(iters):
        for m in range(K):  # neurons
            ind_pixels = ind_B[m]

            B[ind_pixels, m] = np.clip(B[ind_pixels, m] +
                                       ((U[m, ind_pixels] - V[m].dot(B[ind_pixels].T)) /
                                        V[m, m]), 0, np.inf)
    return B


# %%lars_regression_noise_ipyparallel
def regression_ipyparallel(pars):
    """update spatial footprints and background through Basis Pursuit Denoising

       for each pixel i solve the problem
           [A(i,:),b(i)] = argmin sum(A(i,:))
       subject to
           || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

       for each pixel the search is limited to a few spatial components

       Parameters:
       ----------
       C_name: string
            memmap C

       Y_name: string
            memmap Y

       idxs_Y: np.array
           indices of the Calcium traces for each computed components

       idxs_C: np.array
           indices of the Calcium traces for each computed components

       method_least_square:
           method to perform the regression for the basis pursuit denoising.
                'nnls_L0'. Nonnegative least square with L0 penalty
                'lasso_lars' lasso lars function from scikit learn
                'lasso_lars_old' lasso lars from old implementation, will be deprecated



       Returns:
       --------
       px: np.ndarray
            positions o the regression

       idxs_C: np.ndarray
           indices of the Calcium traces for each computed components

       a: learned weight

       Raises:
       -------
       Exception('Least Square Method not found!'
       """

    # /!\ need to import since it is run from within the server
    import numpy as np
    import sys
    import gc
    from sklearn import linear_model

    Y_name, C_name, noise_sn, idxs_C, idxs_Y, method_least_square, cct = pars
    # we load from the memmap file
    if isinstance(Y_name, basestring):
        Y, _, _ = load_memmap(Y_name)
        Y = np.array(Y[idxs_Y, :])
    else:
        Y = Y_name[idxs_Y, :]
    if isinstance(C_name, basestring):
        C = np.load(C_name, mmap_mode='r')
        C = np.array(C)
    else:
        C = C_name

    _, T = np.shape(C)  # initialize values
    As = []

    for y, px in zip(Y, idxs_Y):
        c = C[idxs_C[px], :]
        idx_only_neurons = idxs_C[px]
        if len(idx_only_neurons) > 0:
            cct_ = cct[idx_only_neurons[idx_only_neurons < len(cct)]]
        else:
            cct_ = []

        if np.size(c) > 0:
            sn = noise_sn[px] ** 2 * T
            if method_least_square == 'lasso_lars_old':  # lasso lars from old implementation, will be deprecated
                a = lars_regression_noise_old(y, c.T, 1, sn)[2]

            elif method_least_square == 'nnls_L0':  # Nonnegative least square with L0 penalty
                a = nnls_L0(c.T, y, 1.2 * sn)

            elif method_least_square == 'lasso_lars':  # lasso lars function from scikit learn
                lambda_lasso = 0 if np.size(cct_) == 0 else \
                    .5 * noise_sn[px] * np.sqrt(np.max(cct_)) / T
                clf = linear_model.LassoLars(alpha=lambda_lasso, positive=True)
                a_lrs = clf.fit(np.array(c.T), np.ravel(y))
                a = a_lrs.coef_

            else:
                raise Exception(
                    'Least Square Method not found!' + method_least_square)

            if not np.isscalar(a):
                a = a.T

            As.append((px, idxs_C[px], a))

    if isinstance(Y_name, basestring):
        del Y
    if isinstance(C_name, basestring):
        del C
    if isinstance(Y_name, basestring):
        gc.collect()

    return As





# %%
def construct_ellipse_parallel(pars):
    """update spatial footprints and background through Basis Pursuit Denoising

    for each pixel i solve the problem
        [A(i,:),b(i)] = argmin sum(A(i,:))
    subject to
        || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

    for each pixel the search is limited to a few spatial components

    Parameters:
    ----------
    [parsed]
    cm[i]:
        center of mass of each neuron

    A[:, i]: the A of each components

    Vr:

    dims:
        the dimension of each A's ( same usually )

    dist:
        computed distance matrix

    min_size: [optional] int

    max_size: [optional] int

    Returns:
    --------
    dist: np.ndarray
         new estimate of spatial footprints

    Raise:
    -------
    Exception('You cannot pass empty (all zeros) components!')
    """
    Coor, cm, A_i, Vr, dims, dist, max_size, min_size, d = pars
    dist_cm = coo_matrix(np.hstack([Coor[c].reshape(-1, 1) - cm[k]
                                    for k, c in enumerate(['x', 'y', 'z'][:len(dims)])]))
    Vr.append(dist_cm.T * spdiags(A_i.toarray().squeeze(),
                                  0, d, d) * dist_cm / A_i.sum(axis=0))

    if np.sum(np.isnan(Vr)) > 0:
        raise Exception('You cannot pass empty (all zeros) components!')

    D, V = eig(Vr[-1])

    dkk = [np.min((max_size ** 2, np.max((min_size ** 2, dd.real))))
           for dd in D]

    # search indexes for each component
    return np.sqrt(np.sum([old_div((dist_cm * V[:, k]) ** 2, dkk[k]) for k in range(len(dkk))], 0)) <= dist


# %% threshold_components


def threshold_components(A, dims, medw=None, thr_method='nrg', maxthr=0.1, nrgthr=0.9999, extract_cc=True,
                         se=None, ss=None, dview=None):
    """
    Post-processing of spatial components which includes the following steps

    (i) Median filtering
    (ii) Thresholding
    (iii) Morphological closing of spatial support
    (iv) Extraction of largest connected component ( to remove small unconnected pixel )

    Parameters:
    ----------
    A:      np.ndarray
        2d matrix with spatial components

    dims:   tuple
        dimensions of spatial components

    medw: [optional] tuple
        window of median filter

    thr_method: [optional] string
        Method of thresholding:
            'max' sets to zero pixels that have value less than a fraction of the max value
            'nrg' keeps the pixels that contribute up to a specified fraction of the energy

    maxthr: [optional] scalar
        Threshold of max value

    nrgthr: [optional] scalar
        Threshold of energy

    extract_cc: [optional] bool
        Flag to extract connected components (might want to turn to False for dendritic imaging)

    se: [optional] np.intarray
        Morphological closing structuring element

    ss: [optinoal] np.intarray
        Binary element for determining connectivity

    Returns:
    --------
        Ath: np.ndarray
            2d matrix with spatial components thresholded
    """
    if medw is None:
        medw = (3,) * len(dims)
    if se is None:
        se = np.ones((3,) * len(dims), dtype='uint8')
    if ss is None:
        ss = np.ones((3,) * len(dims), dtype='uint8')
    # dims and nm of neurones
    d, nr = np.shape(A)
    # instanciation of A thresh.
    Ath = np.zeros((d, nr))

    pars = []
    # fo each neurons
    for i in range(nr):
        pars.append([scipy.sparse.csc_matrix(A[:, i]), i, dims,
                     medw, d, thr_method, se, ss, maxthr, nrgthr, extract_cc])

    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            res = dview.map_async(
                threshold_components_parallel, pars).get(4294967)
        else:
            res = dview.map_async(threshold_components_parallel, pars)
    else:
        res = list(map(threshold_components_parallel, pars))

    for r in res:
        At, i = r
        Ath[:, i] = At

    return Ath


def threshold_components_parallel(pars):
    """
       Post-processing of spatial components which includes the following steps

       (i) Median filtering
       (ii) Thresholding
       (iii) Morphological closing of spatial support
       (iv) Extraction of largest connected component ( to remove small unconnected pixel )
       /!\ need to be called through the function threshold components

       Parameters:
        ---------
        [parsed]
       A:      np.ndarray
           2d matrix with spatial components

       dims:   tuple
           dimensions of spatial components

       medw: [optional] tuple
           window of median filter

       thr_method: [optional] string
           Method of thresholding:
               'max' sets to zero pixels that have value less than a fraction of the max value
               'nrg' keeps the pixels that contribute up to a specified fraction of the energy

       maxthr: [optional] scalar
           Threshold of max value

       nrgthr: [optional] scalar
           Threshold of energy

       extract_cc: [optional] bool
           Flag to extract connected components (might want to turn to False for dendritic imaging)

       se: [optional] np.intarray
           Morphological closing structuring element

       ss: [optinoal] np.intarray
           Binary element for determining connectivity

       Returns:
        -------
           Ath: np.ndarray
               2d matrix with spatial components thresholded
       """

    A_i, i, dims, medw, d, thr_method, se, ss, maxthr, nrgthr, extract_cc = pars
    A_i = A_i.toarray()
    # we reshape this one dimension column of the 2d components into the 2D that
    A_temp = np.reshape(A_i, dims[::-1])
    # we apply a median filter of size medw
    A_temp = median_filter(A_temp, medw)
    if thr_method == 'max':
        BW = (A_temp > maxthr * np.max(A_temp))
    elif thr_method == 'nrg':
        Asor = np.sort(np.squeeze(np.reshape(A_temp, (d, 1))))[::-1]
        temp = np.cumsum(Asor ** 2)
        ff = np.squeeze(np.where(temp < nrgthr * temp[-1]))
        if ff.size > 0:
            ind = ff if ff.ndim == 0 else ff[-1]
            A_temp[A_temp < Asor[ind]] = 0
            BW = (A_temp >= Asor[ind])
        else:
            BW = np.zeros_like(A_temp)
    # we want to remove the components that are valued 0 in this now 1d matrix
    Ath = np.squeeze(np.reshape(A_temp, (d, 1)))
    Ath2 = np.zeros((d))
    # we do that to have a full closed structure even if the values have been trehsolded
    BW = binary_closing(BW.astype(np.int), structure=se)

    # if we have deleted the element
    if BW.max() == 0:
        return Ath2, i
    #
    # we want to extract the largest connected component ( to remove small unconnected pixel )
    if extract_cc:
        # we extract each future as independent with the cross structuring elemnt
        labeled_array, num_features = label(BW, structure=ss)
        labeled_array = np.squeeze(np.reshape(labeled_array, (d, 1)))
        nrg = np.zeros((num_features, 1))
        # we extract the energy for each component
        for j in range(num_features):
            nrg[j] = np.sum(Ath[labeled_array == j + 1] ** 2)
        indm = np.argmax(nrg)
        Ath2[labeled_array == indm + 1] = Ath[labeled_array == indm + 1]

    else:
        BW = BW.flatten()
        Ath2[BW] = Ath[BW]

    return Ath2, i


# %%


def nnls_L0(X, Yp, noise):
    """
    Nonnegative least square with L0 penalty

    It will basically call the scipy function with some tests
    we want to minimize :
    min|| Yp-W_lam*X||**2 <= noise
    with ||W_lam||_0  penalty
    and W_lam >0

    Parameters:
    ---------
        X: np.array
            the input parameter ((the regressor

        Y: np.array
            ((the regressand

    Returns:
    --------
        W_lam: np.array
            the learned weight matrices ((Models

    """
    W_lam, RSS = scipy.optimize.nnls(X, np.ravel(Yp))
    RSS = RSS * RSS
    if RSS > noise:  # hard noise constraint problem infeasible
        return W_lam

    while 1:
        eliminate = []
        for i in np.where(W_lam[:-1] > 0)[0]:  # W_lam[:-1] to skip background
            mask = W_lam > 0
            mask[i] = 0
            Wtmp, tmp = scipy.optimize.nnls(X * mask, np.ravel(Yp))
            if tmp * tmp < noise:
                eliminate.append([i, tmp])
        if eliminate == []:
            return W_lam
        else:
            W_lam[eliminate[np.argmin(np.array(eliminate)[:, 1])][0]] = 0


# %% lars_regression_noise
def lars_regression_noise_old(Yp, X, positive, noise, verbose=False):
    """
     Run LARS for regression problems with LASSO penalty, with optional positivity constraints
     Author: Andrea Giovannucci. Adapted code from Eftychios Pnevmatikakis


     Parameters:
        -------
       Yp:          Yp[:,t] is the observed data at time t

       X:           the regresion problem is Yp=X*W + noise

       maxcomps:    maximum number of active components to allow

       positive:    a flag to enforce positivity

       noise:       the noise of the observation equation. if it is not
                    provided as an argument, the noise is computed from the
                    variance at the end point of the algorithm. The noise is
                    used in the computation of the Cp criterion.

     Returns:
    -------
       Ws: weights from each iteration
       lambdas: lambda_ values at each iteration

       Cps: C_p estimates
       last_break:     last_break(m) == n means that the last break with m non-zero weights is at Ws(:,:,n)

    See Also:
    -------
        LARS : https://en.wikipedia.org/wiki/Least-angle_regression
        group Lasso :
    """
    # INITAILIZATION
    T = len(Yp)  # of time steps
    k = 1
    Yp = np.squeeze(np.asarray(Yp))
    # necessary for matrix multiplications
    Yp = np.expand_dims(Yp, axis=1)
    _, T = np.shape(Yp)  # of time steps
    _, N = np.shape(X)  # of compartments

    # 1 : Start with all Weights equal to zero.
    maxcomps = N
    W = np.zeros((N, k))
    active_set = np.zeros((N, k))
    visited_set = np.zeros((N, k))
    lambdas = []
    #  Just preallocation. Ws may end with more or less than maxcomp columns
    Ws = []
    r = np.expand_dims(np.dot(X.T, Yp.flatten()), axis=1)  # N-dim vector
    M = np.dot(-X.T, X)  # N x N matrix

    # %% begin main loop
    i = 0
    flag = 0
    while 1:
        if flag == 1:
            W_lam = 0
            break
            # % calculate new gradient component if necessary
        if i > 0 and new >= 0 and visited_set[new] == 0:  # AG NOT CLEAR HERE
            visited_set[new] = 1  # % remember this direction was computed
        # % Compute full gradient of Q
        dQ = r + np.dot(M, W)

        # % Compute new W
        if i == 0:
            if positive:
                dQa = dQ
            else:
                dQa = np.abs(dQ)
            lambda_, new = np.max(dQa), np.argmax(dQa)

            if lambda_ < 0:
                print('All negative directions!')
                break
        else:  # 2 : Find the predictor x_{j} most correlated with y

            # % calculate vector to travel along
            avec, gamma_plus, gamma_minus = calcAvec(
                new, dQ, W, lambda_, active_set, M, positive)
            # % calculate time of travel and next new direction
            if new == -1:  # % if we just dropped a direction we don't allow it to emerge
                if dropped_sign == 1:  # % with the same sign
                    gamma_plus[dropped] = np.inf
                else:
                    gamma_minus[dropped] = np.inf
            # 3 : Increase the coefficient W in the direction of the sign of its correlation with y
            # % don't consider active components
            gamma_plus[active_set == 1] = np.inf
            # % or components outside the range [0, lambda_]
            gamma_plus[gamma_plus <= 0] = np.inf
            gamma_plus[gamma_plus > lambda_] = np.inf
            gp_min, gp_min_ind = np.min(gamma_plus), np.argmin(gamma_plus)

            if positive:
                gm_min = np.inf  # % don't consider new directions that would grow negative
            else:
                gamma_minus[active_set == 1] = np.inf
                gamma_minus[gamma_minus > lambda_] = np.inf
                gamma_minus[gamma_minus <= 0] = np.inf
                gm_min, gm_min_ind = np.min(
                    gamma_minus), np.argmin(gamma_minus)

            [g_min, which] = np.min(gp_min), np.argmin(gp_min)
# % if there are no possible new components, try move to the end
            if g_min == np.inf:
                g_min = lambda_
# % This happens when all the components are already active or, if positive==1,
# when there are no new positive directions

            # % LARS check  (is g_min*avec too large?)
            gamma_zero = old_div(-W[active_set == 1], np.squeeze(avec))
            gamma_zero_full = np.zeros((N, k))
            gamma_zero_full[active_set == 1] = gamma_zero
            gamma_zero_full[gamma_zero_full <= 0] = np.inf
            gz_min, gz_min_ind = np.min(
                gamma_zero_full), np.argmin(gamma_zero_full)
            # 4: Increase Wj,Wk in their joint least squares direction, until some other predictor x_{m}
            # has as much correlation with the residual r. (see 5)
            if gz_min < g_min:
                if verbose:
                    print(('DROPPING active weight:' + str(gz_min_ind)))
                active_set[gz_min_ind] = 0
                dropped = gz_min_ind
                dropped_sign = np.sign(W[dropped])
                W[gz_min_ind] = 0
                avec = avec[gamma_zero != gz_min]
                g_min = gz_min
                new = -1

            elif g_min < lambda_:
                if which == 0:
                    new = gp_min_ind
                    if verbose:
                        print(('new positive component:' + str(new)))

                else:
                    new = gm_min_ind
                    print(('new negative component:' + str(new)))

            W[active_set == 1] = W[active_set == 1] + \
                np.dot(g_min, np.squeeze(avec))
            if positive:
                if any(W < 0):
                    flag = 1

            lambda_ = lambda_ - g_min

        # %  Update weights and lambdas
        lambdas.append(lambda_)
        Ws.append(W.copy())
        # 5 : Take residuals r=y-y_  along the way. Stop when some other predictor x_{k}
        # has as much correlation with  r as x_{j} has.
        if len((Yp - np.dot(X, W)).shape) > 2:
            res = scipy.linalg.norm(np.squeeze(Yp - np.dot(X, W)), 'fro') ** 2
        else:
            res = scipy.linalg.norm(Yp - np.dot(X, W), 'fro') ** 2

            # % Check finishing conditions
        if lambda_ == 0 or (new >= 0 and np.sum(active_set) == maxcomps) or (res < noise):
            if verbose:
                print('end. \n')
            break
        # 6: Continue until: all predictors are in the model
        if new >= 0:
            active_set[new] = 1

        i = i + 1

    Ws_old = Ws
    # end main loop

    #%% final calculation of mus
    Ws = np.asarray(np.swapaxes(np.swapaxes(Ws_old, 0, 1), 1, 2))
    if flag == 0:
        if i > 0:
            Ws = np.squeeze(Ws[:, :, :len(lambdas)])
            w_dir = old_div(-(Ws[:, i] - Ws[:, i - 1]),
                            (lambdas[i] - lambdas[i - 1]))
            Aw = np.dot(X, w_dir)
            y_res = np.squeeze(
                Yp) - np.dot(X, Ws[:, i - 1] + w_dir * lambdas[i - 1])
            ld = scipy.roots([scipy.linalg.norm(Aw) ** 2, -2 * np.dot(Aw.T, y_res),
                              np.dot(y_res.T, y_res) - noise])
            lam = ld[np.intersect1d(
                np.where(ld > lambdas[i]), np.where(ld < lambdas[i - 1]))]
            if len(lam) == 0 or np.any(lam) < 0 or np.any(~np.isreal(lam)):
                lam = np.array([lambdas[i]])

            W_lam = Ws[:, i - 1] + np.dot(w_dir, lambdas[i - 1] - lam[0])
        else:
            warn('LARS REGRESSION NOT SOLVABLE, USING NN LEAST SQUARE')
            W_lam = scipy.optimize.nnls(X, np.ravel(Yp))[0]
            lam = 10

    else:
        W_lam = 0
        Ws = 0
        lambdas = 0
        lam = 0

    return Ws, lambdas, W_lam, lam, flag


# %% auxiliary functions
def calcAvec(new, dQ, W, lambda_, active_set, M, positive):
    """
    calculate the vectoe to travel along

    used in the lars regression function

               Parameters:
               ----------
               Y: np.ndarray (2D or 3D)
                   movie, raw data in 2D or 3D (pixels x time).

               Cf: np.ndarray
                   calcium activity of each neuron + background components

               Returns:
               --------
               C_name: string
                    the memmaped name of Cf

               Y_name: string
                    the memmaped name of Y


    """
    r, c = np.nonzero(active_set)
    Mm = -M.take(r, axis=0).take(r, axis=1)
    Mm = old_div((Mm + Mm.T), 2)
    # % verify that there is no numerical instability
    if len(Mm) > 1:
        eigMm, _ = scipy.linalg.eig(Mm)
        eigMm = np.real(eigMm)
    else:
        eigMm = Mm
    if any(eigMm < 0):
        np.min(eigMm)
        #%error('The matrix Mm has negative eigenvalues')
        flag = 1

    b = np.sign(W)
    if new >= 0:
        b[new] = np.sign(dQ[new])
    b = b[active_set == 1]
    if len(Mm) > 1:
        avec = np.linalg.solve(Mm, b)
    else:
        avec = old_div(b, Mm)

    if positive:
        if new >= 0:
            in_ = np.sum(active_set[:new])
            if avec[in_] < 0:
                # new;
                #%error('new component of a is negative')
                flag = 1

    one_vec = np.ones(W.shape)
    dQa = np.zeros(W.shape)
    for j in range(len(r)):
        dQa = dQa + np.expand_dims(avec[j] * M[:, r[j]], axis=1)

    gamma_plus = old_div((lambda_ - dQ), (one_vec + dQa))
    gamma_minus = old_div((lambda_ + dQ), (one_vec - dQa))

    return avec, gamma_plus, gamma_minus


def test(Y, A_in, C, f, n_pixels_per_process, nb):
    """test the shape of each matrix, reshape them, test the number of pixel per process

        if it doesn't follow the rules it will throw an exception that is not supposed to be catch by spatial.

        Parameters:
        ----------
        Y: np.ndarray (2D or 3D)
            movie, raw data in 2D or 3D (pixels x time).

        C: np.ndarray
            calcium activity of each neuron.

        f: np.ndarray
            temporal profile  of background activity.

        A_in: np.ndarray
            spatial profile of background activity. If A_in is boolean then it defines the spatial support of A.
            Otherwise it is used to determine it through determine_search_location

        n_pixels_per_process: [optional] int
            number of pixels to be processed by each thread


        Returns:
        --------
        same:
         but reshaped and tested

        Raise:
        -------
        Exception('You need to define the input dimensions')
        Exception('Dimension of Matrix Y must be pixels x time')
        Exception('Dimension of Matrix C must be neurons x time')
        Exception('Dimension of Matrix f must be background comps x time ')
        Exception('Either A or C need to be determined')
        Exception('Dimension of Matrix A must be pixels x neurons ')
        Exception('You need to provide estimate of C and f')
        Exception('Not implemented consistently')
        Exception("Failed to delete: " + folder)
        """
    if Y.ndim < 2 and not isinstance(Y, basestring):
        Y = np.atleast_2d(Y)
        if Y.shape[1] == 1:
            raise Exception('Dimension of Matrix Y must be pixels x time')

    if C is not None:
        C = np.atleast_2d(C)
        if C.shape[1] == 1:
            raise Exception('Dimension of Matrix C must be neurons x time')

    if f is not None:

        f = np.atleast_2d(f)
        if f.shape[1] == 1:
            raise Exception(
                'Dimension of Matrix f must be background comps x time ')

    if (A_in is None) and (C is None):
        raise Exception('Either A or C need to be determined')

    if A_in is not None:
        if len(A_in.shape) == 1:
            A_in = np.atleast_2d(A_in).T
            if A_in.shape[0] == 1:
                raise Exception(
                    'Dimension of Matrix A must be pixels x neurons ')

    [d, T] = np.shape(Y)

    if A_in is None:
        A_in = np.ones((d, np.shape(C)[1]), dtype=bool)

    if n_pixels_per_process > d:
        print('The number of pixels per process (n_pixels_per_process)'
              ' is larger than the total number of pixels!! Decreasing suitably.')
        n_pixels_per_process = d

    if f is not None:
        nb = f.shape[0]

    return Y, A_in, C, f, n_pixels_per_process, nb, d, T
# %% determine_search_location
def determine_search_location(A, dims, method='ellipse', min_size=3, max_size=8, dist=3,
                              expandCore=iterate_structure(generate_binary_structure(2, 1), 2).astype(int), dview=None):
    """
    compute the indices of the distance from the cm to search for the spatial component

    does it by following an ellipse from the cm or doing a step by step dilatation around the cm


    Parameters:
    ----------
    [parsed]
     cm[i]:
        center of mass of each neuron

     A[:, i]: the A of each components

     dims:
        the dimension of each A's ( same usually )

     dist:
        computed distance matrix

     dims: [optional] tuple
                x, y[, z] movie dimensions

    method: [optional] string
            method used to expand the search for pixels 'ellipse' or 'dilate'

    expandCore: [optional]  scipy.ndimage.morphology
            if method is dilate this represents the kernel used for expansion

    min_size: [optional] int

    max_size: [optional] int

    dist: [optional] int

    dims: [optional] tuple
             x, y[, z] movie dimensions

    Returns:
    --------
    dist_indicator: np.ndarray
        distance from the cm to search for the spatial footprint

    Raise:
    -------
    Exception('You cannot pass empty (all zeros) components!')
    """

    from scipy.ndimage.morphology import grey_dilation

    # we initialize the values
    if len(dims) == 2:
        d1, d2 = dims
    elif len(dims) == 3:
        d1, d2, d3 = dims
    d, nr = np.shape(A)
    A = csc_matrix(A)
    dist_indicator = scipy.sparse.csc_matrix((d, nr),dtype= np.float32)

    if method == 'ellipse':
        Coor = dict()
        # we create a matrix of size A.x of each pixel coordinate in A.y and inverse
        if len(dims) == 2:
            Coor['x'] = np.kron(np.ones(d2), list(range(d1)))
            Coor['y'] = np.kron(list(range(d2)), np.ones(d1))
        elif len(dims) == 3:
            Coor['x'] = np.kron(np.ones(d3 * d2), list(range(d1)))
            Coor['y'] = np.kron(
                np.kron(np.ones(d3), list(range(d2))), np.ones(d1))
            Coor['z'] = np.kron(list(range(d3)), np.ones(d2 * d1))
        if not dist == np.inf:  # determine search area for each neuron
            cm = np.zeros((nr, len(dims)))  # vector for center of mass
            Vr = []  # cell(nr,1);
            dist_indicator = []
            pars = []
            # for each dim
            for i, c in enumerate(['x', 'y', 'z'][:len(dims)]):
                # mass center in this dim = (coor*A)/sum(A)
                cm[:, i] = old_div(
                    np.dot(Coor[c], A[:, :nr].todense()), A[:, :nr].sum(axis=0))

            # parrallelizing process of the construct ellipse function
            for i in range(nr):
                pars.append([Coor, cm[i], A[:, i], Vr, dims,
                             dist, max_size, min_size, d])
            if dview is None:
                res = list(map(construct_ellipse_parallel, pars))
            else:
                if 'multiprocessing' in str(type(dview)):
                    res = dview.map_async(
                        construct_ellipse_parallel, pars).get(4294967)
                else:
                    res = dview.map_sync(construct_ellipse_parallel, pars)
            for r in res:
                dist_indicator.append(r)

            dist_indicator = (np.asarray(dist_indicator)).squeeze().T

        else:
            raise Exception('Not implemented')
            dist_indicator = True * np.ones((d, nr))

    elif method == 'dilate':
        for i in range(nr):
            A_temp = np.reshape(A[:, i].toarray(), dims[::-1])
            if len(expandCore) > 0:
                if len(expandCore.shape) < len(dims):  # default for 3D
                    expandCore = iterate_structure(
                        generate_binary_structure(len(dims), 1), 2).astype(int)
                A_temp = grey_dilation(A_temp, footprint=expandCore)
            else:
                A_temp = grey_dilation(A_temp, [1] * len(dims))

            dist_indicator[:, i] = scipy.sparse.coo_matrix(np.squeeze(np.reshape(A_temp, (d, 1)))[:,None] > 0)
    else:
        raise Exception('Not implemented')
        dist_indicator = True * np.ones((d, nr))

    return dist_indicator
#%%
def computing_indicator(Y, A_in, b, C, f, nb, method, dims, min_size, max_size, dist, expandCore, dview):
    """compute the indices of the distance from the cm to search for the spatial component (calling determine_search_location)

    does it by following an ellipse from the cm or doing a step by step dilatation around the cm
    if it doesn't follow the rules it will throw an exception that is not supposed to be catch by spatial.


           Parameters:
           ----------
           Y: np.ndarray (2D or 3D)
               movie, raw data in 2D or 3D (pixels x time).

           C: np.ndarray
               calcium activity of each neuron.

           f: np.ndarray
               temporal profile  of background activity.

           A_in: np.ndarray
               spatial profile of background activity. If A_in is boolean then it defines the spatial support of A.
               Otherwise it is used to determine it through determine_search_location

           n_pixels_per_process: [optional] int
               number of pixels to be processed by each thread

           min_size: [optional] int

           max_size: [optional] int

           dist: [optional] int

           dims: [optional] tuple
                x, y[, z] movie dimensions

           method: [optional] string
                method used to expand the search for pixels 'ellipse' or 'dilate'

           expandCore: [optional]  scipy.ndimage.morphology
                if method is dilate this represents the kernel used for expansion


           Returns:
           --------
           same:
            but reshaped and tested

           Raise:
           -------
           Exception('You need to define the input dimensions')

           Exception('Dimension of Matrix Y must be pixels x time')

           Exception('Dimension of Matrix C must be neurons x time')

           Exception('Dimension of Matrix f must be background comps x time ')

           Exception('Either A or C need to be determined')

           Exception('Dimension of Matrix A must be pixels x neurons ')

           Exception('You need to provide estimate of C and f')

           Exception('Not implemented consistently')

           Exception("Failed to delete: " + folder)
           """

    if A_in.dtype == bool:

        dist_indicator = A_in.copy()
        print("spatial support for each components given by the user")
        # we compute C,B,f,Y if we have boolean for A matrix
        if C is None:  # if C is none we approximate C, b and f from the binary mask
            dist_indicator_av = old_div(dist_indicator.astype(
                'float32'), np.sum(dist_indicator.astype('float32'), axis=0))
            px = (np.sum(dist_indicator, axis=1) > 0)
            not_px = 1 - px
            if Y.shape[-1] < 30000:
                f = Y[not_px, :].mean(0)
            else:  # momory mapping fails here for some reasons
                print('estimating f')
                f = 0
                for xxx in not_px:
                    f = (f + Y[xxx]) / 2

            f = np.atleast_2d(f)

            Y_resf = np.dot(Y, f.T)
            b = np.maximum(Y_resf, 0) / (np.linalg.norm(f)**2)
            C = np.maximum(csr_matrix(dist_indicator_av.T).dot(
                Y) - dist_indicator_av.T.dot(b).dot(f), 0)
            A_in = scipy.sparse.coo_matrix(A_in.astype(np.float32))
            nr, _ = np.shape(C)  # number of neurons
            ind2_ = [np.hstack((np.where(iid_)[0], nr + np.arange(f.shape[0])))
                     if np.size(np.where(iid_)[0]) > 0 else [] for iid_ in dist_indicator]

    else:
        if C is None:
            raise Exception('You need to provide estimate of C and f')

        nr, _ = np.shape(C)  # number of neurons

        if b is None:
            dist_indicator = determine_search_location(
                A_in, dims, method=method, min_size=min_size, max_size=max_size, dist=dist, expandCore=expandCore,
                dview=dview)
        else:
            dist_indicator = determine_search_location(
                scipy.sparse.hstack([A_in, scipy.sparse.coo_matrix(b)]), dims, method=method, min_size=min_size, max_size=max_size, dist=dist, expandCore=expandCore,
                dview=dview)

        ind2_ = [np.where(iid_.toarray().squeeze())[0]  for iid_ in dist_indicator.tocsr()]
        ind2_ = [iid_ if (np.size(iid_) > 0) and (np.min(iid_) < nr) else [] for iid_ in ind2_]

    return ind2_, nr, C, f, b, A_in

#%%
def creatememmap(Y, Cf, dview):
    """memmap the C and Y objects in parallel

           the memmaped object will be red during parallelized computation such as the regression function

           Parameters:
           ----------
           Y: np.ndarray (2D or 3D)
               movie, raw data in 2D or 3D (pixels x time).

           Cf: np.ndarray
               calcium activity of each neuron + background components

           Returns:
           --------
           C_name: string
                the memmaped name of Cf

           Y_name: string
                the memmaped name of Y

           Raises:
           -------
           Exception('Not implemented consistently')
           """
    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        tmpf = os.environ.get('SLURM_SUBMIT_DIR')
        print(('cluster temporary folder:' + tmpf))
        folder = tempfile.mkdtemp(dir=tmpf)
    else:
        folder = tempfile.mkdtemp()

    if dview is None:
        Y_name = Y
        C_name = Cf
    else:
        C_name = os.path.join(folder, 'C_temp.npy')
        np.save(C_name, Cf)

        if type(Y) is np.core.memmap:  # if input file is already memory mapped then find the filename
            Y_name = Y.filename
        # if not create a memory mapped version (necessary for parallelization)
        elif isinstance(Y, basestring) or dview is None:
            Y_name = Y
        else:
            Y_name = os.path.join(folder, 'Y_temp.npy')
            np.save(Y_name, Y)
            Y, _, _, _ = load_memmap(Y_name)
            raise Exception('Not implemented consistently')
    return C_name, Y_name, folder


def circular_constraint(img_original):
    img = img_original.copy()
    nr, nc = img.shape
    [rsub, csub] = img.nonzero()
    if len(rsub) == 0:
        return img

    rmin = np.min(rsub)
    rmax = np.max(rsub) + 1
    cmin = np.min(csub)
    cmax = np.max(csub) + 1

    if (rmax - rmin < 1) or (cmax - cmin < 1):
        return img

    if rmin == 0 and rmax == nr and cmin == 0 and cmax == nc:
        ind_max = np.argmax(img)
        y0, x0 = np.unravel_index(ind_max, [nr, nc])
        vmax = img[y0, x0]
        x, y = np.meshgrid(np.arange(nc), np.arange(nr))
        fy, fx = np.gradient(img)
        ind = ((fx * (x0 - x) + fy * (y0 - y) < 0) & (img < vmax / 2))
        img[ind] = 0

        # # remove isolated pixels
        l, _ = label(img)
        ind = binary_dilation(l == l[y0, x0])
        img[~ind] = 0
    else:
        tmp_img = circular_constraint(img[rmin:rmax, cmin:cmax])
        img[rmin:rmax, cmin:cmax] = tmp_img

    return img
# %% lars_regression_noise_parallel
# def basis_denoising(y, c, boh, sn, id2_, px):
#     if np.size(c) > 0:
#         _, _, a, _, _ = lars_regression_noise(y, c, 1, sn)
#     else:
#         return (None, None, None)
#     return a, px, id2_

# def lars_regression_noise_parallel(Y,C,A,noise_sn,idx_Y,idx_C,positive=1):
#
#    _,T=np.shape(C)
#    newY=np.array(Y[idx_Y,:])
#    newC=np.array(C)
#    for px in idx_Y:
#        #
#        c=newC[idx_C[px],:]
#
#        if np.size(c)>0:
#            y=newY[px-idx_Y[0],:]
#            sn=noise_sn[px]**2*T
#           # print y.shape,sn,c.shape
#            _, _, a, _ , _= lars_regression_noise(y, c.T, positive, sn)
#            if np.isscalar(a):
#                A[px,idx_C[px]]=a
#            else:
#                A[px,idx_C[px]]=a.T
