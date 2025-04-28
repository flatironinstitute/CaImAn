#!/usr/bin/env python

"""
Functions for managing spatial components
"""

import cv2
import logging
import numpy as np
import os
import psutil
import scipy
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags
from scipy.linalg import eig
from scipy.ndimage import (
    binary_dilation,
    binary_closing,
    generate_binary_structure,
    grey_dilation,
    iterate_structure,
    label,
    median_filter
)
import shutil
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import tempfile
import time

import caiman.mmapping
import caiman.utils.stats

def update_spatial_components(Y, C=None, f=None, A_in=None, sn=None, dims=None,
                              min_size=3, max_size=8, dist=3,
                              normalize_yyt_one=True, method_exp='dilate',
                              expandCore=None, dview=None, n_pixels_per_process=128,
                              medw=(3, 3), thr_method='max', maxthr=0.1,
                              nrgthr=0.9999, extract_cc=True, b_in=None,
                              se=np.ones((3, 3), dtype=int),
                              ss=np.ones((3, 3), dtype=int), nb=1,
                              method_ls='lasso_lars', update_background_components=True,
                              low_rank_background=True,
                              num_blocks_per_run_spat=20):
    """update spatial footprints and background through Basis Pursuit Denoising

    for each pixel i solve the problem
        [A(i,:),b(i)] = argmin sum(A(i,:))
    subject to
        || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

    for each pixel the search is limited to a few spatial components

    Args:
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

            normalize_yyt_one: bool
                whether to normalize the C and A matrices so that diag(C*C.T) are ones

        update_background_components:bool
            whether to update the background components in the spatial phase

        low_rank_background:bool
            whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
            (to be used with one background per patch)


    Returns:
        A: np.ndarray
             new estimate of spatial footprints

        b: np.ndarray
            new estimate of spatial background

        C: np.ndarray
             temporal components (updated only when spatial components are completely removed)

        f: np.ndarray
            same as f_in except if empty component deleted.

    Raises:
        Exception 'You need to define the input dimensions'

        Exception 'Dimension of Matrix Y must be pixels x time'

        Exception 'Dimension of Matrix C must be neurons x time'

        Exception 'Dimension of Matrix f must be background comps x time '

        Exception 'Either A or C need to be determined'

        Exception 'Dimension of Matrix A must be pixels x neurons'

        Exception 'You need to provide estimate of C and f'

        Exception 'Not implemented consistently'

        Exception "Failed to delete: " + folder
    """
    logger = logging.getLogger("caiman")
    # TODO fix documentation on backend
    logger.info('Initializing update of Spatial Components')

    if expandCore is None:
        expandCore = iterate_structure(
            generate_binary_structure(2, 1), 2).astype(int)

    if dims is None:
        raise Exception('You need to define the input dimensions')

    # shape transformation and tests
    Y, A_in, C, f, n_pixels_per_process, rank_f, d, T = test(
        Y, A_in, C, f, n_pixels_per_process, nb)

    start_time = time.time()
    logger.info('Computing support of spatial components')
    # we compute the indicator from distance indicator
    ind2_, nr, C, f, b_, A_in = computing_indicator(
        Y, A_in, b_in, C, f, nb, method_exp, dims, min_size, max_size, dist, expandCore, dview)

    # remove components that are empty or have a nan
    ff = np.where((np.sum(C, axis=1)==0) + np.isnan(np.sum(C, axis=1)))[0]
    if np.size(ff) > 0:
        logger.info(f"Eliminating empty and nan components: {ff}")
        A_in = caiman.utils.stats.csc_column_remove(A_in, list(ff))
        C = np.delete(C, list(ff), 0)
        # update indices
        ind_list = list(range(nr+nb-np.size(ff)))
        for i in ff:
            ind_list.insert(i, 0)
        ind_list = np.array(ind_list, dtype=int)
        ind2_ = [ind_list[np.setdiff1d(a,ff)] if len(a) else a for a in ind2_]

    nr = np.shape(C)[0]
    if normalize_yyt_one and C is not None:
        C = np.array(C)
        d_ = scipy.sparse.lil_matrix((nr, nr))
        d_.setdiag(np.sqrt(np.sum(C ** 2, 1)))
        A_in = A_in * d_
        C = C/(np.sqrt((C**2).sum(1))[:, np.newaxis] + np.finfo(np.float32).eps)

    if b_in is None:
        b_in = b_

    logger.info('Memory mapping')
    # we create a memory map file if not already the case, we send Cf, a
    # matrix that include background components
    C_name, Y_name, folder = creatememmap(Y, np.vstack((C, f)), dview)

    # we create a pixel group array (chunks for the cnmf)for the parallelization of the process
    logger.info('Updating Spatial Components using lasso lars')
    cct = np.diag(C.dot(C.T))
    pixel_groups = []
    for i in range(0, np.prod(dims) - n_pixels_per_process + 1, n_pixels_per_process):
        pixel_groups.append([Y_name, C_name, sn, ind2_[i:i + n_pixels_per_process], list(
            range(i, i + n_pixels_per_process)), method_ls, cct, ])
    if i + n_pixels_per_process < np.prod(dims):
        pixel_groups.append([Y_name, C_name, sn, ind2_[(i + n_pixels_per_process):np.prod(dims)], list(
            range(i + n_pixels_per_process, np.prod(dims))), method_ls, cct])
    #A_ = scipy.sparse.lil_matrix((d, nr + np.size(f, 0)))
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
    data:list = []
    rows:list = []
    cols:list = []
    for chunk in parallel_result:
        for pars in chunk:
            px, idxs_, a = pars
            #A_[px, idxs_] = a
            nz = np.where(a>0)[0]
            data.extend(a[nz])
            rows.extend(len(nz)*[px])
            cols.extend(idxs_[nz])
    A_ = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(d, nr + np.size(f, 0)))

    logger.info("thresholding components")
    A_ = threshold_components(A_, dims, dview=dview, medw=medw, thr_method=thr_method,
                              maxthr=maxthr, nrgthr=nrgthr, extract_cc=extract_cc, se=se, ss=ss)
    #ff = np.where(np.sum(A_, axis=0) == 0)  # remove empty components
    ff = np.asarray(A_.sum(0) == 0).nonzero()[1]
    if np.size(ff) > 0:
        logger.info(f'removing {ff.shape[0]} empty spatial component(s)')
        if any(ff < nr):
            A_ = caiman.utils.stats.csc_column_remove(A_, list(ff[ff < nr]))
            C = np.delete(C, list(ff[ff < nr]), 0)
            ff -= nr
            nr = nr - len(ff[ff < nr])
        else:
            ff -= nr
        if update_background_components:
            background_ff = list(filter(lambda i: i >= 0, ff))
            f = np.delete(f, background_ff, 0)
            if b_in is not None:
                b_in = np.delete(b_in, background_ff, 1)

    A_ = A_[:, :nr]    
    if update_background_components:
        A_ = csr_matrix(A_)
        logger.info("Computing residuals")
        if 'memmap' in str(type(Y)):
            bl_siz1 = Y.shape[0] // (num_blocks_per_run_spat - 1)
            bl_siz2 = psutil.virtual_memory().available // (4*Y.shape[-1]*(num_blocks_per_run_spat + 1))
            Y_resf = caiman.mmapping.parallel_dot_product(Y, f.T, dview=dview, block_size=min(bl_siz1, bl_siz2), num_blocks_per_run=num_blocks_per_run_spat) - \
                A_.dot(C[:nr].dot(f.T))
        else:
            # Y*f' - A*(C*f')
            Y_resf = np.dot(Y, f.T) - A_.dot(C[:nr].dot(f.T))

        if b_in is None:
            # update baseline based on residual
            #b = np.fmax(Y_resf.dot(np.linalg.inv(f.dot(f.T))), 0)
            b = np.fmax(np.linalg.solve(f.dot(f.T), Y_resf.T), 0).T
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
    # print(("--- %s seconds ---" % (time.time() - start_time)))
    logger.info('Updating done in ' + 
                 '{0}s'.format(str(time.time() - start_time).split(".")[0]))
    try:  # clean up
        # remove temporary file created
        logger.info("Removing created tempfiles")
        shutil.rmtree(folder)
    except:
        raise Exception("Failed to delete: " + folder)

    return csc_matrix(A_), b, C, f

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


def regression_ipyparallel(pars):
    """update spatial footprints and background through Basis Pursuit Denoising

       for each pixel i solve the problem
           [A(i,:),b(i)] = argmin sum(A(i,:))
       subject to
           || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

       for each pixel the search is limited to a few spatial components

       Args:
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

       Returns:
           px: np.ndarray
                positions o the regression

           idxs_C: np.ndarray
               indices of the Calcium traces for each computed components

           a: learned weight

       Raises:
           Exception 'Least Square Method not found!
       """

    # /!\ need to import since it is run from within the server
    import numpy as np
    import sys
    import gc
    from sklearn import linear_model

    Y_name, C_name, noise_sn, idxs_C, idxs_Y, method_least_square, cct = pars
    # we load from the memmap file
    if isinstance(Y_name, str):
        Y, _, _ = caiman.mmapping.load_memmap(Y_name)
        Y = np.array(Y[idxs_Y, :])
    else:
        Y = Y_name[idxs_Y, :]
    if isinstance(C_name, str):
        C = np.load(C_name, mmap_mode='r')
        C = np.array(C)
    else:
        C = C_name

    _, T = np.shape(C)  # initialize values
    As = []
    for y, px, idx_px_from_0 in zip(Y, idxs_Y, range(len(idxs_C))):
        c = C[idxs_C[idx_px_from_0], :]
        idx_only_neurons = idxs_C[idx_px_from_0]
        if len(idx_only_neurons) > 0:
            cct_ = cct[idx_only_neurons[idx_only_neurons < len(cct)]]
        else:
            cct_ = []

        # skip if no components OR pixel has 0 activity
        if np.size(c) > 0 and noise_sn[px] > 0:
            sn = noise_sn[px] ** 2 * T
            if method_least_square == 'lasso_lars_old':
                raise Exception("Obsolete parameter") # Old code, support was removed

            elif method_least_square == 'nnls_L0':  # Nonnegative least square with L0 penalty
                a = nnls_L0(c.T, y, 1.2 * sn)

            elif method_least_square == 'lasso_lars':  # lasso lars function from scikit learn
                lambda_lasso = 0 if np.size(cct_) == 0 else \
                    .5 * noise_sn[px] * np.sqrt(np.max(cct_)) / T
                model = make_pipeline(
                    StandardScaler(with_mean=False),
                    linear_model.LassoLars(alpha=lambda_lasso, positive=True,
                                                 fit_intercept=True)
                    )
                a = model.fit(np.array(c.T), np.ravel(y))['lassolars'].coef_

            else:
                raise Exception(
                    'Least Square Method not found!' + method_least_square)

            if not np.isscalar(a):
                a = a.T

            As.append((px, idxs_C[idx_px_from_0], a))

    if isinstance(Y_name, str):
        del Y
    if isinstance(C_name, str):
        del C
    if isinstance(Y_name, str):
        gc.collect()

    return As

def construct_ellipse_parallel(pars):
    """update spatial footprints and background through Basis Pursuit Denoising


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
    return np.sqrt(np.sum([(dist_cm * V[:, k]) ** 2 / dkk[k] for k in range(len(dkk))], 0)) <= dist

def threshold_components(A, dims, medw=None, thr_method='max', maxthr=0.1, nrgthr=0.9999, extract_cc=True,
                         se=None, ss=None, dview=None) -> np.ndarray:
    """
    Post-processing of spatial components which includes the following steps

    (i) Median filtering
    (ii) Thresholding
    (iii) Morphological closing of spatial support
    (iv) Extraction of largest connected component ( to remove small unconnected pixel )

    Args:
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

        ss: [optional] np.intarray
            Binary element for determining connectivity

    Returns:
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
    # instantiation of A thresh.
    #Ath = np.zeros((d, nr))
    pars = []
    # for each neurons
    A_1 = scipy.sparse.csc_matrix(A)
    for i in range(nr):
        pars.append([A_1[:, i], i, dims,
                     medw, d, thr_method, se, ss, maxthr, nrgthr, extract_cc])

    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            res = dview.map_async(
                threshold_components_parallel, pars).get(4294967)
        else:
            res = dview.map_sync(threshold_components_parallel, pars)
    else:
        res = list(map(threshold_components_parallel, pars))

    res.sort(key=lambda x: x[1])
    indices:list = []
    indptr = [0]
    data:list = []
    for r in res:
        At, i = r
        indptr.append(indptr[-1]+At.indptr[-1])
        indices.extend(At.indices.tolist())
        data.extend(At.data.tolist())

    Ath = csc_matrix((data, indices, indptr), shape=(d, nr))
    return Ath


def threshold_components_parallel(pars):
    """
       Post-processing of spatial components which includes the following steps

       (i) Median filtering
       (ii) Thresholding
       (iii) Morphological closing of spatial support
       (iv) Extraction of largest connected component ( to remove small unconnected pixel )
       ! need to be called through the function threshold components

       Args:
           [parsed] - A list of actual parameters:
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

               ss: [optional] np.intarray
                   Binary element for determining connectivity

       Returns:
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
    BW = binary_closing(BW.astype(int), structure=se.T) # transpose cause A_temp is C-order

    # if we have deleted the element
    if BW.max() == 0:
        return csr_matrix(Ath2), i
    #
    # we want to extract the largest connected component ( to remove small unconnected pixel )
    if extract_cc:
        # we extract each future as independent with the cross structuring element
        labeled_array, num_features = label(BW, structure=ss.T)
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

    return csr_matrix(Ath2), i

def nnls_L0(X, Yp, noise):
    """
    Nonnegative least square with L0 penalty

    It will basically call the scipy function with some tests
    we want to minimize :
    min|| Yp-W_lam*X||**2 <= noise
    with ||W_lam||_0  penalty
    and W_lam >0

    Args:
        X: np.array
            the input parameter ((the regressor

        Y: np.array
            ((the regressand

    Returns:
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

#####
# auxiliary functions

def calcAvec(new, dQ, W, lambda_, active_set, M, positive):
    """
    calculate the vector to travel along

    used in the lars regression function

    Args:
        Y: np.ndarray (2D or 3D)
            movie, raw data in 2D or 3D (pixels x time).

        Cf: np.ndarray
            calcium activity of each neuron + background components

    Returns:
        C_name: string
            the memmapped name of Cf

        Y_name: string
            the memmapped name of Y
    """
    r, c = np.nonzero(active_set)
    Mm = -M.take(r, axis=0).take(r, axis=1)
    Mm = (Mm + Mm.T) / 2
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
        avec = b / Mm

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

    gamma_plus = (lambda_ - dQ) / (one_vec + dQa)
    gamma_minus = (lambda_ + dQ) / (one_vec - dQa)

    return avec, gamma_plus, gamma_minus


def test(Y, A_in, C, f, n_pixels_per_process, nb):
    """test the shape of each matrix, reshape it, and test the number of pixels per process

        if it doesn't follow the rules it will throw an exception that should not be caught by spatial.

        Args:
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
            same:
                but reshaped and tested

        Raises:
            Exception 'You need to define the input dimensions'
            Exception 'Dimension of Matrix Y must be pixels x time'
            Exception 'Dimension of Matrix C must be neurons x time'
            Exception 'Dimension of Matrix f must be background comps x time'
            Exception 'Either A or C need to be determined'
            Exception 'Dimension of Matrix A must be pixels x neurons'
            Exception 'You need to provide estimate of C and f'
            Exception 'Not implemented consistently'
            Exception "Failed to delete: " + folder
        """
    if Y.ndim < 2 and not isinstance(Y, str):
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
    else:
        f = np.zeros((0, Y.shape[1]), dtype=np.float32)

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

def determine_search_location(A, dims, method='ellipse', min_size=3, max_size=8, dist=3,
                              expandCore=iterate_structure(generate_binary_structure(2, 1), 2).astype(int), dview=None):
    """
    compute the indices of the distance from the cm to search for the spatial component

    does this by following an ellipse from the cm or doing a step by step dilatation around the cm

    Args:
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
        dist_indicator: np.ndarray
            distance from the cm to search for the spatial footprint

    Raises:
        Exception 'You cannot pass empty (all zeros) components!'
    """

    logger = logging.getLogger("caiman")

    # we initialize the values
    if len(dims) == 2:
        d1, d2 = dims
    elif len(dims) == 3:
        d1, d2, d3 = dims
    d, nr = np.shape(A)
    A = csc_matrix(A)
#    dist_indicator = scipy.sparse.lil_matrix((d, nr),dtype= np.float32)
#    dist_indicator = scipy.sparse.csc_matrix((d, nr), dtype=np.float32)

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
            Vr:list = []  # cell(nr,1);
            dist_indicator = []
            pars = []
            # for each dim
            for i, c in enumerate(['x', 'y', 'z'][:len(dims)]):
                # mass center in this dim = (coor*A)/sum(A)
                cm[:, i] = np.dot(Coor[c], A[:, :nr].todense()) / A[:, :nr].sum(axis=0)

            # parallelizing process of the construct ellipse function
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

            dist_indicator = scipy.sparse.coo_matrix((np.asarray(dist_indicator)).squeeze().T)

        else:
            raise Exception('Not implemented')
            dist_indicator = True * np.ones((d, nr))

    elif method == 'dilate':
        indptr = [0]
        indices:list = []
        data = []
        if dview is None:
            for i in range(nr):
                A_temp = np.reshape(A[:, i].toarray(), dims[::-1])
                if len(expandCore) > 0:
                    if len(expandCore.shape) < len(dims):  # default for 3D
                        expandCore = iterate_structure(
                            generate_binary_structure(len(dims), 1), 2).astype(int)
                    A_temp = grey_dilation(A_temp, footprint=expandCore)
                else:
                    A_temp = grey_dilation(A_temp, [1] * len(dims))

                nz = np.where(np.squeeze(np.reshape(A_temp, (d, 1)))[:, None] > 0)[0].tolist()
                indptr.append(indptr[-1] + len(nz))
                indices.extend(nz)
                data.extend(len(nz)*[True])
                #dist_indicator[:, i] = scipy.sparse.coo_matrix(np.squeeze(np.reshape(A_temp, (d, 1)))[:, None] > 0)
            dist_indicator = csc_matrix((data, indices, indptr), shape=(d, nr))

        else:
            logger.info('dilate in parallel...')
            pars = []
            for i in range(nr):
                pars.append([A[:, i], dims, expandCore, d])

            if 'multiprocessing' in str(type(dview)):
                parallel_result = dview.map_async(
                    construct_dilate_parallel, pars).get(4294967)
            else:
                parallel_result = dview.map_sync(
                    construct_dilate_parallel, pars)
                dview.results.clear()

            i = 0
            for res in parallel_result:
                indptr.append(indptr[-1] + len(res.row))
                indices.extend(res.row)
                data.extend(len(res.row)*[True])
                #dist_indicator[:, i] = res
                i += 1
            dist_indicator = csc_matrix((data, indices, indptr), shape=(d, nr))

    else:
        raise Exception('Not implemented')
        dist_indicator = True * np.ones((d, nr))

    return csc_matrix(dist_indicator)

def construct_dilate_parallel(pars):
    """
    """

    from scipy.ndimage.morphology import generate_binary_structure, iterate_structure, grey_dilation

    A_i, dims, expandCore, d  = pars
    A_temp = np.reshape(A_i.toarray(), dims[::-1])
    if len(expandCore) > 0:
        if len(expandCore.shape) < len(dims):  # default for 3D
            expandCore = iterate_structure(
                generate_binary_structure(len(dims), 1), 2).astype(int)
        A_temp = grey_dilation(A_temp, footprint=expandCore)
    else:
        A_temp = grey_dilation(A_temp, [1] * len(dims))

    dist_indicator_i = scipy.sparse.coo_matrix(np.squeeze(np.reshape(A_temp, (d, 1)))[:,None] > 0)

    # search indexes for each component
    return dist_indicator_i

def computing_indicator(Y, A_in, b, C, f, nb, method, dims, min_size, max_size, dist, expandCore, dview):
    """compute the indices of the distance from the cm to search for the spatial component (calling determine_search_location)

    does this by following an ellipse from the cm or doing a step by step dilatation around the cm
    if it doesn't follow the rules it will throw an exception that is not supposed to be caught by spatial.


    Args:
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
        same:
            but reshaped and tested

    Raises:
        Exception 'You need to define the input dimensions'

        Exception 'Dimension of Matrix Y must be pixels x time'

        Exception 'Dimension of Matrix C must be neurons x time'

        Exception 'Dimension of Matrix f must be background comps x time'

        Exception 'Either A or C need to be determined'

        Exception 'Dimension of Matrix A must be pixels x neurons'

        Exception 'You need to provide estimate of C and f'

        Exception 'Not implemented consistently'

        Exception 'Failed to delete: " + folder'
           """

    if A_in.dtype == bool:
        dist_indicator = A_in.copy()
        print("spatial support for each components given by the user")
        # we compute C,B,f,Y if we have boolean for A matrix
        if C is None:  # if C is none we approximate C, b and f from the binary mask
            dist_indicator_av = dist_indicator.astype(
                'float32') / np.sum(dist_indicator.astype('float32'), axis=0)
            px = (np.sum(dist_indicator, axis=1) > 0)
            not_px = ~px

            if nb>1:
                    f = NMF(nb, init='nndsvda').fit(np.maximum(Y[not_px, :], 0)).components_
            else:
                if Y.shape[-1] < 30000:
                    f = Y[not_px, :].mean(0)
                else:
                    print('estimating f')
                    f = 0
                    for xxx in np.where(not_px)[0]:
                        f += Y[xxx]
                    f /= not_px.sum()

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
                A_in, dims, method=method, min_size=min_size, max_size=max_size, dist=dist, expandCore=expandCore, dview=dview)
        else:
            dist_indicator = determine_search_location(
                scipy.sparse.hstack([A_in, scipy.sparse.coo_matrix(b)]), dims, method=method, min_size=min_size, max_size=max_size, dist=dist, expandCore=expandCore,
                dview=dview)

        ind2_ = [np.where(iid_.squeeze())[0]  for iid_ in dist_indicator.astype(bool).toarray()]
        ind2_ = [iid_ if (np.size(iid_) > 0) and (np.min(iid_) < nr) else [] for iid_ in ind2_]

    return ind2_, nr, C, f, b, A_in

def creatememmap(Y, Cf, dview):
    """memmap the C and Y objects in parallel

       the memmapped object will be read during parallelized computation such as the regression function

       Args:
           Y: np.ndarray (2D or 3D)
               movie, raw data in 2D or 3D (pixels x time).

           Cf: np.ndarray
               calcium activity of each neuron + background components

       Returns:
           C_name: string
                the memmapped name of Cf

           Y_name: string
                the memmapped name of Y

           Raises:
           Exception 'Not implemented consistently'
           """
    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        tmpf = os.environ.get('SLURM_SUBMIT_DIR')
        print(f'cluster temporary folder: {tmpf}')
        folder = tempfile.mkdtemp(dir=tmpf)
    else:
        folder = tempfile.mkdtemp()

    if dview is None:
        Y_name = Y
        C_name = Cf
    else:
        C_name = os.path.join(folder, 'C_temp.npy')
        np.save(C_name, Cf)

        if isinstance(Y, np.core.memmap):  # if input file is already memory mapped then find the filename
            Y_name = Y.filename
        # if not create a memory mapped version (necessary for parallelization)
        elif isinstance(Y, str) or dview is None:
            Y_name = Y
        else:
            Y_name = os.path.join(folder, 'Y_temp.npy')
            np.save(Y_name, Y)
            Y, _, _, _ = caiman.mmapping.load_memmap(Y_name)
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

    if (rmax - rmin) * (cmax - cmin) <= 1:
        return img

    if rmin == 0 and rmax == nr and cmin == 0 and cmax == nc:
        ind_max = np.argmax(img)
        y0, x0 = np.unravel_index(ind_max, [nr, nc])
        vmax = img[y0, x0]
        x, y = np.meshgrid(np.arange(nc), np.arange(nr))
        try:
            fy, fx = np.gradient(img)
        except ValueError:
            f = np.gradient(img.ravel()).reshape(nr, nc)
            (fy, fx) = (f, 0) if nc == 1 else (0, f)
        ind = ((fx * (x0 - x) + fy * (y0 - y) < 0) & (img < vmax / 3))
        img[ind] = 0

        # # remove isolated pixels
        l, _ = label(img)
        ind = binary_dilation(l == l[y0, x0])
        img[~ind] = 0
    else:
        tmp_img = circular_constraint(img[rmin:rmax, cmin:cmax])
        img[rmin:rmax, cmin:cmax] = tmp_img

    return img

def connectivity_constraint(img_original, thr=.01, sz=5):
    """remove small nonzero pixels and disconnected components"""
    img = img_original.copy()
    ai_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((sz, sz), np.uint8))
    tmp = ai_open > img.max() * thr
    l, _ = label(tmp)
    img[l != l.ravel()[np.argmax(img)]] = 0
    return img

