#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A set of routines for estimating the temporal components, given the spatial
components and temporal components
"""

from builtins import str
from builtins import map
from builtins import range
import logging
from scipy.sparse import spdiags, diags, coo_matrix, csc_matrix  # ,csgraph
import scipy
import numpy as np
import platform
import psutil
from .deconvolution import constrained_foopsi
from .utilities import update_order_greedy
import sys
from ...mmapping import parallel_dot_product

def make_G_matrix(T, g):
    """
    create matrix of autoregression to enforce indicator dynamics

    Args:
        T: positive integer
            number of time-bins

        g: nd.array, vector p x 1
            Discrete time constants

    Output:
        G: sparse diagonal matrix
            Matrix of autoregression
    """
    if type(g) is np.ndarray:
        if len(g) == 1 and g < 0:
            g = 0
        gs = np.matrix(np.hstack((1, -(g[:]).T)))
        ones_ = np.matrix(np.ones((T, 1)))
        G = spdiags((ones_ * gs).T, list(range(0, -len(g) - 1, -1)), T, T)

        return G
    else:
        raise Exception('g must be an array')

def constrained_foopsi_parallel(arg_in):
    """ necessary for parallel computation of the function  constrained_foopsi

        the most likely discretized spike train underlying a fluorescence trace
    """

    Ytemp, nT, jj_, bl, c1, g, sn, argss = arg_in
    T = np.shape(Ytemp)[0]
    cc_, cb_, c1_, gn_, sn_, sp_, lam_ = constrained_foopsi(
        Ytemp, bl=bl, c1=c1, g=g, sn=sn, **argss)
    gd_ = np.max(np.real(np.roots(np.hstack((1, -gn_.T)))))
    gd_vec = gd_**list(range(T))

    C_ = cc_[:].T + cb_ + np.dot(c1_, gd_vec)
    Sp_ = sp_[:T].T
    Ytemp_ = Ytemp - C_.T

    return C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_, lam_

def update_temporal_components(Y, A, b, Cin, fin, bl=None, c1=None, g=None, sn=None, nb=1, ITER=2, block_size_temp=5000, num_blocks_per_run_temp=20, debug=False, dview=None, **kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.

    Args:
        Y: np.ndarray (2D)
            input data with time in the last axis (d x T)

        A: sparse matrix (crc format)
            matrix of temporal components (d x K)

        b: ndarray (dx1)
            current estimate of background component

        Cin: np.ndarray
            current estimate of temporal components (K x T)

        fin: np.ndarray
            current estimate of temporal background (vector of length T)

        g:  np.ndarray
            Global time constant (not used)

        bl: np.ndarray
           baseline for fluorescence trace for each column in A

        c1: np.ndarray
           initial concentration for each column in A

        g:  np.ndarray
           discrete time constant for each column in A

        sn: np.ndarray
           noise level for each column in A

        nb: [optional] int
            Number of background components

        ITER: positive integer
            Maximum number of block coordinate descent loops.

        method_foopsi: string
            Method of deconvolution of neural activity. constrained_foopsi is the only method supported at the moment.

        n_processes: int
            number of processes to use for parallel computation.
             Should be less than the number of processes started with ipcluster.

        backend: 'str'
            single_thread no parallelization
            ipyparallel, parallelization using the ipyparallel cluster.
            You should start the cluster (install ipyparallel and then type
            ipcluster -n 6, where 6 is the number of processes).
            SLURM: using SLURM scheduler

        memory_efficient: Bool
            whether or not to optimize for memory usage (longer running times). necessary with very large datasets

        kwargs: dict
            all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation).
             Some useful parameters are

        p: int
            order of the autoregression model

        method: [optional] string
            solution method for constrained foopsi. Choices are
                'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
                'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)

        solvers: list string
                primary and secondary (if problem unfeasible for approx solution)
                 solvers to be used with cvxpy, default is ['ECOS','SCS']

    Note:
        The temporal components are updated in parallel by default by forming of sequence of vertex covers.

    Returns:
        C:   np.ndarray
                matrix of temporal components (K x T)

        f:   np.array
                vector of temporal background (length T)

        S:   np.ndarray
                matrix of merged deconvolved activity (spikes) (K x T)

        bl:  float
                same as input

        c1:  float
                same as input

        g:   float
                same as input

        sn:  float
                same as input

        YrA: np.ndarray
                matrix of spatial component filtered raw data, after all contributions have been removed.
                YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)

        lam: np.ndarray
            Automatically tuned sparsity parameter
    """

    if 'p' not in kwargs or kwargs['p'] is None:
        raise Exception("You have to provide a value for p")

    # INITIALIZATION OF VARS
    d, T = np.shape(Y)
    nr = np.shape(A)[-1]
    if b is not None:
        # if b.shape[0] < b.shape[1]:
        #     b = b.T
        nb = b.shape[1]
    if bl is None:
        bl = np.repeat(None, nr)

    if c1 is None:
        c1 = np.repeat(None, nr)

    if g is None:
        g = np.repeat(None, nr)

    if sn is None:
        sn = np.repeat(None, nr)

    A = scipy.sparse.hstack((A, b)).tocsc()
    S = np.zeros(np.shape(Cin))
    Cin = np.vstack((Cin, fin))
    C = Cin.copy()
    nA = np.ravel(A.power(2).sum(axis=0))

    logging.info('Generating residuals')
#    dview_res = None if block_size >= 500 else dview
    if 'memmap' in str(type(Y)):
        bl_siz1 = d // (np.maximum(num_blocks_per_run_temp - 1, 1))
        bl_siz2 = int(psutil.virtual_memory().available/(num_blocks_per_run_temp + 1) - 4*A.nnz) // int(4*T)
        # block_size_temp
        YA = parallel_dot_product(Y, A.tocsr(), dview=dview, block_size=min(bl_siz1, bl_siz2),
                                  transpose=True, num_blocks_per_run=num_blocks_per_run_temp) * diags(1. / nA);
    else:
        YA = (A.T.dot(Y).T) * diags(1. / nA)
    AA = ((A.T.dot(A)) * diags(1. / nA)).tocsr()
    YrA = YA - AA.T.dot(Cin).T
    # creating the patch of components to be computed in parrallel
    parrllcomp, len_parrllcomp = update_order_greedy(AA[:nr, :][:, :nr])
    logging.info("entering the deconvolution ")
    C, S, bl, YrA, c1, sn, g, lam = update_iteration(parrllcomp, len_parrllcomp, nb, C, S, bl, nr,
                                                     ITER, YrA, c1, sn, g, Cin, T, nA, dview, debug, AA, kwargs)
    ff = np.where(np.sum(C, axis=1) == 0)  # remove empty components
    if np.size(ff) > 0:  # Eliminating empty temporal components
        ff = ff[0]
        logging.info('removing {0} empty spatial component(s)'.format(len(ff)))
        keep = list(range(A.shape[1]))
        for i in ff:
            keep.remove(i)

        A = A[:, keep]
        C = np.delete(C, list(ff), 0)
        YrA = np.delete(YrA, list(ff), 1)
        S = np.delete(S, list(ff), 0)
        sn = np.delete(sn, list(ff))
        g = np.delete(g, list(ff))
        bl = np.delete(bl, list(ff))
        c1 = np.delete(c1, list(ff))
        lam = np.delete(lam, list(ff))

        background_ff = list(filter(lambda i: i > 0, ff - nr))
        nr = nr - (len(ff) - len(background_ff))

    b = A[:, nr:].toarray()
    A = csc_matrix(A[:, :nr])
    f = C[nr:, :]
    C = C[:nr, :]
    YrA = np.array(YrA[:, :nr]).T
    return C, A, b, f, S, bl, c1, sn, g, YrA, lam


def update_iteration(parrllcomp, len_parrllcomp, nb, C, S, bl, nr,
                     ITER, YrA, c1, sn, g, Cin, T, nA, dview, debug, AA, kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.

    Args:
        YrA: np.ndarray (2D)
            input data with time in the last axis (d x T)

        AA: sparse matrix (crc format)
            matrix of temporal components (d x K)

        Cin: np.ndarray
            current estimate of temporal components (K x T)

        g:  np.ndarray
            Global time constant (not used)

        bl: np.ndarray
           baseline for fluorescence trace for each column in A

        c1: np.ndarray
           initial concentration for each column in A

        g:  np.ndarray
           discrete time constant for each column in A

        sn: np.ndarray
           noise level for each column in A

        nb: [optional] int
            Number of background components

        ITER: positive integer
            Maximum number of block coordinate descent loops.

        backend: 'str'
            single_thread no parallelization
            ipyparallel, parallelization using the ipyparallel cluster.
            You should start the cluster (install ipyparallel and then type
            ipcluster -n 6, where 6 is the number of processes).
            SLURM: using SLURM scheduler

        memory_efficient: Bool
            whether or not to optimize for memory usage (longer running times). necessary with very large datasets

        **kwargs: dict
            all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation).
             Some useful parameters are

        p: int
            order of the autoregression model

        method: [optional] string
            solution method for constrained foopsi. Choices are
                'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
                'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)

        solvers: list string
            primary and secondary (if problem unfeasible for approx solution)
            solvers to be used with cvxpy, default is ['ECOS','SCS']

    Note:
        The temporal components are updated in parallel by default by forming of sequence of vertex covers.

    Returns:
        C:   np.ndarray
            matrix of temporal components (K x T)

        S:   np.ndarray
            matrix of merged deconvolved activity (spikes) (K x T)

        bl:  float
            same as input

        c1:  float
            same as input

        g:   float
            same as input

        sn:  float
            same as input

        YrA: np.ndarray
            matrix of spatial component filtered raw data, after all contributions have been removed.
            YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)
"""

    lam = np.repeat(None, nr)
    for _ in range(ITER):

        for count, jo_ in enumerate(parrllcomp):
            # INITIALIZE THE PARAMS
            jo = np.array(list(jo_))
            Ytemp = YrA[:, jo.flatten()] + Cin[jo, :].T
            Ctemp = np.zeros((np.size(jo), T))
            Stemp = np.zeros((np.size(jo), T))
            nT = nA[jo]
            args_in = [(np.squeeze(np.array(Ytemp[:, jj])), nT[jj], jj, None,
                        None, None, None, kwargs) for jj in range(len(jo))]
            # computing the most likely discretized spike train underlying a fluorescence trace
            if 'multiprocessing' in str(type(dview)):
                results = dview.map_async(
                    constrained_foopsi_parallel, args_in).get(4294967)

            elif dview is not None and platform.system() != 'Darwin':
                if debug:
                    results = dview.map_async(
                        constrained_foopsi_parallel, args_in)
                    results.get()
                    for outp in results.stdout:
                        print((outp[:-1]))
                        sys.stdout.flush()
                    for outp in results.stderr:
                        print((outp[:-1]))
                        sys.stderr.flush()
                else:
                    results = dview.map_sync(
                        constrained_foopsi_parallel, args_in)

            else:
                results = list(map(constrained_foopsi_parallel, args_in))
            # unparsing and updating the result
            for chunk in results:
                C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_, lam_ = chunk
                Ctemp[jj_, :] = C_[None, :]
                Stemp[jj_, :] = Sp_[None, :]
                bl[jo[jj_]] = cb_
                c1[jo[jj_]] = c1_
                sn[jo[jj_]] = sn_
                g[jo[jj_]] = gn_.T if kwargs['p'] > 0 else []
                lam[jo[jj_]] = lam_

            YrA -= AA[jo, :].T.dot(Ctemp - C[jo, :]).T
            C[jo, :] = Ctemp.copy()
            S[jo, :] = Stemp
            logging.info("{0} ".format(np.sum(len_parrllcomp[:count + 1])) +
                         "out of total {0} temporal components ".format(nr) +
                         "updated")

        for ii in np.arange(nr, nr + nb):
            cc = np.maximum(YrA[:, ii] + Cin[ii], -np.Inf)
            YrA -= AA[ii, :].T.dot((cc - Cin[ii])[None, :]).T
            C[ii, :] = cc

        if dview is not None and not('multiprocessing' in str(type(dview))):
            dview.results.clear()

        try:
            if scipy.linalg.norm(Cin - C, 'fro') <= 1e-3*scipy.linalg.norm(C, 'fro'):
                logging.info("stopping: overall temporal component not changing" +
                             " significantly")
                break
            else:  # we keep Cin and do the iteration once more
                Cin = C.copy()
        except ValueError:
            logging.warning("Aborting updating of temporal components due" +
                            " to possible numerical issues.")
            C = Cin.copy()
            break

    return C, S, bl, YrA, c1, sn, g, lam
