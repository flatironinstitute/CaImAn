#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from past.builtins import basestring
from builtins import zip
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm
from math import sqrt

import caiman as cm
from .initialization import imblur, initialize_components, hals
import scipy
from scipy.sparse import coo_matrix, csc_matrix
from caiman.components_evaluation import compute_event_exceptionality
from .utilities import update_order
from caiman.source_extraction.cnmf import oasis
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import cv2
from skimage.feature import peak_local_max
import pylab as plt
from caiman.source_extraction.cnmf.spatial import threshold_components

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    profile
except:
    def profile(a): return a


#%%
def bare_initialization(Y, init_batch=1000, k=1, method_init='greedy_roi', gnb=1,
                        gSig=[5, 5], motion_flag=False, p=1,
                        return_object=True, **kwargs):
    """
    Quick and dirty initialization for OnACID, bypassing entirely CNMF
    Inputs:
    -------
    Y               movie object or np.array
                    matrix of data

    init_batch      int
                    number of frames to process

    method_init     string
                    initialization method

    k               int
                    number of components to find

    gnb             int
                    number of background components

    gSig            [int,int]
                    half-size of component

    motion_flag     bool
                    also perform motion correction

    Output:
    -------
        cnm_init    object
                    caiman CNMF-like object to initialize OnACID
    """

    if Y.ndim == 4:  # 3D data
        Y = Y[:, :, :, :init_batch]
    else:
        Y = Y[:, :, :init_batch]

    Ain, Cin, b_in, f_in, center = initialize_components(
        Y, K=k, gSig=gSig, nb=gnb, method=method_init)
    Ain = coo_matrix(Ain)
    b_in = np.array(b_in)
    Yr = np.reshape(Y, (Ain.shape[0], Y.shape[-1]), order='F')
    nA = (Ain.power(2).sum(axis=0))
    nr = nA.size

    YA = scipy.sparse.spdiags(old_div(1., nA), 0, nr, nr) * \
        (Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in))
    AA = scipy.sparse.spdiags(old_div(1., nA), 0, nr, nr) * (Ain.T.dot(Ain))
    YrA = YA - AA.T.dot(Cin)
    if return_object:
        cnm_init = cm.source_extraction.cnmf.cnmf.CNMF(2, k=k, gSig=gSig, Ain=Ain, Cin=Cin, b_in=np.array(
            b_in), f_in=f_in, method_init=method_init, p=p, **kwargs)
    
        cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.S, cnm_init.YrA = Ain, Cin, b_in, f_in, np.maximum(
            np.atleast_2d(Cin), 0), YrA
        #cnm_init.g = np.array([-np.poly([0.9]*max(p,1))[1:] for gg in np.ones(k)])
        cnm_init.g = np.array([-np.poly([0.9, 0.5][:max(1, p)])[1:]
                               for gg in np.ones(k)])
        cnm_init.bl = np.zeros(k)
        cnm_init.c1 = np.zeros(k)
        cnm_init.neurons_sn = np.std(YrA, axis=-1)
        cnm_init.lam = np.zeros(k)
        cnm_init.dims = Y.shape[:-1]
        cnm_init.initbatch = init_batch
        cnm_init.gnb = gnb

        return cnm_init
    else:
        return Ain, np.array(b_in), Cin, f_in, YrA


#%%
def seeded_initialization(Y, Ain, dims=None, init_batch=1000, order_init=None, gnb=1, p=1,
                          return_object=True, **kwargs):

    """
    Initialization for OnACID based on a set of user given binary masks.
    Inputs:
    -------
    Y               movie object or np.array
                    matrix of data

    Ain             bool np.array
                    2d np.array with binary masks

    dims            tuple
                    dimensions of FOV

    init_batch      int
                    number of frames to process

    gnb             int
                    number of background components

    order_init:     list
                    order of elements to be initalized using rank1 nmf restricted to the support of
                    each component

    Output:
    -------
        cnm_init    object
                    caiman CNMF-like object to initialize OnACID
    """


    if 'ndarray' not in str(type(Ain)):
        Ain = Ain.toarray()

    if dims is None:
        dims = Y.shape[:-1]
    px = (np.sum(Ain > 0, axis=1) > 0)
    not_px = 1 - px
    if 'matrix' in str(type(not_px)):
        not_px = np.array(not_px).flatten()
    Yr = np.reshape(Y, (Ain.shape[0], Y.shape[-1]), order='F')
    model = NMF(n_components=gnb, init='nndsvdar', max_iter=10)
    b_temp = model.fit_transform(np.maximum(Yr[not_px], 0), iter=20)
    f_in = model.components_.squeeze()
    f_in = np.atleast_2d(f_in)
    Y_resf = np.dot(Yr, f_in.T)
#    b_in = np.maximum(Y_resf.dot(np.linalg.inv(f_in.dot(f_in.T))), 0)
    b_in = np.maximum(np.linalg.solve(f_in.dot(f_in.T), Y_resf.T), 0).T
    Yr_no_bg = (Yr - b_in.dot(f_in)).astype(np.float32)

    Cin = np.zeros([Ain.shape[-1],Yr.shape[-1]], dtype = np.float32)
    if order_init is not None: #initialize using rank-1 nmf for each component

        model_comp = NMF(n_components=1, init='nndsvdar', max_iter=50)
        for count, idx_in in enumerate(order_init):
            if count%10 == 0:
                print(count)
            idx_domain = np.where(Ain[:,idx_in])[0]
            Ain[idx_domain,idx_in] = model_comp.fit_transform(\
                                   np.maximum(Yr_no_bg[idx_domain], 0)).squeeze()
            Cin[idx_in] = model_comp.components_.squeeze()
            Yr_no_bg[idx_domain] -= np.outer(Ain[idx_domain, idx_in],Cin[idx_in])
    else:
        Ain = normalize(Ain.astype('float32'), axis=0, norm='l1')
        Cin = np.maximum(Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in), 0)

#    Ain = HALS4shapes(Yr_no_bg, Ain, Cin, iters=5)
    Ain, Cin, b_in, f_in = hals(Yr, Ain, Cin, b_in, f_in, maxIter=8, bSiz=None)
    Ain = csc_matrix(Ain)
    nA = (Ain.power(2).sum(axis=0))
    nr = nA.size

    YA = scipy.sparse.spdiags(old_div(1., nA), 0, nr, nr) * \
        (Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in))
    AA = scipy.sparse.spdiags(old_div(1., nA), 0, nr, nr) * (Ain.T.dot(Ain))
    YrA = YA - AA.T.dot(Cin)
    if return_object:
        cnm_init = cm.source_extraction.cnmf.cnmf.CNMF(
            2, Ain=Ain, Cin=Cin, b_in=np.array(b_in), f_in=f_in, p=1, **kwargs)
        cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.S, cnm_init.YrA = Ain, Cin, b_in, f_in, np.fmax(
            np.atleast_2d(Cin), 0), YrA
    #    cnm_init.g = np.array([[gg] for gg in np.ones(nr)*0.9])
        cnm_init.g = np.array([-np.poly([0.9] * max(p, 1))[1:]
                               for gg in np.ones(nr)])
        cnm_init.bl = np.zeros(nr)
        cnm_init.c1 = np.zeros(nr)
        cnm_init.neurons_sn = np.std(YrA, axis=-1)
        cnm_init.lam = np.zeros(nr)
        cnm_init.dims = Y.shape[:-1]
        cnm_init.initbatch = init_batch
        cnm_init.gnb = gnb
    
        return cnm_init
    else:
        return Ain, np.array(b_in), Cin, f_in, YrA


def HALS4shapes(Yr, A, C, iters=2):
    K = A.shape[-1]
    ind_A = A > 0
    U = C.dot(Yr.T)
    V = C.dot(C.T)
    V_diag = V.diagonal() + np.finfo(float).eps
    for _ in range(iters):
        for m in range(K):  # neurons
            ind_pixels = np.squeeze(ind_A[:, m])
            A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                       ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                        V_diag[m]), 0, np.inf)

    return A


# definitions for demixed time series extraction and denoising/deconvolving
@profile
def HALS4activity(Yr, A, noisyC, AtA=None, iters=5, tol=1e-3, groups=None,
                  order=None):
    """Solves C = argmin_C ||Yr-AC|| using block-coordinate decent. Can use
    groups to update non-overlapping components in parallel or a specified
    order.

    Parameters
    ----------
    Yr : np.array (possibly memory mapped, (x,y,[,z]) x t)
        Imaging data reshaped in matrix format

    A : scipy.sparse.csc_matrix (or np.array) (x,y,[,z]) x # of components)
        Spatial components and background

    noisyC : np.array  (# of components x t)
        Temporal traces (including residuals plus background)

    AtA : np.array, optional (# of components x # of components)
        A.T.dot(A) Overlap matrix of shapes A.

    iters : int, optional
        Maximum number of iterations.

    tol : float, optional
        Change tolerance level

    groups : list of sets
        grouped components to be updated simultaneously

    order : list
        Update components in that order (used if nonempty and groups=None)

    Output:
    -------
    C : np.array (# of components x t)
        solution of HALS

    noisyC : np.array (# of components x t)
        solution of HALS + residuals, i.e, (C + YrA)
    """

    AtY = A.T.dot(Yr)
    num_iters = 0
    C_old = np.zeros_like(noisyC)
    C = noisyC.copy()
    if AtA is None:
        AtA = A.T.dot(A)
    AtAd = AtA.diagonal() + np.finfo(np.float32).eps

    # faster than np.linalg.norm
    def norm(c): return sqrt(c.ravel().dot(c.ravel()))
    while (norm(C_old - C) >= tol * norm(C_old)) and (num_iters < iters):
        C_old[:] = C
        if groups is None:
            if order is None:
                order = list(range(AtY.shape[0]))
            for m in order:
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / AtAd[m]
                C[m] = np.maximum(noisyC[m], 0)
        else:
            for m in groups:
                noisyC[m] = C[m] + ((AtY[m] - AtA[m].dot(C)).T/AtAd[m]).T
                C[m] = np.maximum(noisyC[m], 0)
        num_iters += 1
    return C, noisyC


@profile
def demix_and_deconvolve(C, noisyC, AtY, AtA, OASISinstances, iters=3, n_refit=0):
    """
    Solve C = argmin_C ||Y-AC|| subject to C following AR(p) dynamics
    using OASIS within block-coordinate decent
    Newly fits the last elements in buffers C and AtY and possibly refits
    earlier elements.
    Parameters
    ----------
    C : ndarray of float
        Buffer containing the denoised fluorescence intensities.
        All elements up to and excluding the last one have been denoised in
        earlier calls.
    noisyC : ndarray of float
        Buffer containing the undenoised fluorescence intensities.
    AtY : ndarray of float
        Buffer containing the projections of data Y on shapes A.
    AtA : ndarray of float
        Overlap matrix of shapes A.
    OASISinstances : list of OASIS objects
        Objects for deconvolution and denoising
    iters : int, optional
        Number of iterations.
    n_refit : int, optional
        Number of previous OASIS pools to refit
        0 fits only last pool, np.inf all pools fully (i.e. starting) within buffer
    """
    AtA += np.finfo(float).eps
    T = OASISinstances[0].t + 1
    len_buffer = C.shape[1]
    nb = AtY.shape[0] - len(OASISinstances)
    if n_refit == 0:
        for i in range(iters):
            for m in range(AtY.shape[0]):
                noisyC[m, -1] = C[m, -1] + \
                    (AtY[m, -1] - AtA[m].dot(C[:, -1])) / AtA[m, m]
                if m >= nb and i > 0:
                    n = m - nb
                    if i == iters - 1:  # commit
                        OASISinstances[n].fit_next(noisyC[m, -1])
                        l = OASISinstances[n].get_l_of_last_pool()
                        if l < len_buffer:
                            C[m, -l:] = OASISinstances[n].get_c_of_last_pool()
                        else:
                            C[m] = OASISinstances[n].get_c(len_buffer)
                    else:  # temporary non-commited update of most recent frame
                        C[m] = OASISinstances[n].fit_next_tmp(
                            noisyC[m, -1], len_buffer)
                else:
                    # no need to enforce max(c, 0) for background, is it?
                    C[m, -1] = np.maximum(noisyC[m, -1], 0)
    else:
        # !threshold .1 assumes normalized A (|A|_2=1)
        overlap = np.sum(AtA[nb:, nb:] > .1, 0) > 1

        def refit(o, c):
            # remove last pools
            tmp = 0
            while tmp < n_refit and o.t - o.get_l_of_last_pool() > T - len_buffer:
                o.remove_last_pool()
                tmp += 1
            # refit last pools
            for cc in c[o.t - T + len_buffer:-1]:
                o.fit_next(cc)
        for i in range(iters):
            for m in range(AtY.shape[0]):
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / AtA[m, m]
                if m >= nb:
                    n = m - nb
                    if overlap[n]:
                        refit(OASISinstances[n], noisyC[m])
                    if i == iters - 1:  # commit
                        OASISinstances[n].fit_next(noisyC[m, -1])
                        C[m] = OASISinstances[n].get_c(len_buffer)
                    else:  # temporary non-commited update of most recent frame
                        C[m] = OASISinstances[n].fit_next_tmp(
                            noisyC[m, -1], len_buffer)
                else:
                    # no need to enforce max(c, 0) for background, is it?
                    C[m] = noisyC[m]
    return C, noisyC, OASISinstances


#%% Estimate shapes on small initial batch
def init_shapes_and_sufficient_stats(Y, A, C, b, f, bSiz=3):
    # smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    nb = b.shape[1]  # number of background components
    if isinstance(bSiz, (int, float)):
        bSiz = [bSiz] * len(dims)

    # ind_A = uniform_filter(np.reshape(A, dims + (K,), order='F'), size=bSiz + [0])
    # ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
    # ind_A = [np.where(a)[0] for a in ind_A.T]
    Ab = np.hstack([b, A])
    # Ab = scipy.sparse.hstack([A.astype('float32'), b.astype('float32')]).tocsc()  might be faster
    # closing of shapes to not have holes in index matrix ind_A.
    # do this somehow smarter & faster, e.g. smooth only within patch !!
    #a = Ab[:,0]

    A_smooth = np.transpose([gaussian_filter(np.array(a).reshape(
        dims, order='F'), 0).ravel(order='F') for a in Ab.T])
    A_smooth[A_smooth < 1e-2] = 0
    # set explicity zeros of Ab to small value, s.t. ind_A and Ab.indptr match
    Ab += 1e-6 * A_smooth
    Ab = scipy.sparse.csc_matrix(Ab)
    ind_A = [Ab.indices[Ab.indptr[m]:Ab.indptr[m + 1]]
             for m in range(nb, nb + K)]
    Cf = np.r_[f.reshape(nb, -1), C]
    CY = Cf.dot(np.reshape(Y, (np.prod(dims), T), order='F').T)
    CC = Cf.dot(Cf.T)
    # # hals
    # for _ in range(5):
    #     for m in range(K):  # neurons
    #         ind_pixels = ind_A[m]
    #         Ab[ind_pixels, m] = np.clip(
    #             Ab[ind_pixels, m] + ((CY[m, ind_pixels] - CC[m].dot(Ab[ind_pixels].T)) / CC[m, m]),
    #             0, np.inf)
    #     for m in range(K, K + nb):  # background
    #         Ab[:, m] = np.clip(Ab[:, m] + ((CY[m] - CC[m].dot(Ab.T)) /
    #                                        CC[m, m]), 0, np.inf)
    return Ab, ind_A, CY, CC


@profile
def update_shapes(CY, CC, Ab, ind_A, sn=None, q=0.5, indicator_components=None, Ab_dense=None, update_bkgrd=True, iters=5):

    D, M = Ab.shape
    N = len(ind_A)
    nb = M - N
    if sn is None:
        L = np.zeros((M,D))
    else:
        L = norm.ppf(q)*np.outer(np.sqrt(CC.diagonal()), sn)
        L[:nb] = 0
    for _ in range(iters):  # it's presumably better to run just 1 iter but update more neurons
        if indicator_components is None:
            idx_comp = range(nb, M)
        else:
            idx_comp = np.where(indicator_components)[0] + nb

        if Ab_dense is None:
            for m in idx_comp:  # neurons
                ind_pixels = ind_A[m - nb]

                tmp = np.maximum(Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] +

                    ((CY[m, ind_pixels] - L[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels]) / CC[m, m]), 0)


                if tmp.dot(tmp) > 0:
                    tmp *= 1e-3 / \
                        min(1e-3, sqrt(tmp.dot(tmp)) + np.finfo(float).eps)
                    tmp = tmp / max(1, sqrt(tmp.dot(tmp)))
                    Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] = tmp

#                    Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] = np.maximum(
#                        Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] +
#                        ((CY[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels]) / CC[m, m]), 0)
                    # normalize
#                    Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] /= \
#                        max(1, sqrt(Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]]
#                                    .dot(Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]])))
                    ind_A[m -
                          nb] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]
                # N.B. Ab[ind_pixels].dot(CC[m]) is slower for csc matrix due to indexing rows
        else:
            for m in idx_comp:  # neurons
                ind_pixels = ind_A[m - nb]
                tmp = np.maximum(Ab_dense[ind_pixels, m] + ((CY[m, ind_pixels] - L[m, ind_pixels] -
                                                             Ab_dense[ind_pixels].dot(CC[m])) /
                                                            CC[m, m]), 0)
                # normalize
                if tmp.dot(tmp) > 0:
                    tmp *= 1e-3 / \
                        min(1e-3, sqrt(tmp.dot(tmp)) + np.finfo(float).eps)
                    Ab_dense[ind_pixels, m] = tmp / max(1, sqrt(tmp.dot(tmp)))
                    Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]
                            ] = Ab_dense[ind_pixels, m]
                    ind_A[m -
                          nb] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]
            # Ab.data[Ab.indptr[nb]:] = np.concatenate(
            #     [Ab_dense[ind_A[m - nb], m] for m in range(nb, M)])
            # N.B. why does selecting only overlapping neurons help surprisingly little, i.e
            # Ab[ind_pixels][:, overlap[m]].dot(CC[overlap[m], m])
            # where overlap[m] are the indices of all neurons overlappping with & including m?
            # sparsify ??
        if update_bkgrd:
            for m in range(nb):  # background
                sl = slice(Ab.indptr[m], Ab.indptr[m + 1])
                ind_pixels = Ab.indices[sl]
                Ab.data[sl] = np.maximum(
                    Ab.data[sl] + ((CY[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels]) / CC[m, m]), 0)
                if Ab_dense is not None:
                    Ab_dense[ind_pixels, m] = Ab.data[sl]

    return Ab, ind_A, Ab_dense


#%%
class RingBuffer(np.ndarray):
    """ implements ring buffer efficiently"""

    def __new__(cls, input_array, num_els):
        obj = np.asarray(input_array).view(cls)
        obj.max_ = num_els
        obj.cur = 0
        if input_array.shape[0] != num_els:
            print([input_array.shape[0], num_els])
            raise Exception('The first dimension should equal num_els')

        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.max_ = getattr(obj, 'max_', None)
        self.cur = getattr(obj, 'cur', None)

    def append(self, x):
        self[self.cur] = x
        self.cur = (self.cur + 1) % self.max_

    def get_ordered(self):
        return np.concatenate([self[self.cur:], self[:self.cur]], axis=0)

    def get_first(self):
        return self[self.cur]

    def get_last_frames(self, num_frames):
        if self.cur >= num_frames:
            return self[self.cur - num_frames:self.cur]
        else:
            return np.concatenate([self[(self.cur - num_frames):], self[:self.cur]], axis=0)


#%%
def csc_append(a, b):
    """ Takes in 2 csc_matrices and appends the second one to the right of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csc and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.concatenate((a.data, b.data))
    a.indices = np.concatenate((a.indices, b.indices))
    a.indptr = np.concatenate((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0], a.shape[1] + b.shape[1])


def corr(a, b):
    """
    faster correlation than np.corrcoef, especially for smaller arrays
    be aware of side effects and pass a copy if necessary!
    """
    a -= a.mean()
    b -= b.mean()
    return a.dot(b) / sqrt(a.dot(a) * b.dot(b) + np.finfo(float).eps)


def rank1nmf(Ypx, ain):
    """
    perform a fast rank 1 NMF
    """
    # cin_old = -1
    for _ in range(15):
        cin_res = ain.T.dot(Ypx)  # / ain.dot(ain)
        cin = np.maximum(cin_res, 0)
        ain = np.maximum(Ypx.dot(cin.T), 0)
        ain /= sqrt(ain.dot(ain) + np.finfo(float).eps)
        # nc = cin.dot(cin)
        # ain = np.maximum(Ypx.dot(cin.T) / nc, 0)
        # tmp = cin - cin_old
        # if tmp.dot(tmp) < 1e-6 * nc:
        #     break
        # cin_old = cin.copy()
    cin_res = ain.T.dot(Ypx)  # / ain.dot(ain)
    cin = np.maximum(cin_res, 0)
    return ain, cin, cin_res


#%%
def get_candidate_components(sv, dims, Yres_buf, min_num_trial=3, gSig=(5, 5),
                             gHalf=(5, 5), sniper_mode=True, rval_thr=0.85,
                             patch_size=50, loaded_model=None, test_both=False,
                             thresh_CNN_noisy=0.5, use_peak_max=False, thresh_std_peak_resid = 1):
    """
    Extract new candidate components from the residual buffer and test them
    using space correlation or the CNN classifier. The function runs the CNN
    classifier in batch mode which can bring speed improvements when
    multiple components are considered in each timestep.
    """
    Ain = []
    Ain_cnn = []
    Cin = []
    Cin_res = []
    idx = []
    ijsig_all = []
    cnn_pos = []
    local_maxima = []
    Y_patch = []
    ksize = tuple([int(3 * i / 2) * 2 + 1 for i in gSig])
    half_crop_cnn = tuple([int(np.minimum(gs*2, patch_size/2)) for gs in gSig])
    compute_corr = test_both

    if use_peak_max:

        img_select_peaks = sv.reshape(dims).copy()
#        plt.subplot(1,3,1)
#        plt.cla()
#        plt.imshow(img_select_peaks)

        img_select_peaks = cv2.GaussianBlur(img_select_peaks , ksize=ksize, sigmaX=gSig[0],
                                                        sigmaY=gSig[1], borderType=cv2.BORDER_REPLICATE) \
                    - cv2.boxFilter(img_select_peaks, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
        thresh_img_sel = np.median(img_select_peaks) + thresh_std_peak_resid  * np.std(img_select_peaks)

#        plt.subplot(1,3,2)
#        plt.cla()
#        plt.imshow(img_select_peaks*(img_select_peaks>thresh_img_sel))
#        plt.pause(.05)
#        threshold_abs = np.median(img_select_peaks) + np.std(img_select_peaks)

#        img_select_peaks -= np.min(img_select_peaks)
#        img_select_peaks /= np.max(img_select_peaks)
#        img_select_peaks *= 2**15
#        img_select_peaks = img_select_peaks.astype(np.uint16)
#        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(half_crop_cnn[0]//2,half_crop_cnn[0]//2))
#        img_select_peaks = clahe.apply(img_select_peaks)

        local_maxima = peak_local_max(img_select_peaks,
                                      min_distance=np.max(np.array(gSig)).astype(np.int),
                                      num_peaks=min_num_trial,threshold_abs=thresh_img_sel, exclude_border = False)
        min_num_trial = np.minimum(len(local_maxima),min_num_trial)


    for i in range(min_num_trial):
        if use_peak_max:
            ij = local_maxima[i]
        else:
            ind = np.argmax(sv)
            ij = np.unravel_index(ind, dims, order='C')
            local_maxima.append(ij)

        ij = [min(max(ij_val,g_val),dim_val-g_val-1) for ij_val, g_val, dim_val in zip(ij,gHalf,dims)]

        ij_cnn = [min(max(ij_val,g_val),dim_val-g_val-1) for ij_val, g_val, dim_val in zip(ij,half_crop_cnn,dims)]

        ind = np.ravel_multi_index(ij, dims, order='C')

        ijSig = [[max(i - g, 0), min(i+g+1,d)] for i, g, d in zip(ij, gHalf, dims)]

        ijsig_all.append(ijSig)
        ijSig_cnn = [[max(i - g, 0), min(i+g+1,d)] for i, g, d in zip(ij_cnn, half_crop_cnn, dims)]

        indeces = np.ravel_multi_index(np.ix_(*[np.arange(ij[0], ij[1])
                        for ij in ijSig]), dims, order='F').ravel(order = 'C')

        indeces_ = np.ravel_multi_index(np.ix_(*[np.arange(ij[0], ij[1])
                        for ij in ijSig]), dims, order='C').ravel(order = 'C')

        Ypx = Yres_buf.T[indeces, :]
        ain = np.maximum(np.mean(Ypx, 1), 0)

        if sniper_mode:
            indeces_cnn = np.ravel_multi_index(np.ix_(*[np.arange(ij[0], ij[1])
                            for ij in ijSig_cnn]), dims, order='F').ravel(order = 'C')
            Ypx_cnn = Yres_buf.T[indeces_cnn, :]
            ain_cnn = Ypx_cnn.mean(1)

        else:
            compute_corr = True  # determine when to compute corr coef

        na = ain.dot(ain)
        sv[indeces_] /= 1  # 0
        if na:
            ain /= sqrt(na)
            Ain.append(ain)
            Y_patch.append(Ypx)
            idx.append(ind)
            if sniper_mode:
                Ain_cnn.append(ain_cnn)

    if sniper_mode & (len(Ain_cnn) > 0):
        Ain_cnn = np.stack(Ain_cnn)
        Ain2 = Ain_cnn.copy()
        Ain2 -= np.median(Ain2,axis=1)[:,None]
        Ain2 /= np.std(Ain2,axis=1)[:,None]
        Ain2 = np.reshape(Ain2,(-1,) + tuple(np.diff(ijSig_cnn).squeeze()),order= 'F')
        Ain2 = np.stack([cv2.resize(ain,(patch_size ,patch_size)) for ain in Ain2])
        predictions = loaded_model.predict(Ain2[:,:,:,np.newaxis], batch_size=min_num_trial, verbose=0)
        keep_cnn = list(np.where(predictions[:, 0] > thresh_CNN_noisy)[0])
        discard = list(np.where(predictions[:, 0] <= thresh_CNN_noisy)[0])
        cnn_pos = Ain2[keep_cnn]
    else:
        keep_cnn = []  # list(range(len(Ain_cnn)))

    if compute_corr:
        keep_corr = []
        for i, (ain, Ypx) in enumerate(zip(Ain, Y_patch)):
            ain, cin, cin_res = rank1nmf(Ypx, ain)
            Ain[i] = ain
            Cin.append(cin)
            Cin_res.append(cin_res)
            rval = corr(ain.copy(), np.mean(Ypx, -1))
            if rval > rval_thr:
                keep_corr.append(i)
        keep_final = list(set().union(keep_cnn, keep_corr))
        if len(keep_final) > 0:
            Ain = np.stack(Ain)[keep_final]
        else:
            Ain = []
        Cin = [Cin[kp] for kp in keep_final]
        Cin_res = [Cin_res[kp] for kp in keep_final]
        idx = list(np.array(idx)[keep_final])
    else:
        Ain = [Ain[kp] for kp in keep_cnn]
        Y_patch = [Y_patch[kp] for kp in keep_cnn]
        idx = list(np.array(idx)[keep_cnn])
        for i, (ain, Ypx) in enumerate(zip(Ain, Y_patch)):
            ain, cin, cin_res = rank1nmf(Ypx, ain)
            Ain[i] = ain
            Cin.append(cin)
            Cin_res.append(cin_res)

    return Ain, Cin, Cin_res, idx, ijsig_all, cnn_pos, local_maxima


#%%
@profile
def update_num_components(t, sv, Ab, Cf, Yres_buf, Y_buf, rho_buf,
                          dims, gSig, gSiz, ind_A, CY, CC, groups, oases, gnb=1,
                          rval_thr=0.875, bSiz=3, robust_std=False,
                          N_samples_exceptionality=5, remove_baseline=True,
                          thresh_fitness_delta=-80, thresh_fitness_raw=-20,
                          thresh_overlap=0.25, batch_update_suff_stat=False,
                          sn=None, g=None, thresh_s_min=None, s_min=None,
                          Ab_dense=None, max_num_added=1, min_num_trial=1,
                          loaded_model=None, thresh_CNN_noisy=0.99,
                          sniper_mode=False, use_peak_max=False,
                          test_both=False):
    """
    Checks for new components in the residual buffer and incorporates them if they pass the acceptance tests
    """

    ind_new = []
    gHalf = np.array(gSiz) // 2

    # number of total components (including background)
    M = np.shape(Ab)[-1]
    N = M - gnb                 # number of coponents (without background)

    sv -= rho_buf.get_first()
    # update variance of residual buffer
    sv += rho_buf.get_last_frames(1).squeeze()
    sv = np.maximum(sv, 0)

    Ains, Cins, Cins_res, inds, ijsig_all, cnn_pos, local_max = get_candidate_components(sv, dims, Yres_buf=Yres_buf,
                                                          min_num_trial=min_num_trial, gSig=gSig,
                                                          gHalf=gHalf, sniper_mode=sniper_mode, rval_thr=rval_thr, patch_size=50,
                                                          loaded_model=loaded_model, thresh_CNN_noisy=thresh_CNN_noisy,
                                                          use_peak_max=use_peak_max, test_both=test_both)

    ind_new_all = ijsig_all

    num_added = len(inds)
    cnt = 0
    for ind, ain, cin, cin_res in zip(inds, Ains, Cins, Cins_res):
        cnt += 1
        ij = np.unravel_index(ind, dims)

        ijSig = [[max(i - temp_g, 0), min(i + temp_g + 1, d)] for i, temp_g, d in zip(ij, gHalf, dims)]
        dims_ain = (np.abs(np.diff(ijSig[1])[0]), np.abs(np.diff(ijSig[0])[0]))

        indeces = np.ravel_multi_index(
                np.ix_(*[np.arange(ij[0], ij[1])
                       for ij in ijSig]), dims, order='F').ravel()

        # use sparse Ain only later iff it is actually added to Ab
        Ain = np.zeros((np.prod(dims), 1), dtype=np.float32)
        Ain[indeces, :] = ain[:, None]

        cin_circ = cin.get_ordered()
        useOASIS = False  # whether to use faster OASIS for cell detection
        accepted = True   # flag indicating new component has not been rejected yet

        if Ab_dense is None:
            ff = np.where((Ab.T.dot(Ain).T > thresh_overlap)
                          [:, gnb:])[1] + gnb
        else:
            ff = np.where(Ab_dense[indeces, gnb:].T.dot(
                ain).T > thresh_overlap)[0] + gnb

        if ff.size > 0:
#                accepted = False
            cc = [corr(cin_circ.copy(), cins) for cins in Cf[ff, :]]
            if np.any(np.array(cc) > .25) and accepted:
                accepted = False         # reject component as duplicate

        if s_min is None:
            s_min = 0
        # use s_min * noise estimate * sqrt(1-sum(gamma))
        elif s_min < 0:
            # the formula has been obtained by running OASIS with s_min=0 and lambda=0 on Gaussin noise.
            # e.g. 1 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the root mean square (non-zero) spike size, sqrt(<s^2>)
            #      2 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the 95% percentile of (non-zero) spike sizes
            #      3 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the 99.7% percentile of (non-zero) spike sizes
            s_min = -s_min * sqrt((ain**2).dot(sn[indeces]**2)) * sqrt(1 - np.sum(g))

        cin_res = cin_res.get_ordered()
        if accepted:
            if useOASIS:
                oas = oasis.OASIS(g=g, s_min=s_min,
                                  num_empty_samples=t + 1 - len(cin_res))
                for yt in cin_res:
                    oas.fit_next(yt)
                accepted = oas.get_l_of_last_pool() <= t
            else:
                fitness_delta, erfc_delta, std_rr, _ = compute_event_exceptionality(
                    np.diff(cin_res)[None, :], robust_std=robust_std, N=N_samples_exceptionality)
                if remove_baseline:
                    num_samps_bl = min(len(cin_res) // 5, 800)
                    bl = scipy.ndimage.percentile_filter(
                        cin_res, 8, size=num_samps_bl)
                else:
                    bl = 0
                fitness_raw, erfc_raw, std_rr, _ = compute_event_exceptionality(
                    (cin_res - bl)[None, :], robust_std=robust_std,
                    N=N_samples_exceptionality)
                accepted = (fitness_delta < thresh_fitness_delta) or (
                    fitness_raw < thresh_fitness_raw)

#        if accepted:
#            dims_ain = (np.abs(np.diff(ijSig[1])[0]), np.abs(np.diff(ijSig[0])[0]))
#            thrcomp = threshold_components(ain[:,None],
#                                 dims_ain, medw=None, thr_method='max', maxthr=0.2,
#                                 nrgthr=0.99, extract_cc=True,
#                                 se=None, ss=None)
#
#            sznr = np.sum(thrcomp>0)
#            accepted = (sznr >= np.pi*(np.prod(gSig)/4))
#            if not accepted:
#                print('Rejected because of size')

        if accepted:
            # print('adding component' + str(N + 1) + ' at timestep ' + str(t))
            num_added += 1
            ind_new.append(ijSig)

            if oases is not None:
                if not useOASIS:
                    # lambda from Selesnick's 3*sigma*|K| rule
                    # use noise estimate from init batch or use std_rr?
                    #                    sn_ = sqrt((ain**2).dot(sn[indeces]**2)) / sqrt(1 - g**2)
                    sn_ = std_rr
                    oas = oasis.OASIS(np.ravel(g)[0], 3 * sn_ /
                                      (sqrt(1 - g**2) if np.size(g) == 1 else
                                       sqrt((1 + g[1]) * ((1 - g[1])**2 - g[0]**2) / (1 - g[1])))
                                      if s_min == 0 else 0,
                                      s_min, num_empty_samples=t +
                                      1 - len(cin_res),
                                      g2=0 if np.size(g) == 1 else g[1])
                    for yt in cin_res:
                        oas.fit_next(yt)

                oases.append(oas)

            Ain_csc = scipy.sparse.csc_matrix((ain, (indeces, [0] * len(indeces))),
                                              (np.prod(dims), 1), dtype=np.float32)
            if Ab_dense is None:
                groups = update_order(Ab, Ain, groups)[0]
            else:
                groups = update_order(Ab_dense[indeces], ain, groups)[0]
                Ab_dense = np.hstack((Ab_dense,Ain))
            # faster version of scipy.sparse.hstack
            csc_append(Ab, Ain_csc)
            ind_A.append(Ab.indices[Ab.indptr[M]:Ab.indptr[M + 1]])

            tt = t * 1.
            Y_buf_ = Y_buf
            cin_ = cin
            Cf_ = Cf
            cin_circ_ = cin_circ

            CY[M, indeces] = cin_.dot(Y_buf_[:, indeces]) / tt

            # preallocate memory for speed up?
            CC1 = np.hstack([CC, Cf_.dot(cin_circ_ / tt)[:, None]])
            CC2 = np.hstack(
                [(Cf_.dot(cin_circ_)).T, cin_circ_.dot(cin_circ_)]) / tt
            CC = np.vstack([CC1, CC2])
            Cf = np.vstack([Cf, cin_circ])

            N = N + 1
            M = M + 1

            Yres_buf[:, indeces] -= np.outer(cin, ain)
            # vb = imblur(np.reshape(Ain, dims, order='F'), sig=gSig,
            #             siz=gSiz, nDimBlur=2).ravel()
            # restrict blurring to region where component is located
#            vb = np.reshape(Ain, dims, order='F')
            slices = tuple(slice(max(0, ijs[0] - 2*sg), min(d, ijs[1] + 2*sg))
                           for ijs, sg, d in zip(ijSig, gSiz//2, dims))  # is 2 enough?

            slice_within = tuple(slice(ijs[0] - sl.start, ijs[1] - sl.start)
                           for ijs, sl in zip(ijSig, slices))


            ind_vb = np.ravel_multi_index(
               np.ix_(*[np.arange(ij[0], ij[1])
                      for ij in ijSig]), dims, order='C').ravel()




            vb_buf = [imblur(np.maximum(0,vb.reshape(dims,order='F')[slices][slice_within]), sig=gSig, siz=gSiz, nDimBlur=len(dims)) for vb in Yres_buf]

            vb_buf2 = np.stack([vb.ravel() for vb in vb_buf])

#            ind_vb = np.ravel_multi_index(
#                    np.ix_(*[np.arange(s.start, s.stop)
#                           for s in slices_small]), dims).ravel()

            rho_buf[:, ind_vb] = vb_buf2**2


            sv[ind_vb] = np.sum(rho_buf[:, ind_vb], 0)
#            sv = np.sum([imblur(vb.reshape(dims,order='F'), sig=gSig, siz=gSiz, nDimBlur=len(dims))**2 for vb in Yres_buf], 0).reshape(-1)
#            plt.subplot(1,5,4)
#            plt.cla()
#            plt.imshow(sv.reshape(dims), vmax=30)
#            plt.pause(.05)
#            plt.subplot(1,5,5)
#            plt.cla()
#            plt.imshow(Yres_buf.mean(0).reshape(dims,order='F'))
#            plt.imshow(np.sum([imblur(vb.reshape(dims,order='F'),\
#                                       sig=gSig, siz=gSiz, nDimBlur=len(dims))**2\
#                                        for vb in Yres_buf],axis=0), vmax=30)
#            plt.pause(.05)

    #print(np.min(sv))
#    plt.subplot(1,3,3)
#    plt.cla()
#    plt.imshow(Yres_buf.mean(0).reshape(dims, order = 'F'))
#    plt.pause(.05)
    return Ab, Cf, Yres_buf, rho_buf, CC, CY, ind_A, sv, groups, ind_new, ind_new_all, sv, cnn_pos


#%% remove components online

def remove_components_online(ind_rem, gnb, Ab, use_dense, Ab_dense, AtA, CY,
                             CC, M, N, noisyC, OASISinstances, C_on, exp_comps):

    """
    Remove components indexed by ind_r (indexing starts at zero)

    ind_rem list
        indeces of components to be removed (starting from zero)
    gnb int
        number of global background components
    Ab  csc_matrix
        matrix of components + background
    use_dense bool
        use dense representation
    Ab_dense ndarray
    """

    ind_rem.sort()
    ind_rem = [ind + gnb for ind in ind_rem[::-1]]
    ind_keep = list(set(range(Ab.shape[-1])) - set(ind_rem))
    ind_keep.sort()

    if use_dense:
        Ab_dense = np.delete(Ab_dense, ind_rem, axis=1)
    else:
        Ab_dense = []
    AtA = np.delete(AtA, ind_rem, axis=0)
    AtA = np.delete(AtA, ind_rem, axis=1)
    CY = np.delete(CY, ind_rem, axis=0)
    CC = np.delete(CC, ind_rem, axis=0)
    CC = np.delete(CC, ind_rem, axis=1)
    M -= len(ind_rem)
    N -= len(ind_rem)
    exp_comps -= len(ind_rem)
    noisyC = np.delete(noisyC, ind_rem, axis=0)
    for ii in ind_rem:
        del OASISinstances[ii - gnb]
#        #del self.ind_A[ii-self.gnb]

    C_on = np.delete(C_on, ind_rem, axis=0)
    Ab = scipy.sparse.csc_matrix(Ab[:, ind_keep])
    ind_A = list(
        [(Ab.indices[Ab.indptr[ii]:Ab.indptr[ii+1]]) for ii in range(gnb, M)])
    groups = list(map(list, update_order(Ab)[0]))

    return Ab, Ab_dense, CC, CY, M, N, noisyC, OASISinstances, C_on, exp_comps, ind_A, groups, AtA
#%%


def initialize_movie_online(Y, K, gSig, rf, stride, base_name,
                            p=1, merge_thresh=0.95, rval_thr_online=0.9, thresh_fitness_delta_online=-30, thresh_fitness_raw_online=-50,
                            rval_thr_init=.5, thresh_fitness_delta_init=-20, thresh_fitness_raw_init=-20,
                            rval_thr_refine=0.95, thresh_fitness_delta_refine=-100, thresh_fitness_raw_refine=-100,
                            final_frate=10, Npeaks=10, single_thread=True, dview=None, n_processes=None):
    """
    Initialize movie using CNMF on minibatch. See CNMF parameters
    """

    _, d1, d2 = Y.shape
    dims = (d1, d2)
    Yr = Y.to_2D().T
    # merging threshold, max correlation allowed
    # order of the autoregressive system
    #T = Y.shape[0]
    base_name = base_name + '.mmap'
    fname_new = Y.save(base_name, order='C')
    #%
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.reshape(Yr, dims + (T,), order='F')
    Cn2 = cm.local_correlations(Y)
#    pl.imshow(Cn2)
    #%
    #% RUN ALGORITHM ON PATCHES
#    pl.close('all')
    cnm_init = cm.source_extraction.cnmf.CNMF(n_processes, method_init='greedy_roi', k=K, gSig=gSig, merge_thresh=merge_thresh,
                                              p=0, dview=dview, Ain=None, rf=rf, stride=stride, method_deconvolution='oasis', skip_refinement=False,
                                              normalize_init=False, options_local_NMF=None,
                                              minibatch_shape=100, minibatch_suff_stat=5,
                                              update_num_comps=True, rval_thr=rval_thr_online, thresh_fitness_delta=thresh_fitness_delta_online, thresh_fitness_raw=thresh_fitness_raw_online,
                                              batch_update_suff_stat=True, max_comp_update_shape=5)

    cnm_init = cnm_init.fit(images)
    A_tot = cnm_init.A
    C_tot = cnm_init.C
    YrA_tot = cnm_init.YrA
    b_tot = cnm_init.b
    f_tot = cnm_init.f

    print(('Number of components:' + str(A_tot.shape[-1])))

    #%

    traces = C_tot + YrA_tot
    #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
    #        traces_b=np.diff(traces,axis=1)
    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = cm.components_evaluation.evaluate_components(
        Y, traces, A_tot, C_tot, b_tot, f_tot, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

    idx_components_r = np.where(r_values >= rval_thr_init)[0]
    idx_components_raw = np.where(fitness_raw < thresh_fitness_raw_init)[0]
    idx_components_delta = np.where(
        fitness_delta < thresh_fitness_delta_init)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(('Keeping ' + str(len(idx_components)) +
           ' and discarding  ' + str(len(idx_components_bad))))

    A_tot = A_tot.tocsc()[:, idx_components]
    C_tot = C_tot[idx_components]
    #%
    cnm_refine = cm.source_extraction.cnmf.CNMF(n_processes, method_init='greedy_roi', k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, rf=None, stride=None,
                                                p=p, dview=dview, Ain=A_tot, Cin=C_tot, f_in=f_tot, method_deconvolution='oasis', skip_refinement=True,
                                                normalize_init=False, options_local_NMF=None,
                                                minibatch_shape=100, minibatch_suff_stat=5,
                                                update_num_comps=True, rval_thr=rval_thr_refine, thresh_fitness_delta=thresh_fitness_delta_refine, thresh_fitness_raw=thresh_fitness_raw_refine,
                                                batch_update_suff_stat=True, max_comp_update_shape=5)

    cnm_refine = cnm_refine.fit(images)
    #%
    A, C, b, f, YrA, sn = cnm_refine.A, cnm_refine.C, cnm_refine.b, cnm_refine.f, cnm_refine.YrA, cnm_refine.sn
    #%
    final_frate = 10
    Npeaks = 10
    traces = C + YrA

    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
        cm.components_evaluation.evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                                                     N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

    idx_components_r = np.where(r_values >= rval_thr_refine)[0]
    idx_components_raw = np.where(fitness_raw < thresh_fitness_raw_refine)[0]
    idx_components_delta = np.where(
        fitness_delta < thresh_fitness_delta_refine)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(' ***** ')
    print((len(traces)))
    print((len(idx_components)))
    #%
    cnm_refine.idx_components = idx_components
    cnm_refine.idx_components_bad = idx_components_bad
    cnm_refine.r_values = r_values
    cnm_refine.fitness_raw = fitness_raw
    cnm_refine.fitness_delta = fitness_delta
    cnm_refine.Cn2 = Cn2

    #%

#    cnm_init.dview = None
#    save_object(cnm_init,fls[0][:-4]+ '_DS_' + str(ds)+ '_init.pkl')

    return cnm_refine, Cn2, fname_new
