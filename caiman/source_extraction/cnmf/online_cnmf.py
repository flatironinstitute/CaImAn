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


try:
    profile
except:
    profile = lambda a: a


#%%
def bare_initialization(Y, init_batch = 1000, k = 1, method_init = 'greedy_roi', gnb = 1,
                        gSig = [5,5], motion_flag = False, **kwargs):
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
    
    if motion_flag:
        Y = Y[:,:,:init_batch]
    else:
        Y = Y[:,:,:init_batch]           
    
    Ain, Cin, b_in, f_in, center = initialize_components(Y, K=k, gSig=gSig, nb = gnb, method= method_init)
    Ain = coo_matrix(Ain)
    b_in = np.array(b_in)
    Yr = np.reshape(Y, (Ain.shape[0],Y.shape[-1]), order='F')
    nA = (Ain.power(2).sum(axis=0))
    nr = nA.size

    YA = scipy.sparse.spdiags(old_div(1.,nA),0,nr,nr)*(Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in))
    AA = scipy.sparse.spdiags(old_div(1.,nA),0,nr,nr)*(Ain.T.dot(Ain))
    YrA = YA - AA.T.dot(Cin)
    
    cnm_init = cm.source_extraction.cnmf.cnmf.CNMF(2, k=k, gSig=gSig, Ain=Ain, Cin=Cin, b_in=np.array(b_in), f_in=f_in, method_init = method_init, **kwargs)
    cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.S, cnm_init.YrA = Ain, Cin, b_in, f_in, np.maximum(np.atleast_2d(Cin),0), YrA
    cnm_init.g = np.array([[gg] for gg in np.ones(k)*0.9])
    cnm_init.bl = np.zeros(k)
    cnm_init.c1 = np.zeros(k)
    cnm_init.neurons_sn = np.std(YrA,axis=-1)
    cnm_init.lam = np.zeros(k)
    cnm_init.dims = Y.shape[:-1]
    cnm_init.initbatch = init_batch
    cnm_init.gnb = gnb
    
    return cnm_init


#%%
def seeded_initialization(Y, Ain, dims = None, init_batch = 1000, gnb = 1, **kwargs):
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
                    
    Output:
    -------
        cnm_init    object
                    caiman CNMF-like object to initialize OnACID
    """

    def HALS4shapes(Yr, A, C, iters=2):
        K = A.shape[-1]
        ind_A = A>0
        U = C.dot(Yr.T)
        V = C.dot(C.T)
        for _ in range(iters):
            for m in range(K):  # neurons
                ind_pixels = np.squeeze(ind_A[:, m])
                A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                           ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                            V[m, m]), 0, np.inf)
                
        return A    

    if dims is None:
        dims = Y.shape[:-1]
               
    px = (np.sum(Ain > 0, axis = 1) > 0);
    not_px = 1-px
    
    Yr = np.reshape(Y,(Ain.shape[0],Y.shape[-1]), order='F')
    model = NMF(n_components = gnb, init='nndsvdar')    
    b_temp = model.fit_transform(np.maximum(Yr[not_px,:], 0))
    f_in = model.components_.squeeze()
    f_in = np.atleast_2d(f_in)
    Y_resf = np.dot(Yr, f_in.T)
    b_in = np.maximum(Y_resf, 0)/(np.linalg.norm(f_in)**2)
    
    Ain = normalize(Ain.astype('float32'),axis=0,norm='l1')

    Cin = np.maximum(Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in), 0)
    Ain = HALS4shapes(Yr - b_in.dot(f_in), Ain, Cin, iters=5)
    Ain, Cin, b_in, f_in = hals(Yr, Ain, Cin, b_in, f_in)
    Ain = csc_matrix(Ain)
    nA = (Ain.power(2).sum(axis=0))
    nr = nA.size

    YA = scipy.sparse.spdiags(old_div(1.,nA),0,nr,nr)*(Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in))
    AA = scipy.sparse.spdiags(old_div(1.,nA),0,nr,nr)*(Ain.T.dot(Ain))
    YrA = YA - AA.T.dot(Cin)
    
    cnm_init = cm.source_extraction.cnmf.cnmf.CNMF(2, Ain=Ain, Cin=Cin, b_in=np.array(b_in), f_in=f_in, **kwargs)
    cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.S, cnm_init.YrA = Ain, Cin, b_in, f_in, np.fmax(np.atleast_2d(Cin),0), YrA
    cnm_init.g = np.array([[gg] for gg in np.ones(nr)*0.9])
    cnm_init.bl = np.zeros(nr)
    cnm_init.c1 = np.zeros(nr)
    cnm_init.neurons_sn = np.std(YrA,axis=-1)
    cnm_init.lam = np.zeros(nr)
    cnm_init.dims = Y.shape[:-1]
    cnm_init.initbatch = init_batch
    cnm_init.gnb = gnb
    
    return cnm_init
       

# definitions for demixed time series extraction and denoising/deconvolving
@profile
def HALS4activity(Yr, A, C, AtA, iters=5, tol=1e-3, groups=None):
    """Solve C = argmin_C ||Yr-AC|| using block-coordinate decent"""

    AtY = A.T.dot(Yr)
    num_iters = 0
    C_old = np.zeros(C.shape, dtype=np.float32)
    norm = lambda c: sqrt(c.ravel().dot(c.ravel()))  # faster than np.linalg.norm
    while (norm(C_old - C) >= tol * norm(C_old)) and (num_iters < iters):
        C_old[:] = C
        if groups is None:
            for m in range(len(AtY)):
                C[m] = max(C[m] + (AtY[m] - AtA[m].dot(C)) / AtA[m, m], 0)
        else:
            for m in groups:
                C[m] = np.maximum(C[m] + (AtY[m] - AtA[m].dot(C)) / AtA.diagonal()[m], 0)
        num_iters += 1
    return C


@profile
def demix_and_deconvolve(C, noisyC, AtY, AtA, OASISinstances, iters=3, n_refit=0):
    """
    Solve C = argmin_C ||Y-AC|| subject to C following AR(p) dynamics
    using OASIS within block-coordinate decent
    Newly fits the last elements in buffers C and AtY and possibly refits earlier elements.

    Parameters
    ----------
    C : ndarray of float
        Buffer containing the denoised fluorescence intensities.
        All elements up to and excluding the last one have been denoised in earlier calls.
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
                noisyC[m, -1] = C[m, -1] + (AtY[m, -1] - AtA[m].dot(C[:, -1])) / AtA[m, m]
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
                        C[m] = OASISinstances[n].fit_next_tmp(noisyC[m, -1], len_buffer)                
                else:
                    C[m, -1] = np.maximum(noisyC[m, -1],0)  # no need to enforce max(c, 0) for background, is it?
    else:
        overlap = np.sum(AtA[nb:, nb:] > .1, 0) > 1  # !threshold .1 assumes normalized A (|A|_2=1)

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
                        C[m] = OASISinstances[n].fit_next_tmp(noisyC[m, -1], len_buffer)
                else:
                    C[m] = noisyC[m]  # no need to enforce max(c, 0) for background, is it?
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
    A_smooth = np.transpose([gaussian_filter(a.reshape(
        dims, order='F'), 2).ravel(order='F') for a in Ab.T])
    A_smooth[A_smooth < 1e-2] = 0
    # set explicity zeros of Ab to small value, s.t. ind_A and Ab.indptr match
    Ab += 1e-6 * A_smooth
    Ab = scipy.sparse.csc_matrix(Ab)
    ind_A = [Ab.indices[Ab.indptr[m]:Ab.indptr[m + 1]] for m in range(nb, nb + K)]
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
def update_shapes(CY, CC, Ab, ind_A, indicator_components=None, Ab_dense=None, update_bkgrd=True, iters=3):
    D, M = Ab.shape
    N = len(ind_A)
    nb = M - N
    for _ in range(iters):  # it's presumably better to run just 1 iter but update more neurons
        if indicator_components is None:
            idx_comp = range(nb, M)
        else:
            idx_comp = np.where(indicator_components)[0] + nb

        if Ab_dense is None:
            for m in idx_comp:  # neurons
                ind_pixels = ind_A[m - nb]
                Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] = np.maximum(
                    Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] +
                    ((CY[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels]) / CC[m, m]), 0)
                # normalize
                Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] /= \
                    max(1, sqrt(Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]]
                                .dot(Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]])))
                ind_A[m-nb] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]
                # N.B. Ab[ind_pixels].dot(CC[m]) is slower for csc matrix due to indexing rows
        else:
            for m in idx_comp:  # neurons
                ind_pixels = ind_A[m - nb]
                tmp = np.maximum(Ab_dense[ind_pixels, m] + ((CY[m, ind_pixels] -
                                                             Ab_dense[ind_pixels].dot(CC[m])) /
                                                            CC[m, m]), 0)
                # normalize
                #if tmp.dot(tmp) > 0:
                tmp *= 1e-3/min(1e-3,sqrt(tmp.dot(tmp))+np.finfo(float).eps)
                Ab_dense[ind_pixels, m] = tmp / max(1, sqrt(tmp.dot(tmp)))                
                Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] = Ab_dense[ind_pixels, m]
                ind_A[m-nb] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]
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
    # cin_old = -1
    for _ in range(15):
        cin_res = ain.T.dot(Ypx)  # / ain.dot(ain)
        cin = np.maximum(cin_res, 0)
        ain = np.maximum(Ypx.dot(cin.T), 0)
        ain /= sqrt(ain.dot(ain)+ np.finfo(float).eps)
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
@profile
def update_num_components(t, sv, Ab, Cf, Yres_buf, Y_buf, rho_buf,
                          dims, gSig, gSiz, ind_A, CY, CC, groups, oases, gnb=1,
                          rval_thr=0.875, bSiz=3, robust_std=False,
                          N_samples_exceptionality=5, remove_baseline=True,
                          thresh_fitness_delta=-20, thresh_fitness_raw=-20, thresh_overlap=0.5,
                          batch_update_suff_stat=False, sn=None, g=None, lam=0, thresh_s_min=None,
                          s_min=None, Ab_dense=None, max_num_added=1):

    gHalf = np.array(gSiz) // 2

    M = np.shape(Ab)[-1]
    N = M - gnb

#    Yres = np.array(Yres_buf).T
#    Y = np.array(Y_buf).T
#    rhos = np.array(rho_buf).T

    first = True

    sv -= rho_buf.get_first()
    sv += rho_buf.get_last_frames(1).squeeze()

    num_added = 0
    while num_added < max_num_added:

        if first:
            sv_ = sv.copy()  # np.sum(rho_buf,0)
            first = False

        ind = np.argmax(sv_)
        ij = np.unravel_index(ind, dims)
        # ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, dims[c])]
        #          for c in range(len(ij))]
        # better than above expensive call of numpy and loop creation
        ijSig = [[max(ij[0] - gHalf[0], 0), min(ij[0] + gHalf[0] + 1, dims[0])],
                 [max(ij[1] - gHalf[1], 0), min(ij[1] + gHalf[1] + 1, dims[1])]]

        # xySig = np.meshgrid(*[np.arange(s[0], s[1]) for s in ijSig], indexing='xy')
        # arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze()
        #                 for s in xySig], dtype=np.int)
        # indeces = np.ravel_multi_index(arr, dims, order='F')
        indeces = np.ravel_multi_index(np.ix_(np.arange(ijSig[0][0], ijSig[0][1]),
                                              np.arange(ijSig[1][0], ijSig[1][1])),
                                       dims, order='F').ravel()

        Ypx = Yres_buf.T[indeces, :]

        ain = np.maximum(np.mean(Ypx, 1), 0)
        na = ain.dot(ain)
        if not na:
            break

        ain /= sqrt(na)

#        new_res = sv_.copy()
#        new_res[ np.ravel_multi_index(arr, dims, order='C')] = 10000
#        cv2.imshow('untitled', 0.1*cv2.resize(new_res.reshape(dims,order = 'C'),(512,512))/2000)
#        cv2.waitKey(1)

#        for iter_ in range(15):
#            cin_res = ain.T.dot(Ypx) / ain.dot(ain)
#            cin = np.maximum(cin_res, 0)
#            ain = np.maximum(Ypx.dot(cin.T) / cin.dot(cin), 0)

        ain, cin, cin_res = rank1nmf(Ypx, ain)  # expects and returns normalized ain

        rval = corr(ain.copy(), np.mean(Ypx, -1))
#        print(rval)
        if rval > rval_thr:
            # na = sqrt(ain.dot(ain))
            # ain /= na
            # cin = na * cin
            # use sparse Ain only later iff it is actually added to Ab
            Ain = np.zeros((np.prod(dims), 1), dtype=np.float32)
            # Ain = scipy.sparse.csc_matrix((np.prod(dims), 1), dtype=np.float32)
            Ain[indeces, :] = ain[:, None]

            cin_circ = cin.get_ordered()

    #        indeces_good = (Ain[indeces]>0.01).nonzero()[0]

            # rval = np.corrcoef(ain, np.mean(Ypx, -1))[0, 1]

# rval =
# np.corrcoef(Ain[indeces_good].toarray().squeeze(),np.mean(Yres[indeces_good,:],-1))[0,1]

        # if rval > rval_thr:
            #            pl.cla()
            #            _ = cm.utils.visualization.plot_contours(Ain, sv.reshape(dims), thr=0.95)
            #            pl.pause(0.01)

            useOASIS = False  # whether to use faster OASIS for cell detection
            if Ab_dense is None:
                ff = np.where((Ab.T.dot(Ain).T > thresh_overlap)[:, gnb:])[1] + gnb
            else:
                ff = np.where(Ab_dense[indeces, gnb:].T.dot(ain).T > thresh_overlap)[0] + gnb
            if ff.size > 0:
                cc = [corr(cin_circ.copy(), cins) for cins in Cf[ff, :]]

                if np.any(np.array(cc) > .8):
                    #                    repeat = False
                    # vb = imblur(np.reshape(Ain, dims, order='F'),
                    #             sig=gSig, siz=gSiz, nDimBlur=2)
                    # restrict blurring to region where component is located
                    vb = np.reshape(Ain, dims, order='F')
                    slices = tuple(slice(max(0, ijs[0] - 2 * sg), min(d, ijs[1] + 2 * sg))
                                   for ijs, sg, d in zip(ijSig, gSig, dims))  # is 2 enough?
                    vb[slices] = imblur(vb[slices], sig=gSig, siz=gSiz, nDimBlur=2)
                    sv_ -= (vb.ravel()**2) * cin.dot(cin)

#                    pl.imshow(np.reshape(sv,dims));pl.pause(0.001)
                    # print('Overlap at step' + str(t) + ' ' + str(cc))
                    break

            if s_min is None:  # use thresh_s_min * noise estimate * sqrt(1-sum(gamma))
            # the formula has been obtained by running OASIS with s_min=0 and lambda=0 on Gaussin noise.
            # e.g. 1 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the root mean square (non-zero) spike size, sqrt(<s^2>)
            #      2 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the 95% percentile of (non-zero) spike sizes
            #      3 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the 99.7% percentile of (non-zero) spike sizes
                s_min = 0 if thresh_s_min is None else thresh_s_min * \
                    sqrt((ain**2).dot(sn[indeces]**2)) * sqrt(1 - np.sum(g))

            cin_res = cin_res.get_ordered()
            if useOASIS:
                oas = oasis.OASIS(g=g, s_min=s_min,
                                  num_empty_samples=t + 1 - len(cin_res))
                for yt in cin_res:
                    oas.fit_next(yt)
                foo = oas.get_l_of_last_pool() <= t
                # cc=oas.c
                # print([np.corrcoef(cin_circ,cins)[0,1] for cins in Cf[overlap[0] > 0]])
                # print([np.corrcoef(cc,cins)[0,1] for cins in Cf[overlap[0] > 0, ]])
                # import matplotlib.pyplot as plt
                # plt.plot(cin_res); plt.plot(cc); plt.show()
                # import pdb;pdb.set_trace()
            else:
                fitness_delta, erfc_delta, std_rr, _ = compute_event_exceptionality(
                    np.diff(cin_res)[None, :], robust_std=robust_std, N=N_samples_exceptionality)
                if remove_baseline:
                    num_samps_bl = min(len(cin_res) // 5, 800)
                    bl = scipy.ndimage.percentile_filter(cin_res, 8, size=num_samps_bl)
                else:
                    bl = 0
                fitness_raw, erfc_raw, std_rr, _ = compute_event_exceptionality(
                    (cin_res - bl)[None, :], robust_std=robust_std, N=N_samples_exceptionality)
                foo = (fitness_delta < thresh_fitness_delta) or (fitness_raw < thresh_fitness_raw)

            if foo:
                # print('adding component' + str(N + 1) + ' at timestep ' + str(t))
                num_added += 1
#                ind_a = uniform_filter(np.reshape(Ain.toarray(), dims, order='F'), size=bSiz)
#                ind_a = np.reshape(ind_a > 1e-10, (np.prod(dims),), order='F')
#                indeces_good = np.where(ind_a)[0]#np.where(determine_search_location(Ain,dims))[0]
                if not useOASIS:
                    # TODO: decide on a line to use for setting the parameters
                    # # lambda from init batch (e.g. mean of lambdas)  or  s_min if either s_min or s_min_thresh are given
                    # oas = oasis.OASIS(g=np.ravel(g)[0], lam if s_min==0 else 0, s_min, num_empty_samples=t + 1 - len(cin_res),
                    #                   g2=0 if np.size(g) == 1 else g[1])
                    # or
                    # lambda from Selesnick's 3*sigma*|K| rule
                    # use noise estimate from init batch or use std_rr?
#                    sn_ = sqrt((ain**2).dot(sn[indeces]**2)) / sqrt(1 - g**2)
                    sn_ = std_rr
                    oas = oasis.OASIS(np.ravel(g)[0], 3 * sn_ /
                                      (sqrt(1 - g**2) if np.size(g) == 1 else 
                                        sqrt((1 + g[1]) * ((1 - g[1])**2 - g[0]**2) / (1-g[1])))
                                      if s_min == 0 else 0,
                                      s_min, num_empty_samples=t + 1 - len(cin_res),
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
                csc_append(Ab, Ain_csc)  # faster version of scipy.sparse.hstack
                ind_A.append(Ab.indices[Ab.indptr[M]:Ab.indptr[M + 1]])

#                ccf = Cf[:,-minibatch_suff_stat:]
#                CY = ((t*1.-1)/t) * CY + (1./t) * np.dot(ccf, Yr[:, t-minibatch_suff_stat:t].T)/minibatch_suff_stat
#                CC = ((t*1.-1)/t) * CC + (1./t) * ccf.dot(ccf.T)/minibatch_suff_stat

                tt = t * 1.
#                if batch_update_suff_stat and Y_buf.cur<len(Y_buf)-1:
#                   Y_buf_ = Y_buf[Y_buf.cur+1:,:]
#                   cin_ = cin[Y_buf.cur+1:]
#                   n_fr_ = len(cin_)
#                   cin_circ_= cin_circ[-n_fr_:]
#                   Cf_ = Cf[:,-n_fr_:]
#                else:
                Y_buf_ = Y_buf
                cin_ = cin
                Cf_ = Cf
                cin_circ_ = cin_circ

#                CY[M, :] = Y_buf_.T.dot(cin_)[None, :] / tt
                # much faster: exploit that we only access CY[m, ind_pixels],
                # hence update only these
                CY[M, indeces] = cin_.dot(Y_buf_[:, indeces]) / tt
#                CY = np.vstack([CY[:N,:], Y_buf.T.dot(cin / tt)[None,:], CY[ N:,:]])
#                YC = CY.T
#                YC = np.hstack([YC[:, :N], Y_buf.T.dot(cin / tt)[:, None], YC[:, N:]])
#                CY = YC.T

                # preallocate memory for speed up?
                CC1 = np.hstack([CC, Cf_.dot(cin_circ_ / tt)[:, None]])
                CC2 = np.hstack([(Cf_.dot(cin_circ_)).T, cin_circ_.dot(cin_circ_)]) / tt
                CC = np.vstack([CC1, CC2])
                Cf = np.vstack([Cf, cin_circ])

                N = N + 1
                M = M + 1

                Yres_buf[:, indeces] -= np.outer(cin, ain)
                # vb = imblur(np.reshape(Ain, dims, order='F'), sig=gSig,
                #             siz=gSiz, nDimBlur=2).ravel()
                # restrict blurring to region where component is located
                vb = np.reshape(Ain, dims, order='F')
                slices = tuple(slice(max(0, ijs[0] - 2 * sg), min(d, ijs[1] + 2 * sg))
                               for ijs, sg, d in zip(ijSig, gSig, dims))  # is 2 enough?
                vb[slices] = imblur(vb[slices], sig=gSig, siz=gSiz, nDimBlur=2)
                vb = vb.ravel()

                # ind_vb = np.where(vb)[0]
                ind_vb = np.ravel_multi_index(np.ix_(*[np.arange(s.start, s.stop)
                                                       for s in slices]), dims).ravel()

                updt_res = (vb[None, ind_vb].T**2).dot(cin[None, :]**2).T
                rho_buf[:, ind_vb] -= updt_res
                updt_res_sum = np.sum(updt_res, 0)
                sv[ind_vb] -= updt_res_sum
                sv_[ind_vb] -= updt_res_sum

            else:

                num_added = max_num_added

        else:

            num_added = max_num_added

    return Ab, Cf, Yres_buf, rho_buf, CC, CY, ind_A, sv, groups


#%%
def initialize_movie_online(Y, K, gSig, rf, stride, base_name, 
                     p = 1, merge_thresh = 0.95, rval_thr_online = 0.9, thresh_fitness_delta_online = -30, thresh_fitness_raw_online = -50, 
                     rval_thr_init = .5, thresh_fitness_delta_init = -20, thresh_fitness_raw_init = -20,
                     rval_thr_refine = 0.95, thresh_fitness_delta_refine = -100, thresh_fitness_raw_refine = -100,
                     final_frate = 10, Npeaks = 10, single_thread = True, dview = None, n_processes = None):      
    
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
    fname_new = Y.save(base_name, order= 'C')
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
                        update_num_comps=True, rval_thr = rval_thr_online, thresh_fitness_delta=thresh_fitness_delta_online, thresh_fitness_raw=thresh_fitness_raw_online,
                        batch_update_suff_stat = True,max_comp_update_shape = 5)
    
    
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
    idx_components_delta = np.where(fitness_delta < thresh_fitness_delta_init)[0]
    
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
                        update_num_comps=True, rval_thr = rval_thr_refine, thresh_fitness_delta=thresh_fitness_delta_refine, thresh_fitness_raw=thresh_fitness_raw_refine,
                        batch_update_suff_stat = True,max_comp_update_shape = 5)
    
    
    
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
    idx_components_delta = np.where(fitness_delta < thresh_fitness_delta_refine)[0]

    
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
