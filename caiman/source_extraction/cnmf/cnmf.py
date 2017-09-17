# -*- coding: utf-8 -*-
"""
Constrained Nonnegative Matrix Factorization

Created on Fri Aug 26 15:44:32 2016

@author: agiovann


"""
from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import object
import numpy as np
from .utilities import CNMFSetParms, update_order
from .pre_processing import preprocess_data
from .initialization import initialize_components
from .merging import merge_components
from .spatial import update_spatial_components
from .temporal import update_temporal_components
from .map_reduce import run_CNMF_patches
from .initialization import imblur
import cv2
from .online_cnmf import RingBuffer, HALS4activity, demix_and_deconvolve
from .online_cnmf import init_shapes_and_sufficient_stats, update_shapes, update_num_components
import scipy
import oasis
import pylab as pl
from time import time

try:
    profile
except:
    profile = lambda a: a


class CNMF(object):
    """
    Source extraction using constrained non-negative matrix factorization.
    """

    def __init__(self, n_processes, k=5, gSig=[4, 4], merge_thresh=0.8, p=2, dview=None, Ain=None, Cin=None, f_in=None, do_merge=True,
                 ssub=2, tsub=2, p_ssub=1, p_tsub=1, method_init='greedy_roi', alpha_snmf=None,
                 rf=None, stride=None, memory_fact=1, gnb=1, only_init_patch=False,
                 method_deconvolution='oasis', n_pixels_per_process=4000, block_size=20000, check_nan=True,
                 skip_refinement=False, normalize_init=True, options_local_NMF=None,
                 remove_very_bad_comps = False, border_pix = 0, low_rank_background = True, update_background_components = True,
                 rolling_sum = True, rolling_length = 100,
                 minibatch_shape=100, minibatch_suff_stat=3,
                 update_num_comps=True, rval_thr=0.9, thresh_fitness_delta=-20,
                 thresh_fitness_raw=-40, thresh_overlap=.5,
                 max_comp_update_shape=np.inf, num_times_comp_updated=np.inf,
                 batch_update_suff_stat=False, thresh_s_min=None, s_min=None):
        """

        Constructor of the CNMF method

        Parameters:
        -----------


        n_processes: int
           number of processed used (if in parallel this controls memory usage)

        k: int
           number of neurons expected per FOV (or per patch if patches_pars is  None)

        gSig: tuple
            expected half size of neurons

        merge_thresh: float
            merging threshold, max correlation allowed

        p: int
            order of the autoregressive process used to estimate deconvolution

        dview: Direct View object
            for parallelization pruposes when using ipyparallel

        Ain: ndarray
            if know, it is the initial estimate of spatial filters

        ssub: int
            downsampleing factor in space

        tsub: int
             downsampling factor in time

        p_ssub: int
            downsampling factor in space for patches

        p_tsub: int
             downsampling factor in time for patches

        method_init: str
           can be greedy_roi or sparse_nmf

        alpha_snmf: float
            weight of the sparsity regularization

        rf: int
            half-size of the patches in pixels. rf=25, patches are 50x50

        stride: int
            amount of overlap between the patches in pixels

        memory_fact: float
            unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system

        gnb: int
            number of global background components

        N_samples_fitness: int
            number of samples over which exceptional events are computed (See utilities.evaluate_components)

        expected_comps: int
            number of expected components (try to exceed the expected)

        num_times_comp_updated:
            number of times each component is updated. In inf components are updated at every initbatch time steps

        max_comp_update_shape:
            threshold number of components after which selective updating starts (using the parameter num_times_comp_updated)

            

        Returns:
        --------
        self
        """

        self.k = k  # number of neurons expected per FOV (or per patch if patches_pars is not None
        self.gSig = gSig  # expected half size of neurons
        self.merge_thresh = merge_thresh  # merging threshold, max correlation allowed
        self.p = p  # order of the autoregressive system
        self.dview = dview
        self.Ain = Ain
        self.Cin = Cin
        self.f_in = f_in

        self.ssub = ssub
        self.tsub = tsub
        self.p_ssub = p_ssub
        self.p_tsub = p_tsub
        self.method_init = method_init
        self.n_processes = n_processes
        self.rf = rf  # half-size of the patches in pixels. rf=25, patches are 50x50
        self.stride = stride  # amount of overlap between the patches in pixels
        self.memory_fact = memory_fact  # unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
        self.gnb = gnb
        self.do_merge = do_merge
        self.alpha_snmf = alpha_snmf
        self.only_init = only_init_patch
        self.method_deconvolution = method_deconvolution
        self.n_pixels_per_process = n_pixels_per_process
        self.block_size = block_size
        self.check_nan = check_nan
        self.skip_refinement = skip_refinement
        self.normalize_init = normalize_init
        self.options_local_NMF = options_local_NMF
        self.A = None
        self.C = None
        self.S = None
        self.b = None
        self.f = None
        self.sn = None
        self.g = None


#        self.num_total_frames = num_total_frames
        self.minibatch_shape = minibatch_shape
        self.minibatch_suff_stat = minibatch_suff_stat
        self.update_num_comps = update_num_comps
        self.rval_thr = rval_thr
        self.thresh_s_min = thresh_s_min
        self.s_min = s_min
        self.thresh_fitness_delta = thresh_fitness_delta
        self.thresh_fitness_raw = thresh_fitness_raw
        self.thresh_overlap = thresh_overlap
        self.num_times_comp_updated = num_times_comp_updated
        self.max_comp_update_shape = max_comp_update_shape
        self.batch_update_suff_stat = batch_update_suff_stat

    def fit(self, images):
        """
        This method uses the cnmf algorithm to find sources in data.

        Parameters
        ----------
        images : mapped np.ndarray of shape (t,x,y[,z]) containing the images that vary over time.

        Returns
        --------
        self

        """
        T = images.shape[0]
        self.initbatch = T
        dims = images.shape[1:]
        Yr = images.reshape([T, np.prod(dims)], order='F').T
        Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])

        print((T,) + dims)
        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        try:
            Y.filename = images.filename
            Yr.filename = images.filename
        except AttributeError:  # no filename if data is not mem-mapped but in memory
            pass

        options = CNMFSetParms(Y, self.n_processes, p=self.p, gSig=self.gSig, K=self.k,
                               ssub=self.ssub, tsub=self.tsub, p_ssub=self.p_ssub,
                               p_tsub=self.p_tsub, method_init=self.method_init,
                               n_pixels_per_process=self.n_pixels_per_process,
                               block_size=self.block_size, check_nan=self.check_nan,
                               nb=self.gnb, normalize_init=self.normalize_init,
                               options_local_NMF=self.options_local_NMF)

        self.options = options

        if self.rf is None:  # no patches
            print('preprocessing ...')

            Yr, sn, g, psx = preprocess_data(Yr, dview=self.dview, **options['preprocess_params'])

            if self.Ain is None:
                print('initializing ...')
                if self.alpha_snmf is not None:
                    options['init_params']['alpha_snmf'] = self.alpha_snmf

                self.Ain, self.Cin, self.b_in, self.f_in, center = initialize_components(
                    Y, **options['init_params'])

            if self.only_init:  # only return values after initialization

                nA = np.squeeze(np.array(np.sum(np.square(self.Ain), axis=0)))

                nr = nA.size
                Cin = scipy.sparse.coo_matrix(self.Cin)

                YA = (self.Ain.T.dot(Yr).T) * scipy.sparse.spdiags(1. / nA, 0, nr, nr)
                AA = ((self.Ain.T.dot(self.Ain)) * scipy.sparse.spdiags(1. / nA, 0, nr, nr))

                self.YrA = YA - Cin.T.dot(AA)
                self.C = Cin.todense()

                self.bl = None
                self.c1 = None
                self.neurons_sn = None
                self.g = g
                self.A = self.Ain
                self.b = self.b_in
                self.f = self.f_in
                self.sn = sn

                return self

            print('update spatial ...')
            A, b, Cin, self.f_in = update_spatial_components(
                Yr, self.Cin, self.f_in, self.Ain, sn=sn, dview=self.dview,
                **options['spatial_params'])

            print('update temporal ...')
            if not self.skip_refinement:
                # set this to zero for fast updating without deconvolution
                options['temporal_params']['p'] = 0
            else:
                options['temporal_params']['p'] = self.p

            options['temporal_params']['method'] = self.method_deconvolution

            C, A, b, f, S, bl, c1, neurons_sn, g, YrA, lam = update_temporal_components(
                Yr, A, b, Cin, self.f_in, dview=self.dview, **options['temporal_params'])

            if not self.skip_refinement:

                if self.do_merge:
                    print('merge components ...')
                    A, C, nr, merged_ROIs, S, bl, c1, sn1, g1 = merge_components(
                        Yr, A, b, C, f, S, sn, options['temporal_params'],
                        options['spatial_params'], dview=self.dview, bl=bl, c1=c1, sn=neurons_sn,
                        g=g, thr=self.merge_thresh, mx=50, fast_merge=True)

                print((A.shape))

                print('update spatial ...')

                A, b, C, f = update_spatial_components(
                    Yr, C, f, A, sn=sn, dview=self.dview, **options['spatial_params'])

                # set it back to original value to perform full deconvolution
                options['temporal_params']['p'] = self.p
                print('update temporal ...')
                C, A, b, f, S, bl, c1, neurons_sn, g1, YrA, lam = update_temporal_components(
                    Yr, A, b, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None,
                    **options['temporal_params'])

            else:

                C, f, S, bl, c1, neurons_sn, g1, YrA = C, f, S, bl, c1, neurons_sn, g, YrA

        else:  # use patches

            if self.stride is None:
                self.stride = np.int(self.rf * 2 * .1)
                print(('**** Setting the stride to 10% of 2*rf automatically:' + str(self.stride)))

            if type(images) is np.ndarray:
                raise Exception(
                    'You need to provide a memory mapped file as input if you use patches!!')

            if self.only_init:
                options['patch_params']['only_init'] = True

            if self.alpha_snmf is not None:
                options['init_params']['alpha_snmf'] = self.alpha_snmf

            A, C, YrA, b, f, sn, optional_outputs = run_CNMF_patches(
                images.filename, dims + (T,), options, rf=self.rf, stride=self.stride,
                dview=self.dview, memory_fact=self.memory_fact, gnb=self.gnb)

            options = CNMFSetParms(Y, self.n_processes, p=self.p, gSig=self.gSig, K=A.shape[-1],
                                   thr=self.merge_thresh, n_pixels_per_process=self.n_pixels_per_process,
                                   block_size=self.block_size, check_nan=self.check_nan)

            options['temporal_params']['method'] = self.method_deconvolution

            print("merging")
            merged_ROIs = [0]
            while len(merged_ROIs) > 0:
                A, C, nr, merged_ROIs, S, bl, c1, sn_n, g = merge_components(
                    Yr, A, [], np.array(C), [], np.array(C), [], options['temporal_params'],
                    options['spatial_params'], dview=self.dview, thr=self.merge_thresh, mx=np.Inf)

            print("update temporal")
            C, A, b, f, S, bl, c1, neurons_sn, g1, YrA, lam = update_temporal_components(
                Yr, A, b, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None,
                **options['temporal_params'])

        self.A = A
        self.C = C
        self.b = b
        self.f = f
        self.S = S
        self.YrA = YrA
        self.sn = sn
        self.g = g1
        self.bl = bl
        self.c1 = c1
        self.neurons_sn = neurons_sn
        self.lam = lam
        self.dims = dims

        return self

    def _prepare_object(self, Yr, T, expected_comps, new_dims=None, idx_components=None,
                        g=None, lam=None, s_min=None, bl=None, use_dense=True):

        self.expected_comps = expected_comps

        if idx_components is None:
            idx_components = range(self.A.shape[-1])

        self.A2 = self.A.tocsc()[:, idx_components]
        self.C2 = self.C[idx_components]
        self.b2 = self.b
        self.f2 = self.f
        self.S2 = self.S[idx_components]
        self.YrA2 = self.YrA[idx_components]
        self.g2 = self.g[idx_components]
        self.bl2 = self.bl[idx_components]
        self.c12 = self.c1[idx_components]
        self.neurons_sn2 = self.neurons_sn[idx_components]
        self.lam2 = self.lam[idx_components]
        self.dims2 = self.dims

        self.N = self.A2.shape[-1]
        self.M = self.gnb + self.N

        if Yr.shape[-1] != self.initbatch:
            raise Exception(
                'The movie size used for initialization does not match with the minibatch size')

        if new_dims is not None:

            new_Yr = np.zeros([np.prod(new_dims), T])
            for ffrr in range(T):
                tmp = cv2.resize(Yr[:, ffrr].reshape(self.dims2, order='F'), new_dims[::-1])
                print(tmp.shape)
                new_Yr[:, ffrr] = tmp.reshape([np.prod(new_dims)], order='F')
            Yr = new_Yr
            A_new = scipy.sparse.csc_matrix(
                (np.prod(new_dims), self.A2.shape[-1]), dtype=np.float32)
            for neur in range(N):
                a = self.A2.tocsc()[:, neur].toarray()
                a = a.reshape(self.dims2, order='F')
                a = cv2.resize(a, new_dims[::-1]).reshape([-1, 1], order='F')

                A_new[:, neur] = scipy.sparse.csc_matrix(a)

            self.A2 = A_new
            self.b2 = self.b2.reshape(self.dims2, order='F')
            self.b2 = cv2.resize(self.b2, new_dims[::-1]).reshape([-1, 1], order='F')

            self.dims2 = new_dims

        nA = np.ravel(np.sqrt(self.A2.power(2).sum(0)))
        self.A2 /= nA
        self.C2 *= nA[:, None]
        self.YrA2 *= nA[:, None]
        self.S2 *= nA[:, None]
        self.neurons_sn2 *= nA
        self.lam2 *= nA
        z = np.sqrt([b.T.dot(b) for b in self.b2.T])
        self.f2 *= z[:, None]
        self.b2 /= z

        self.noisyC = np.zeros((self.gnb + expected_comps, T), dtype=np.float32)
        self.C_on = np.zeros((expected_comps, T), dtype=np.float32)

#        self.noisyC[:, :self.initbatch] = np.vstack(
#            [self.C[:, :self.initbatch] + self.YrA, self.f])
#        Ab_ = scipy.sparse.csc_matrix(np.c_[self.A2, self.b2])
#        AtA_ = Ab_.T.dot(Ab_)
#        self.noisyC[:,:self.initbatch] = hals_full(Yr[:, :self.initbatch], Ab_, np.r_[self.C,self.f], iters=3)
#
#        for t in xrange(self.initbatch):
#            if t % 100 == 0:
#                print(t)
#            self.noisyC[:, t] = HALS4activity(Yr[:, t], Ab_, np.ones(N + 1) if t == 0 else
# self.noisyC[:, t - 1].copy(), AtA_, iters=30 if t == 0 else 5)
        self.noisyC[self.gnb:self.M, :self.initbatch] = self.C2 + self.YrA2
        self.noisyC[:self.gnb, :self.initbatch] = self.f2

         # next line requires some estimate of the spike size, e.g. running OASIS with penalty=0
         # or s_min from histogram a la Deneux et al (2016)
#        self.OASISinstances = [oasis.OASIS(
#            g=g if g is not None else (gam[0] if self.p == 1 else gam),
#            s_min=self.thresh_s_min * sn if s_min is None else s_min,
#            b=b if bl is None else bl)
#            for gam, sn, b in zip(self.g2, self.neurons_sn2, self.bl2)]
#         # using L1 instead of min spikesize with lambda obtained from fit on init batch
        self.OASISinstances = [oasis.OASIS(
            g=g if g is not None else (gam[0] if self.p == 1 else gam),
            lam=l if lam is None else lam,
            s_min=0 if s_min is None else s_min,
            b=b if bl is None else bl)
            for gam, l, b in zip(self.g2, self.lam2, self.bl2)]

        for i, o in enumerate(self.OASISinstances):
            o.fit(self.noisyC[i + self.gnb, :self.initbatch])
            self.C_on[i, :self.initbatch] = o.c

        self.Ab, self.ind_A, self.CY, self.CC = init_shapes_and_sufficient_stats(
            Yr[:, :self.initbatch].reshape(self.dims2 + (-1,), order='F'), self.A2,
            self.C_on[:self.N, :self.initbatch], self.b2, self.noisyC[:self.gnb, :self.initbatch])

        self.CY, self.CC = self.CY * 1. / self.initbatch, 1 * self.CC / self.initbatch

        self.A2 = scipy.sparse.csc_matrix(self.A2.astype(np.float32), dtype=np.float32)
        self.C2 = self.C2.astype(np.float32)
        self.f2 = self.f2.astype(np.float32)
        self.b2 = self.b2.astype(np.float32)
        self.Ab = scipy.sparse.csc_matrix(self.Ab.astype(np.float32), dtype=np.float32)
        self.noisyC = self.noisyC.astype(np.float32)
        self.CY = self.CY.astype(np.float32)
        self.CC = self.CC.astype(np.float32)
        print('Expecting ' + str(self.expected_comps) + ' components')
        self.CY.resize([self.expected_comps + 1, self.CY.shape[-1]])
        if use_dense:
            self.Ab_dense = np.zeros((self.CY.shape[-1], self.expected_comps + 1),
                                     dtype=np.float32)
            self.Ab_dense[:, :self.Ab.shape[1]] = self.Ab.toarray()
        self.C_on = np.vstack([self.noisyC[:self.gnb, :], self.C_on.astype(np.float32)])

        self.gSiz = np.add(np.multiply(self.gSig, 2), 1)

        self.Yr_buf = RingBuffer(Yr[:, self.initbatch - self.minibatch_shape:
                                    self.initbatch].T.copy(), self.minibatch_shape)
        self.Yres_buf = RingBuffer(self.Yr_buf.get_ordered() - self.Ab.dot(
            self.C_on[:self.M, self.initbatch - self.minibatch_shape:self.initbatch]).T, self.minibatch_shape)
        self.rho_buf = imblur(self.Yres_buf.get_ordered().T.reshape(
            self.dims2 + (-1,), order='F'), sig=self.gSig, siz=self.gSiz, nDimBlur=2)**2
        self.rho_buf = np.reshape(self.rho_buf, (self.dims2[0] * self.dims2[1], -1)).T
        self.rho_buf = RingBuffer(self.rho_buf, self.minibatch_shape)
        self.AtA = (self.Ab.T.dot(self.Ab)).toarray()
        self.AtY_buf = self.Ab.T.dot(self.Yr_buf.T)
        self.sv = np.sum(self.rho_buf.get_last_frames(self.initbatch), 0)
        self.groups = map(list, update_order(self.Ab)[0])
        # self.update_counter = np.zeros(self.N)
        self.update_counter = .5**(-np.linspace(0, 1, self.N, dtype=np.float32))
        self.time_neuron_added = []
        for nneeuu in range(self.N):
            self.time_neuron_added.append((nneeuu, self.initbatch))
        self.time_spend = 0
        return self

    @profile
    def fit_next(self, t, frame_in, num_iters_hals=3, use_dense=True,
                 simultaneously=False, n_refit=0):
        """
        This method fits the next frame using the online cnmf algorithm and updates the object.

        Parameters
        ----------
        t : int
            time measured in number of frames

        frame_in : array
            flattened array of shape (x*y[*z],) containing the t-th image.

        num_iters_hals: int, optional
            maximal number of iterations for HALS (NNLS via blockCD)

        simultaneously : bool, optional
            If true, demix and denoise/deconvolve simultaneously. Slower but can be more accurate.

        n_refit : int, optional
            Number of pools (cf. oasis.pyx) prior to the last one that are refitted when
            simultaneously demixing and denoising/deconvolving.
        """

        t_start = time()

        # locally scoped variables for brevity of code and faster look up
        nb_ = self.gnb
        Ab_ = self.Ab
        mbs = self.minibatch_shape
        frame = frame_in.astype(np.float32)
#        print(np.max(1/scipy.sparse.linalg.norm(self.Ab,axis = 0)))
        self.Yr_buf.append(frame)

        if not simultaneously:
            # get noisy fluor value via NNLS (project data on shapes & demix)
            C_in = self.noisyC[:self.M, t - 1].copy()
            self.noisyC[:self.M, t] = HALS4activity(
                frame, self.Ab, C_in, self.AtA, iters=num_iters_hals, groups=self.groups)
            # denoise & deconvolve
            for i, o in enumerate(self.OASISinstances):
                o.fit_next(self.noisyC[nb_ + i, t])
                self.C_on[nb_ + i, t - o.get_l_of_last_pool() + 1: t + 1] = o.get_c_of_last_pool()
            self.C_on[:nb_, t] = self.noisyC[:nb_, t]
        else:
            # update buffer, initialize C with previous value
            self.C_on[:, t] = self.C_on[:, t - 1]
            self.noisyC[:, t] = self.C_on[:, t - 1]
            self.AtY_buf = np.concatenate((self.AtY_buf[:, 1:], self.Ab.T.dot(frame)[:, None]), 1) \
                if n_refit else self.Ab.T.dot(frame)[:, None]
            # demix, denoise & deconvolve
            (self.C_on[:self.M, t + 1 - mbs:t + 1], self.noisyC[:self.M, t + 1 - mbs:t + 1],
                self.OASISinstances) = demix_and_deconvolve(
                self.C_on[:self.M, t + 1 - mbs:t + 1],
                self.noisyC[:self.M, t + 1 - mbs:t + 1],
                self.AtY_buf, self.AtA, self.OASISinstances, iters=num_iters_hals,
                n_refit=n_refit)
            for i, o in enumerate(self.OASISinstances):
                self.C_on[nb_ + i, t - o.get_l_of_last_pool() + 1: t + 1] = o.get_c_of_last_pool()


#        cv2.imshow('untitled', 3*cv2.resize(self.Ab.sum(1).reshape(self.dims,order = 'F'),(512,512)))
#        cv2.waitKey(1)
#
        if self.update_num_comps:

            res_frame = frame - self.Ab.dot(self.noisyC[:self.M, t])
#            cv2.imshow('untitled', 0.1*cv2.resize(res_frame.reshape(self.dims,order = 'F'),(512,512)))
#            cv2.waitKey(1)
#
            self.Yres_buf.append(res_frame)

            res_frame = np.reshape(res_frame, self.dims2, order='F')

            rho = imblur(res_frame, sig=self.gSig, siz=self.gSiz, nDimBlur=2)**2
            rho = np.reshape(rho, np.prod(self.dims2))
            self.rho_buf.append(rho)

            self.Ab, Cf_temp, self.Yres_buf, self.rhos_buf, self.CC, self.CY, self.ind_A, self.sv, self.groups = update_num_components(
                t, self.sv, self.Ab, self.C_on[:self.M, (t - mbs + 1):(t + 1)],
                self.Yres_buf, self.Yr_buf, self.rho_buf, self.dims2,
                self.gSig, self.gSiz, self.ind_A, self.CY, self.CC, rval_thr=self.rval_thr,
                thresh_fitness_delta=self.thresh_fitness_delta,
                thresh_fitness_raw=self.thresh_fitness_raw, thresh_overlap=self.thresh_overlap,
                groups=self.groups, batch_update_suff_stat=self.batch_update_suff_stat, gnb=self.gnb,
                sn=self.sn, g=np.mean(self.g) if self.p == 1 else np.mean(self.g, 0),
                lam=self.lam.mean(), thresh_s_min=self.thresh_s_min, s_min=self.s_min,
                Ab_dense=self.Ab_dense[:, :self.M] if use_dense else None,
                oases=self.OASISinstances)

            num_added = len(self.ind_A) - self.N

            if num_added > 0:
                self.N += num_added
                self.M += num_added
                if self.N >= self.expected_comps:
                    self.expected_comps += 200
                    self.CY.resize([self.expected_comps + nb_, self.CY.shape[-1]])
                    self.C_on.resize([self.expected_comps + nb_, self.C_on.shape[-1]])
                    self.noisyC.resize([self.expected_comps + nb_, self.C_on.shape[-1]])
                    if use_dense:  # resize won't work due to contingency issue
                        # self.Ab_dense.resize([self.CY.shape[-1], self.expected_comps+nb_])
                        self.Ab_dense = np.zeros((self.CY.shape[-1], self.expected_comps + nb_),
                                                 dtype=np.float32)
                        self.Ab_dense[:, :Ab_.shape[1]] = Ab_.toarray()
                    print('Increasing number of expected components to:' +
                          str(self.expected_comps))
                self.update_counter.resize(self.N)
                self.AtA = (Ab_.T.dot(Ab_)).toarray()

                self.noisyC[self.M - num_added:self.M, t - mbs +
                            1:t + 1] = Cf_temp[self.M - num_added:self.M]

                for _ct in range(self.M - num_added, self.M):
                    self.time_neuron_added.append((_ct - nb_, t))
                    # N.B. OASISinstances are already updated within update_num_components
                    self.C_on[_ct, t - mbs + 1:t +
                              1] = self.OASISinstances[_ct - nb_].get_c(mbs)
                    if simultaneously and n_refit:
                        self.AtY_buf = np.concatenate((
                            self.AtY_buf, [Ab_.data[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]].dot(
                                self.Yr_buf.T[Ab_.indices[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]]])]))
                    if use_dense:  # much faster than Ab_[:, self.N + nb_ - num_added:].toarray()
                        self.Ab_dense[Ab_.indices[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]],
                                      _ct] = Ab_.data[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]]

                # set the update counter to 0 for components that are overlaping the newly added
                if use_dense:
                    idx_overlap = np.concatenate([
                        self.Ab_dense[self.ind_A[_ct], nb_:self.M - num_added].T.dot(
                            self.Ab_dense[self.ind_A[_ct], _ct + nb_]).nonzero()[0]
                        for _ct in range(self.N - num_added, self.N)])
                else:
                    idx_overlap = Ab_.T.dot(
                        Ab_[:, -num_added:])[nb_:-num_added].nonzero()[0]
                self.update_counter[idx_overlap] = 0

        if (t - self.initbatch) % mbs == mbs - 1 and\
                self.batch_update_suff_stat:
            # faster update using minibatch of frames

            ccf = self.C_on[:self.M, t - mbs + 1:t + 1]
            y = self.Yr_buf  # .get_last_frames(mbs)[:]

            # much faster: exploit that we only access CY[m, ind_pixels], hence update only these
            n0 = mbs
            t0 = 0 * self.initbatch
            w1 = (t - n0 + t0) * 1. / (t + t0)  # (1 - 1./t)#mbs*1. / t
            w2 = 1. / (t + t0)  # 1.*mbs /t
            for m in range(self.N):
                self.CY[m + nb_, self.ind_A[m]] *= w1
                self.CY[m + nb_, self.ind_A[m]] += w2 * ccf[m + nb_].dot(y[:, self.ind_A[m]])

            self.CY[:nb_] = self.CY[:nb_] * w1 + w2 * ccf[:nb_].dot(y)   # background
            self.CC = self.CC * w1 + w2 * ccf.dot(ccf.T)

        if not self.batch_update_suff_stat:

            ccf = self.C_on[:self.M, t - self.minibatch_suff_stat:t - self.minibatch_suff_stat + 1]
            y = self.Yr_buf.get_last_frames(self.minibatch_suff_stat)[:1]
            # much faster: exploit that we only access CY[m, ind_pixels], hence update only these
            for m in range(self.N):
                self.CY[m + nb_, self.ind_A[m]] *= (1 - 1. / t)
                self.CY[m + nb_, self.ind_A[m]] += ccf[m + nb_].dot(y[:, self.ind_A[m]]) / t
            self.CY[:nb_] = self.CY[:nb_] * (1 - 1. / t) + ccf[:nb_].dot(y / t)
            self.CC = self.CC * (1 - 1. / t) + ccf.dot(ccf.T / t)

        # update shapes
        if True:  # False:  # bulk shape update
            if (t - self.initbatch) % mbs == mbs - 1:
                print('Updating Shapes')

                if self.N > self.max_comp_update_shape:
                    indicator_components = np.where(self.update_counter <=
                                                    self.num_times_comp_updated)[0]
                    # np.random.choice(self.N,10,False)
                    self.update_counter[indicator_components] += 1
                else:
                    indicator_components = None

                if use_dense:
                    # update dense Ab and sparse Ab simultaneously;
                    # this is faster than calling update_shapes with sparse Ab only
                    Ab_, self.ind_A, self.Ab_dense[:, :self.M] = update_shapes(
                        self.CY, self.CC, self.Ab, self.ind_A,
                        indicator_components, self.Ab_dense[:, :self.M])
                else:
                    Ab_, self.ind_A, _ = update_shapes(self.CY, self.CC, Ab_, self.ind_A,
                                                       indicator_components=indicator_components)

                self.AtA = (Ab_.T.dot(Ab_)).toarray()

            self.Ab = Ab_

        else:  # distributed shape update
            self.update_counter *= .5**(1. / mbs)
            # if not num_added:
            if (not num_added) and (time() - t_start < self.time_spend / (t - self.initbatch + 1)):
                candidates = np.where(self.update_counter <= 1)[0]
                if len(candidates):
                    indicator_components = candidates[:self.N // mbs + 1]
                    self.update_counter[indicator_components] += 1

                    if use_dense:
                        # update dense Ab and sparse Ab simultaneously;
                        # this is faster than calling update_shapes with sparse Ab only
                        Ab_, self.ind_A, self.Ab_dense[:, :self.M] = update_shapes(
                            self.CY, self.CC, self.Ab, self.ind_A,
                            indicator_components, self.Ab_dense[:, :self.M],
                            update_bkgrd=(t % mbs == 0))
                    else:
                        Ab_, self.ind_A, _ = update_shapes(
                            self.CY, self.CC, Ab_, self.ind_A,
                            indicator_components=indicator_components,
                            update_bkgrd=(t % mbs == 0))

                    self.AtA = (Ab_.T.dot(Ab_)).toarray()

                self.Ab = Ab_
            self.time_spend += time() - t_start


def scale(y):
    return (y - np.mean(y)) / (np.max(y) - np.min(y))
