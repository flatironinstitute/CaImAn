#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Constrained Nonnegative Matrix Factorization

The general file class which is used to produce a factorization of the Y matrix being the video
it computes it using all the files inside of cnmf folder.
Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
 of the structure of the class

 it is calling everyfunction from the cnmf folder
 you can find out more at how the functions are called and how they are laid out at the ipython notebook

See Also:
------------

@url http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
.. mage:: docs/img/quickintro.png
@author andrea giovannucci
"""
#\package Caiman/utils
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Fri Aug 26 15:44:32 2016

from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import object
import numpy as np
from .utilities import CNMFSetParms, update_order, normalize_AC, compute_residuals, detrend_df_f
from .pre_processing import preprocess_data
from .initialization import initialize_components, imblur
from .merging import merge_components
from .spatial import update_spatial_components
from .temporal import update_temporal_components, constrained_foopsi_parallel
from caiman.components_evaluation import estimate_components_quality_auto, select_components_from_metrics
from caiman.motion_correction import motion_correct_iteration_fast
from .map_reduce import run_CNMF_patches
from .oasis import OASIS
import caiman
from caiman import components_evaluation, mmapping
import cv2
from .online_cnmf import RingBuffer, HALS4activity, HALS4shapes, demix_and_deconvolve, remove_components_online
from .online_cnmf import init_shapes_and_sufficient_stats, update_shapes, update_num_components, bare_initialization, seeded_initialization
import scipy
import psutil
import pylab as pl
from time import time
import logging
import sys
import inspect

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    profile
except:
    def profile(a): return a


class CNMF(object):
    """  Source extraction using constrained non-negative matrix factorization.


    The general class which is used to produce a factorization of the Y matrix being the video
    it computes it using all the files inside of cnmf folder.
    Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
     of the structure of the class

    it is calling everyfunction from the cnmf folder
    you can find out more at how the functions are called and how they are laid out at the ipython notebook

    See Also:
    ------------
    @url http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
    .. image:: docs/img/quickintro.png
    @author andrea giovannucci
    """

    def __init__(self, n_processes, k=5, gSig=[4, 4], gSiz=None, merge_thresh=0.8, p=2, dview=None,
                 Ain=None, Cin=None, b_in=None, f_in=None, do_merge=True,
                 ssub=2, tsub=2, p_ssub=1, p_tsub=1, method_init='greedy_roi', alpha_snmf=None,
                 rf=None, stride=None, memory_fact=1, gnb=1, nb_patch=1, only_init_patch=False,
                 method_deconvolution='oasis', n_pixels_per_process=4000, block_size=5000, num_blocks_per_run=20,
                 check_nan=True, skip_refinement=False, normalize_init=True, options_local_NMF=None,
                 minibatch_shape=100, minibatch_suff_stat=3,
                 update_num_comps=True, rval_thr=0.9, thresh_fitness_delta=-20,
                 thresh_fitness_raw=-40, thresh_overlap=.5,
                 max_comp_update_shape=np.inf, num_times_comp_updated=np.inf,
                 batch_update_suff_stat=False, s_min=None,
                 remove_very_bad_comps=False, border_pix=0, low_rank_background=True,
                 update_background_components=True, rolling_sum=True, rolling_length=100,
                 min_corr=.85, min_pnr=20, ring_size_factor=1.5,
                 center_psf=False, use_dense=True, deconv_flag=True,
                 simultaneously=False, n_refit=0, del_duplicates=False, N_samples_exceptionality=5,
                 max_num_added=1, min_num_trial=2, thresh_CNN_noisy=0.5,
                 fr=30, decay_time=0.4, min_SNR=2.5, ssub_B=2, init_iter=2):
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

        dview: Direct View object
            for parallelization pruposes when using ipyparallel

        p: int
            order of the autoregressive process used to estimate deconvolution

        Ain: ndarray
            if know, it is the initial estimate of spatial filters

        ssub: int
            downsampleing factor in space

        tsub: int
             downsampling factor in time

        p_ssub: int
            downsampling factor in space for patches

        method_init: str
           can be greedy_roi or sparse_nmf

        alpha_snmf: float
            weight of the sparsity regularization

        p_tsub: int
             downsampling factor in time for patches

        rf: int
            half-size of the patches in pixels. rf=25, patches are 50x50

        gnb: int
            number of global background components

        nb_patch: int
            number of background components per patch

        stride: int
            amount of overlap between the patches in pixels

        memory_fact: float
            unitless number accounting how much memory should be used. You will
             need to try different values to see which one would work the default is OK for a 16 GB system

        N_samples_fitness: int
            number of samples over which exceptional events are computed (See utilities.evaluate_components)

        only_init_patch= boolean
            only run initialization on patches

        method_deconvolution = 'oasis' or 'cvxpy'
            method used for deconvolution. Suggested 'oasis' see
            Friedrich J, Zhou P, Paninski L. Fast Online Deconvolution of Calcium Imaging Data.
            PLoS Comput Biol. 2017; 13(3):e1005423.

        n_pixels_per_process: int.
            Number of pixels to be processed in parallel per core (no patch mode). Decrease if memory problems

        block_size: int.
            Number of pixels to be used to perform residual computation in blocks. Decrease if memory problems

        num_blocks_per_run: int
            In case of memory problems you can reduce this numbers, controlling the number of blocks processed in parallel during residual computing

        check_nan: Boolean.
            Check if file contains NaNs (costly for very large files so could be turned off)

        skip_refinement:
            Bool. If true it only performs one iteration of update spatial update temporal instead of two

        normalize_init=Bool.
            Differences in intensities on the FOV might caus troubles in the initialization when patches are not used,
             so each pixels can be normalized by its median intensity

        options_local_NMF:
            experimental, not to be used

        remove_very_bad_comps:Bool
            whether to remove components with very low values of component quality directly on the patch.
             This might create some minor imprecisions.
            Howeverm benefits can be considerable if done because if many components (>2000) are created
            and joined together, operation that causes a bottleneck

        border_pix:int
            number of pixels to not consider in the borders

        low_rank_background:bool
            if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)
             In the False case all the nonzero elements of the background components are updated using hals (to be used with one background per patch)

        update_background_components:bool
            whether to update the background components during the spatial phase

        min_corr: float
            minimal correlation peak for 1-photon imaging initialization

        min_pnr: float
            minimal peak  to noise ratio for 1-photon imaging initialization

        ring_size_factor: float
            it's the ratio between the ring radius and neuron diameters.

                max_comp_update_shape:
                         threshold number of components after which selective updating starts (using the parameter num_times_comp_updated)

            num_times_comp_updated:
            number of times each component is updated. In inf components are updated at every initbatch time steps

        expected_comps: int
            number of expected components (try to exceed the expected)

        deconv_flag : bool, optional
            If True, deconvolution is also performed using OASIS

        simultaneously : bool, optional
            If true, demix and denoise/deconvolve simultaneously. Slower but can be more accurate.

        n_refit : int, optional
            Number of pools (cf. oasis.pyx) prior to the last one that are refitted when
            simultaneously demixing and denoising/deconvolving.

        N_samples_exceptionality : int, optional
            Number of consecutives intervals to be considered when testing new neuron candidates

        del_duplicates: bool
            whether to delete the duplicated created in initialization

        max_num_added : int, optional
            maximum number of components to be added at each step in OnACID

        min_num_trial : int, optional
            minimum numbers of attempts to include a new components in OnACID

        thresh_CNN_noisy: float
            threshold on the per patch CNN classifier for online algorithm

        ssub_B: int, optional
            downsampleing factor for 1-photon imaging background computation

        init_iter: int, optional
            number of iterations for 1-photon imaging initialization

        Returns:
        --------
        self
        """

        # number of neurons expected per FOV (or per patch if patches_pars is not None
        self.k = k
        self.gSig = gSig  # expected half size of neurons
        self.gSiz = gSiz  # expected half size of neurons
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
        # unitless number accounting how much memory should be used.
        #  You will need to try different values to see which one would work the default is OK for a 16 GB system
        self.memory_fact = memory_fact
        self.gnb = gnb
        self.do_merge = do_merge
        self.alpha_snmf = alpha_snmf
        self.only_init = only_init_patch
        self.method_deconvolution = method_deconvolution
        self.n_pixels_per_process = n_pixels_per_process
        self.block_size = block_size
        self.num_blocks_per_run = num_blocks_per_run
        self.check_nan = check_nan
        self.skip_refinement = skip_refinement
        self.normalize_init = normalize_init
        self.options_local_NMF = options_local_NMF
        self.b_in = b_in
        self.A = None
        self.C = None
        self.S = None
        self.b = None
        self.f = None
        self.sn = None
        self.g = None
        self.remove_very_bad_comps = remove_very_bad_comps
        self.border_pix = border_pix
        self.low_rank_background = low_rank_background
        self.update_background_components = update_background_components
        self.rolling_sum = rolling_sum
        self.rolling_length = rolling_length
        self.minibatch_shape = minibatch_shape
        self.minibatch_suff_stat = minibatch_suff_stat
        self.update_num_comps = update_num_comps
        self.rval_thr = rval_thr
        self.s_min = s_min
        self.thresh_fitness_delta = thresh_fitness_delta
        self.thresh_fitness_raw = thresh_fitness_raw
        self.thresh_overlap = thresh_overlap
        self.num_times_comp_updated = num_times_comp_updated
        self.max_comp_update_shape = max_comp_update_shape
        self.batch_update_suff_stat = batch_update_suff_stat
        self.use_dense = use_dense
        self.deconv_flag = deconv_flag
        self.simultaneously = simultaneously
        self.n_refit = n_refit
        self.N_samples_exceptionality = N_samples_exceptionality
        self.max_num_added = max_num_added
        self.min_num_trial = min_num_trial
        self.thresh_CNN_noisy = thresh_CNN_noisy
        self.fr = fr
        self.min_SNR = min_SNR
        self.decay_time = decay_time

        self.min_corr = min_corr
        self.min_pnr = min_pnr
        self.ring_size_factor = ring_size_factor
        self.center_psf = center_psf
        self.nb_patch = nb_patch
        self.del_duplicates = del_duplicates

        self.params = CNMFSetParms((1, 1, 1), n_processes, p=p, gSig=gSig, gSiz=gSiz,
                                    K=k, ssub=ssub, tsub=tsub,
                                    p_ssub=p_ssub, p_tsub=p_tsub, method_init=method_init,
                                    n_pixels_per_process=n_pixels_per_process,
                                    check_nan=check_nan, nb=gnb,
                                    nb_patch=nb_patch, normalize_init=normalize_init,
                                    options_local_NMF=options_local_NMF,
                                    remove_very_bad_comps=remove_very_bad_comps,
                                    low_rank_background=low_rank_background,
                                    update_background_components=update_background_components,
                                    rolling_sum=self.rolling_sum,
                                    min_corr=min_corr, min_pnr=min_pnr,
                                    ring_size_factor=ring_size_factor, center_psf=center_psf,
                                    fr=fr, min_SNR=min_SNR, decay_time=decay_time,
                                    ssub_B=ssub_B, init_iter=init_iter)
        self.params.merging['thr'] = merge_thresh
        self.params.temporal['s_min'] = s_min

    def fit(self, images):
        """
        This method uses the cnmf algorithm to find sources in data.

        it is calling everyfunction from the cnmf folder
        you can find out more at how the functions are called and how they are laid out at the ipython notebook

        Parameters:
        ----------
        images : mapped np.ndarray of shape (t,x,y[,z]) containing the images that vary over time.

        Returns:
        --------
        self: updated using the cnmf algorithm with C,A,S,b,f computed according to the given initial values

        Raise:
        ------
        raise Exception('You need to provide a memory mapped file as input if you use patches!!')

        See Also:
        --------

        ..image::docs/img/quickintro.png

        http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3

        """
        # Todo : to compartiment
        T = images.shape[0]
        self.initbatch = T
        dims = images.shape[1:]
        Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])
        Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
        if np.isfortran(Yr):
            raise Exception('The file is in F order, it should be in C order (see save_memmap function')

        print((T,) + dims)

        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        try:
            Y.filename = images.filename
            Yr.filename = images.filename
        except AttributeError:  # if no memmapping cause working with small data
            pass

        # update/set all options that depend on data dimensions
        # number of rows, columns [and depths]
        self.params.spatial['dims'] = dims
        self.params.spatial['medw'] = (
            3,) * len(dims)  # window of median filter
        # Morphological closing structuring element
        self.params.spatial['se'] = np.ones(
            (3,) * len(dims), dtype=np.uint8)
        # Binary element for determining connectivity
        self.params.spatial['ss'] = np.ones(
            (3,) * len(dims), dtype=np.uint8)

        print(('using ' + str(self.n_processes) + ' processes'))
        if self.n_pixels_per_process is None:
            avail_memory_per_process = psutil.virtual_memory()[
                1] / 2.**30 / self.n_processes
            mem_per_pix = 3.6977678498329843e-09
            self.n_pixels_per_process = np.int(
                avail_memory_per_process / 8. / mem_per_pix / T)
            self.n_pixels_per_process = np.int(np.minimum(
                self.n_pixels_per_process, np.prod(dims) // self.n_processes))
        self.params.preprocess['n_pixels_per_process'] = self.n_pixels_per_process
        self.params.spatial['n_pixels_per_process'] = self.n_pixels_per_process

#        if self.block_size is None:
#            self.block_size = self.n_pixels_per_process
#
#        if self.num_blocks_per_run is None:
#           self.num_blocks_per_run = 20

        # number of pixels to process at the same time for dot product. Make it
        # smaller if memory problems
        self.params.temporal['block_size'] = self.block_size
        self.params.temporal['num_blocks_per_run'] = self.num_blocks_per_run
        self.params.spatial['block_size'] = self.block_size
        self.params.spatial['num_blocks_per_run'] = self.num_blocks_per_run

        print(('using ' + str(self.n_pixels_per_process) + ' pixels per process'))
        print(('using ' + str(self.block_size) + ' block_size'))

        if self.rf is None:  # no patches
            print('preprocessing ...')
            Yr, sn, g, psx = preprocess_data(
                Yr, dview=self.dview, **self.params.preprocess)

            self.sn = sn

            if self.Ain is None:
                print('initializing ...')
                if self.alpha_snmf is not None:
                    self.params.init['alpha_snmf'] = self.alpha_snmf

                self.initialize(Y, sn)

            if self.only_init:  # only return values after initialization
                self.A = self.Ain
                self.C = self.Cin
                self.b = self.b_in
                self.f = self.f_in
                if self.center_psf:
                    try:
                        self.S, self.bl, self.c1, self.neurons_sn, self.g, self.YrA = self.extra_1p
                    except:
                        self.S, self.bl, self.c1, self.neurons_sn, self.g, self.YrA, self.W, self.b0 = self.extra_1p
                else:
                    self.compute_residuals(Yr)
                    self.g = g
                    self.bl = None
                    self.c1 = None
                    self.neurons_sn = None

                self.A = self.Ain
                self.C = self.Cin

                if self.remove_very_bad_comps:
                    print('removing bad components : ')
                    final_frate = 10
                    r_values_min = 0.5  # threshold on space consistency
                    fitness_min = -15  # threshold on time variability
                    fitness_delta_min = -15
                    Npeaks = 10
                    traces = np.array(self.C)
                    print('estimating the quality...')
                    idx_components, idx_components_bad, fitness_raw,\
                        fitness_delta, r_values = components_evaluation.estimate_components_quality(
                            traces, Y, self.A, self.C, self.b_in, self.f_in,
                            final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
                            fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, return_all=True, N=5)

                    print(('Keeping ' + str(len(idx_components)) +
                           ' and discarding  ' + str(len(idx_components_bad))))
                    self.C = self.C[idx_components]
                    self.A = self.A[:, idx_components]
                    self.YrA = self.YrA[idx_components]

                self.sn = sn
                self.b = self.b_in
                self.f = self.f_in

                self.A, self.C, self.YrA, self.b, self.f, self.neurons_sn = normalize_AC(
                    self.A, self.C, self.YrA, self.b, self.f, self.neurons_sn)
                return self

            print('update spatial ...')
            self.update_spatial(Yr, use_init=True, dview=self.dview, **self.params.spatial)

            print('update temporal ...')
            if not self.skip_refinement:
                # set this to zero for fast updating without deconvolution
                self.params.temporal['p'] = 0
            else:
                self.params.temporal['p'] = self.p
            print('deconvolution ...')
            self.params.temporal['method'] = self.method_deconvolution

            self.update_temporal(Yr, dview=self.dview, **self.params.temporal)

            if not self.skip_refinement:
                print('refinement...')
                if self.do_merge:
                    print('merge components ...')
                    print(self.A.shape)
                    print(self.C.shape)
                    self.merge_comps(Yr, mx=50, fast_merge=True)

                print((self.A.shape))
                print(self.C.shape)
                print('update spatial ...')

                self.update_spatial(Yr, use_init=False, dview=self.dview, **self.params.spatial)
                # set it back to original value to perform full deconvolution
                self.params.temporal['p'] = self.p
                print('update temporal ...')
                self.update_temporal(Yr, use_init=False, dview=self.dview, **self.params.temporal)

            else:
                # todo : ask for those..
                C, f, S, bl, c1, neurons_sn, g1, YrA, lam = self.C, self.f, self.S, self.bl, self.c1, self.neurons_sn, self.g, self.YrA, self.lam

        else:  # use patches
            if self.stride is None:
                self.stride = np.int(self.rf * 2 * .1)
                print(
                    ('**** Setting the stride to 10% of 2*rf automatically:' + str(self.stride)))

            if type(images) is np.ndarray:
                raise Exception(
                    'You need to provide a memory mapped file as input if you use patches!!')

            if self.only_init:
                self.params.patch['only_init'] = True

            if self.alpha_snmf is not None:
                self.params.init['alpha_snmf'] = self.alpha_snmf

            A, C, YrA, b, f, sn, optional_outputs = run_CNMF_patches(images.filename, dims + (T,),
                                                                     self.params, rf=self.rf, stride=self.stride,
                                                                     dview=self.dview, memory_fact=self.memory_fact,
                                                                     gnb=self.gnb, border_pix=self.border_pix,
                                                                     low_rank_background=self.low_rank_background,
                                                                     del_duplicates=self.del_duplicates)

            self.A, self.C, self.YrA, self.b, self.f, self.sn, self.optional_outputs = A, C, YrA, b, f, sn, optional_outputs
            self.bl, self.c1, self.g, self.neurons_sn = None, None, None, None
            print("merging")
            self.merged_ROIs = [0]

            if self.center_psf:  # merge taking best neuron
                if self.nb_patch > 0:

                    while len(self.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.Inf, thr=self.merge_thresh, fast_merge=True)

                    print("update temporal")
                    self.update_temporal(Yr, use_init=False, **self.params.temporal)

                    self.params.spatial['se'] = np.ones((1,) * len(dims), dtype=np.uint8)
                    print('update spatial ...')
                    self.update_spatial(Yr, use_init=False, dview=self.dview, **self.params.spatial)

                    print("update temporal")
                    self.update_temporal(Yr, use_init=False, **self.params.temporal)
                else:
                    while len(self.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.Inf, thr=self.merge_thresh, fast_merge=True)
                        if len(self.merged_ROIs) > 0:
                            not_merged = np.setdiff1d(list(range(len(self.YrA))),
                                                      np.unique(np.concatenate(self.merged_ROIs)))
                            self.YrA = np.concatenate([self.YrA[not_merged],
                                                       np.array([self.YrA[m].mean(0) for m in self.merged_ROIs])])
            else:
                
                while len(self.merged_ROIs) > 0:
                    self.merge_comps(Yr, mx=np.Inf, thr=self.merge_thresh)

                print("update temporal")
                self.update_temporal(Yr, use_init=False, **self.params.temporal)

        self.A, self.C, self.YrA, self.b, self.f, self.neurons_sn = normalize_AC(
            self.A, self.C, self.YrA, self.b, self.f, self.neurons_sn)

        return self

    def _prepare_object(self, Yr, T, expected_comps, new_dims=None,
                        idx_components=None, g=None, lam=None, s_min=None,
                        bl=None, use_dense=True, N_samples_exceptionality=5,
                        max_num_added=1, min_num_trial=1, path_to_model=None,
                        sniper_mode=False, use_peak_max=False,
                        test_both=False, q=0.5, **kwargs):

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
        self.N_samples_exceptionality = N_samples_exceptionality
        self.max_num_added = max_num_added
        self.min_num_trial = min_num_trial
        self.q = q   # sparsity parameter (between 0.5 and 1)

        self.N = self.A2.shape[-1]
        self.M = self.gnb + self.N

        if expected_comps <= self.N + max_num_added:
            expected_comps = self.N + max_num_added + 200
        self.expected_comps = expected_comps

        if Yr.shape[-1] != self.initbatch:
            raise Exception(
                'The movie size used for initialization does not match with the minibatch size')

        if new_dims is not None:

            new_Yr = np.zeros([np.prod(new_dims), T])
            for ffrr in range(T):
                tmp = cv2.resize(Yr[:, ffrr].reshape(
                    self.dims2, order='F'), new_dims[::-1])
                print(tmp.shape)
                new_Yr[:, ffrr] = tmp.reshape([np.prod(new_dims)], order='F')
            Yr = new_Yr
            A_new = scipy.sparse.csc_matrix(
                (np.prod(new_dims), self.A2.shape[-1]), dtype=np.float32)
            for neur in range(self.N):
                a = self.A2.tocsc()[:, neur].toarray()
                a = a.reshape(self.dims2, order='F')
                a = cv2.resize(a, new_dims[::-1]).reshape([-1, 1], order='F')

                A_new[:, neur] = scipy.sparse.csc_matrix(a)

            self.A2 = A_new
            self.b2 = self.b2.reshape(self.dims2, order='F')
            self.b2 = cv2.resize(
                self.b2, new_dims[::-1]).reshape([-1, 1], order='F')

            self.dims2 = new_dims

        nA = np.ravel(np.sqrt(self.A2.power(2).sum(0)))
        self.A2 /= nA
        self.C2 *= nA[:, None]
        self.YrA2 *= nA[:, None]
#        self.S2 *= nA[:, None]
        self.neurons_sn2 *= nA
        if self.p:
            self.lam2 *= nA
        z = np.sqrt([b.T.dot(b) for b in self.b2.T])
        self.f2 *= z[:, None]
        self.b2 /= z

        self.noisyC = np.zeros(
            (self.gnb + expected_comps, T), dtype=np.float32)
        self.C_on = np.zeros((expected_comps, T), dtype=np.float32)

        self.noisyC[self.gnb:self.M, :self.initbatch] = self.C2 + self.YrA2
        self.noisyC[:self.gnb, :self.initbatch] = self.f2

        if self.p:
            # if no parameter for calculating the spike size threshold is given, then use L1 penalty
            if s_min is None and self.s_min is None:
                use_L1 = True
            else:
                use_L1 = False

            self.OASISinstances = [OASIS(
                g=np.ravel(0.01) if self.p == 0 else (
                    np.ravel(g)[0] if g is not None else gam[0]),
                lam=0 if not use_L1 else (l if lam is None else lam),
                s_min=0 if use_L1 else (s_min if s_min is not None else
                                        (self.s_min if self.s_min > 0 else
                                         (-self.s_min * sn * np.sqrt(1 - np.sum(gam))))),
                b=b if bl is None else bl,
                g2=0 if self.p < 2 else (np.ravel(g)[1] if g is not None else gam[1]))
                for gam, l, b, sn in zip(self.g2, self.lam2, self.bl2, self.neurons_sn2)]

            for i, o in enumerate(self.OASISinstances):
                o.fit(self.noisyC[i + self.gnb, :self.initbatch])
                self.C_on[i, :self.initbatch] = o.c
        else:
            self.C_on[:self.N, :self.initbatch] = self.C2
        import pdb
        #pdb.set_trace()
        self.Ab, self.ind_A, self.CY, self.CC = init_shapes_and_sufficient_stats(
            Yr[:, :self.initbatch].reshape(
                self.dims2 + (-1,), order='F'), self.A2,
            self.C_on[:self.N, :self.initbatch], self.b2, self.noisyC[:self.gnb, :self.initbatch])

        self.CY, self.CC = self.CY * 1. / self.initbatch, 1 * self.CC / self.initbatch

        self.A2 = scipy.sparse.csc_matrix(
            self.A2.astype(np.float32), dtype=np.float32)
        self.C2 = self.C2.astype(np.float32)
        self.f2 = self.f2.astype(np.float32)
        self.b2 = self.b2.astype(np.float32)
        self.Ab = scipy.sparse.csc_matrix(
            self.Ab.astype(np.float32), dtype=np.float32)
        self.noisyC = self.noisyC.astype(np.float32)
        self.CY = self.CY.astype(np.float32)
        self.CC = self.CC.astype(np.float32)
        print('Expecting ' + str(self.expected_comps) + ' components')
        self.CY.resize([self.expected_comps + self.gnb, self.CY.shape[-1]])
        if use_dense:
            self.Ab_dense = np.zeros((self.CY.shape[-1], self.expected_comps + self.gnb),
                                     dtype=np.float32)
            self.Ab_dense[:, :self.Ab.shape[1]] = self.Ab.toarray()
        self.C_on = np.vstack(
            [self.noisyC[:self.gnb, :], self.C_on.astype(np.float32)])

        self.gSiz = np.add(np.multiply(np.ceil(self.gSig).astype(np.int), 2), 1)

        self.Yr_buf = RingBuffer(Yr[:, self.initbatch - self.minibatch_shape:
                                    self.initbatch].T.copy(), self.minibatch_shape)
        self.Yres_buf = RingBuffer(self.Yr_buf - self.Ab.dot(
            self.C_on[:self.M, self.initbatch - self.minibatch_shape:self.initbatch]).T, self.minibatch_shape)
        self.sn = np.array(np.std(self.Yres_buf,axis=0))
        self.vr = np.array(np.var(self.Yres_buf,axis=0))
        self.mn = self.Yres_buf.mean(0)
        self.Yres_buf = self.Yres_buf
        self.mean_buff = self.Yres_buf.mean(0)
        self.ind_new = []
        self.rho_buf = imblur(np.maximum(self.Yres_buf.T,0).reshape(
            self.dims2 + (-1,), order='F'), sig=self.gSig, siz=self.gSiz, nDimBlur=len(self.dims2))**2
        self.rho_buf = np.reshape(
            self.rho_buf, (np.prod(self.dims2), -1)).T
        self.rho_buf = RingBuffer(self.rho_buf, self.minibatch_shape)
        self.AtA = (self.Ab.T.dot(self.Ab)).toarray()
        self.AtY_buf = self.Ab.T.dot(self.Yr_buf.T)
        self.sv = np.sum(self.rho_buf.get_last_frames(
            min(self.initbatch, self.minibatch_shape) - 1), 0)
        self.groups = list(map(list, update_order(self.Ab)[0]))
        # self.update_counter = np.zeros(self.N)
        self.update_counter = .5**(-np.linspace(0, 1,
                                                self.N, dtype=np.float32))
        self.time_neuron_added = []
        for nneeuu in range(self.N):
            self.time_neuron_added.append((nneeuu, self.initbatch))
        self.time_spend = 0
        # setup per patch classifier

        if path_to_model is None or sniper_mode is False:
            loaded_model = None
            sniper_mode = False
        else:
            import keras
            from keras.models import model_from_json
            path = path_to_model.split(".")[:-1]
            json_path = ".".join(path + ["json"])
            model_path = ".".join(path + ["h5"])

            json_file = open(json_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_path)
            opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
            loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=opt, metrics=['accuracy'])

        self.loaded_model = loaded_model
        self.sniper_mode = sniper_mode
        self.test_both = test_both
        self.use_peak_max = use_peak_max
        return self

    @profile
    def fit_next(self, t, frame_in, num_iters_hals=3):
        """
        This method fits the next frame using the online cnmf algorithm and
        updates the object.

        Parameters
        ----------
        t : int
            time measured in number of frames

        frame_in : array
            flattened array of shape (x*y[*z],) containing the t-th image.

        num_iters_hals: int, optional
            maximal number of iterations for HALS (NNLS via blockCD)


        """

        t_start = time()

        # locally scoped variables for brevity of code and faster look up
        nb_ = self.gnb
        Ab_ = self.Ab
        mbs = self.minibatch_shape
        frame = frame_in.astype(np.float32)
#        print(np.max(1/scipy.sparse.linalg.norm(self.Ab,axis = 0)))
        self.Yr_buf.append(frame)
        if len(self.ind_new) > 0:
            self.mean_buff = self.Yres_buf.mean(0)

        if (not self.simultaneously) or self.p == 0:
            # get noisy fluor value via NNLS (project data on shapes & demix)
            C_in = self.noisyC[:self.M, t - 1].copy()
            self.C_on[:self.M, t], self.noisyC[:self.M, t] = HALS4activity(
                frame, self.Ab, C_in, self.AtA, iters=num_iters_hals, groups=self.groups)
            if self.p:
                # denoise & deconvolve
                for i, o in enumerate(self.OASISinstances):
                    o.fit_next(self.noisyC[nb_ + i, t])
                    self.C_on[nb_ + i, t - o.get_l_of_last_pool() +
                              1: t + 1] = o.get_c_of_last_pool()

        else:
            # update buffer, initialize C with previous value
            self.C_on[:, t] = self.C_on[:, t - 1]
            self.noisyC[:, t] = self.C_on[:, t - 1]
            self.AtY_buf = np.concatenate((self.AtY_buf[:, 1:], self.Ab.T.dot(frame)[:, None]), 1) \
                if self.n_refit else self.Ab.T.dot(frame)[:, None]
            # demix, denoise & deconvolve
            (self.C_on[:self.M, t + 1 - mbs:t + 1], self.noisyC[:self.M, t + 1 - mbs:t + 1],
                self.OASISinstances) = demix_and_deconvolve(
                self.C_on[:self.M, t + 1 - mbs:t + 1],
                self.noisyC[:self.M, t + 1 - mbs:t + 1],
                self.AtY_buf, self.AtA, self.OASISinstances, iters=num_iters_hals,
                n_refit=self.n_refit)
            for i, o in enumerate(self.OASISinstances):
                self.C_on[nb_ + i, t - o.get_l_of_last_pool() + 1: t +
                          1] = o.get_c_of_last_pool()

        #self.mean_buff = self.Yres_buf.mean(0)
        res_frame = frame - self.Ab.dot(self.C_on[:self.M, t])
        mn_ = self.mn.copy()
        self.mn = (t-1)/t*self.mn + res_frame/t
        self.vr = (t-1)/t*self.vr + (res_frame - mn_)*(res_frame - self.mn)/t
        self.sn = np.sqrt(self.vr)

        if self.update_num_comps:

            self.mean_buff += (res_frame-self.Yres_buf[self.Yres_buf.cur])/self.minibatch_shape
#            cv2.imshow('untitled', 0.1*cv2.resize(res_frame.reshape(self.dims,order = 'F'),(512,512)))
#            cv2.waitKey(1)
#
            self.Yres_buf.append(res_frame)

            res_frame = np.reshape(res_frame, self.dims2, order='F')

            rho = imblur(np.maximum(res_frame,0), sig=self.gSig,
                         siz=self.gSiz, nDimBlur=len(self.dims2))**2

            rho = np.reshape(rho, np.prod(self.dims2))
            self.rho_buf.append(rho)

            self.Ab, Cf_temp, self.Yres_buf, self.rhos_buf, self.CC, self.CY, self.ind_A, self.sv, self.groups, self.ind_new, self.ind_new_all, self.sv, self.cnn_pos = update_num_components(
                t, self.sv, self.Ab, self.C_on[:self.M, (t - mbs + 1):(t + 1)],
                self.Yres_buf, self.Yr_buf, self.rho_buf, self.dims2,
                self.gSig, self.gSiz, self.ind_A, self.CY, self.CC, rval_thr=self.rval_thr,
                thresh_fitness_delta=self.thresh_fitness_delta,
                thresh_fitness_raw=self.thresh_fitness_raw, thresh_overlap=self.thresh_overlap,
                groups=self.groups, batch_update_suff_stat=self.batch_update_suff_stat, gnb=self.gnb,
                sn=self.sn, g=np.mean(
                    self.g2) if self.p == 1 else np.mean(self.g2, 0),
                s_min=self.s_min,
                Ab_dense=self.Ab_dense[:, :self.M] if self.use_dense else None,
                oases=self.OASISinstances if self.p else None, N_samples_exceptionality=self.N_samples_exceptionality,
                max_num_added=self.max_num_added, min_num_trial=self.min_num_trial,
                loaded_model = self.loaded_model, thresh_CNN_noisy = self.thresh_CNN_noisy,
                sniper_mode=self.sniper_mode, use_peak_max=self.use_peak_max,
                test_both=self.test_both)

            num_added = len(self.ind_A) - self.N

            if num_added > 0:
                self.N += num_added
                self.M += num_added
                if self.N + self.max_num_added > self.expected_comps:
                    self.expected_comps += 200
                    self.CY.resize(
                        [self.expected_comps + nb_, self.CY.shape[-1]])
                    # refcheck can trigger "ValueError: cannot resize an array references or is referenced
                    #                       by another array in this way.  Use the resize function"
                    # np.resize didn't work, but refcheck=False seems fine
                    self.C_on.resize(
                        [self.expected_comps + nb_, self.C_on.shape[-1]], refcheck=False)
                    self.noisyC.resize(
                        [self.expected_comps + nb_, self.C_on.shape[-1]])
                    if self.use_dense:  # resize won't work due to contingency issue
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
                    if self.p:
                        # N.B. OASISinstances are already updated within update_num_components
                        self.C_on[_ct, t - mbs + 1: t +
                                  1] = self.OASISinstances[_ct - nb_].get_c(mbs)
                    else:
                        self.C_on[_ct, t - mbs + 1: t + 1] = np.maximum(0,
                                                                        self.noisyC[_ct, t - mbs + 1: t + 1])
                    if self.simultaneously and self.n_refit:
                        self.AtY_buf = np.concatenate((
                            self.AtY_buf, [Ab_.data[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]].dot(
                                self.Yr_buf.T[Ab_.indices[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]]])]))
                    # much faster than Ab_[:, self.N + nb_ - num_added:].toarray()
                    if self.use_dense:
                        self.Ab_dense[Ab_.indices[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]],
                                      _ct] = Ab_.data[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]]

                # set the update counter to 0 for components that are overlaping the newly added
                if self.use_dense:
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
                self.CY[m + nb_, self.ind_A[m]] += w2 * \
                    ccf[m + nb_].dot(y[:, self.ind_A[m]])

            self.CY[:nb_] = self.CY[:nb_] * w1 + \
                w2 * ccf[:nb_].dot(y)   # background
            self.CC = self.CC * w1 + w2 * ccf.dot(ccf.T)

        if not self.batch_update_suff_stat:

            ccf = self.C_on[:self.M, t - self.minibatch_suff_stat:t -
                            self.minibatch_suff_stat + 1]
            y = self.Yr_buf.get_last_frames(self.minibatch_suff_stat + 1)[:1]
            # much faster: exploit that we only access CY[m, ind_pixels], hence update only these
            for m in range(self.N):
                self.CY[m + nb_, self.ind_A[m]] *= (1 - 1. / t)
                self.CY[m + nb_, self.ind_A[m]] += ccf[m +
                                                       nb_].dot(y[:, self.ind_A[m]]) / t
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

                if self.use_dense:
                    # update dense Ab and sparse Ab simultaneously;
                    # this is faster than calling update_shapes with sparse Ab only
                    Ab_, self.ind_A, self.Ab_dense[:, :self.M] = update_shapes(
                        self.CY, self.CC, self.Ab, self.ind_A,
                        indicator_components=indicator_components,
                        Ab_dense=self.Ab_dense[:, :self.M],
                        sn=self.sn, q=self.q)
                else:
                    Ab_, self.ind_A, _ = update_shapes(self.CY, self.CC, Ab_, self.ind_A,
                                                       indicator_components=indicator_components,
                                                       sn=self.sn, q=self.q)

                self.AtA = (Ab_.T.dot(Ab_)).toarray()

                ind_zero = list(np.where(self.AtA.diagonal() < 1e-10)[0])
                if len(ind_zero) > 0:
                    ind_zero.sort()
                    ind_zero = ind_zero[::-1]
                    ind_keep = list(set(range(Ab_.shape[-1])) - set(ind_zero))
                    ind_keep.sort()

                    if self.use_dense:
                        self.Ab_dense = np.delete(
                            self.Ab_dense, ind_zero, axis=1)
                    self.AtA = np.delete(self.AtA, ind_zero, axis=0)
                    self.AtA = np.delete(self.AtA, ind_zero, axis=1)
                    self.CY = np.delete(self.CY, ind_zero, axis=0)
                    self.CC = np.delete(self.CC, ind_zero, axis=0)
                    self.CC = np.delete(self.CC, ind_zero, axis=1)
                    self.M -= len(ind_zero)
                    self.N -= len(ind_zero)
                    self.noisyC = np.delete(self.noisyC, ind_zero, axis=0)
                    for ii in ind_zero:
                        del self.OASISinstances[ii - self.gnb]
                        #del self.ind_A[ii-self.gnb]

                    self.C_on = np.delete(self.C_on, ind_zero, axis=0)
                    self.AtY_buf = np.delete(self.AtY_buf, ind_zero, axis=0)
                    print(1)
                    #import pdb
                    # pdb.set_trace()
                    #Ab_ = Ab_[:,ind_keep]
                    Ab_ = scipy.sparse.csc_matrix(Ab_[:, ind_keep])
                    #Ab_ = scipy.sparse.csc_matrix(self.Ab_dense[:,:self.M])
                    self.Ab_dense_copy = self.Ab_dense
                    self.Ab_copy = Ab_
                    self.Ab = Ab_
                    self.ind_A = list(
                        [(self.Ab.indices[self.Ab.indptr[ii]:self.Ab.indptr[ii + 1]]) for ii in range(self.gnb, self.M)])
                    self.groups = list(map(list, update_order(Ab_)[0]))

                if self.n_refit:
                    self.AtY_buf = Ab_.T.dot(self.Yr_buf.T)

        else:  # distributed shape update
            self.update_counter *= .5**(1. / mbs)
            # if not num_added:
            if (not num_added) and (time() - t_start < self.time_spend / (t - self.initbatch + 1)):
                candidates = np.where(self.update_counter <= 1)[0]
                if len(candidates):
                    indicator_components = candidates[:self.N // mbs + 1]
                    self.update_counter[indicator_components] += 1

                    if self.use_dense:
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

    def remove_components(self, ind_rm):
        """remove a specified list of components from the OnACID CNMF object.

        Parameters:
        -----------
        ind_rm :    list
                    indeces of components to be removed
        """

        self.Ab, self.Ab_dense, self.CC, self.CY, self.M,\
        self.N, self.noisyC, self.OASISinstances, self.C_on,\
        self.expected_comps, self.ind_A,\
        self.groups, self.AtA = remove_components_online(
                ind_rm, self.gnb, self.Ab, self.use_dense, self.Ab_dense,
                self.AtA, self.CY, self.CC, self.M, self.N, self.noisyC,
                self.OASISinstances, self.C_on, self.expected_comps)

    def compute_residuals(self, Yr):
        """compute residual for each component (variable YrA)

         Parameters:
         -----------
         Yr :    np.ndarray
                 movie in format pixels (d) x frames (T)

        """

        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if 'array' not in str(type(self.b)):
            self.b = self.b.toarray()
        if 'array' not in str(type(self.C)):
            self.C = self.C.toarray()
        if 'array' not in str(type(self.f)):
            self.f = self.f.toarray()

        Ab = scipy.sparse.hstack((self.A, self.b)).tocsc()
        nA2 = np.ravel(Ab.power(2).sum(axis=0))
        nA2_inv_mat = scipy.sparse.spdiags(
            1. / nA2, 0, nA2.shape[0], nA2.shape[0])
        Cf = np.vstack((self.C, self.f))
        if 'numpy.ndarray' in str(type(Yr)):
            YA = (Ab.T.dot(Yr)).T * nA2_inv_mat
        else:
            YA = mmapping.parallel_dot_product(Yr, Ab, dview=self.dview, block_size=2000,
                                           transpose=True, num_blocks_per_run=5) * nA2_inv_mat

        AA = Ab.T.dot(Ab) * nA2_inv_mat
        self.YrA = (YA - (AA.T.dot(Cf)).T)[:, :self.A.shape[-1]].T

        return self

    def normalize_components(self):
        """ normalize components such that spatial components have norm 1
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if 'array' not in str(type(self.b)):
            self.b = self.b.toarray()
        if 'array' not in str(type(self.C)):
            self.C = self.C.toarray()
        if 'array' not in str(type(self.f)):
            self.f = self.f.toarray()

        nA = np.sqrt(np.ravel(self.A.power(2).sum(axis=0)))
        nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
        nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
        self.A = self.A * nA_inv_mat
        self.C = nA_mat * self.C
        if self.YrA is not None:
            self.YrA = nA_mat * self.YrA
        if self.bl is not None:
            self.bl = nA * self.bl
        if self.c1 is not None:
            self.c1 = nA * self.c1
        if self.neurons_sn is not None:
            self.neurons_sn *= nA * self.neurons_sn

        nB = np.sqrt(np.ravel((self.b**2).sum(axis=0)))
        nB_mat = scipy.sparse.spdiags(nB, 0, nB.shape[0], nB.shape[0])
        nB_inv_mat = scipy.sparse.spdiags(1. / nB, 0, nB.shape[0], nB.shape[0])
        self.b = self.b * nB_inv_mat
        self.f = nB_mat * self.f

    def plot_contours(self, img=None, idx=None, crd=None, thr_method='max',
                      thr='0.2'):
        """view contour plots for each spatial footprint. 
        Parameters:
        -----------
        img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)
        idx :   list
                list of accepted components

        crd :   list
                list of coordinates (if empty they are computed)

        thr_method : str
                     thresholding method for computing contours ('max', 'nrg')

        thr : float
                threshold value
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if img is None:
            img = np.reshape(np.array(self.A.mean(1)), self.dims, order='F')
        if not hasattr(self, 'coordinates'):
            self.coordinates = caiman.utils.visualization.get_contours(self.A, self.dims, thr=thr, thr_method=thr_method)
        pl.figure()
        if idx is None:
            caiman.utils.visualization.plot_contours(self.A, img, coordinates=self.coordinates)
        else:
            if not isinstance(idx, list):
                idx = idx.tolist()
            coor_g = [self.coordinates[cr] for cr in idx]
            bad = list(set(range(self.A.shape[1])) - set(idx))
            coor_b = [self.coordinates[cr] for cr in bad]
            pl.subplot(1, 2, 1)
            caiman.utils.visualization.plot_contours(self.A[:, idx], img,
                                                     coordinates=coor_g)
            pl.title('Accepted Components')
            bad = list(set(range(self.A.shape[1])) - set(idx))
            pl.subplot(1, 2, 2)
            caiman.utils.visualization.plot_contours(self.A[:, bad], img,
                                                     coordinates=coor_b)
            pl.title('Rejected Components')
        return self

    def view_components(self, Yr, dims, img=None, idx=None):
        """view spatial and temporal components interactively

        Parameters:
        -----------
        Yr :    np.ndarray
                movie in format pixels (d) x frames (T)

        dims :  tuple
                dimensions of the FOV

        img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)

        idx :   list
                list of components to be plotted


        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if 'array' not in str(type(self.b)):
            self.b = self.b.toarray()

        pl.ion()
        nr, T = self.C.shape

        if self.YrA is None:
            self.compute_residuals(Yr)

        if img is None:
            img = np.reshape(np.array(self.A.mean(axis=1)), dims, order='F')

        if idx is None:
            caiman.utils.visualization.view_patches_bar(Yr, self.A, self.C,
                    self.b, self.f, dims[0], dims[1], YrA=self.YrA, img=img)
        else:
            caiman.utils.visualization.view_patches_bar(Yr, self.A.tocsc()[:,idx], self.C[idx], self.b, self.f, dims[
                                                    0], dims[1], YrA=self.YrA[idx], img=img)

    def detrend_df_f(self, quantileMin=8, frames_window=500,
                     flag_auto=True, use_fast=False, use_residuals=True):
        """Computes DF/F normalized fluorescence for the extracted traces. See
        caiman.source.extraction.utilities.detrend_df_f for details

        Parameters:
        -----------
        quantile_min: float
            quantile used to estimate the baseline (values in [0,100])

        frames_window: int
            number of frames for computing running quantile

        flag_auto: bool
            flag for determining quantile automatically (different for each
            trace)

        use_fast: bool
            flag for using approximate fast percentile filtering

        use_residuals: bool
            flag for using non-deconvolved traces in DF/F calculation

        Returns:
        --------
        self: CNMF object
            self.F_dff contains the DF/F normalized traces
        """

        if self.C is None:
            logging.warning("There are no components for DF/F extraction!")
            return self

        if use_residuals:
            R = self.YrA
        else:
            R = None

        self.F_dff = detrend_df_f(self.A, self.b, self.C, self.f, R,
                                  quantileMin=quantileMin,
                                  frames_window=frames_window,
                                  flag_auto=flag_auto, use_fast=use_fast)
        return self

    def deconvolve(self, p=None, method=None, bas_nonneg=None,
                   noise_method=None, optimize_g=0, s_min=None, **kwargs):
        """Performs deconvolution on already extracted traces using
        constrained foopsi.
        """

        p = self.p if p is None else p
        method = self.method_deconvolution if method is None else method
        bas_nonneg = (self.params.temporal['bas_nonneg']
                      if bas_nonneg is None else bas_nonneg)
        noise_method = (self.params.temporal['noise_method']
                        if noise_method is None else noise_method)
        s_min = self.s_min if s_min is None else s_min

        F = self.C + self.YrA
        args = dict()
        args['p'] = p
        args['method'] = method
        args['bas_nonneg'] = bas_nonneg
        args['noise_method'] = noise_method
        args['s_min'] = s_min
        args['optimize_g'] = optimize_g
        args['noise_range'] = self.params.temporal['noise_range']
        args['fudge_factor'] = self.params.temporal['fudge_factor']

        args_in = [(F[jj], None, jj, None, None, None, None,
                    args) for jj in range(F.shape[0])]

        if 'multiprocessing' in str(type(self.dview)):
            results = self.dview.map_async(
                constrained_foopsi_parallel, args_in).get(4294967)
        elif self.dview is not None:
            results = self.dview.map_sync(constrained_foopsi_parallel, args_in)
        else:
            results = list(map(constrained_foopsi_parallel, args_in))

        if sys.version_info >= (3, 0):
            results = list(zip(*results))
        else:  # python 2
            results = zip(*results)

        order = list(results[7])
        self.C = np.stack([results[0][i] for i in order])
        self.S = np.stack([results[1][i] for i in order])
        self.bl = [results[3][i] for i in order]
        self.c1 = [results[4][i] for i in order]
        self.g = [results[5][i] for i in order]
        self.neuron_sn = [results[6][i] for i in order]
        self.lam = [results[8][i] for i in order]
        self.YrA = F - self.C
        return self

    def evaluate_components(self, imgs, fr=None, decay_time=None, min_SNR=None,
                            rval_thr=None, use_cnn=None, min_cnn_thr=None):
        """Computes the quality metrics for each component and stores the
        indeces of the components that pass user specified thresholds. The
        various thresholds and parameters can be passed as inputs. If left
        empty then they are read from self.params.quality']
        Parameters:
        -----------
        imgs: np.array (possibly memory mapped, t,x,y[,z])
            Imaging data

        fr: float
            Imaging rate

        decay_time: float
            length of decay of typical transient (in seconds)

        min_SNR: float
            trace SNR threshold

        rval_thr: float
            space correlation threshold

        use_cnn: bool
            flag for using the CNN classifier

        min_cnn_thr: float
            CNN classifier threshold

        Returns:
        --------
        self: CNMF object
            self.idx_components: np.array
                indeces of accepted components
            self.idx_components_bad: np.array
                indeces of rejected components
            self.SNR_comp: np.array
                SNR values for each temporal trace
            self.r_values: np.array
                space correlation values for each component
            self.cnn_preds: np.array
                CNN classifier values for each component
        """
        dims = imgs.shape[1:]
        fr = self.params.quality['fr'] if fr is None else fr
        decay_time = (self.params.quality['decay_time']
                      if decay_time is None else decay_time)
        min_SNR = (self.params.quality['min_SNR']
                   if min_SNR is None else min_SNR)
        rval_thr = (self.params.quality['rval_thr']
                    if rval_thr is None else rval_thr)
        use_cnn = (self.params.quality['use_cnn']
                   if use_cnn is None else use_cnn)
        min_cnn_thr = (self.params.quality['min_cnn_thr']
                       if min_cnn_thr is None else min_cnn_thr)

        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
        estimate_components_quality_auto(imgs, self.A, self.C, self.b, self.f,
                                         self.YrA, fr, decay_time, self.gSig,
                                         dims, dview=self.dview,
                                         min_SNR=min_SNR,
                                         r_values_min=rval_thr,
                                         use_cnn=use_cnn,
                                         thresh_cnn_min=min_cnn_thr)
        self.idx_components = idx_components
        self.idx_components_bad = idx_components_bad
        self.SNR_comp = SNR_comp
        self.r_values = r_values
        self.cnn_preds = cnn_preds

        return self

    def filter_components(self, imgs, fr=None, decay_time=None, min_SNR=None,
                          SNR_lowest=None, rval_thr=None, rval_lowest=None,
                          use_cnn=None, min_cnn_thr=None,
                          cnn_lowest=None, gSig_range=None):
        """Filters components based on given thresholds without re-computing
        the quality metrics. If the quality metrics are not present then it
        calls self.evaluate components.
        Parameters:
        -----------
        imgs: np.array (possibly memory mapped, t,x,y[,z])
            Imaging data

        fr: float
            Imaging rate

        decay_time: float
            length of decay of typical transient (in seconds)

        min_SNR: float
            trace SNR threshold

        SNR_lowest: float
            minimum required trace SNR

        rval_thr: float
            space correlation threshold

        rval_lowest: float
            minimum required space correlation

        use_cnn: bool
            flag for using the CNN classifier

        min_cnn_thr: float
            CNN classifier threshold

        cnn_lowest: float
            minimum required CNN threshold

        gSig_range: list
            gSig scale values for CNN classifier

        Returns:
        --------
        self: CNMF object
            self.idx_components: np.array
                indeces of accepted components
            self.idx_components_bad: np.array
                indeces of rejected components
            self.SNR_comp: np.array
                SNR values for each temporal trace
            self.r_values: np.array
                space correlation values for each component
            self.cnn_preds: np.array
                CNN classifier values for each component
        """
        dims = imgs.shape[1:]
        fr = self.params.quality['fr'] if fr is None else fr
        decay_time = (self.params.quality['decay_time']
                      if decay_time is None else decay_time)
        min_SNR = (self.params.quality['min_SNR']
                   if min_SNR is None else min_SNR)
        SNR_lowest = (self.params.quality['SNR_lowest']
                      if SNR_lowest is None else SNR_lowest)
        rval_thr = (self.params.quality['rval_thr']
                    if rval_thr is None else rval_thr)
        rval_lowest = (self.params.quality['rval_lowest']
                       if rval_lowest is None else rval_lowest)
        use_cnn = (self.params.quality['use_cnn']
                   if use_cnn is None else use_cnn)
        min_cnn_thr = (self.params.quality['min_cnn_thr']
                       if min_cnn_thr is None else min_cnn_thr)
        cnn_lowest = (self.params.quality['cnn_lowest']
                      if cnn_lowest is None else cnn_lowest)
        gSig_range = (self.params.quality['gSig_range']
                      if gSig_range is None else gSig_range)

        if not hasattr(self, 'idx_components'):
            self.evaluate_components(imgs, fr=fr, decay_time=decay_time,
                                     min_SNR=min_SNR, rval_thr=rval_thr,
                                     use_cnn=use_cnn,
                                     min_cnn_thr=min_cnn_thr)

        self.idx_components, self.idx_components_bad, self.cnn_preds = \
        select_components_from_metrics(self.A, dims, self.gSig, self.r_values,
                                       self.SNR_comp, r_values_min=rval_thr,
                                       r_values_lowest=rval_lowest,
                                       min_SNR=min_SNR,
                                       min_SNR_reject=SNR_lowest,
                                       thresh_cnn_min=min_cnn_thr,
                                       thresh_cnn_lowest=cnn_lowest,
                                       use_cnn=use_cnn, gSig_range=gSig_range,
                                       predictions=self.cnn_preds)

        return self

    def play_movie(self, imgs, q_max=99.75, q_min=2, gain_res=1,
                   magnification=1, include_bck=True,
                   frame_range=slice(None, None, None)):

        """Displays a movie with three panels (original data (left panel),
        reconstructed data (middle panel), residual (right panel))
        Parameters:
        -----------
        imgs: np.array (possibly memory mapped, t,x,y[,z])
            Imaging data

        q_max: float (values in [0, 100])
            percentile for maximum plotting value

        q_min: float (values in [0, 100])
            percentile for minimum plotting value

        gain_res: float
            amplification factor for residual movie

        magnification: float
            magnification factor for whole movie

        include_bck: bool
            flag for including background in original and reconstructed movie

        frame_rage: range or slice or list
            display only a subset of frames


        Returns:
        --------
        self (to stop the movie press 'q')
        """
        dims = imgs.shape[1:]
        if 'movie' not in str(type(imgs)):
            imgs = caiman.movie(imgs)
        Y_rec = self.A.dot(self.C[:, frame_range])
        Y_rec = Y_rec.reshape(dims + (-1,), order='F')
        Y_rec = Y_rec.transpose([2, 0, 1])
        if self.gnb == -1 or self.gnb > 0:
            B = self.b.dot(self.f[:, frame_range])
            if 'matrix' in str(type(B)):
                B = B.toarray()
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        elif self.gnb == -2:
            B = self.W.dot(imgs[frame_range] - self.A.dot(self.C[:, frame_range]))
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        else:
            B = np.zeros_like(Y_rec)
        if self.border_pix > 0:
            imgs = imgs[:, self.border_pix:-self.border_pix, self.border_pix:-self.border_pix]
            B = B[:, self.border_pix:-self.border_pix, self.border_pix:-self.border_pix]
            Y_rec = Y_rec[:, self.border_pix:-self.border_pix, self.border_pix:-self.border_pix]

        Y_res = imgs[frame_range] - Y_rec - B

        
        caiman.concatenate((imgs[frame_range] - (not include_bck)*B, Y_rec + include_bck*B, Y_res*gain_res), axis=2).play(q_min=q_min, q_max=q_max, magnification=magnification)

        return self

    def HALS4traces(self, Yr, groups=None, use_groups=False, order=None,
                    update_bck=True, bck_non_neg=True, **kwargs):
        """Solves C, f = argmin_C ||Yr-AC-bf|| using block-coordinate decent.
        Can use groups to update non-overlapping components in parallel or a
        specified order.

        Parameters
        ----------
        Yr : np.array (possibly memory mapped, (x,y,[,z]) x t)
            Imaging data reshaped in matrix format

        groups : list of sets
            grouped components to be updated simultaneously

        use_groups : bool
            flag for using groups

        order : list
            Update components in that order (used if nonempty and groups=None)

        update_bck : bool
            Flag for updating temporal background components

        bck_non_neg : bool
            Require temporal background to be non-negative

        Output:
        -------
        self (updated values for self.C, self.f, self.YrA)
        """
        if update_bck:
            Ab = scipy.sparse.hstack([self.b, self.A]).tocsc()
            try:
                Cf = np.vstack([self.f, self.C + self.YrA])
            except():
                Cf = np.vstack([self.f, self.C])
        else:
            Ab = self.A
            try:
                Cf = self.C + self.YrA
            except():
                Cf = self.C
            Yr = Yr - self.b.dot(self.f)
        if (groups is None) and use_groups:
            groups = list(map(list, update_order(Ab)[0]))
        self.groups = groups
        C, noisyC = HALS4activity(Yr, Ab, Cf, groups=self.groups, order=order,
                                  **kwargs)
        if update_bck:
            if bck_non_neg:
                self.f = C[:self.gnb]
            else:
                self.f = noisyC[:self.gnb]
            self.C = C[self.gnb:]
            self.YrA = noisyC[self.gnb:] - self.C
        else:
            self.C = C
            self.YrA = noisyC - self.C
        return self

    def HALS4footprints(self, Yr, update_bck=True, num_iter=2):
        """Uses hierarchical alternating least squares to update shapes and
        background
        Parameters:
        -----------
        Yr: np.array (possibly memory mapped, (x,y,[,z]) x t)
            Imaging data reshaped in matrix format

        update_bck: bool
            flag for updating spatial background components

        num_iter: int
            number of iterations

        Returns:
        --------
        self (updated values for self.A and self.b)
        """
        if update_bck:
            Ab = np.hstack([self.b, self.A.toarray()])
            try:
                Cf = np.vstack([self.f, self.C + self.YrA])
            except():
                Cf = np.vstack([self.f, self.C])
        else:
            Ab = self.A.toarray()
            try:
                Cf = self.C + self.YrA
            except():
                Cf = self.C
            Yr = Yr - self.b.dot(self.f)
        Ab = HALS4shapes(Yr, Ab, Cf, iters=num_iter)
        if update_bck:
            self.A = scipy.sparse.csc_matrix(Ab[:, self.gnb:])
            self.b = Ab[:, :self.gnb]
        else:
            self.A = scipy.sparse.csc_matrix(Ab)

        return self

    def update_temporal(self, Y, use_init=True, **kwargs):
        """Updates temporal components

        Parameters:
        -----------
        Y:  np.array (d1*d2) x T
            input data
        
        """
        lc = locals()
        pr = inspect.signature(self.update_temporal)
        params = [k for k, v in pr.parameters.items() if '=' in str(v)]
        kw2 = {k: lc[k] for k in params}
        try:
            kwargs_new = {**kw2, **kwargs}
        except():  # python 2.7
            kwargs_new = kw2.copy()
            kwargs_new.update(kwargs)
        self.update_options('temporal', kwargs_new)
        Cin = self.Cin if use_init else self.C
        f_in = self.f_in if use_init else self.f
        self.C, self.A, self.b, self.f, self.S, \
        self.bl, self.c1, self.neurons_sn, \
        self.g, self.YrA, self.lam = update_temporal_components(
                Y, self.A, self.b, Cin, f_in, dview=self.dview,
                **self.params.temporal)
        return self
    
    def update_spatial(self, Y, use_init=True, **kwargs):
        """Updates spatial components

        Parameters:
        -----------
        Y:  np.array (d1*d2) x T
            input data
        use_init: bool
            use Cin, f_in for computing A, b otherwise use C, f
            
        Returns:
        ---------
        self
            modified values self.A, self.b possibly self.C, self.f
        """
        lc = locals()
        pr = inspect.signature(self.update_spatial)
        params = [k for k, v in pr.parameters.items() if '=' in str(v)]
        kw2 = {k: lc[k] for k in params}
        try:
            kwargs_new = {**kw2, **kwargs}
        except():  # python 2.7
            kwargs_new = kw2.copy()
            kwargs_new.update(kwargs)
        self.update_options('spatial', kwargs_new)
        for key in kwargs_new:
            if hasattr(self, key):
                setattr(self, key, kwargs_new[key])
        C = self.Cin if use_init else self.C
        f = self.f_in if use_init else self.f
        Ain = self.Ain if use_init else self.A
        b_in = self.b_in if use_init else self.b_in
        self.A, self.b, C, f =\
            update_spatial_components(Y, C=C, f=f, A_in=Ain, b_in=b_in, dview=self.dview,
                                      sn=self.sn, **self.params.spatial)
        if use_init:
            self.Cin, self.f_in = C, f
        else:
            self.C, self.f = C, f
        return self

    def merge_comps(self, Y, thr=None, mx=50, fast_merge=True):
        """merges components
        """
        if thr is not None:
            self.merge_thresh = thr
        self.A, self.C, self.nr, self.merged_ROIs, self.S,\
        self.bl, self.c, self.neurons_sn, self.g =\
            merge_components(Y, self.A, self.b, self.C, self.f, self.S,
                             self.sn, self.params.temporal,
                             self.params.spatial, dview=self.dview,
                             bl=self.bl, c1=self.c1, sn=self.neurons_sn, 
                             g=self.g, thr=self.merge_thresh, mx=mx,
                             fast_merge=fast_merge)
            
        return self
    
    def initialize(self, Y, sn, **kwargs):
        """Component initialization
        """
        self.update_options('init', kwargs)
        if self.center_psf:
            self.Ain, self.Cin, self.b_in, self.f_in, self.center,\
            self.extra_1p = initialize_components(
                Y, sn=sn, options_total=self.params.to_dict(),
                **self.params.init)
        else:
            self.Ain, self.Cin, self.b_in, self.f_in, self.center =\
            initialize_components(Y, sn=sn, options_total=self.params.to_dict(),
                                  **self.params.init)
        
        return self
    
    def update_options(self, subdict, kwargs):
        """modifies a specified subdictionary in self.params. If a specified
        parameter does not exist it gets created.
        Parameters:
        -----------
        subdict: string
            Name of subdictionary ('patch', 'preprocess',
                                   'init', 'spatial', 'merging',
                                   'temporal', 'quality', 'online')

        kwargs: dict
            Dictionary with parameters to be modified

        Returns:
        --------
        self (updated values for self.A and self.b)
        """
        if hasattr(self.params, subdict):
            d = getattr(self.params, subdict)
            for key in kwargs:
                if key not in d:
                    logging.warning("The key %s you provided does not exist! Adding it anyway..", key)
                else:
                    d[key] = kwargs[key]
        else:
            logging.warning("The subdictionary you provided does not exist!")
        return self

    def fit_online(self, fls, init_batch=200, epochs=1, motion_correct=True,
                   thresh_fitness_raw=None, **kwargs):
        """Implements the caiman online algorithm on the list of files fls. The
        files are taken in alpha numerical order and are assumed to each have
        the same number of frames (except the last one that can be shorter).
        Caiman online is initialized using the seeded or bare initialization
        methods.
        Parameters:
        -----------
        fls: list
            list of files to be processed

        init_batch: int
            number of frames to be processed during initialization

        epochs: int
            number of passes over the data

        motion_correct: bool
            flag for performing motion correction

        thresh_fitness_raw: float
            threshold for trace SNR (leave to None for automatic computation)

        kwargs: dict
            additional parameters used to modify self.params.online']
            see options.['online'] for details

        Returns:
        --------
        self (results of caiman online)
        """
        lc = locals()
        pr = inspect.signature(self.fit_online)
        params = [k for k, v in pr.parameters.items() if '=' in str(v)]
        kw2 = {k: lc[k] for k in params}
        try:
            kwargs_new = {**kw2, **kwargs}
        except():  # python 2.7
            kwargs_new = kw2.copy()
            kwargs_new.update(kwargs)
        self.update_options('online', kwargs_new)
        for key in kwargs_new:
            if hasattr(self, key):
                setattr(self, key, kwargs_new[key])
        if thresh_fitness_raw is None:
            thresh_fitness_raw = scipy.special.log_ndtr(
                    -self.params.online['min_SNR']) *\
                    self.params.online['N_samples_exceptionality']
            self.thresh_fitness_raw = thresh_fitness_raw
            self.params.online['thresh_fitness_raw'] = thresh_fitness_raw
        if isinstance(fls, str):
            fls = [fls]
        Y = caiman.load(fls[0], subindices=slice(0, init_batch,
                        None)).astype(np.float32)
        ds_factor = np.maximum(self.params.online['ds_factor'], 1)
        if ds_factor > 1:
            Y.resize(1./ds_factor)
        mc_flag = self.params.online['motion_correct']
        shifts = []  # store motion shifts here
        time_new_comp = []
        if mc_flag:
            max_shifts = self.params.online['max_shifts']
            mc = Y.motion_correct(max_shifts, max_shifts)
            Y = mc[0].astype(np.float32)
            shifts.extend(mc[1])

        img_min = Y.min()
        Y -= img_min
        img_norm = np.std(Y, axis=0)
        img_norm += np.median(img_norm)  # normalize data to equalize the FOV
        Y = Y/img_norm[None, :, :]

        _, d1, d2 = Y.shape
        Yr = Y.to_2D().T        # convert data into 2D array

        if self.params.online['init_method'] == 'bare':
            self.A, self.b, self.C, self.f, self.YrA = bare_initialization(
                    Y.transpose(1, 2, 0), gnb=self.gnb, k=self.k,
                    gSig=self.gSig, return_object=False)
            self.S = np.zeros_like(self.C)
            nr = self.C.shape[0]
            self.g = np.array([-np.poly([0.9] * max(self.p, 1))[1:]
                               for gg in np.ones(nr)])
            self.bl = np.zeros(nr)
            self.c1 = np.zeros(nr)
            self.neurons_sn = np.std(self.YrA, axis=-1)
            self.lam = np.zeros(nr)
        elif self.params.online['init_method'] == 'cnmf':
            self.rf = None
            self.dview = None
            self.fit(np.array(Y))
        elif self.params.online['init_method'] == 'seeded':
            self.A, self.b, self.C, self.f, self.YrA = seeded_initialization(
                    Y.transpose(1, 2, 0), self.Ain, gnb=self.gnb, k=self.k,
                    gSig=self.gSig, return_object=False)
            self.S = np.zeros_like(self.C)
            nr = self.C.shape[0]
            self.g = np.array([-np.poly([0.9] * max(self.p, 1))[1:]
                               for gg in np.ones(nr)])
            self.bl = np.zeros(nr)
            self.c1 = np.zeros(nr)
            self.neurons_sn = np.std(self.YrA, axis=-1)
            self.lam = np.zeros(nr)
        else:
            raise Exception('Unknown initialization method!')
        self.dims = Y.shape[1:]
        self.initbatch = init_batch
        epochs = self.params.online['epochs']
        T1 = caiman.load(fls[0]).shape[0]*len(fls)*epochs
        self._prepare_object(Yr, T1, **self.params.online)
        extra_files = len(fls) - 1
        init_files = 1
        t = init_batch
        self.Ab_epoch = []
        if extra_files == 0:     # check whether there are any additional files
            process_files = fls[:init_files]     # end processing at this file
            init_batc_iter = [init_batch]         # place where to start
        else:
            process_files = fls[:init_files + extra_files]   # additional files
            # where to start reading at each file
            init_batc_iter = [init_batch] + [0]*extra_files
        for iter in range(epochs):
            if iter > 0:
                # if not on first epoch process all files from scratch
                process_files = fls[:init_files + extra_files]
                init_batc_iter = [0] * (extra_files + init_files)

            for file_count, ffll in enumerate(process_files):
                print('Now processing file ' + ffll)
                Y_ = caiman.load(ffll, subindices=slice(
                                    init_batc_iter[file_count], None, None))

                old_comps = self.N     # number of existing components
                for frame_count, frame in enumerate(Y_):   # process each file
                    if np.isnan(np.sum(frame)):
                        raise Exception('Frame ' + str(frame_count) +
                                        ' contains NaN')
                    if t % 100 == 0:
                        print('Epoch: ' + str(iter + 1) + '. ' + str(t) +
                              ' frames have beeen processed in total. ' +
                              str(self.N - old_comps) +
                              ' new components were added. Total # of components is '
                              + str(self.Ab.shape[-1] - self.gnb))
                        old_comps = self.N

                    frame_ = frame.copy().astype(np.float32)
                    if ds_factor > 1:
                        frame_ = cv2.resize(frame_, img_norm.shape[::-1])
                    frame_ -= img_min     # make data non-negative

                    if mc_flag:    # motion correct
                        templ = self.Ab.dot(
                            self.C_on[:self.M, t-1]).reshape(self.dims, order='F')*img_norm
                        frame_cor, shift = motion_correct_iteration_fast(
                            frame_, templ, max_shifts, max_shifts)
                        shifts.append(shift)
                    else:
                        templ = None
                        frame_cor = frame_

                    frame_cor = frame_cor/img_norm    # normalize data-frame
                    self.fit_next(t, frame_cor.reshape(-1, order='F'))
                    t += 1
            self.Ab_epoch.append(self.Ab.copy())
        self.A, self.b = self.Ab[:, self.gnb:], self.Ab[:, :self.gnb].toarray()
        self.C, self.f = self.C_on[self.gnb:self.M, t - t //
                         epochs:t], self.C_on[:self.gnb, t - t // epochs:t]
        noisyC = self.noisyC[self.gnb:self.M, t - t // epochs:t]
        self.YrA = noisyC - self.C
        self.bl = [osi.b for osi in self.OASISinstances] if hasattr(
            self, 'OASISinstances') else [0] * self.C.shape[0]
        self.shifts = shifts
        
        return self
