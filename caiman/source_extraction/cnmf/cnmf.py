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
from .utilities import CNMFSetParms, update_order, normalize_AC, compute_residuals
from .pre_processing import preprocess_data
from .initialization import initialize_components, imblur
from .merging import merge_components
from .spatial import update_spatial_components
from .temporal import update_temporal_components
from .map_reduce import run_CNMF_patches
from .oasis import OASIS
import caiman
from caiman import components_evaluation, mmapping
import cv2
from .online_cnmf import RingBuffer, HALS4activity, demix_and_deconvolve
from .online_cnmf import init_shapes_and_sufficient_stats, update_shapes, update_num_components
import scipy
import psutil
import pylab as pl
from time import time

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
                 batch_update_suff_stat=False, thresh_s_min=None, s_min=None,
                 remove_very_bad_comps=False, border_pix=0, low_rank_background=True,
                 update_background_components=True, rolling_sum=True, rolling_length=100,
                 min_corr=.85, min_pnr=20, deconvolve_options_init=None, ring_size_factor=1.5,
                 center_psf=False, use_dense=True, deconv_flag=True,
                 simultaneously=False, n_refit=0, del_duplicates=False, N_samples_exceptionality=5,
                 max_num_added=1, min_num_trial=2, thresh_CNN_noisy=0.99, 
                 ssub_B=2, compute_B_3x=False, init_iter=2):
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

         deconvolve_options: dict
            all options for deconvolving temporal traces, in general just pass options['temporal_params']    

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

        compute_B_3x: bool, optional=False, 
            whether to compute background 3x or only 2x for 1-photon imaging

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
        self.thresh_s_min = thresh_s_min
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

        self.min_corr = min_corr
        self.min_pnr = min_pnr
        self.deconvolve_options_init = deconvolve_options_init
        self.ring_size_factor = ring_size_factor
        self.center_psf = center_psf
        self.nb_patch = nb_patch
        self.del_duplicates = del_duplicates

        self.options = CNMFSetParms((1, 1, 1), n_processes, p=p, gSig=gSig, gSiz=gSiz,
                                    K=k, ssub=ssub, tsub=tsub,
                                    p_ssub=p_ssub, p_tsub=p_tsub, method_init=method_init,
                                    n_pixels_per_process=n_pixels_per_process,
                                    check_nan=check_nan, nb=gnb,
                                    nb_patch=nb_patch, normalize_init=normalize_init,
                                    options_local_NMF=options_local_NMF,
                                    remove_very_bad_comps=remove_very_bad_comps,
                                    low_rank_background=low_rank_background,
                                    update_background_components=update_background_components, rolling_sum=self.rolling_sum,
                                    min_corr=min_corr, min_pnr=min_pnr, deconvolve_options_init=deconvolve_options_init,
                                    ring_size_factor=ring_size_factor, center_psf=center_psf,
                                    ssub_B=ssub_B, compute_B_3x=compute_B_3x, init_iter=init_iter)

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
        print((T,) + dims)

        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        try:
            Y.filename = images.filename
            Yr.filename = images.filename
        except AttributeError:  # if no memmapping cause working with small data
            pass

        # update/set all options that depend on data dimensions
        # number of rows, columns [and depths]
        self.options['spatial_params']['dims'] = dims
        self.options['spatial_params']['medw'] = (
            3,) * len(dims)  # window of median filter
        # Morphological closing structuring element
        self.options['spatial_params']['se'] = np.ones(
            (3,) * len(dims), dtype=np.uint8)
        # Binary element for determining connectivity
        self.options['spatial_params']['ss'] = np.ones(
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
        self.options['preprocess_params']['n_pixels_per_process'] = self.n_pixels_per_process
        self.options['spatial_params']['n_pixels_per_process'] = self.n_pixels_per_process

#        if self.block_size is None:
#            self.block_size = self.n_pixels_per_process
#
#        if self.num_blocks_per_run is None:
#           self.num_blocks_per_run = 20

        # number of pixels to process at the same time for dot product. Make it
        # smaller if memory problems
        self.options['temporal_params']['block_size'] = self.block_size
        self.options['temporal_params']['num_blocks_per_run'] = self.num_blocks_per_run
        self.options['spatial_params']['block_size'] = self.block_size
        self.options['spatial_params']['num_blocks_per_run'] = self.num_blocks_per_run

        print(('using ' + str(self.n_pixels_per_process) + ' pixels per process'))
        print(('using ' + str(self.block_size) + ' block_size'))

        options = self.options

        if self.rf is None:  # no patches
            print('preprocessing ...')
            Yr, sn, g, psx = preprocess_data(
                Yr, dview=self.dview, **options['preprocess_params'])

            if self.Ain is None:
                print('initializing ...')
                if self.alpha_snmf is not None:
                    options['init_params']['alpha_snmf'] = self.alpha_snmf

                self.Ain, self.Cin, self.b_in, self.f_in, center, extra_1p = initialize_components(
                    Y, sn=sn, options_total=options, **options['init_params'])
                
            if self.only_init:  # only return values after initialization

                if self.ring_size_factor is None:
                    self.YrA = compute_residuals(
                        Yr, self.Ain, self.b_in, self.Cin, self.f_in,
                        dview=self.dview, block_size=1000, num_blocks_per_run=5)
                    self.g = g
                    self.bl = None
                    self.c1 = None
                    self.neurons_sn = None
                else:
                    self.S, self.bl, self.c1, self.neurons_sn, self.g, self.YrA = extra_1p

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

                self.A, self.C, self.YrA, self.b, self.f = normalize_AC(
                    self.A, self.C, self.YrA, self.b, self.f)
                return self

            print('update spatial ...')
            A, b, Cin, self.f_in = update_spatial_components(Yr, C=self.Cin, f=self.f_in, b_in=self.b_in, A_in=self.Ain,
                                                             sn=sn, dview=self.dview, **options['spatial_params'])

            print('update temporal ...')
            if not self.skip_refinement:
                # set this to zero for fast updating without deconvolution
                options['temporal_params']['p'] = 0
            else:
                options['temporal_params']['p'] = self.p
            print('deconvolution ...')
            options['temporal_params']['method'] = self.method_deconvolution

            C, A, b, f, S, bl, c1, neurons_sn, g, YrA, lam = update_temporal_components(
                Yr, A, b, Cin, self.f_in, dview=self.dview, **options['temporal_params'])

            if not self.skip_refinement:
                print('refinement...')
                if self.do_merge:
                    print('merge components ...')
                    A, C, nr, merged_ROIs, S, bl, c1, sn1, g1 = merge_components(
                        Yr, A, b, C, f, S, sn, options[
                            'temporal_params'], options['spatial_params'],
                        dview=self.dview, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=self.merge_thresh,
                        mx=50, fast_merge=True)
                print((A.shape))
                print('update spatial ...')
                A, b, C, f = update_spatial_components(
                    Yr, C=C, f=f, A_in=A, sn=sn, b_in=b, dview=self.dview, **options['spatial_params'])
                # set it back to original value to perform full deconvolution
                options['temporal_params']['p'] = self.p
                print('update temporal ...')
                C, A, b, f, S, bl, c1, neurons_sn, g1, YrA, lam = update_temporal_components(
                    Yr, A, b, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
            else:
                g1 = g
                # todo : ask for those..
                C, f, S, bl, c1, neurons_sn, g1, YrA = C, f, S, bl, c1, neurons_sn, g, YrA

        else:  # use patches
            if self.stride is None:
                self.stride = np.int(self.rf * 2 * .1)
                print(
                    ('**** Setting the stride to 10% of 2*rf automatically:' + str(self.stride)))

            if type(images) is np.ndarray:
                raise Exception(
                    'You need to provide a memory mapped file as input if you use patches!!')

            if self.only_init:
                options['patch_params']['only_init'] = True

            if self.alpha_snmf is not None:
                options['init_params']['alpha_snmf'] = self.alpha_snmf

            A, C, YrA, b, f, sn, optional_outputs = run_CNMF_patches(images.filename, dims + (T,),
                                                                     options, rf=self.rf, stride=self.stride,
                                                                     dview=self.dview, memory_fact=self.memory_fact,
                                                                     gnb=self.gnb, border_pix=self.border_pix,
                                                                     low_rank_background=self.low_rank_background,
                                                                     del_duplicates=self.del_duplicates)

            # options = CNMFSetParms(Y, self.n_processes, p=self.p, gSig=self.gSig, K=A.shape[
            #                        -1], thr=self.merge_thresh, n_pixels_per_process=self.n_pixels_per_process,
            #                        block_size=self.block_size, check_nan=self.check_nan)

            # options['temporal_params']['method'] = self.method_deconvolution


#            options['spatial_params']['se'] = np.ones((1,) * len(dims), dtype=np.uint8)
#            options['spatial_params']['update_background_components'] = True
#            print('update spatial ...')
#            A, b, C, f = update_spatial_components(
#                    Yr, C = C, f = f, A_in = A, sn=sn, b_in = b, dview=self.dview, **options['spatial_params'])

            if self.center_psf:  # merge taking best neuron
                print("merging")
                merged_ROIs = [0]
                while len(merged_ROIs) > 0:
                    A, C, nr, merged_ROIs, S, bl, c1, sn_n, g = merge_components(Yr, A, [], np.array(C), [], np.array(
                        C), [], options['temporal_params'], options['spatial_params'], dview=self.dview,
                        thr=self.merge_thresh, mx=np.Inf, fast_merge=True)

                print("update temporal")
                C, A, b, f, S, bl, c1, neurons_sn, g1, YrA, lam = update_temporal_components(
                    Yr, A, b, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

                options['spatial_params']['se'] = np.ones(
                    (1,) * len(dims), dtype=np.uint8)
#                options['spatial_params']['update_background_components'] = True
                print('update spatial ...')
                A, b, C, f = update_spatial_components(
                    Yr, C=C, f=f, A_in=A, sn=sn, b_in=b, dview=self.dview, **options['spatial_params'])

                print("update temporal")
                C, A, b, f, S, bl, c1, neurons_sn, g1, YrA, lam = update_temporal_components(
                    Yr, A, b, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

            else:

                print("merging")
                merged_ROIs = [0]
                while len(merged_ROIs) > 0:
                    A, C, nr, merged_ROIs, S, bl, c1, sn_n, g = merge_components(Yr, A, [], np.array(C), [], np.array(
                        C), [], options['temporal_params'], options['spatial_params'], dview=self.dview,
                        thr=self.merge_thresh, mx=np.Inf)

                print("update temporal")
                C, A, b, f, S, bl, c1, neurons_sn, g1, YrA, lam = update_temporal_components(
                    Yr, A, b, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

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

        self.A, self.C, self.YrA, self.b, self.f = normalize_AC(
            self.A, self.C, self.YrA, self.b, self.f)

        return self

    def _prepare_object(self, Yr, T, expected_comps, new_dims=None, idx_components=None,
                        g=None, lam=None, s_min=None, bl=None, use_dense=True, N_samples_exceptionality=5,
                        max_num_added=1, min_num_trial=1, path_to_model = None,
                        sniper_mode = False):

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

        if self.p:
            # if no parameter for calculating the spike size threshold is given, then use L1 penalty
            if s_min is None and self.s_min is None and self.thresh_s_min is None:
                use_L1 = True
            else:
                use_L1 = False

            self.OASISinstances = [OASIS(
                g=np.ravel(0.01) if self.p == 0 else (
                    np.ravel(g)[0] if g is not None else gam[0]),
                lam=0 if not use_L1 else (l if lam is None else lam),
                # if no explicit value for s_min,  use thresh_s_min * noise estimate * sqrt(1-gamma)
                s_min=0 if use_L1 else (s_min if s_min is not None else
                                        (self.s_min if self.s_min is not None else
                                         (self.thresh_s_min * sn * np.sqrt(1 - np.sum(gam))))),
                b=b if bl is None else bl,
                g2=0 if self.p < 2 else (np.ravel(g)[1] if g is not None else gam[1]))
                for gam, l, b, sn in zip(self.g2, self.lam2, self.bl2, self.neurons_sn2)]

            for i, o in enumerate(self.OASISinstances):
                o.fit(self.noisyC[i + self.gnb, :self.initbatch])
                self.C_on[i, :self.initbatch] = o.c
        else:
            self.C_on[:self.N, :self.initbatch] = self.C2

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

        self.gSiz = np.add(np.multiply(self.gSig, 2), 1)

        self.Yr_buf = RingBuffer(Yr[:, self.initbatch - self.minibatch_shape:
                                    self.initbatch].T.copy(), self.minibatch_shape)
        self.Yres_buf = RingBuffer(self.Yr_buf - self.Ab.dot(
            self.C_on[:self.M, self.initbatch - self.minibatch_shape:self.initbatch]).T, self.minibatch_shape)
        self.rho_buf = imblur(self.Yres_buf.T.reshape(
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
        import keras
        from keras.models import model_from_json
        
        # prepare CNN
        
        path = path_to_model.split(".")[:-1]
        json_path = ".".join(path + ["json"])
        model_path = ".".join(path + ["h5"])
        try:
            json_file = open(json_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_path)
            opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
            loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=opt, metrics=['accuracy'])   
        except:
            print('No model found')
            loaded_model = None
            sniper_mode = False
                
        self.loaded_model = loaded_model
        self.sniper_mode = sniper_mode
        return self

    @profile
    def fit_next(self, t, frame_in, num_iters_hals=3):
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


        """

        t_start = time()

        # locally scoped variables for brevity of code and faster look up
        nb_ = self.gnb
        Ab_ = self.Ab
        mbs = self.minibatch_shape
        frame = frame_in.astype(np.float32)
#        print(np.max(1/scipy.sparse.linalg.norm(self.Ab,axis = 0)))
        self.Yr_buf.append(frame)

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

            rho = imblur(res_frame, sig=self.gSig,
                         siz=self.gSiz, nDimBlur=len(self.dims2))**2
            rho = np.reshape(rho, np.prod(self.dims2))
            self.rho_buf.append(rho)

            self.Ab, Cf_temp, self.Yres_buf, self.rhos_buf, self.CC, self.CY, self.ind_A, self.sv, self.groups, self.ind_new = update_num_components(
                t, self.sv, self.Ab, self.C_on[:self.M, (t - mbs + 1):(t + 1)],
                self.Yres_buf, self.Yr_buf, self.rho_buf, self.dims2,
                self.gSig, self.gSiz, self.ind_A, self.CY, self.CC, rval_thr=self.rval_thr,
                thresh_fitness_delta=self.thresh_fitness_delta,
                thresh_fitness_raw=self.thresh_fitness_raw, thresh_overlap=self.thresh_overlap,
                groups=self.groups, batch_update_suff_stat=self.batch_update_suff_stat, gnb=self.gnb,
                sn=self.sn, g=np.mean(
                    self.g2) if self.p == 1 else np.mean(self.g2, 0),
                thresh_s_min=self.thresh_s_min, s_min=self.s_min,
                Ab_dense=self.Ab_dense[:, :self.M] if self.use_dense else None,
                oases=self.OASISinstances if self.p else None, N_samples_exceptionality=self.N_samples_exceptionality,
                max_num_added=self.max_num_added, min_num_trial=self.min_num_trial,
                loaded_model = self.loaded_model, thresh_CNN_noisy = self.thresh_CNN_noisy,
                sniper_mode = self.sniper_mode)

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
            y = self.Yr_buf.get_last_frames(self.minibatch_suff_stat)[:1]
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
                        indicator_components, self.Ab_dense[:, :self.M])
                else:
                    Ab_, self.ind_A, _ = update_shapes(self.CY, self.CC, Ab_, self.ind_A,
                                                       indicator_components=indicator_components)

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

    def view_patches(self, Yr, dims, img=None):
        """view spatial and temporal components interactively

         Parameters:
         -----------
         Yr :    np.ndarray
                 movie in format pixels (d) x frames (T)

         dims :  tuple
                 dimensions of the FOV

         img :   np.ndarray
                 background image for contour plotting. Default is the mean image of all spatial components (d1 x d2)

        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if 'array' not in str(type(self.b)):
            self.b = self.b.toarray()

        pl.ion()
        nr, T = self.C.shape
        #nb = self.f.shape[0]

        if self.YrA is None:
            self.compute_residuals(Yr)

        if img is None:
            img = np.reshape(np.array(self.A.mean(axis=1)), dims, order='F')

        caiman.utils.visualization.view_patches_bar(Yr, self.A, self.C, self.b, self.f, dims[
                                                    0], dims[1], YrA=self.YrA, img=img)


def scale(y):
    return (y - np.mean(y)) / (np.max(y) - np.min(y))
