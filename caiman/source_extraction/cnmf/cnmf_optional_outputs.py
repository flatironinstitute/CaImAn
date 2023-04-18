#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Constrained Nonnegative Matrix Factorization

A similar CNMF class that will output data that are not usually present in the regular CNMF process

Created on Fri Aug 26 15:44:32 2016

@author: agiovann


"""

import numpy as np
from .utilities import local_correlations, order_components, evaluate_components
from caiman.source_extraction.cnmf.params import CNMFParams
from .pre_processing import preprocess_data
from .initialization import initialize_components
from .merging import merge_components
from .spatial import update_spatial_components
from .temporal import update_temporal_components
from .map_reduce import run_CNMF_patches


class CNMF(object):
    """
    Source extraction using constrained non-negative matrix factorization.
    """

    def __init__(self, n_processes, k=5, gSig=[4, 4], merge_thresh=0.8, p=2, dview=None, Ain=None, Cin=None,
                 f_in=None, do_merge=True,
                 ssub=2, tsub=2, p_ssub=1, p_tsub=1, method_init='greedy_roi', alpha_snmf=None,
                 rf=None, stride=None, memory_fact=1,
                 N_samples_fitness=5, robust_std=False, fitness_threshold=-10, corr_threshold=0,
                 only_init_patch=False):
        """ 
        Constructor of the CNMF method

        A similar CNMF class that will output data that are not usually present in the regular CNMF process

        Args:
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

            stride: int
                amount of overlap between the patches in pixels

            memory_fact: float
                unitless number accounting how much memory should be used. You will need to try
                 different values to see which one would work the default is OK for a 16 GB system

            N_samples_fitness: int 
                number of samples over which exceptional events are computed (See utilities.evaluate_components) 

            robust_std: bool
                whether to use robust std estimation for fitness (See utilities.evaluate_components)        

            fitness_threshold: float
                fitness threshold to decide which components to keep. The lower the stricter
                 is the inclusion criteria (See utilities.evaluate_components)

        Returns:
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
        # unitless number accounting how much memory should be used. You will need to try different
        #  values to see which one would work the default is OK for a 16 GB system
        self.memory_fact = memory_fact
        self.N_samples_fitness = N_samples_fitness
        self.robust_std = robust_std
        self.fitness_threshold = fitness_threshold
        self.corr_threshold = corr_threshold
        self.do_merge = do_merge
        self.alpha_snmf = alpha_snmf
        self.only_init = only_init_patch

        self.A = None
        self.C = None
        self.S = None
        self.b = None
        self.f = None

    def fit(self, images):
        """
        This method uses the cnmf algorithm to find sources in data.

        Args:
            images : mapped np.ndarray of shape (t,x,y) containing the images that vary over time.

        Returns:
            self 
        """

        T, d1, d2 = images.shape
        dims = (d1, d2)
        Yr = images.reshape([T, np.prod(dims)], order='F').T
        Y = np.transpose(images, [1, 2, 0])
        print((T, d1, d2))

        options = CNMFParams(dims, K=self.k, gSig=self.gSig, ssub=self.ssub, tsub=self.tsub, p=self.p,
                             p_ssub=self.p_ssub, p_tsub=self.p_tsub, method_init=self.method_init, normalize_init=True)

        self.options = options

        if self.rf is None:

            Yr, sn, g, psx = preprocess_data(
                Yr, dview=self.dview, **options['preprocess_params'])

            if self.Ain is None:
                if self.alpha_snmf is not None:
                    options['init_params']['alpha_snmf'] = self.alpha_snmf

                self.Ain, self.Cin, self.b_in, self.f_in, center = initialize_components(
                    Y, **options['init_params'])

            A, b, Cin, self.f_in = update_spatial_components(Yr, self.Cin, self.f_in, self.Ain, sn=sn,
                                                             dview=self.dview, **options['spatial_params'])

            # set this to zero for fast updating without deconvolution
            options['temporal_params']['p'] = 0

            C, A, b, f, S, bl, c1, neurons_sn, g, YrA = update_temporal_components(Yr, A, b, Cin, self.f_in,
                                                                                   dview=self.dview, **options['temporal_params'])

            if self.do_merge:
                A, C, nr, merged_ROIs, S, bl, c1, sn1, g1 = merge_components(Yr, A, b, C, f, S, sn, options['temporal_params'],
                                                                             options['spatial_params'], dview=self.dview, bl=bl, c1=c1, sn=neurons_sn, g=g,
                                                                             thr=self.merge_thresh, mx=50, fast_merge=True)

            print((A.shape))

            A, b, C, f = update_spatial_components(
                Yr, C, f, A, sn=sn, dview=self.dview, dims=self.dims,  **options['spatial_params'])
            # set it back to original value to perform full deconvolution
            options['temporal_params']['p'] = self.p

            C, A, b, f, S, bl, c1, neurons_sn, g1, YrA = update_temporal_components(Yr, A, b, C, f,
                                                                                    dview=self.dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

        else:  # use patches
            if self.stride is None:
                self.stride = int(self.rf * 2 * .1)
                print(
                    ('**** Setting the stride to 10% of 2*rf automatically:' + str(self.stride)))

            if isinstance(images, np.ndarray):
                raise Exception(
                    'You must provide a memory mapped file as input if you use patches!!')

            if self.only_init:
                options['patch_params']['only_init'] = True

            A, C, YrA, b, f, sn, optional_outputs = run_CNMF_patches(images.filename, (d1, d2, T), options, rf=self.rf,
                                                                     stride=self.stride,
                                                                     dview=self.dview, memory_fact=self.memory_fact)

            self.optional_outputs = optional_outputs

            options = CNMFParams(dims, K=A.shape[-1], gSig=self.gSig, p=self.p, thr=self.merge_thresh)
            pix_proc = np.minimum(int((d1 * d2) / self.n_processes / (
                (T / 2000.))), int((d1 * d2) // self.n_processes))  # regulates the amount of memory used

            options['spatial_params']['n_pixels_per_process'] = pix_proc
            options['temporal_params']['n_pixels_per_process'] = pix_proc
            merged_ROIs = [0]
            self.merged_ROIs = []
            while len(merged_ROIs) > 0:
                A, C, nr, merged_ROIs, S, bl, c1, sn, g = merge_components(Yr, A, [],
                                                                           np.array(C), [], np.array(
                                                                               C), [], options['temporal_params'], options['spatial_params'],
                                                                           dview=self.dview, thr=self.merge_thresh, mx=np.Inf)

                self.merged_ROIs.append(merged_ROIs)

            C, A, b, f, S, bl, c1, neurons_sn, g2, YrA = update_temporal_components(
                Yr, A, np.atleast_2d(b).T, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

        self.A = A
        self.C = C
        self.b = b
        self.f = f
        self.YrA = YrA
        self.sn = sn

        return self
