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
import cv2
import scipy
import psutil
from time import time
import logging
import sys
import inspect

import caiman
from .map_reduce import run_CNMF_patches
from .oasis import OASIS
from .estimates import Estimates
from .utilities import update_order, normalize_AC, get_file_size
from .params import CNMFParams
from .pre_processing import preprocess_data
from .initialization import initialize_components, imblur, downscale
from .merging import merge_components
from .spatial import update_spatial_components
from .temporal import update_temporal_components, constrained_foopsi_parallel
from ...components_evaluation import estimate_components_quality_auto, select_components_from_metrics, estimate_components_quality
from ...motion_correction import motion_correct_iteration_fast
from ... import mmapping
from ...utils.utils import save_dict_to_hdf5, load_dict_from_hdf5
#from .online_cnmf import RingBuffer, HALS4activity, HALS4shapes, demix_and_deconvolve, remove_components_online
#from .online_cnmf import init_shapes_and_sufficient_stats, update_shapes, update_num_components, bare_initialization, seeded_initialization

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
                 thresh_fitness_raw=None, thresh_overlap=.5,
                 max_comp_update_shape=np.inf, num_times_comp_updated=np.inf,
                 batch_update_suff_stat=False, s_min=None,
                 remove_very_bad_comps=False, border_pix=0, low_rank_background=True,
                 update_background_components=True, rolling_sum=True, rolling_length=100,
                 min_corr=.85, min_pnr=20, ring_size_factor=1.5,
                 center_psf=False, use_dense=True, deconv_flag=True,
                 simultaneously=False, n_refit=0, del_duplicates=False, N_samples_exceptionality=None,
                 max_num_added=3, min_num_trial=2, thresh_CNN_noisy=0.5,
                 fr=30, decay_time=0.4, min_SNR=2.5, ssub_B=2, init_iter=2,
                 sniper_mode=False, use_peak_max=False, test_both=False,
                 expected_comps=500, params=None):
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

        self.dview = dview

        # these are movie properties that will be refactored into the Movie object
        self.dims = None

        # these are member variables related to the CNMF workflow
        self.skip_refinement = skip_refinement
        self.remove_very_bad_comps = remove_very_bad_comps
        
        if params is None:
            self.params = CNMFParams(
                border_pix=border_pix, del_duplicates=del_duplicates, low_rank_background=low_rank_background,
                memory_fact=memory_fact, n_processes=n_processes, nb_patch=nb_patch, only_init_patch=only_init_patch, p_ssub=p_ssub, p_tsub=p_tsub,
                remove_very_bad_comps=remove_very_bad_comps, rf=rf, stride=stride,
                check_nan=check_nan, n_pixels_per_process=n_pixels_per_process,
                k=k, center_psf=center_psf, gSig=gSig, gSiz=gSiz,
                init_iter=init_iter, method_init=method_init, min_corr=min_corr,  min_pnr=min_pnr,
                gnb=gnb, normalize_init=normalize_init, options_local_NMF=options_local_NMF,
                ring_size_factor=ring_size_factor, rolling_length=rolling_length, rolling_sum=rolling_sum,
                ssub=ssub, ssub_B=ssub_B, tsub=tsub,
                block_size=block_size, num_blocks_per_run=num_blocks_per_run,
                update_background_components=update_background_components,
                method_deconvolution=method_deconvolution, p=p, s_min=s_min,
                do_merge=do_merge, merge_thresh=merge_thresh,
                decay_time=decay_time, fr=fr, min_SNR=min_SNR, rval_thr=rval_thr,
                N_samples_exceptionality=N_samples_exceptionality, batch_update_suff_stat=batch_update_suff_stat,
                expected_comps=expected_comps, max_comp_update_shape=max_comp_update_shape, max_num_added=max_num_added,
                min_num_trial=min_num_trial, minibatch_shape=minibatch_shape, minibatch_suff_stat=minibatch_suff_stat,
                n_refit=n_refit, num_times_comp_updated=num_times_comp_updated, simultaneously=simultaneously,
                sniper_mode=sniper_mode, test_both=test_both, thresh_CNN_noisy=thresh_CNN_noisy,
                thresh_fitness_delta=thresh_fitness_delta, thresh_fitness_raw=thresh_fitness_raw, thresh_overlap=thresh_overlap,
                update_num_comps=update_num_comps, use_dense=use_dense, use_peak_max=use_peak_max
            )
        else:
            self.params = params
        self.estimates = Estimates(A=Ain, C=Cin, b=b_in, f=f_in,
                                   dims=self.params.data['dims'])


    
    def fit(self, images, indeces=[slice(None), slice(None)]):
        """
        This method uses the cnmf algorithm to find sources in data.

        it is calling everyfunction from the cnmf folder
        you can find out more at how the functions are called and how they are laid out at the ipython notebook

        Parameters:
        ----------
        images : mapped np.ndarray of shape (t,x,y[,z]) containing the images that vary over time.
        
        indeces: list of slice objects along dimensions (x,y[,z]) for processing only part of the FOV

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
        # Todo : to compartment
        if isinstance(indeces, slice):
            indeces = [indeces]
        indeces = [slice(None)] + indeces
        if len(indeces) < len(images.shape):
            indeces = indeces + [slice(None)]*(len(images.shape) - len(indeces))
        dims_orig = images.shape[1:]
        dims_sliced = images[indeces].shape[1:]
        is_sliced = (dims_orig != dims_sliced)
        if self.params.get('patch', 'rf') is None and (is_sliced or 'ndarray' in str(type(images))):
            images = images[indeces]
            self.dview = None
            logging.warning("Parallel processing in a single patch\
                            is not available for loaded in memory or sliced\
                            data.")
            
        T = images.shape[0]
        self.params.set('online', {'init_batch': T})
        self.dims = images.shape[1:]
        #self.params.data['dims'] = images.shape[1:]
        Y = np.transpose(images, list(range(1, len(self.dims) + 1)) + [0])
        Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
        if np.isfortran(Yr):
            raise Exception('The file is in F order, it should be in C order (see save_memmap function')

        print((T,) + self.dims)

        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        try:
            Y.filename = images.filename
            Yr.filename = images.filename
        except AttributeError:  # if no memmapping cause working with small data
            pass

        # update/set all options that depend on data dimensions
        # number of rows, columns [and depths]
        self.params.set('spatial', {'medw': (3,) * len(self.dims),
                                    'se': np.ones((3,) * len(self.dims), dtype=np.uint8),
                                    'ss': np.ones((3,) * len(self.dims), dtype=np.uint8)
                                    })

        print(('using ' + str(self.params.get('patch', 'n_processes')) + ' processes'))
        if self.params.get('preprocess', 'n_pixels_per_process') is None:
            avail_memory_per_process = psutil.virtual_memory()[
                1] / 2.**30 / self.params.get('patch', 'n_processes')
            mem_per_pix = 3.6977678498329843e-09
            npx_per_proc = np.int(avail_memory_per_process / 8. / mem_per_pix / T)
            npx_per_proc = np.int(np.minimum(npx_per_proc, np.prod(self.dims) // self.params.get('patch', 'n_processes')))
            self.params.set('preprocess', {'n_pixels_per_process': npx_per_proc})

        self.params.set('spatial', {'n_pixels_per_process': self.params.get('preprocess', 'n_pixels_per_process')})

        print(('using ' + str(self.params.get('preprocess', 'n_pixels_per_process')) + ' pixels per process'))
        print(('using ' + str(self.params.get('spatial', 'block_size')) + ' block_size'))

        if self.params.get('patch', 'rf') is None:  # no patches
            print('preprocessing ...')
            Yr = self.preprocess(Yr)
            if self.estimates.A is None:
                print('initializing ...')
                self.initialize(Y)

            if self.params.get('patch', 'only_init'):  # only return values after initialization
                if not self.params.get('init', 'center_psf'):
                    self.compute_residuals(Yr)
                    self.estimates.bl = None
                    self.estimates.c1 = None
                    self.estimates.neurons_sn = None


                if self.remove_very_bad_comps:
                    print('removing bad components : ')
                    #todo update these parameters to reflect the new version of Params
                    final_frate = 10
                    r_values_min = 0.5  # threshold on space consistency
                    fitness_min = -15  # threshold on time variability
                    fitness_delta_min = -15
                    Npeaks = 10
                    traces = np.array(self.estimates.C)
                    print('estimating the quality...')
                    idx_components, idx_components_bad, fitness_raw,\
                        fitness_delta, r_values = estimate_components_quality(
                            traces, Y, self.estimates.A, self.estimates.C, self.estimates.b, self.estimates.f,
                            final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
                            fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, return_all=True, N=5)

                    print(('Keeping ' + str(len(idx_components)) +
                           ' and discarding  ' + str(len(idx_components_bad))))
                    self.estimates.C = self.estimates.C[idx_components]
                    self.estimates.A = self.estimates.A[:, idx_components]
                    self.estimates.YrA = self.estimates.YrA[idx_components]

                self.estimates.normalize_components()

                return self

            print('update spatial ...')
            self.update_spatial(Yr, use_init=True)

            print('update temporal ...')
            if not self.skip_refinement:
                # set this to zero for fast updating without deconvolution
                self.params.set('temporal', {'p': 0})
            else:
                self.params.set('temporal', {'p': self.params.get('preprocess', 'p')})
            print('deconvolution ...')

            self.update_temporal(Yr)

            if not self.skip_refinement:
                print('refinement...')
                if self.params.get('merging', 'do_merge'):
                    print('merge components ...')
                    print(self.estimates.A.shape)
                    print(self.estimates.C.shape)
                    self.merge_comps(Yr, mx=50, fast_merge=True)

                print((self.estimates.A.shape))
                print(self.estimates.C.shape)
                print('update spatial ...')

                self.update_spatial(Yr, use_init=False)
                # set it back to original value to perform full deconvolution
                self.params.set('temporal', {'p': self.params.get('preprocess', 'p')})
                print('update temporal ...')
                self.update_temporal(Yr, use_init=False)
            # else:
            #     todo : ask for those..
                # C, f, S, bl, c1, neurons_sn, g1, YrA, lam = self.estimates.C, self.estimates.f, self.estimates.S, self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, self.estimates.g, self.estimates.YrA, self.estimates.lam

            # embed in the whole FOV
            if is_sliced:
                FOV = np.zeros(dims_orig, order='C')
                FOV[indeces[1:]] = 1
                FOV = FOV.flatten(order='F')
                ind_nz = np.where(FOV>0)[0].tolist()
                self.estimates.A = self.estimates.A.tocsc()
                A_data = self.estimates.A.data
                A_ind = np.array(ind_nz)[self.estimates.A.indices]
                A_ptr = self.estimates.A.indptr
                A_FOV = scipy.sparse.csc_matrix((A_data, A_ind, A_ptr),
                                                shape=(FOV.shape[0], self.estimates.A.shape[-1]))
                b_FOV = np.zeros((FOV.shape[0], self.estimates.b.shape[-1]))
                b_FOV[ind_nz] = self.estimates.b
                self.estimates.A = A_FOV
                self.estimates.b = b_FOV
            
        else:  # use patches
            if self.params.get('patch', 'stride') is None:
                self.params.set('patch', {'stride': np.int(self.params.get('patch', 'rf') * 2 * .1)})
                print(
                    ('**** Setting the stride to 10% of 2*rf automatically:' + str(self.params.get('patch', 'stride'))))

            if type(images) is np.ndarray:
                raise Exception(
                    'You need to provide a memory mapped file as input if you use patches!!')

            self.estimates.A, self.estimates.C, self.estimates.YrA, self.estimates.b, self.estimates.f, \
                self.estimates.sn, self.estimates.optional_outputs = run_CNMF_patches(
                    images.filename, self.dims + (T,), self.params,
                    dview=self.dview, memory_fact=self.params.get('patch', 'memory_fact'),
                    gnb=self.params.get('init', 'nb'), border_pix=self.params.get('patch', 'border_pix'),
                    low_rank_background=self.params.get('patch', 'low_rank_background'),
                    del_duplicates=self.params.get('patch', 'del_duplicates'),
                    indeces=indeces)

            self.estimates.bl, self.estimates.c1, self.estimates.g, self.estimates.neurons_sn = None, None, None, None
            print("merging")
            self.estimates.merged_ROIs = [0]


            if self.params.get('init', 'center_psf'):  # merge taking best neuron
                if self.params.get('patch', 'nb_patch') > 0:

                    while len(self.estimates.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.Inf, fast_merge=True)

                    print("update temporal")
                    self.update_temporal(Yr, use_init=False)

                    self.params.set('spatial', {'se': np.ones((1,) * len(self.dims), dtype=np.uint8)})
                    print('update spatial ...')
                    self.update_spatial(Yr, use_init=False)

                    print("update temporal")
                    self.update_temporal(Yr, use_init=False)
                else:
                    while len(self.estimates.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.Inf, fast_merge=True)
                        if len(self.estimates.merged_ROIs) > 0:
                            not_merged = np.setdiff1d(list(range(len(self.estimates.YrA))),
                                                      np.unique(np.concatenate(self.estimates.merged_ROIs)))
                            self.estimates.YrA = np.concatenate([self.estimates.YrA[not_merged],
                                                       np.array([self.estimates.YrA[m].mean(0) for m in self.estimates.merged_ROIs])])
            else:
                
                while len(self.estimates.merged_ROIs) > 0:
                    self.merge_comps(Yr, mx=np.Inf)

                print("update temporal")
                self.update_temporal(Yr, use_init=False)

#        self.estimates.A, self.estimates.C, self.estimates.YrA, self.estimates.b, self.estimates.f, self.estimates.neurons_sn = normalize_AC(
#            self.estimates.A, self.estimates.C, self.estimates.YrA, self.estimates.b, self.estimates.f, self.estimates.neurons_sn)
        self.estimates.normalize_components()
        return self


    def save(self,filename):
        '''save object in hdf5 file format
        Parameters:
        -----------
        filename: str
            path to the hdf5 file containing the saved object
        '''
        if '.hdf5' in filename:
            # keys_types = [(k, type(v)) for k, v in self.__dict__.items()]
            ptpt = self.optional_outputs
            self.optional_outputs = None
            save_dict_to_hdf5(self.__dict__, filename)
            self.optional_outputs = ptpt

        else:
            raise Exception("Filename not supported")



#    def _prepare_object(self, Yr, T, new_dims=None,
#                        idx_components=None, g=None, lam=None, s_min=None,
#                        bl=None):
#
#        if idx_components is None:
#            idx_components = range(self.estimates.A.shape[-1])
#
#        self.estimates.A = self.estimates.A.tocsc()[:, idx_components]
#        self.estimates.C = self.estimates.C[idx_components]
#        self.estimates.b = self.estimates.b
#        self.estimates.f = self.estimates.f
#        self.estimates.S = self.estimates.S[idx_components]
#        self.estimates.YrA = self.estimates.YrA[idx_components]
#        self.estimates.g = self.estimates.g[idx_components]
#        self.estimates.bl = self.estimates.bl[idx_components]
#        self.estimates.c1 = self.estimates.c1[idx_components]
#        self.estimates.neurons_sn = self.estimates.neurons_sn[idx_components]
#        self.estimates.lam = self.estimates.lam[idx_components]
#        self.dims = self.dims
#        self.N = self.estimates.A.shape[-1]
#        self.M = self.params.get('init', 'nb') + self.N
#
#        expected_comps = self.params.get('online', 'expected_comps')
#        if expected_comps <= self.N + self.params.get('online', 'max_num_added'):
#            expected_comps = self.N + self.params.get('online', 'max_num_added') + 200
#            self.params.set('online', {'expected_comps': expected_comps})
#
#        if Yr.shape[-1] != self.params.get('online', 'init_batch'):
#            raise Exception(
#                'The movie size used for initialization does not match with the minibatch size')
#
#        if new_dims is not None:
#
#            new_Yr = np.zeros([np.prod(new_dims), T])
#            for ffrr in range(T):
#                tmp = cv2.resize(Yr[:, ffrr].reshape(
#                    self.dims, order='F'), new_dims[::-1])
#                print(tmp.shape)
#                new_Yr[:, ffrr] = tmp.reshape([np.prod(new_dims)], order='F')
#            Yr = new_Yr
#            A_new = scipy.sparse.csc_matrix(
#                (np.prod(new_dims), self.estimates.A.shape[-1]), dtype=np.float32)
#            for neur in range(self.N):
#                a = self.estimates.A.tocsc()[:, neur].toarray()
#                a = a.reshape(self.dims, order='F')
#                a = cv2.resize(a, new_dims[::-1]).reshape([-1, 1], order='F')
#
#                A_new[:, neur] = scipy.sparse.csc_matrix(a)
#
#            self.estimates.A = A_new
#            self.estimates.b = self.estimates.b.reshape(self.dims, order='F')
#            self.estimates.b = cv2.resize(
#                self.estimates.b, new_dims[::-1]).reshape([-1, 1], order='F')
#
#            self.dims = new_dims
#
#        nA = np.ravel(np.sqrt(self.estimates.A.power(2).sum(0)))
#        self.estimates.A /= nA
#        self.estimates.C *= nA[:, None]
#        self.estimates.YrA *= nA[:, None]
##        self.estimates.S *= nA[:, None]
#        self.estimates.neurons_sn *= nA
#        if self.params.get('preprocess', 'p'):
#            self.estimates.lam *= nA
#        z = np.sqrt([b.T.dot(b) for b in self.estimates.b.T])
#        self.estimates.f *= z[:, None]
#        self.estimates.b /= z
#
#        self.estimates.noisyC = np.zeros(
#            (self.params.get('init', 'nb') + expected_comps, T), dtype=np.float32)
#        self.estimates.C_on = np.zeros((expected_comps, T), dtype=np.float32)
#
#        self.estimates.noisyC[self.params.get('init', 'nb'):self.M, :self.params.get('online', 'init_batch')] = self.estimates.C + self.estimates.YrA
#        self.estimates.noisyC[:self.params.get('init', 'nb'), :self.params.get('online', 'init_batch')] = self.estimates.f
#
#        if self.params.get('preprocess', 'p'):
#            # if no parameter for calculating the spike size threshold is given, then use L1 penalty
#            if s_min is None and self.params.get('temporal', 's_min') is None:
#                use_L1 = True
#            else:
#                use_L1 = False
#
#            self.estimates.OASISinstances = [OASIS(
#                g=np.ravel(0.01) if self.params.get('preprocess', 'p') == 0 else (
#                    np.ravel(g)[0] if g is not None else gam[0]),
#                lam=0 if not use_L1 else (l if lam is None else lam),
#                s_min=0 if use_L1 else (s_min if s_min is not None else
#                                        (self.params.get('temporal', 's_min') if self.params.get('temporal', 's_min') > 0 else
#                                         (-self.params.get('temporal', 's_min') * sn * np.sqrt(1 - np.sum(gam))))),
#                b=b if bl is None else bl,
#                g2=0 if self.params.get('preprocess', 'p') < 2 else (np.ravel(g)[1] if g is not None else gam[1]))
#                for gam, l, b, sn in zip(self.estimates.g, self.estimates.lam, self.estimates.bl, self.estimates.neurons_sn)]
#
#            for i, o in enumerate(self.estimates.OASISinstances):
#                o.fit(self.estimates.noisyC[i + self.params.get('init', 'nb'), :self.params.get('online', 'init_batch')])
#                self.estimates.C_on[i, :self.params.get('online', 'init_batch')] = o.c
#        else:
#            self.estimates.C_on[:self.N, :self.params.get('online', 'init_batch')] = self.estimates.C
#
#        self.estimates.Ab, self.ind_A, self.estimates.CY, self.estimates.CC = init_shapes_and_sufficient_stats(
#            Yr[:, :self.params.get('online', 'init_batch')].reshape(
#                self.params.get('data', 'dims') + (-1,), order='F'), self.estimates.A,
#            self.estimates.C_on[:self.N, :self.params.get('online', 'init_batch')], self.estimates.b, self.estimates.noisyC[:self.params.get('init', 'nb'), :self.params.get('online', 'init_batch')])
#
#        self.estimates.CY, self.estimates.CC = self.estimates.CY * 1. / self.params.get('online', 'init_batch'), 1 * self.estimates.CC / self.params.get('online', 'init_batch')
#
#        self.estimates.A = scipy.sparse.csc_matrix(
#            self.estimates.A.astype(np.float32), dtype=np.float32)
#        self.estimates.C = self.estimates.C.astype(np.float32)
#        self.estimates.f = self.estimates.f.astype(np.float32)
#        self.estimates.b = self.estimates.b.astype(np.float32)
#        self.estimates.Ab = scipy.sparse.csc_matrix(
#            self.estimates.Ab.astype(np.float32), dtype=np.float32)
#        self.estimates.noisyC = self.estimates.noisyC.astype(np.float32)
#        self.estimates.CY = self.estimates.CY.astype(np.float32)
#        self.estimates.CC = self.estimates.CC.astype(np.float32)
#        print('Expecting ' + str(expected_comps) + ' components')
#        self.estimates.CY.resize([expected_comps + self.params.get('init', 'nb'), self.estimates.CY.shape[-1]])
#        if self.params.get('online', 'use_dense'):
#            self.estimates.Ab_dense = np.zeros((self.estimates.CY.shape[-1], expected_comps + self.params.get('init', 'nb')),
#                                     dtype=np.float32)
#            self.estimates.Ab_dense[:, :self.estimates.Ab.shape[1]] = self.estimates.Ab.toarray()
#        self.estimates.C_on = np.vstack(
#            [self.estimates.noisyC[:self.params.get('init', 'nb'), :], self.estimates.C_on.astype(np.float32)])
#
#        self.params.set('init', {'gSiz': np.add(np.multiply(np.ceil(self.params.get('init', 'gSig')).astype(np.int), 2), 1)})
#
#        self.estimates.Yr_buf = RingBuffer(Yr[:, self.params.get('online', 'init_batch') - self.params.get('online', 'minibatch_shape'):
#                                    self.params.get('online', 'init_batch')].T.copy(), self.params.get('online', 'minibatch_shape'))
#        self.estimates.Yres_buf = RingBuffer(self.estimates.Yr_buf - self.estimates.Ab.dot(
#            self.estimates.C_on[:self.M, self.params.get('online', 'init_batch') - self.params.get('online', 'minibatch_shape'):self.params.get('online', 'init_batch')]).T, self.params.get('online', 'minibatch_shape'))
#        self.estimates.sn = np.array(np.std(self.estimates.Yres_buf,axis=0))
#        self.estimates.vr = np.array(np.var(self.estimates.Yres_buf,axis=0))
#        self.estimates.mn = self.estimates.Yres_buf.mean(0)
#        self.estimates.mean_buff = self.estimates.Yres_buf.mean(0)
#        self.estimates.ind_new = []
#        self.estimates.rho_buf = imblur(np.maximum(self.estimates.Yres_buf.T,0).reshape(
#            self.params.get('data', 'dims') + (-1,), order='F'), sig=self.params.get('init', 'gSig'), siz=self.params.get('init', 'gSiz'), nDimBlur=len(self.params.get('data', 'dims')))**2
#        self.estimates.rho_buf = np.reshape(
#            self.estimates.rho_buf, (np.prod(self.params.get('data', 'dims')), -1)).T
#        self.estimates.rho_buf = RingBuffer(self.estimates.rho_buf, self.params.get('online', 'minibatch_shape'))
#        self.estimates.AtA = (self.estimates.Ab.T.dot(self.estimates.Ab)).toarray()
#        self.estimates.AtY_buf = self.estimates.Ab.T.dot(self.estimates.Yr_buf.T)
#        self.estimates.sv = np.sum(self.estimates.rho_buf.get_last_frames(
#            min(self.params.get('online', 'init_batch'), self.params.get('online', 'minibatch_shape')) - 1), 0)
#        self.estimates.groups = list(map(list, update_order(self.estimates.Ab)[0]))
#        # self.update_counter = np.zeros(self.N)
#        self.update_counter = .5**(-np.linspace(0, 1,
#                                                self.N, dtype=np.float32))
#        self.time_neuron_added = []
#        for nneeuu in range(self.N):
#            self.time_neuron_added.append((nneeuu, self.params.get('online', 'init_batch')))
#        self.time_spend = 0
#        # setup per patch classifier
#
#        if self.params.get('online', 'path_to_model') is None or self.params.get('online', 'sniper_mode') is False:
#            loaded_model = None
#            self.params.set('online', {'sniper_mode': False})
#        else:
#            import keras
#            from keras.models import model_from_json
#            path = self.params.get('online', 'path_to_model').split(".")[:-1]
#            json_path = ".".join(path + ["json"])
#            model_path = ".".join(path + ["h5"])
#
#            json_file = open(json_path, 'r')
#            loaded_model_json = json_file.read()
#            json_file.close()
#            loaded_model = model_from_json(loaded_model_json)
#            loaded_model.load_weights(model_path)
#            opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#            loaded_model.compile(loss=keras.losses.categorical_crossentropy,
#                                 optimizer=opt, metrics=['accuracy'])
#
#        self.loaded_model = loaded_model
#        return self
#
#    @profile
#    def fit_next(self, t, frame_in, num_iters_hals=3):
#        """
#        This method fits the next frame using the online cnmf algorithm and
#        updates the object.
#
#        Parameters
#        ----------
#        t : int
#            time measured in number of frames
#
#        frame_in : array
#            flattened array of shape (x*y[*z],) containing the t-th image.
#
#        num_iters_hals: int, optional
#            maximal number of iterations for HALS (NNLS via blockCD)
#
#
#        """
#
#        t_start = time()
#
#        # locally scoped variables for brevity of code and faster look up
#        nb_ = self.params.get('init', 'nb')
#        Ab_ = self.estimates.Ab
#        mbs = self.params.get('online', 'minibatch_shape')
#        expected_comps = self.params.get('online', 'expected_comps')
#        frame = frame_in.astype(np.float32)
##        print(np.max(1/scipy.sparse.linalg.norm(self.estimates.Ab,axis = 0)))
#        self.estimates.Yr_buf.append(frame)
#        if len(self.estimates.ind_new) > 0:
#            self.estimates.mean_buff = self.estimates.Yres_buf.mean(0)
#
#        if (not self.params.get('online', 'simultaneously')) or self.params.get('preprocess', 'p') == 0:
#            # get noisy fluor value via NNLS (project data on shapes & demix)
#            C_in = self.estimates.noisyC[:self.M, t - 1].copy()
#            self.estimates.C_on[:self.M, t], self.estimates.noisyC[:self.M, t] = HALS4activity(
#                frame, self.estimates.Ab, C_in, self.estimates.AtA, iters=num_iters_hals, groups=self.estimates.groups)
#            if self.params.get('preprocess', 'p'):
#                # denoise & deconvolve
#                for i, o in enumerate(self.estimates.OASISinstances):
#                    o.fit_next(self.estimates.noisyC[nb_ + i, t])
#                    self.estimates.C_on[nb_ + i, t - o.get_l_of_last_pool() +
#                              1: t + 1] = o.get_c_of_last_pool()
#
#        else:
#            # update buffer, initialize C with previous value
#            self.estimates.C_on[:, t] = self.estimates.C_on[:, t - 1]
#            self.estimates.noisyC[:, t] = self.estimates.C_on[:, t - 1]
#            self.estimates.AtY_buf = np.concatenate((self.estimates.AtY_buf[:, 1:], self.estimates.Ab.T.dot(frame)[:, None]), 1) \
#                if self.params.get('online', 'n_refit') else self.estimates.Ab.T.dot(frame)[:, None]
#            # demix, denoise & deconvolve
#            (self.estimates.C_on[:self.M, t + 1 - mbs:t + 1], self.estimates.noisyC[:self.M, t + 1 - mbs:t + 1],
#                self.estimates.OASISinstances) = demix_and_deconvolve(
#                self.estimates.C_on[:self.M, t + 1 - mbs:t + 1],
#                self.estimates.noisyC[:self.M, t + 1 - mbs:t + 1],
#                self.estimates.AtY_buf, self.estimates.AtA, self.estimates.OASISinstances, iters=num_iters_hals,
#                n_refit=self.params.get('online', 'n_refit'))
#            for i, o in enumerate(self.estimates.OASISinstances):
#                self.estimates.C_on[nb_ + i, t - o.get_l_of_last_pool() + 1: t +
#                          1] = o.get_c_of_last_pool()
#
#        #self.estimates.mean_buff = self.estimates.Yres_buf.mean(0)
#        res_frame = frame - self.estimates.Ab.dot(self.estimates.C_on[:self.M, t])
#        mn_ = self.estimates.mn.copy()
#        self.estimates.mn = (t-1)/t*self.estimates.mn + res_frame/t
#        self.estimates.vr = (t-1)/t*self.estimates.vr + (res_frame - mn_)*(res_frame - self.estimates.mn)/t
#        self.estimates.sn = np.sqrt(self.estimates.vr)
#
#        if self.params.get('online', 'update_num_comps'):
#
#            self.estimates.mean_buff += (res_frame-self.estimates.Yres_buf[self.estimates.Yres_buf.cur])/self.params.get('online', 'minibatch_shape')
##            cv2.imshow('untitled', 0.1*cv2.resize(res_frame.reshape(self.dims,order = 'F'),(512,512)))
##            cv2.waitKey(1)
##
#            self.estimates.Yres_buf.append(res_frame)
#
#            res_frame = np.reshape(res_frame, self.params.get('data', 'dims'), order='F')
#
#            rho = imblur(np.maximum(res_frame,0), sig=self.params.get('init', 'gSig'),
#                         siz=self.params.get('init', 'gSiz'), nDimBlur=len(self.params.get('data', 'dims')))**2
#
#            rho = np.reshape(rho, np.prod(self.params.get('data', 'dims')))
#            self.estimates.rho_buf.append(rho)
#
#            self.estimates.Ab, Cf_temp, self.estimates.Yres_buf, self.rhos_buf, self.estimates.CC, self.estimates.CY, self.ind_A, self.estimates.sv, self.estimates.groups, self.estimates.ind_new, self.ind_new_all, self.estimates.sv, self.cnn_pos = update_num_components(
#                t, self.estimates.sv, self.estimates.Ab, self.estimates.C_on[:self.M, (t - mbs + 1):(t + 1)],
#                self.estimates.Yres_buf, self.estimates.Yr_buf, self.estimates.rho_buf, self.params.get('data', 'dims'),
#                self.params.get('init', 'gSig'), self.params.get('init', 'gSiz'), self.ind_A, self.estimates.CY, self.estimates.CC, rval_thr=self.params.get('online', 'rval_thr'),
#                thresh_fitness_delta=self.params.get('online', 'thresh_fitness_delta'),
#                thresh_fitness_raw=self.params.get('online', 'thresh_fitness_raw'), thresh_overlap=self.params.get('online', 'thresh_overlap'),
#                groups=self.estimates.groups, batch_update_suff_stat=self.params.get('online', 'batch_update_suff_stat'), gnb=self.params.get('init', 'nb'),
#                sn=self.estimates.sn, g=np.mean(
#
#                    self.estimates.g) if self.params.get('preprocess', 'p') == 1 else np.mean(self.estimates.g, 0),
#                s_min=self.params.get('temporal', 's_min'),
#                Ab_dense=self.estimates.Ab_dense[:, :self.M] if self.params.get('online', 'use_dense') else None,
#                oases=self.estimates.OASISinstances if self.params.get('preprocess', 'p') else None,
#                N_samples_exceptionality=self.params.get('online', 'N_samples_exceptionality'),
#                max_num_added=self.params.get('online', 'max_num_added'), min_num_trial=self.params.get('online', 'min_num_trial'),
#                loaded_model = self.loaded_model, thresh_CNN_noisy = self.params.get('online', 'thresh_CNN_noisy'),
#                sniper_mode=self.params.get('online', 'sniper_mode'), use_peak_max=self.params.get('online', 'use_peak_max'),
#                test_both=self.params.get('online', 'test_both'))
#
#            num_added = len(self.ind_A) - self.N
#
#            if num_added > 0:
#                self.N += num_added
#                self.M += num_added
#                if self.N + self.params.get('online', 'max_num_added') > expected_comps:
#                    expected_comps += 200
#                    self.params.set('online', {'expected_comps': expected_comps})
#                    self.estimates.CY.resize(
#                        [expected_comps + nb_, self.estimates.CY.shape[-1]])
#                    # refcheck can trigger "ValueError: cannot resize an array references or is referenced
#                    #                       by another array in this way.  Use the resize function"
#                    # np.resize didn't work, but refcheck=False seems fine
#                    self.estimates.C_on.resize(
#                        [expected_comps + nb_, self.estimates.C_on.shape[-1]], refcheck=False)
#                    self.estimates.noisyC.resize(
#                        [expected_comps + nb_, self.estimates.C_on.shape[-1]])
#                    if self.params.get('online', 'use_dense'):  # resize won't work due to contingency issue
#                        # self.estimates.Ab_dense.resize([self.estimates.CY.shape[-1], expected_comps+nb_])
#                        self.estimates.Ab_dense = np.zeros((self.estimates.CY.shape[-1], expected_comps + nb_),
#                                                 dtype=np.float32)
#                        self.estimates.Ab_dense[:, :Ab_.shape[1]] = Ab_.toarray()
#                    print('Increasing number of expected components to:' +
#                          str(expected_comps))
#                self.update_counter.resize(self.N)
#                self.estimates.AtA = (Ab_.T.dot(Ab_)).toarray()
#
#                self.estimates.noisyC[self.M - num_added:self.M, t - mbs +
#                            1:t + 1] = Cf_temp[self.M - num_added:self.M]
#
#                for _ct in range(self.M - num_added, self.M):
#                    self.time_neuron_added.append((_ct - nb_, t))
#                    if self.params.get('preprocess', 'p'):
#                        # N.B. OASISinstances are already updated within update_num_components
#                        self.estimates.C_on[_ct, t - mbs + 1: t +
#                                  1] = self.estimates.OASISinstances[_ct - nb_].get_c(mbs)
#                    else:
#                        self.estimates.C_on[_ct, t - mbs + 1: t + 1] = np.maximum(0,
#                                                                        self.estimates.noisyC[_ct, t - mbs + 1: t + 1])
#                    if self.params.get('online', 'simultaneously') and self.params.get('online', 'n_refit'):
#                        self.estimates.AtY_buf = np.concatenate((
#                            self.estimates.AtY_buf, [Ab_.data[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]].dot(
#                                self.estimates.Yr_buf.T[Ab_.indices[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]]])]))
#                    # much faster than Ab_[:, self.N + nb_ - num_added:].toarray()
#                    if self.params.get('online', 'use_dense'):
#                        self.estimates.Ab_dense[Ab_.indices[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]],
#                                      _ct] = Ab_.data[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]]
#
#                # set the update counter to 0 for components that are overlaping the newly added
#                if self.params.get('online', 'use_dense'):
#                    idx_overlap = np.concatenate([
#                        self.estimates.Ab_dense[self.ind_A[_ct], nb_:self.M - num_added].T.dot(
#                            self.estimates.Ab_dense[self.ind_A[_ct], _ct + nb_]).nonzero()[0]
#                        for _ct in range(self.N - num_added, self.N)])
#                else:
#                    idx_overlap = Ab_.T.dot(
#                        Ab_[:, -num_added:])[nb_:-num_added].nonzero()[0]
#                self.update_counter[idx_overlap] = 0
#
#        if (t - self.params.get('online', 'init_batch')) % mbs == mbs - 1 and\
#                self.params.get('online', 'batch_update_suff_stat'):
#            # faster update using minibatch of frames
#
#            ccf = self.estimates.C_on[:self.M, t - mbs + 1:t + 1]
#            y = self.estimates.Yr_buf  # .get_last_frames(mbs)[:]
#
#            # much faster: exploit that we only access CY[m, ind_pixels], hence update only these
#            n0 = mbs
#            t0 = 0 * self.params.get('online', 'init_batch')
#            w1 = (t - n0 + t0) * 1. / (t + t0)  # (1 - 1./t)#mbs*1. / t
#            w2 = 1. / (t + t0)  # 1.*mbs /t
#            for m in range(self.N):
#                self.estimates.CY[m + nb_, self.ind_A[m]] *= w1
#                self.estimates.CY[m + nb_, self.ind_A[m]] += w2 * \
#                    ccf[m + nb_].dot(y[:, self.ind_A[m]])
#
#            self.estimates.CY[:nb_] = self.estimates.CY[:nb_] * w1 + \
#                w2 * ccf[:nb_].dot(y)   # background
#            self.estimates.CC = self.estimates.CC * w1 + w2 * ccf.dot(ccf.T)
#
#        if not self.params.get('online', 'batch_update_suff_stat'):
#
#            ccf = self.estimates.C_on[:self.M, t - self.params.get('online', 'minibatch_suff_stat'):t - self.params.get('online', 'minibatch_suff_stat') + 1]
#            y = self.estimates.Yr_buf.get_last_frames(self.params.get('online', 'minibatch_suff_stat') + 1)[:1]
#            # much faster: exploit that we only access CY[m, ind_pixels], hence update only these
#            for m in range(self.N):
#                self.estimates.CY[m + nb_, self.ind_A[m]] *= (1 - 1. / t)
#                self.estimates.CY[m + nb_, self.ind_A[m]] += ccf[m +
#                                                       nb_].dot(y[:, self.ind_A[m]]) / t
#            self.estimates.CY[:nb_] = self.estimates.CY[:nb_] * (1 - 1. / t) + ccf[:nb_].dot(y / t)
#            self.estimates.CC = self.estimates.CC * (1 - 1. / t) + ccf.dot(ccf.T / t)
#
#        # update shapes
#        if True:  # False:  # bulk shape update
#            if (t - self.params.get('online', 'init_batch')) % mbs == mbs - 1:
#                print('Updating Shapes')
#
#                if self.N > self.params.get('online', 'max_comp_update_shape'):
#                    indicator_components = np.where(self.update_counter <=
#                                                    self.params.get('online', 'num_times_comp_updated'))[0]
#                    # np.random.choice(self.N,10,False)
#                    self.update_counter[indicator_components] += 1
#                else:
#                    indicator_components = None
#
#                if self.params.get('online', 'use_dense'):
#                    # update dense Ab and sparse Ab simultaneously;
#                    # this is faster than calling update_shapes with sparse Ab only
#                    Ab_, self.ind_A, self.estimates.Ab_dense[:, :self.M] = update_shapes(
#                        self.estimates.CY, self.estimates.CC, self.estimates.Ab, self.ind_A,
#                        indicator_components=indicator_components,
#                        Ab_dense=self.estimates.Ab_dense[:, :self.M],
#                        sn=self.estimates.sn, q=0.5)
#                else:
#                    Ab_, self.ind_A, _ = update_shapes(self.estimates.CY, self.estimates.CC, Ab_, self.ind_A,
#                                                       indicator_components=indicator_components,
#                                                       sn=self.estimates.sn, q=0.5)
#
#                self.estimates.AtA = (Ab_.T.dot(Ab_)).toarray()
#
#                ind_zero = list(np.where(self.estimates.AtA.diagonal() < 1e-10)[0])
#                if len(ind_zero) > 0:
#                    ind_zero.sort()
#                    ind_zero = ind_zero[::-1]
#                    ind_keep = list(set(range(Ab_.shape[-1])) - set(ind_zero))
#                    ind_keep.sort()
#
#                    if self.params.get('online', 'use_dense'):
#                        self.estimates.Ab_dense = np.delete(
#                            self.estimates.Ab_dense, ind_zero, axis=1)
#                    self.estimates.AtA = np.delete(self.estimates.AtA, ind_zero, axis=0)
#                    self.estimates.AtA = np.delete(self.estimates.AtA, ind_zero, axis=1)
#                    self.estimates.CY = np.delete(self.estimates.CY, ind_zero, axis=0)
#                    self.estimates.CC = np.delete(self.estimates.CC, ind_zero, axis=0)
#                    self.estimates.CC = np.delete(self.estimates.CC, ind_zero, axis=1)
#                    self.M -= len(ind_zero)
#                    self.N -= len(ind_zero)
#                    self.estimates.noisyC = np.delete(self.estimates.noisyC, ind_zero, axis=0)
#                    for ii in ind_zero:
#                        del self.estimates.OASISinstances[ii - self.params.get('init', 'nb')]
#                        #del self.ind_A[ii-self.params.init['nb']]
#
#                    self.estimates.C_on = np.delete(self.estimates.C_on, ind_zero, axis=0)
#                    self.estimates.AtY_buf = np.delete(self.estimates.AtY_buf, ind_zero, axis=0)
#                    #Ab_ = Ab_[:,ind_keep]
#                    Ab_ = scipy.sparse.csc_matrix(Ab_[:, ind_keep])
#                    #Ab_ = scipy.sparse.csc_matrix(self.estimates.Ab_dense[:,:self.M])
#                    self.Ab_dense_copy = self.estimates.Ab_dense
#                    self.Ab_copy = Ab_
#                    self.estimates.Ab = Ab_
#                    self.ind_A = list(
#                        [(self.estimates.Ab.indices[self.estimates.Ab.indptr[ii]:self.estimates.Ab.indptr[ii + 1]]) for ii in range(self.params.get('init', 'nb'), self.M)])
#                    self.estimates.groups = list(map(list, update_order(Ab_)[0]))
#
#                if self.params.get('online', 'n_refit'):
#                    self.estimates.AtY_buf = Ab_.T.dot(self.estimates.Yr_buf.T)
#
#        else:  # distributed shape update
#            self.update_counter *= .5**(1. / mbs)
#            # if not num_added:
#            if (not num_added) and (time() - t_start < self.time_spend / (t - self.params.get('online', 'init_batch') + 1)):
#                candidates = np.where(self.update_counter <= 1)[0]
#                if len(candidates):
#                    indicator_components = candidates[:self.N // mbs + 1]
#                    self.update_counter[indicator_components] += 1
#
#                    if self.params.get('online', 'use_dense'):
#                        # update dense Ab and sparse Ab simultaneously;
#                        # this is faster than calling update_shapes with sparse Ab only
#                        Ab_, self.ind_A, self.estimates.Ab_dense[:, :self.M] = update_shapes(
#                            self.estimates.CY, self.estimates.CC, self.estimates.Ab, self.ind_A,
#                            indicator_components, self.estimates.Ab_dense[:, :self.M],
#                            update_bkgrd=(t % mbs == 0))
#                    else:
#                        Ab_, self.ind_A, _ = update_shapes(
#                            self.estimates.CY, self.estimates.CC, Ab_, self.ind_A,
#                            indicator_components=indicator_components,
#                            update_bkgrd=(t % mbs == 0))
#
#                    self.estimates.AtA = (Ab_.T.dot(Ab_)).toarray()
#
#                self.estimates.Ab = Ab_
#            self.time_spend += time() - t_start

    def remove_components(self, ind_rm):
        """remove a specified list of components from the OnACID CNMF object.

        Parameters:
        -----------
        ind_rm :    list
                    indeces of components to be removed
        """

        self.estimates.Ab, self.estimates.Ab_dense, self.estimates.CC, self.estimates.CY, self.M,\
            self.N, self.estimates.noisyC, self.estimates.OASISinstances, self.estimates.C_on,\
            expected_comps, self.ind_A,\
            self.estimates.groups, self.estimates.AtA = remove_components_online(
                ind_rm, self.params.get('init', 'nb'), self.estimates.Ab,
                self.params.get('online', 'use_dense'), self.estimates.Ab_dense,
                self.estimates.AtA, self.estimates.CY, self.estimates.CC, self.M, self.N,
                self.estimates.noisyC, self.estimates.OASISinstances, self.estimates.C_on,
                self.params.get('online', 'expected_comps'))
        self.params.set('online', {'expected_comps': expected_comps})

    def compute_residuals(self, Yr):
        """compute residual for each component (variable YrA)

         Parameters:
         -----------
         Yr :    np.ndarray
                 movie in format pixels (d) x frames (T)

        """

        if 'csc_matrix' not in str(type(self.estimates.A)):
            self.estimates.A = scipy.sparse.csc_matrix(self.estimates.A)
        if 'array' not in str(type(self.estimates.b)):
            self.estimates.b = self.estimates.b.toarray()
        if 'array' not in str(type(self.estimates.C)):
            self.estimates.C = self.estimates.C.estimates.toarray()
        if 'array' not in str(type(self.estimates.f)):
            self.estimates.f = self.estimates.f.toarray()

        Ab = scipy.sparse.hstack((self.estimates.A, self.estimates.b)).tocsc()
        nA2 = np.ravel(Ab.power(2).sum(axis=0))
        nA2_inv_mat = scipy.sparse.spdiags(
            1. / nA2, 0, nA2.shape[0], nA2.shape[0])
        Cf = np.vstack((self.estimates.C, self.estimates.f))
        if 'numpy.ndarray' in str(type(Yr)):
            YA = (Ab.T.dot(Yr)).T * nA2_inv_mat
        else:
            YA = mmapping.parallel_dot_product(Yr, Ab, dview=self.dview, block_size=2000,
                                           transpose=True, num_blocks_per_run=5) * nA2_inv_mat

        AA = Ab.T.dot(Ab) * nA2_inv_mat
        self.estimates.YrA = (YA - (AA.T.dot(Cf)).T)[:, :self.estimates.A.shape[-1]].T

        return self


    def deconvolve(self, p=None, method=None, bas_nonneg=None,
                   noise_method=None, optimize_g=0, s_min=None, **kwargs):
        """Performs deconvolution on already extracted traces using
        constrained foopsi.
        """

        p = self.params.get('preprocess', 'p') if p is None else p
        method = self.params.get('temporal', 'method') if method is None else method
        bas_nonneg = (self.params.get('temporal', 'bas_nonneg')
                      if bas_nonneg is None else bas_nonneg)
        noise_method = (self.params.get('temporal', 'noise_method')
                        if noise_method is None else noise_method)
        s_min = self.params.get('temporal', 's_min') if s_min is None else s_min

        F = self.estimates.C + self.estimates.YrA
        args = dict()
        args['p'] = p
        args['method'] = method
        args['bas_nonneg'] = bas_nonneg
        args['noise_method'] = noise_method
        args['s_min'] = s_min
        args['optimize_g'] = optimize_g
        args['noise_range'] = self.params.get('temporal', 'noise_range')
        args['fudge_factor'] = self.params.get('temporal', 'fudge_factor')

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
        self.estimates.C = np.stack([results[0][i] for i in order])
        self.estimates.S = np.stack([results[1][i] for i in order])
        self.estimates.bl = [results[3][i] for i in order]
        self.estimates.c1 = [results[4][i] for i in order]
        self.estimates.g = [results[5][i] for i in order]
        self.estimates.neuron_sn = [results[6][i] for i in order]
        self.estimates.lam = [results[8][i] for i in order]
        self.estimates.YrA = F - self.estimates.C
        return self

#    def evaluate_components(self, imgs, **kwargs):
#        """Computes the quality metrics for each component and stores the
#        indeces of the components that pass user specified thresholds. The
#        various thresholds and parameters can be passed as inputs. If left
#        empty then they are read from self.params.quality']
#        Parameters:
#        -----------
#        imgs: np.array (possibly memory mapped, t,x,y[,z])
#            Imaging data
#
#        fr: float
#            Imaging rate
#
#        decay_time: float
#            length of decay of typical transient (in seconds)
#
#        min_SNR: float
#            trace SNR threshold
#
#        rval_thr: float
#            space correlation threshold
#
#        use_cnn: bool
#            flag for using the CNN classifier
#
#        min_cnn_thr: float
#            CNN classifier threshold
#
#        Returns:
#        --------
#        self: CNMF object
#            self.idx_components: np.array
#                indeces of accepted components
#            self.idx_components_bad: np.array
#                indeces of rejected components
#            self.SNR_comp: np.array
#                SNR values for each temporal trace
#            self.r_values: np.array
#                space correlation values for each component
#            self.cnn_preds: np.array
#                CNN classifier values for each component
#        """
#        dims = imgs.shape[1:]
#        self.params.set('quality', kwargs)
#        opts = self.params.get_group('quality')
#        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
#        estimate_components_quality_auto(imgs, self.estimates.A, self.estimates.C, self.estimates.b, self.estimates.f,
#                                         self.estimates.YrA, 
#                                         self.params.get('data', 'fr'),
#                                         self.params.get('data', 'decay_time'),
#                                         self.params.get('init', 'gSig'),
#                                         dims, dview=self.dview,
#                                         min_SNR=opts['min_SNR'],
#                                         r_values_min=opts['rval_thr'],
#                                         use_cnn=opts['use_cnn'],
#                                         thresh_cnn_min=opts['min_cnn_thr'])
#        self.estimates.idx_components = idx_components
#        self.estimates.idx_components_bad = idx_components_bad
#        self.estimates.SNR_comp = SNR_comp
#        self.estimates.r_values = r_values
#        self.estimates.cnn_preds = cnn_preds
#
#        return self
#
#
#    def filter_components(self, imgs, **kwargs):
#        """Filters components based on given thresholds without re-computing
#        the quality metrics. If the quality metrics are not present then it
#        calls self.evaluate components.
#        Parameters:
#        -----------
#        imgs: np.array (possibly memory mapped, t,x,y[,z])
#            Imaging data
#
#        fr: float
#            Imaging rate
#
#        decay_time: float
#            length of decay of typical transient (in seconds)
#
#        min_SNR: float
#            trace SNR threshold
#
#        SNR_lowest: float
#            minimum required trace SNR
#
#        rval_thr: float
#            space correlation threshold
#
#        rval_lowest: float
#            minimum required space correlation
#
#        use_cnn: bool
#            flag for using the CNN classifier
#
#        min_cnn_thr: float
#            CNN classifier threshold
#
#        cnn_lowest: float
#            minimum required CNN threshold
#
#        gSig_range: list
#            gSig scale values for CNN classifier
#
#        Returns:
#        --------
#        self: CNMF object
#            self.idx_components: np.array
#                indeces of accepted components
#            self.idx_components_bad: np.array
#                indeces of rejected components
#            self.SNR_comp: np.array
#                SNR values for each temporal trace
#            self.r_values: np.array
#                space correlation values for each component
#            self.cnn_preds: np.array
#                CNN classifier values for each component
#        """
#        dims = imgs.shape[1:]
#        self.params.set('quality', kwargs)
#
#        opts = self.params.get_group('quality')
#        self.estimates.idx_components, self.estimates.idx_components_bad, self.estimates.cnn_preds = \
#        select_components_from_metrics(self.estimates.A, dims, self.params.get('init', 'gSig'), self.estimates.r_values,
#                                       self.estimates.SNR_comp, predictions=self.estimates.cnn_preds,
#                                       r_values_min=opts['rval_thr'],
#                                       r_values_lowest=opts['rval_lowest'],
#                                       min_SNR=opts['min_SNR'],
#                                       min_SNR_reject=opts['SNR_lowest'],
#                                       thresh_cnn_min=opts['min_cnn_thr'],
#                                       thresh_cnn_lowest=opts['cnn_lowest'],
#                                       use_cnn=opts['use_cnn'],
#                                       gSig_range=opts['gSig_range'])
#
#        return self

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
        self (updated values for self.estimates.C, self.estimates.f, self.estimates.YrA)
        """
        if update_bck:
            Ab = scipy.sparse.hstack([self.estimates.b, self.estimates.A]).tocsc()
            try:
                Cf = np.vstack([self.estimates.f, self.estimates.C + self.estimates.YrA])
            except():
                Cf = np.vstack([self.estimates.f, self.estimates.C])
        else:
            Ab = self.estimates.A
            try:
                Cf = self.estimates.C + self.estimates.YrA
            except():
                Cf = self.estimates.C
            Yr = Yr - self.estimates.b.dot(self.estimates.f)
        if (groups is None) and use_groups:
            groups = list(map(list, update_order(Ab)[0]))
        self.estimates.groups = groups
        C, noisyC = HALS4activity(Yr, Ab, Cf, groups=self.estimates.groups, order=order,
                                  **kwargs)
        if update_bck:
            if bck_non_neg:
                self.estimates.f = C[:self.params.get('init', 'nb')]
            else:
                self.estimates.f = noisyC[:self.params.get('init', 'nb')]
            self.estimates.C = C[self.params.get('init', 'nb'):]
            self.estimates.YrA = noisyC[self.params.get('init', 'nb'):] - self.estimates.C
        else:
            self.estimates.C = C
            self.estimates.YrA = noisyC - self.estimates.C
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
        self (updated values for self.estimates.A and self.estimates.b)
        """
        if update_bck:
            Ab = np.hstack([self.estimates.b, self.estimates.A.toarray()])
            try:
                Cf = np.vstack([self.estimates.f, self.estimates.C + self.estimates.YrA])
            except():
                Cf = np.vstack([self.estimates.f, self.estimates.C])
        else:
            Ab = self.estimates.A.toarray()
            try:
                Cf = self.estimates.C + self.estimates.YrA
            except():
                Cf = self.estimates.C
            Yr = Yr - self.estimates.b.dot(self.estimates.f)
        Ab = HALS4shapes(Yr, Ab, Cf, iters=num_iter)
        if update_bck:
            self.estimates.A = scipy.sparse.csc_matrix(Ab[:, self.params.get('init', 'nb'):])
            self.estimates.b = Ab[:, :self.params.get('init', 'nb')]
        else:
            self.estimates.A = scipy.sparse.csc_matrix(Ab)

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
        self.params.set('temporal', kwargs_new)


        self.estimates.C, self.estimates.A, self.estimates.b, self.estimates.f, self.estimates.S, \
        self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, \
        self.estimates.g, self.estimates.YrA, self.estimates.lam = update_temporal_components(
                Y, self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, dview=self.dview,
                **self.params.get_group('temporal'))
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
            modified values self.estimates.A, self.estimates.b possibly self.estimates.C, self.estimates.f
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
        self.params.set('spatial', kwargs_new)
        for key in kwargs_new:
            if hasattr(self, key):
                setattr(self, key, kwargs_new[key])

        self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f =\
            update_spatial_components(Y, C=self.estimates.C, f=self.estimates.f, A_in=self.estimates.A,
                                      b_in=self.estimates.b, dview=self.dview,
                                      sn=self.estimates.sn, dims=self.dims, **self.params.get_group('spatial'))

        return self

    def merge_comps(self, Y, mx=50, fast_merge=True):
        """merges components
        """
        self.estimates.A, self.estimates.C, self.estimates.nr, self.estimates.merged_ROIs, self.estimates.S, \
        self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, self.estimates.g=\
            merge_components(Y, self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, self.estimates.S,
                             self.estimates.sn, self.params.get_group('temporal'),
                             self.params.get_group('spatial'), dview=self.dview,
                             bl=self.estimates.bl, c1=self.estimates.c1, sn=self.estimates.neurons_sn,
                             g=self.estimates.g, thr=self.params.get('merging', 'merge_thr'), mx=mx,
                             fast_merge=fast_merge)
            
        return self
    
    def initialize(self, Y, **kwargs):
        """Component initialization
        """
        self.params.set('init', kwargs)
        estim = self.estimates
        if self.params.get('init', 'center_psf'):
            estim.A, estim.C, estim.b, estim.f, estim.center, \
            extra_1p = initialize_components(
                Y, sn=estim.sn, options_total=self.params.to_dict(),
                **self.params.get_group('init'))
            try:
                estim.S, estim.bl, estim.c1, estim.neurons_sn, \
                estim.g, estim.YrA = extra_1p
            except:
                estim.S, estim.bl, estim.c1, estim.neurons_sn, \
                estim.g, estim.YrA, estim.W, estim.b0 = extra_1p
        else:
            estim.A, estim.C, estim.b, estim.f, estim.center =\
            initialize_components(Y, sn=estim.sn, options_total=self.params.to_dict(),
                                  **self.params.get_group('init'))

        self.estimates = estim

        return self

#    def initialize_online(self):
#        fls = self.params.get('data', 'fnames')
#        opts = self.params.get_group('online')
#        print(opts['init_batch'])
#        Y = load(fls[0], subindices=slice(0, opts['init_batch'],
#                 None)).astype(np.float32)
#        ds_factor = np.maximum(opts['ds_factor'], 1)
#        if ds_factor > 1:
#            Y.resize(1./ds_factor)
#        mc_flag = self.params.get('online', 'motion_correct')
#        self.estimates.shifts = []  # store motion shifts here
#        self.estimates.time_new_comp = []
#        if mc_flag:
#            max_shifts = self.params.get('online', 'max_shifts')
#            mc = Y.motion_correct(max_shifts, max_shifts)
#            Y = mc[0].astype(np.float32)
#            self.estimates.shifts.extend(mc[1])
#
#        img_min = Y.min()
#        
#        if self.params.get('online', 'normalize'):
#            Y -= img_min
#        img_norm = np.std(Y, axis=0)
#        img_norm += np.median(img_norm)  # normalize data to equalize the FOV
#        if self.params.get('online', 'normalize'):
#            Y = Y/img_norm[None, :, :]
#        _, d1, d2 = Y.shape
#        Yr = Y.to_2D().T        # convert data into 2D array
#        self.img_min = img_min
#        self.img_norm = img_norm
#        if self.params.get('online', 'init_method') == 'bare':
#            self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, self.estimates.YrA = bare_initialization(
#                    Y.transpose(1, 2, 0), gnb=self.params.get('init', 'nb'), k=self.params.get('init', 'K'),
#                    gSig=self.params.get('init', 'gSig'), return_object=False)
#            self.estimates.S = np.zeros_like(self.estimates.C)
#            nr = self.estimates.C.shape[0]
#            self.estimates.g = np.array([-np.poly([0.9] * max(self.params.get('preprocess', 'p'), 1))[1:]
#                               for gg in np.ones(nr)])
#            self.estimates.bl = np.zeros(nr)
#            self.estimates.c1 = np.zeros(nr)
#            self.estimates.neurons_sn = np.std(self.estimates.YrA, axis=-1)
#            self.estimates.lam = np.zeros(nr)
#        elif self.params.get('online', 'init_method') == 'cnmf':
#            if self.params.get('patch', 'rf') is None:
#                self.dview = None
#                self.fit(np.array(Y))
#            else:
#                f_new = mmapping.save_memmap(fls[:1], base_name='Yr', order='C',
#                                             slices=[slice(0, opts['init_batch']), None, None])
#                Yrm, dims_, T_ = mmapping.load_memmap(f_new)
#                Y = np.reshape(Yrm.T, [T_] + list(dims_), order='F')
#                self.fit(Y)
#                if self.params.get('online', 'normalize'):
#                    self.estimates.A /= self.img_norm.reshape(-1, order='F')[:, np.newaxis]
#                    self.estimates.b /= self.img_norm.reshape(-1, order='F')[:, np.newaxis]
#                    self.estimates.A = scipy.sparse.csc_matrix(self.estimates.A)
#            
#        elif self.params.get('online', 'init_method') == 'seeded':
#            self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, self.estimates.YrA = seeded_initialization(
#                    Y.transpose(1, 2, 0), self.estimates.A, gnb=self.params.get('init', 'nb'), k=self.params.get('init', 'k'),
#                    gSig=self.params.get('init', 'gSig'), return_object=False)
#            self.estimates.S = np.zeros_like(self.estimates.C)
#            nr = self.estimates.C.shape[0]
#            self.estimates.g = np.array([-np.poly([0.9] * max(self.params.get('preprocess', 'p'), 1))[1:]
#                               for gg in np.ones(nr)])
#            self.estimates.bl = np.zeros(nr)
#            self.estimates.c1 = np.zeros(nr)
#            self.estimates.neurons_sn = np.std(self.estimates.YrA, axis=-1)
#            self.estimates.lam = np.zeros(nr)
#        else:
#            raise Exception('Unknown initialization method!')
#        dims, Ts = get_file_size(fls[0])
#        self.params.set('data', {'dims': dims})
#        T1 = np.array(Ts).sum()*self.params.get('online', 'epochs')
#        self._prepare_object(Yr, T1)
#        return self
#
#
#    #def fit_online(self, fls, init_batch=200, epochs=1, motion_correct=True, **kwargs):
#    def fit_online(self, **kwargs):
#        """Implements the caiman online algorithm on the list of files fls. The
#        files are taken in alpha numerical order and are assumed to each have
#        the same number of frames (except the last one that can be shorter).
#        Caiman online is initialized using the seeded or bare initialization
#        methods.
#        Parameters:
#        -----------
#        fls: list
#            list of files to be processed
#
#        init_batch: int
#            number of frames to be processed during initialization
#
#        epochs: int
#            number of passes over the data
#
#        motion_correct: bool
#            flag for performing motion correction
#
#        kwargs: dict
#            additional parameters used to modify self.params.online']
#            see options.['online'] for details
#
#        Returns:
#        --------
#        self (results of caiman online)
#        """
##        lc = locals()
##        pr = inspect.signature(self.fit_online)
##        params = [k for k, v in pr.parameters.items() if '=' in str(v)]
##        kw2 = {k: lc[k] for k in params}
##        try:
##            kwargs_new = {**kw2, **kwargs}
##        except():  # python 2.7
##            kwargs_new = kw2.copy()
##            kwargs_new.update(kwargs)
##        self.params.set('data', {'fnames': fls})
##        self.params.set('online', kwargs_new)
##        for key in kwargs_new:
##            if hasattr(self, key):
##                setattr(self, key, kwargs_new[key])
#        fls = self.params.get('data', 'fnames')
#        init_batch = self.params.get('online', 'init_batch')
#        epochs = self.params.get('online', 'epochs')
#        self.initialize_online()
#        extra_files = len(fls) - 1
#        init_files = 1
#        t = init_batch
#        self.Ab_epoch = []
#        max_shifts = self.params.get('online', 'max_shifts')
#        if extra_files == 0:     # check whether there are any additional files
#            process_files = fls[:init_files]     # end processing at this file
#            init_batc_iter = [init_batch]         # place where to start
#        else:
#            process_files = fls[:init_files + extra_files]   # additional files
#            # where to start reading at each file
#            init_batc_iter = [init_batch] + [0]*extra_files
#        for iter in range(epochs):
#            if iter > 0:
#                # if not on first epoch process all files from scratch
#                process_files = fls[:init_files + extra_files]
#                init_batc_iter = [0] * (extra_files + init_files)
#
#            for file_count, ffll in enumerate(process_files):
#                print('Now processing file ' + ffll)
#                Y_ = load(ffll, subindices=slice(
#                                    init_batc_iter[file_count], None, None))
#
#                old_comps = self.N     # number of existing components
#                for frame_count, frame in enumerate(Y_):   # process each file
#                    if np.isnan(np.sum(frame)):
#                        raise Exception('Frame ' + str(frame_count) +
#                                        ' contains NaN')
#                    if t % 100 == 0:
#                        print('Epoch: ' + str(iter + 1) + '. ' + str(t) +
#                              ' frames have beeen processed in total. ' +
#                              str(self.N - old_comps) +
#                              ' new components were added. Total # of components is '
#                              + str(self.estimates.Ab.shape[-1] - self.params.get('init', 'nb')))
#                        old_comps = self.N
#
#                    frame_ = frame.copy().astype(np.float32)
#                    if self.params.get('online', 'ds_factor') > 1:
#                        frame_ = cv2.resize(frame_, self.img_norm.shape[::-1])
#                    
#                    if self.params.get('online', 'normalize'):
#                        frame_ -= self.img_min     # make data non-negative
#
#                    if self.params.get('online', 'motion_correct'):    # motion correct
#                        templ = self.estimates.Ab.dot(
#                            self.estimates.C_on[:self.M, t-1]).reshape(self.params.get('data', 'dims'), order='F')*self.img_norm
#                        frame_cor, shift = motion_correct_iteration_fast(
#                            frame_, templ, max_shifts, max_shifts)
#                        self.estimates.shifts.append(shift)
#                    else:
#                        templ = None
#                        frame_cor = frame_
#
#                    if self.params.get('online', 'normalize'):
#                        frame_cor = frame_cor/self.img_norm    # normalize data-frame
#                    self.fit_next(t, frame_cor.reshape(-1, order='F'))
#                    t += 1
#            self.Ab_epoch.append(self.estimates.Ab.copy())
#        if self.params.get('online', 'normalize'):
#            self.estimates.Ab /= 1./self.img_norm.reshape(-1, order='F')[:,np.newaxis]
#            self.estimates.Ab = scipy.sparse.csc_matrix(self.estimates.Ab)
#        self.estimates.A, self.estimates.b = self.estimates.Ab[:, self.params.get('init', 'nb'):], self.estimates.Ab[:, :self.params.get('init', 'nb')].toarray()
#        self.estimates.C, self.estimates.f = self.estimates.C_on[self.params.get('init', 'nb'):self.M, t - t //
#                         epochs:t], self.estimates.C_on[:self.params.get('init', 'nb'), t - t // epochs:t]
#        noisyC = self.estimates.noisyC[self.params.get('init', 'nb'):self.M, t - t // epochs:t]
#        self.estimates.YrA = noisyC - self.estimates.C
#        self.estimates.bl = [osi.b for osi in self.estimates.OASISinstances] if hasattr(
#            self, 'OASISinstances') else [0] * self.estimates.C.shape[0]
#
#        return self

    def preprocess(self, Yr):
        Yr, self.estimates.sn, self.estimates.g, self.estimates.psx = preprocess_data(
            Yr, dview=self.dview, **self.params.get_group('preprocess'))
        return Yr


def load_CNMF(filename, n_processes=1, dview = None):
    '''load object saved with the CNMF save method
    Parameters:
    ----------
    filename: str
        hdf5 file name containing the saved object
    dview: multiprocessingor ipyparallel object
        useful to set up parllelization in the objects

    '''
    new_obj = CNMF(n_processes)
    for key,val in load_dict_from_hdf5(filename).items():
        if key == 'params':
            prms = CNMFSetParms((1, 1, 1), n_processes)
            prms.spatial = val['spatial']
            prms.temporal = val['temporal']
            prms.patch = val['patch']
            prms.preprocess = val['preprocess']
            prms.init = val['init']
            prms.merging = val['merging']
            prms.quality = val['quality']
            setattr(new_obj, key, prms)
        elif key == 'dview':
            setattr(new_obj, key, dview)
        else:
            setattr(new_obj, key, val)

    return new_obj

