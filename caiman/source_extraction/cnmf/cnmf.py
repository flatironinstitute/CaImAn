#!/usr/bin/env python

""" Constrained Nonnegative Matrix Factorization

The general file class which is used to produce a factorization of the Y matrix being the video
it computes it using all the files inside of cnmf folder.
Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
 of the structure of the class

 it is calling everyfunction from the cnmf folder
 you can find out more at how the functions are called and how they are laid out at the ipython notebook

See Also:
    http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
"""

from copy import deepcopy
import cv2
import glob
import inspect
import logging
import numpy as np
import os
import pathlib
import psutil
import scipy
import sys

import caiman
from caiman.components_evaluation import estimate_components_quality
import caiman.mmapping
from caiman.motion_correction import MotionCorrect
import caiman.paths
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.initialization import initialize_components, compute_W
from caiman.source_extraction.cnmf.map_reduce import run_CNMF_patches
from caiman.source_extraction.cnmf.merging import merge_components
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.source_extraction.cnmf.pre_processing import preprocess_data
from caiman.source_extraction.cnmf.spatial import update_spatial_components
from caiman.source_extraction.cnmf.temporal import update_temporal_components, constrained_foopsi_parallel
from caiman.source_extraction.cnmf.utilities import update_order
from caiman.utils.utils import save_dict_to_hdf5, load_dict_from_hdf5


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

    See Also:
    @url http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
    .. image:: docs/img/quickintro.png
    """
    def __init__(self, n_processes, k=5, gSig=[4, 4], gSiz=None, merge_thresh=0.8, p=2, dview=None,
                 Ain=None, Cin=None, b_in=None, f_in=None, do_merge=True,
                 ssub=2, tsub=2, p_ssub=1, p_tsub=1, method_init='greedy_roi', alpha_snmf=0.5,
                 rf=None, stride=None, memory_fact=1, gnb=1, nb_patch=1, only_init_patch=False,
                 method_deconvolution='oasis', n_pixels_per_process=4000, block_size_temp=5000, num_blocks_per_run_temp=20,
                 block_size_spat=5000, num_blocks_per_run_spat=20,
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
                 expected_comps=500, max_merge_area=None, params=None):
        """
        Constructor of the CNMF method

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
                for parallelization purposes when using ipyparallel

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

            num_blocks_per_run_spat: int
                In case of memory problems you can reduce this numbers, controlling the number of blocks processed in parallel during residual computing

            num_blocks_per_run_temp: int
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
                However benefits can be considerable if done because if many components (>2000) are created
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

            max_merge_area: int, optional
                maximum area (in pixels) of merged components, used to determine whether to merge components during fitting process
        """

        self.dview = dview

        # these are movie properties that will be refactored into the Movie object
        self.dims = None
        self.empty_merged = None

        # these are member variables related to the CNMF workflow
        self.skip_refinement = skip_refinement
        self.remove_very_bad_comps = remove_very_bad_comps

        if params is None:
            self.params = CNMFParams(
                border_pix=border_pix, del_duplicates=del_duplicates, low_rank_background=low_rank_background,
                memory_fact=memory_fact, n_processes=n_processes, nb_patch=nb_patch, only_init_patch=only_init_patch,
                p_ssub=p_ssub, p_tsub=p_tsub, remove_very_bad_comps=remove_very_bad_comps, rf=rf, stride=stride,
                check_nan=check_nan, n_pixels_per_process=n_pixels_per_process,
                k=k, center_psf=center_psf, gSig=gSig, gSiz=gSiz,
                init_iter=init_iter, method_init=method_init, min_corr=min_corr,  min_pnr=min_pnr,
                gnb=gnb, normalize_init=normalize_init, options_local_NMF=options_local_NMF,
                ring_size_factor=ring_size_factor, rolling_length=rolling_length, rolling_sum=rolling_sum,
                ssub=ssub, ssub_B=ssub_B, tsub=tsub,
                block_size_spat=block_size_spat, num_blocks_per_run_spat=num_blocks_per_run_spat,
                block_size_temp=block_size_temp, num_blocks_per_run_temp=num_blocks_per_run_temp,
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
                update_num_comps=update_num_comps, use_dense=use_dense, use_peak_max=use_peak_max, alpha_snmf=alpha_snmf,
                max_merge_area=max_merge_area
            )
        else:
            self.params = params
            params.set('patch', {'n_processes': n_processes})

        self.estimates = Estimates(A=Ain, C=Cin, b=b_in, f=f_in,
                                   dims=self.params.data['dims'])

    def fit_file(self, motion_correct=False, indices=None, include_eval=False):
        """
        This method packages the analysis pipeline (motion correction, memory
        mapping, patch based CNMF processing and component evaluation) in a
        single method that can be called on a specific (sequence of) file(s).
        It is assumed that the CNMF object already contains a params object
        where the location of the files and all the relevant parameters have
        been specified. The method will perform the last step, i.e. component
        evaluation, if the flag "include_eval" is set to `True`.

        Args:
            motion_correct (bool)
                flag for performing motion correction
            indices (list of slice objects)
                perform analysis only on a part of the FOV
            include_eval (bool)
                flag for performing component evaluation
        Returns:
            cnmf object with the current estimates
        """
        if indices is None:
            indices = (slice(None), slice(None))
        fnames = self.params.get('data', 'fnames')
        if os.path.exists(fnames[0]):
            _, extension = os.path.splitext(fnames[0])[:2]
            extension = extension.lower()
        else:
            logging.warning("Error: File not found, with file list:\n" + fnames[0])
            raise Exception('File not found!')

        base_name = pathlib.Path(fnames[0]).stem + "_memmap_"
        if extension == '.mmap':
            fname_new = fnames[0]
            Yr, dims, T = caiman.mmapping.load_memmap(fnames[0])
            if np.isfortran(Yr):
                raise Exception('The file should be in C order (see save_memmap function)')
        else:
            data_set_name = self.params.get('data', 'var_name_hdf5')
            if motion_correct:
                mc = MotionCorrect(fnames, dview=self.dview, **self.params.motion)
                mc.motion_correct(save_movie=True)
                fname_mc = mc.fname_tot_els if self.params.motion['pw_rigid'] else mc.fname_tot_rig
                if self.params.get('motion', 'pw_rigid'):
                    b0 = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                            np.max(np.abs(mc.y_shifts_els)))).astype(int)
                    if self.params.get('motion', 'is3D'):
                        self.estimates.shifts = [mc.x_shifts_els, mc.y_shifts_els, mc.z_shifts_els]
                    else:
                        self.estimates.shifts = [mc.x_shifts_els, mc.y_shifts_els]
                else:
                    b0 = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
                    self.estimates.shifts = mc.shifts_rig
                # TODO - b0 is currently direction inspecific, which can cause
                # sub-optimal behavior. See
                # https://github.com/flatironinstitute/CaImAn/pull/618#discussion_r313960370
                # for further details.
                # b0 = 0 if self.params.get('motion', 'border_nan') == 'copy' else 0
                b0 = 0
                fname_new = caiman.mmapping.save_memmap(fname_mc, base_name=base_name, order='C',
                                                 var_name_hdf5=data_set_name, border_to_0=b0)
            else:
                fname_new = caiman.mmapping.save_memmap(fnames, base_name=base_name, var_name_hdf5=data_set_name, order='C')
            Yr, dims, T = caiman.mmapping.load_memmap(fname_new)

        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        self.mmap_file = fname_new
        if not include_eval:
            return self.fit(images, indices=indices)

        fit_cnm = self.fit(images, indices=indices)
        Cn = caiman.summary_images.local_correlations(images[::max(T//1000, 1)], swap_dim=False)
        Cn[np.isnan(Cn)] = 0
        fit_cnm.save(fname_new[:-5] + '_init.hdf5')
        #fit_cnm.params.change_params({'p': self.params.get('preprocess', 'p')})
        # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
        cnm2 = fit_cnm.refit(images, dview=self.dview)
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=self.dview)
        # update object with selected components
        #cnm2.estimates.select_components(use_object=True)
        # Extract DF/F values
        cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
        cnm2.estimates.Cn = Cn
        cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')

        # XXX Why are we stopping the cluster here? What started it? Why remove log files?
        caiman.cluster.stop_server(dview=self.dview)
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        return cnm2


    def refit(self, images, dview=None):
        """
        Refits the data using CNMF initialized from a previous iteration

        Args:
            images
            dview
        Returns:
            cnm
                A new CNMF object
        """
        cnm = CNMF(self.params.patch['n_processes'], params=self.params, dview=dview)
        cnm.params.patch['rf'] = None
        cnm.params.patch['only_init'] = False
        estimates = deepcopy(self.estimates)
        estimates.select_components(use_object=True)
        estimates.coordinates = None
        cnm.estimates = estimates
        cnm.mmap_file = self.mmap_file
        return cnm.fit(images)

    def fit(self, images, indices=(slice(None), slice(None))):
        """
        This method uses the cnmf algorithm to find sources in data.

        Args:
            images : mapped np.ndarray of shape (t,x,y[,z]) containing the images that vary over time.

            indices: list of slice objects along dimensions (x,y[,z]) for processing only part of the FOV

        Returns:
            self: updated using the cnmf algorithm with C,A,S,b,f computed according to the given initial values

        http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3

        """
        # Todo : to compartment
        if isinstance(indices, slice):
            indices = [indices]
        if isinstance(indices, tuple):
            indices = list(indices)
        indices = [slice(None)] + indices
        if len(indices) < len(images.shape):
            indices = indices + [slice(None)]*(len(images.shape) - len(indices))
        dims_orig = images.shape[1:]
        dims_sliced = images[tuple(indices)].shape[1:]
        is_sliced = (dims_orig != dims_sliced)
        if self.params.get('patch', 'rf') is None and (is_sliced or 'ndarray' in str(type(images))):
            images = images[tuple(indices)]
            self.dview = None
            logging.info("Parallel processing in a single patch "
                            "is not available for loaded in memory or sliced" +
                            " data.")

        T = images.shape[0]
        self.params.set('online', {'init_batch': T})
        self.dims = images.shape[1:]
        #self.params.data['dims'] = images.shape[1:]
        Y = np.transpose(images, list(range(1, len(self.dims) + 1)) + [0])
        Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
        if np.isfortran(Yr):
            raise Exception('The file is in F order, it should be in C order (see save_memmap function)')

        logging.info((T,) + self.dims)

        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        try:
            Y.filename = images.filename
            Yr.filename = images.filename
            self.mmap_file = images.filename
        except AttributeError:  # if no memmapping cause working with small data
            pass

        # update/set all options that depend on data dimensions
        # number of rows, columns [and depths]
        # self.params.set('spatial', {'medw': (3,) * len(self.dims),
        #                             'se': np.ones((3,) * len(self.dims), dtype=np.uint8),
        #                             'ss': np.ones((3,) * len(self.dims), dtype=np.uint8)
        #                             })

        logging.info(('Using ' + str(self.params.get('patch', 'n_processes')) + ' processes'))
        if self.params.get('preprocess', 'n_pixels_per_process') is None:
            avail_memory_per_process = psutil.virtual_memory()[
                1] / 2.**30 / self.params.get('patch', 'n_processes')
            mem_per_pix = 3.6977678498329843e-09
            npx_per_proc = int(avail_memory_per_process / 8. / mem_per_pix / T)
            npx_per_proc = int(np.minimum(npx_per_proc, np.prod(self.dims) // self.params.get('patch', 'n_processes')))
            self.params.set('preprocess', {'n_pixels_per_process': npx_per_proc})

        self.params.set('spatial', {'n_pixels_per_process': self.params.get('preprocess', 'n_pixels_per_process')})

        logging.info('using ' + str(self.params.get('preprocess', 'n_pixels_per_process')) + ' pixels per process')
        logging.info('using ' + str(self.params.get('spatial', 'block_size_spat')) + ' block_size_spat')
        logging.info('using ' + str(self.params.get('temporal', 'block_size_temp')) + ' block_size_temp')

        if self.params.get('patch', 'rf') is None:  # no patches
            logging.info('preprocessing ...')
            Yr = self.preprocess(Yr)
            if self.estimates.A is None:
                logging.info('initializing ...')
                self.initialize(Y)

            if self.params.get('patch', 'only_init'):  # only return values after initialization
                if not (self.params.get('init', 'method_init') == 'corr_pnr' and
                    self.params.get('init', 'ring_size_factor') is not None):
                    self.compute_residuals(Yr)
                    self.estimates.bl = None
                    self.estimates.c1 = None
                    self.estimates.neurons_sn = None


                if self.remove_very_bad_comps:
                    logging.info('removing bad components : ')
                    final_frate = 10
                    r_values_min = 0.5  # threshold on space consistency
                    fitness_min = -15  # threshold on time variability
                    fitness_delta_min = -15
                    Npeaks = 10
                    traces = np.array(self.C)
                    logging.info('estimating the quality...')
                    idx_components, idx_components_bad, fitness_raw,\
                        fitness_delta, r_values = estimate_components_quality(
                            traces, Y, self.estimates.A, self.estimates.C, self.estimates.b, self.estimates.f,
                            final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
                            fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, return_all=True, N=5)

                    logging.info(('Keeping ' + str(len(idx_components)) +
                           ' and discarding  ' + str(len(idx_components_bad))))
                    self.estimates.C = self.estimates.C[idx_components]
                    self.estimates.A = self.estimates.A[:, idx_components] # type: ignore # not provable that self.initialise provides a value
                    self.estimates.YrA = self.estimates.YrA[idx_components]

                self.estimates.normalize_components()

                return self

            logging.info('update spatial ...')
            self.update_spatial(Yr, use_init=True)

            logging.info('update temporal ...')
            if not self.skip_refinement:
                # set this to zero for fast updating without deconvolution
                self.params.set('temporal', {'p': 0})
            else:
                self.params.set('temporal', {'p': self.params.get('preprocess', 'p')})
                logging.info('deconvolution ...')

            self.update_temporal(Yr)

            if not self.skip_refinement:
                logging.info('refinement...')
                if self.params.get('merging', 'do_merge'):
                    logging.info('merging components ...')
                    self.merge_comps(Yr, mx=50, fast_merge=True, max_merge_area=self.params.get('merging', 'max_merge_area'))

                logging.info('Updating spatial ...')

                self.update_spatial(Yr, use_init=False)
                # set it back to original value to perform full deconvolution
                self.params.set('temporal', {'p': self.params.get('preprocess', 'p')})
                logging.info('update temporal ...')
                self.update_temporal(Yr, use_init=False)
            # else:
            #     todo : ask for those..
                # C, f, S, bl, c1, neurons_sn, g1, YrA, lam = self.estimates.C, self.estimates.f, self.estimates.S, self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, self.estimates.g, self.estimates.YrA, self.estimates.lam

            # embed in the whole FOV
            if is_sliced:
                FOV = np.zeros(dims_orig, order='C')
                FOV[indices[1:]] = 1
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
                self.params.set('patch', {'stride': int(self.params.get('patch', 'rf') * 2 * .1)})
                logging.info(
                    ('Setting the stride to 10% of 2*rf automatically:' + str(self.params.get('patch', 'stride'))))

            if not isinstance(images, np.memmap):
                raise Exception(
                    'You need to provide a memory mapped file as input if you use patches!!')

            self.estimates.A, self.estimates.C, self.estimates.YrA, self.estimates.b, self.estimates.f, \
                self.estimates.sn, self.estimates.optional_outputs = run_CNMF_patches(
                    images.filename, self.dims + (T,), self.params,
                    dview=self.dview, memory_fact=self.params.get('patch', 'memory_fact'),
                    gnb=self.params.get('init', 'nb'), border_pix=self.params.get('patch', 'border_pix'),
                    low_rank_background=self.params.get('patch', 'low_rank_background'),
                    del_duplicates=self.params.get('patch', 'del_duplicates'),
                    indices=indices)

            self.estimates.bl, self.estimates.c1, self.estimates.g, self.estimates.neurons_sn = None, None, None, None
            logging.info("merging")
            self.estimates.merged_ROIs = [0]


            if self.params.get('init', 'center_psf'):  # merge taking best neuron
                if self.params.get('patch', 'nb_patch') > 0:

                    while len(self.estimates.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.Inf, fast_merge=True)

                    logging.info("update temporal")
                    self.update_temporal(Yr, use_init=False)

                    self.params.set('spatial', {'se': np.ones((1,) * len(self.dims), dtype=np.uint8)})
                    logging.info('update spatial ...')
                    self.update_spatial(Yr, use_init=False)

                    logging.info("update temporal")
                    self.update_temporal(Yr, use_init=False)
                else:
                    while len(self.estimates.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.Inf, fast_merge=True)
                        #if len(self.estimates.merged_ROIs) > 0:
                            #not_merged = np.setdiff1d(list(range(len(self.estimates.YrA))),
                            #                          np.unique(np.concatenate(self.estimates.merged_ROIs)))
                            #self.estimates.YrA = np.concatenate([self.estimates.YrA[not_merged],
                            #                           np.array([self.estimates.YrA[m].mean(0) for ind, m in enumerate(self.estimates.merged_ROIs) if not self.empty_merged[ind]])])
                    if self.params.get('init', 'nb') == 0:
                        self.estimates.W, self.estimates.b0 = compute_W(
                            Yr, self.estimates.A.toarray(), self.estimates.C, self.dims,
                            self.params.get('init', 'ring_size_factor') *
                            self.params.get('init', 'gSiz')[0],
                            ssub=self.params.get('init', 'ssub_B'),
                            robust=self.params.get('init', 'robust'),
                            sn_pixel=self.estimates.sn,
                            zeta=self.params.get('init', 'zeta'))
                    if len(self.estimates.C):
                        self.deconvolve()
                        self.estimates.C = self.estimates.C.astype(np.float32)
                    else:
                        self.estimates.S = self.estimates.C
            else:
                while len(self.estimates.merged_ROIs) > 0:
                    self.merge_comps(Yr, mx=np.Inf)

                logging.info("update temporal")
                self.update_temporal(Yr, use_init=False)

        self.estimates.normalize_components()
        return self


    def save(self, filename):
        '''save object in hdf5 file format

        Args:
            filename: str
                path to the hdf5 file containing the saved object
        '''

        if '.hdf5' in filename:
            filename = caiman.paths.fn_relocated(filename)
            save_dict_to_hdf5(self.__dict__, filename)
        else:
            raise Exception("File extension not supported for cnmf.save")

    def remove_components(self, ind_rm):
        """
        Remove a specified list of components from the CNMF object.

        Args:
            ind_rm :    list
                        indices of components to be removed
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
        """
        Compute residual trace for each component (variable YrA).
        WARNING: At the moment this method is valid only for the 2p processing
        pipeline

         Args:
             Yr :    np.ndarray
                     movie in format pixels (d) x frames (T)
        """
        block_size, num_blocks_per_run = self.params.get('temporal', 'block_size_temp'), self.params.get('temporal', 'num_blocks_per_run_temp')
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
            1. / (nA2 + np.finfo(np.float32).eps), 0, nA2.shape[0], nA2.shape[0])
        Cf = np.vstack((self.estimates.C, self.estimates.f))
        if 'numpy.ndarray' in str(type(Yr)):
            YA = (Ab.T.dot(Yr)).T * nA2_inv_mat
        else:
            YA = caiman.mmapping.parallel_dot_product(Yr, Ab, dview=self.dview, block_size=block_size,
                                           transpose=True, num_blocks_per_run=num_blocks_per_run) * nA2_inv_mat

        AA = Ab.T.dot(Ab) * nA2_inv_mat
        self.estimates.YrA = (YA - (AA.T.dot(Cf)).T)[:, :self.estimates.A.shape[-1]].T
        self.estimates.R = self.estimates.YrA

        return self


    def deconvolve(self, p=None, method_deconvolution=None, bas_nonneg=None,
                   noise_method=None, optimize_g=0, s_min=None, **kwargs):
        """Performs deconvolution on already extracted traces using
        constrained foopsi.
        """

        p = self.params.get('preprocess', 'p') if p is None else p
        method_deconvolution = (self.params.get('temporal', 'method_deconvolution')
                if method_deconvolution is None else method_deconvolution)
        bas_nonneg = (self.params.get('temporal', 'bas_nonneg')
                      if bas_nonneg is None else bas_nonneg)
        noise_method = (self.params.get('temporal', 'noise_method')
                        if noise_method is None else noise_method)
        s_min = self.params.get('temporal', 's_min') if s_min is None else s_min

        F = self.estimates.C + self.estimates.YrA
        args = dict()
        args['p'] = p
        args['method_deconvolution'] = method_deconvolution
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
        self.estimates.g = [results[6][i] for i in order]
        self.estimates.neurons_sn = [results[5][i] for i in order]
        self.estimates.lam = [results[8][i] for i in order]
        self.estimates.YrA = F - self.estimates.C
        return self

    def HALS4traces(self, Yr, groups=None, use_groups=False, order=None,
                    update_bck=True, bck_non_neg=True, **kwargs):
        """Solves C, f = argmin_C ||Yr-AC-bf|| using block-coordinate decent.
        Can use groups to update non-overlapping components in parallel or a
        specified order.

        Args:
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

        Returns:
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
                                  **kwargs) # FIXME: this function is not defined in this scope
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

        Args:
            Yr: np.array (possibly memory mapped, (x,y,[,z]) x t)
                Imaging data reshaped in matrix format

            update_bck: bool
                flag for updating spatial background components

            num_iter: int
                number of iterations

        Returns:
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
        Ab = HALS4shapes(Yr, Ab, Cf, iters=num_iter) # FIXME: this function is not defined in this scope
        if update_bck:
            self.estimates.A = scipy.sparse.csc_matrix(Ab[:, self.params.get('init', 'nb'):])
            self.estimates.b = Ab[:, :self.params.get('init', 'nb')]
        else:
            self.estimates.A = scipy.sparse.csc_matrix(Ab)

        return self

    def update_temporal(self, Y, use_init=True, **kwargs):
        """Updates temporal components

        Args:
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
        self.estimates.R = self.estimates.YrA
        return self

    def update_spatial(self, Y, use_init=True, **kwargs):
        """Updates spatial components

        Args:
            Y:  np.array (d1*d2) x T
                input data
            use_init: bool
                use Cin, f_in for computing A, b otherwise use C, f

        Returns:
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

    def merge_comps(self, Y, mx=50, fast_merge=True, max_merge_area=None):
        """merges components
        """
        self.estimates.A, self.estimates.C, self.estimates.nr, self.estimates.merged_ROIs, self.estimates.S, \
        self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, self.estimates.g, self.empty_merged, \
        self.estimates.YrA =\
            merge_components(Y, self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.YrA,
                             self.estimates.f, self.estimates.S, self.estimates.sn, self.params.get_group('temporal'),
                             self.params.get_group('spatial'), dview=self.dview,
                             bl=self.estimates.bl, c1=self.estimates.c1, sn=self.estimates.neurons_sn,
                             g=self.estimates.g, thr=self.params.get('merging', 'merge_thr'), mx=mx,
                             fast_merge=fast_merge, merge_parallel=self.params.get('merging', 'merge_parallel'),
                             max_merge_area=max_merge_area)

        return self

    def initialize(self, Y, **kwargs):
        """Component initialization
        """
        self.params.set('init', kwargs)
        estim = self.estimates
        if (self.params.get('init', 'method_init') == 'corr_pnr' and
                self.params.get('init', 'ring_size_factor') is not None):
            estim.A, estim.C, estim.b, estim.f, estim.center, \
                extra_1p = initialize_components(
                    Y, sn=estim.sn, options_total=self.params.to_dict(),
                    **self.params.get_group('init'))
            try:
                estim.S, estim.bl, estim.c1, estim.neurons_sn, \
                    estim.g, estim.YrA, estim.lam = extra_1p
            except:
                estim.S, estim.bl, estim.c1, estim.neurons_sn, \
                    estim.g, estim.YrA, estim.lam, estim.W, estim.b0 = extra_1p
        else:
            estim.A, estim.C, estim.b, estim.f, estim.center =\
                initialize_components(Y, sn=estim.sn, options_total=self.params.to_dict(),
                                      **self.params.get_group('init'))

        self.estimates = estim

        return self

    def preprocess(self, Yr):
        """
        Examines data to remove corrupted pixels and computes the noise level
        estimate for each pixel.

        Args:
            Yr: np.array (or memmap.array)
                2d array of data (pixels x timesteps) typically in memory
                mapped form
        """
        Yr, self.estimates.sn, self.estimates.g, self.estimates.psx = preprocess_data(
            Yr, dview=self.dview, **self.params.get_group('preprocess'))
        return Yr


def load_CNMF(filename:str, n_processes=1, dview=None):
    '''load object saved with the CNMF save method

    Args:
        filename:
            hdf5 (or nwb) file name containing the saved object
        dview: multiprocessing or ipyparallel object
            used to set up parallelization, default None
    '''
    new_obj = CNMF(n_processes)
    if os.path.splitext(filename)[1].lower() in ('.hdf5', '.h5'):
        filename = caiman.paths.fn_relocated(filename)
        for key, val in load_dict_from_hdf5(filename).items():
            if key == 'params':
                prms = CNMFParams()
                for subdict in val.keys():
                    prms.set(subdict, val[subdict])
                setattr(new_obj, key, prms)
            elif key == 'dview':
                setattr(new_obj, key, dview)
            elif key == 'estimates':
                estims = Estimates()
                for kk, vv in val.items():
                    if kk == 'discarded_components':
                        if vv is not None and vv != b'NoneType':
                            discarded_components = Estimates()
                            for kk__, vv__ in vv.items():
                                setattr(discarded_components, kk__, vv__)
                            setattr(estims, kk, discarded_components)
                    else:
                        setattr(estims, kk, vv)

                setattr(new_obj, key, estims)
            else:
                setattr(new_obj, key, val)
        if new_obj.estimates.dims is None or new_obj.estimates.dims == b'NoneType':
            new_obj.estimates.dims = new_obj.dims
    elif os.path.splitext(filename)[1].lower() == '.nwb':
        from pynwb import NWBHDF5IO
        with NWBHDF5IO(filename, 'r') as io:
            nwb = io.read()
            ophys = nwb.processing['ophys']
            rrs_group = ophys.data_interfaces['Fluorescence'].roi_response_series
            rrs = rrs_group['RoiResponseSeries']
            C = rrs.data[:].T
            rois = rrs.rois
            roi_indices = rois.data
            A = rois.table['image_mask'][roi_indices, ...]
            dims = A.shape[1:]
            A = A.reshape((A.shape[0], -1)).T
            A = scipy.sparse.csc_matrix(A)
            if 'Background_Fluorescence_Response' in rrs_group:
                brs = rrs_group['Background_Fluorescence_Response']
                f = brs.data[:].T
                brois = brs.rois
                broi_indices = brois.data
                b = brois.table['image_mask'][broi_indices, ...]
                b = b.reshape((b.shape[0], -1)).T
            else:
                b = None #np.zeros(mov.shape[1:])
                f = None
            estims = Estimates(A=A, b=b, C=C, f=f)
            estims.YrA = ophys.data_interfaces['residuals'].data[:].T

            frame_rate = ophys.data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']. \
                imaging_plane.imaging_rate

            if 'r' in rois.table:
                estims.r_values = rois.table['r'][roi_indices]
            if 'snr' in rois.table:
                estims.SNR_comp = rois.table['snr'][roi_indices]
            if 'cnn' in rois.table:
                estims.cnn_preds = rois.table['cnn'][roi_indices]
            if 'keep' in rois.table:
                keep = rois.table['keep'][roi_indices]
                estims.idx_components = np.where(keep)[0]
            if 'accepted' in rois.table:
                accepted = rois.table['accepted'][roi_indices]
                estims.accepted_list = np.where(accepted==True)[0]
            if 'rejected' in rois.table:
                rejected = rois.table['rejected'][roi_indices]
                estims.rejected_list = np.where(rejected==True)[0]                
                print(estims.rejected_list)
            estims.nr = len(roi_indices)

            if 'summary_images' in ophys.data_interfaces:
                if 'Cn' in ophys.data_interfaces['summary_images']:
                    estims.Cn = ophys.data_interfaces['summary_images']['Cn']
            if hasattr(nwb.acquisition['TwoPhotonSeries'], 'external_file'):
                setattr(new_obj, 'mmap_file', nwb.acquisition['TwoPhotonSeries'].external_file[0])
            else:
                setattr(new_obj, 'mmap_file', filename)

            estims.dims = dims
            prms = CNMFParams(dims=dims)
            prms.set('data', {'fr': frame_rate})

            setattr(new_obj, 'params', prms)
            setattr(new_obj, 'dview', dview)
            setattr(new_obj, 'estimates', estims)

    else:
        raise NotImplementedError('unsupported file extension')

    return new_obj
