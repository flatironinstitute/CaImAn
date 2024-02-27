#!/usr/bin/env python

import json
import logging
import numpy as np
import os
import pkg_resources
from pprint import pformat
import scipy
from scipy.ndimage import generate_binary_structure, iterate_structure
from typing import Optional

import caiman.utils.utils
import caiman.base.movies
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf.utilities import dict_compare


class CNMFParams(object):
    """Class for setting and changing the various parameters."""
    
    def __init__(self, fnames=None, dims=None, dxy=(1, 1),
                 border_pix=0, del_duplicates=False, low_rank_background=True,
                 memory_fact=1, n_processes=1, nb_patch=1, p_ssub=2, p_tsub=2,
                 remove_very_bad_comps=False, rf=None, stride=None,
                 check_nan=True, n_pixels_per_process=None,
                 k=30, alpha_snmf=0.5, center_psf=False, gSig=[5, 5], gSiz=None,
                 init_iter=2, method_init='greedy_roi', min_corr=.85,
                 min_pnr=20, gnb=1, normalize_init=True, options_local_NMF=None,
                 ring_size_factor=1.5, rolling_length=100, rolling_sum=True,
                 ssub=2, ssub_B=2, tsub=2,
                 block_size_spat=5000, num_blocks_per_run_spat=20,
                 block_size_temp=5000, num_blocks_per_run_temp=20,
                 update_background_components=True,
                 method_deconvolution='oasis', p=2, s_min=None,
                 do_merge=True, merge_thresh=0.8,
                 decay_time=0.4, fr=30, min_SNR=2.5, rval_thr=0.8,
                 N_samples_exceptionality=None, batch_update_suff_stat=False,
                 expected_comps=500, iters_shape=5, max_comp_update_shape=np.inf,
                 max_num_added=5, min_num_trial=5, minibatch_shape=100, minibatch_suff_stat=5,
                 n_refit=0, num_times_comp_updated=np.inf, simultaneously=False,
                 sniper_mode=False, test_both=False, thresh_CNN_noisy=0.5,
                 thresh_fitness_delta=-50, thresh_fitness_raw=None, thresh_overlap=0.5,
                 update_freq=200, update_num_comps=True, use_dense=True, use_peak_max=True,
                 only_init_patch=True, var_name_hdf5='mov', max_merge_area=None, 
                 use_corr_img=False,
                 params_from_file:Optional[str]=None,
                 params_dict={},
                 ):
        """Class for setting the processing parameters. All parameters for CNMF, online-CNMF, quality testing,
        and motion correction can be set here and then used in the various processing pipeline steps.

        Params have default values; users can override the defaults in two intended ways:
            A) During initialisation of the object, people can pass a nested dictionary through the
               params_dict parameter, or the name of a jsonfile containing the same nested dictionary
               through the params_from_file parameter
            B) If the CNMFParams object already exists, they can call its change_params() method to pass in
               a dict or change_params_from_jsonfile() to pass in a filename
        With both of these, people only need to name and override values they wish to change; all others keep
        their defaults.

        All other means of changing parameters are deprecated (including other constructor arguments)
        and will be removed in some future version of Caiman (whether they give a deprecation warning or not). 

        Args:
            params_from_file
                name of a json file used to initialise the object
            params_dict
                a dictionary used to initialise the object

            Any parameter that is not set uses a default value
            All other arguments are deprecated and should not be used.

        Object Structure:
          CNMFParams.data (these represent features of the data and other misc settings):
            fnames
                list of complete paths to files that need to be processed

            dims: (int, int), default: computed from fnames
                dimensions of the FOV in pixels

            fr: float, default: 30
                imaging rate in frames per second

            decay_time: float, default: 0.4
                length of typical transient in seconds

            dxy: (float, float)
                spatial resolution of FOV in pixels per um

            var_name_hdf5: str, default: 'mov'
                if loading from hdf5 name of the variable to load

            caiman_version: str
                version of CaImAn being used. Please do not override this

            last_commit: str
                hash of last commit in the caiman repo. Pleaes do not override this.

            mmap_F: list[str]
                paths to F-order memory mapped files after motion correction

            mmap_C: str
                path to C-order memory mapped file after motion correction

          CNMFParams.patch (these control how the data is divided into patches):
            border_pix: int, default: 0
                Number of pixels to exclude around each border.

            del_duplicates: bool, default: False
                Delete duplicate components in the overlapping regions between neighboring patches. If False,
                then merging is used.

            in_memory: bool, default: True
                Whether to load patches in memory

            low_rank_background: bool, default: True
                Whether to update the background using a low rank approximation.
                If False all the nonzero elements of the background components are updated using hals
                (to be used with one background per patch)

            memory_fact: float, default: 1
                unitless number for increasing the amount of available memory

            n_processes: int
                Number of processes used for processing patches in parallel

            nb_patch: int, default: 1
                Number of (local) background components per patch

            only_init: bool, default: True
                whether to run only the initialization

            p_patch: int, default: 0
                order of AR dynamics when processing within a patch

            remove_very_bad_comps: bool, default: True
                Whether to remove (very) bad quality components during patch processing

            rf: int or list or None, default: None
                Half-size of patch in pixels. If None, no patches are constructed and the whole FOV is processed jointly.
                If list, it should be a list of two elements corresponding to the height and width of patches

            skip_refinement: bool, default: False
                Whether to skip refinement of components (deprecated?)

            p_ssub: float, default: 2
                Spatial downsampling factor

            stride: int or None, default: None
                Overlap between neighboring patches in pixels.

            p_tsub: float, default: 2
                Temporal downsampling factor

          CNMFParams.preprocess (these control preprocessing steps for the data):
            check_nan: bool, default: True
                whether to check for NaNs

            compute_g': bool, default: False
                whether to estimate global time constant

            include_noise: bool, default: False
                    flag for using noise values when estimating g

            lags: int, default: 5
                number of lags to be considered for time constant estimation

            max_num_samples_fft: int, default: 3*1024
                Chunk size for computing the PSD of the data (for memory considerations)

            n_pixels_per_process: int, default: 1000
                Number of pixels to be allocated to each process

            noise_method: 'mean'|'median'|'logmexp', default: 'mean'
                PSD averaging method for computing the noise std

            noise_range: [float, float], default: [.25, .5]
                range of normalized frequencies over which to compute the PSD for noise determination

            p: int, default: 2
                 order of AR indicator dynamics

            pixels: list, default: None
                 pixels to be excluded due to saturation

            sn: np.array or None, default: None
                noise level for each pixel

          CNMFParams.init (these control how CNMF should be initialised):
            K: int, default: 30
                number of components to be found (per patch or whole FOV depending on whether rf=None)

            SC_kernel: {'heat', 'cos', binary'}, default: 'heat'
                kernel for graph affinity matrix

            SC_sigma: float, default: 1
                variance for SC kernel

            SC_thr: float, default: 0,
                threshold for affinity matrix

            SC_normalize: bool, default: True
                standardize entries prior to computing the affinity matrix

            SC_use_NN: bool, default: False
                sparsify affinity matrix by using only nearest neighbors

            SC_nnn: int, default: 20
                number of nearest neighbors to use

            alpha_snmf: float, default: 0.5
                sparse NMF sparsity regularization weight

            center_psf: bool, default: False
                whether to use 1p data processing mode. Set to true for 1p

            gSig: [int, int], default: [5, 5]
                radius of average neurons (in pixels)

            gSiz: [int, int], default: [int(round((x * 2) + 1)) for x in gSig],
                half-size of bounding box for each neuron

            init_iter: int, default: 2
                number of iterations during corr_pnr (1p) initialization

            kernel: np.array or None, default: None
                user specified template for greedyROI

            lambda_gnmf: float, default: 1.
                regularization weight for graph NMF

            maxIter: int, default: 5
                number of HALS iterations during initialization

            max_iter_snmf : int, default: 500
                maximum number of iterations for sparse NMF initialization

            method_init: 'greedy_roi'|'corr_pnr'|'sparse_NMF'|'local_NMF' default: 'greedy_roi'
                initialization method. use 'corr_pnr' for 1p processing and 'sparse_NMF' for dendritic processing.

            min_corr: float, default: 0.85
                minimum value of correlation image for determining a candidate component during corr_pnr

            min_pnr: float, default: 20
                minimum value of psnr image for determining a candidate component during corr_pnr

            nIter: int, default: 5
                number of rank-1 refinement iterations during greedy_roi initialization

            nb: int, default: 1
                number of background components

            normalize_init: bool, default: True
                whether to equalize the movies during initialization

            options_local_NMF: dict
                dictionary with parameters to pass to local_NMF initializer

            perc_baseline_snmf: float, default: 20
                percentile to be removed from the data in sparse_NMF prior to decomposition

            ring_size_factor: float, default: 1.5
                radius of ring (*gSig) for computing background during corr_pnr

            rolling_length: int, default: 100
                width of rolling window for rolling sum option

            rolling_sum: bool, default: True
                use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi

            seed_method: str {'auto', 'manual', 'semi'}
                methods for choosing seed pixels during greedy_roi or corr_pnr initialization
                'semi' detects nr components automatically and allows to add more manually
                if running as notebook 'semi' and 'manual' require a backend that does not
                inline figures, e.g. %matplotlib tk

            sigma_smooth_snmf : (float, float, float), default: (.5,.5,.5)
                std of Gaussian kernel for smoothing data in sparse_NMF

            ssub: float, default: 2
                spatial downsampling factor

            ssub_B: float, default: 2
                downsampling factor for background during corr_pnr

            tsub: float, default: 2
                temporal downsampling factor

          CNMFParams.spatial (these control how the algorithms handle spatial components):
            block_size_spat : int, default: 5000
                Number of pixels to process at the same time for dot product. Reduce if you face memory problems

            dist: float, default: 3
                expansion factor of ellipse

            expandCore: morphological element, default: None(?)
                morphological element for expanding footprints under dilate

            extract_cc: bool, default: True
                whether to extract connected components during thresholding
                (might want to turn to False for dendritic imaging)

            maxthr: float, default: 0.1
                Max threshold

            medw: (int, int) default: None
                window of median filter (set to (3,)*len(dims) in cnmf.fit)

            method_exp: 'dilate'|'ellipse', default: 'dilate'
                method for expanding footprint of spatial components

            method_ls: 'lasso_lars'|'nnls_L0', default: 'lasso_lars'
                'nnls_L0'. Nonnegative least square with L0 penalty
                'lasso_lars' lasso lars function from scikit learn

            n_pixels_per_process: int, default: 1000
                number of pixels to be processed by each worker

            normalize_yyt_one: bool, default: True
                Whether to normalize the C and A matrices so that diag(C*C.T) = 1 during update spatial

            nrgthr: float, default: 0.9999
                Energy threshold

            num_blocks_per_run_spat: int, default: 20
                Parallelization of A'*Y operation

            se: np.array or None, default: None
                 Morphological closing structuring element (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

            ss: np.array or None, default: None
                Binary element for determining connectivity (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

            thr_method: 'nrg'|'max', default: 'nrg'
                thresholding method

            update_background_components: bool, default: True
                whether to update the spatial background components


          CNMFParams.temporal (these control how the algorithms handle temporal components):
            ITER: int, default: 2
                block coordinate descent iterations

            bas_nonneg: bool, default: True
                whether to set a non-negative baseline (otherwise b >= min(y))

            block_size_temp : int, default: 5000
                Number of pixels to process at the same time for dot product. Reduce if you face memory problems

            fudge_factor: float (close but smaller than 1) default: .96
                bias correction factor for discrete time constants

            lags: int, default: 5
                number of autocovariance lags to be considered for time constant estimation

            optimize_g: bool, default: False
                flag for optimizing time constants

            memory_efficient:
                (undocumented)

            method_deconvolution: 'oasis'|'cvxpy'|'oasis', default: 'oasis'
                method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
                if method cvxpy, primary and secondary (if problem unfeasible for approx solution)

            noise_method: 'mean'|'median'|'logmexp', default: 'mean'
                PSD averaging method for computing the noise std

            noise_range: [float, float], default: [.25, .5]
                range of normalized frequencies over which to compute the PSD for noise determination

            num_blocks_per_run_temp: int, default: 20
                Parallelization of A'*Y operation

            p: 0|1|2, default: 2
                order of AR indicator dynamics

            s_min: float or None, default: None
                Minimum spike threshold amplitude (computed in the code if used).

            solvers: 'ECOS'|'SCS', default: ['ECOS', 'SCS']
                 solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'

            verbosity: bool, default: False
                whether to be verbose

          CNMFParams.merging (these control how components are merged):
            do_merge: bool, default: True
                Whether or not to merge

            merge_thr: float, default: 0.8
                Trace correlation threshold for merging two components.

            merge_parallel: bool, default: False
                Perform merging in parallel

            max_merge_area: int or None, default: None
                maximum area (in pixels) of merged components, used to determine whether to merge components during fitting process

          CNMFParams.quality (these control how quality of traces are evaluated):
            SNR_lowest: float, default: 0.5
                minimum required trace SNR. Traces with SNR below this will get rejected

            cnn_lowest: float, default: 0.1
                minimum required CNN threshold. Components with score lower than this will get rejected.

            gSig_range: list or integers, default: None
                gSig scale values for CNN classifier. In not None, multiple values are tested in the CNN classifier.

            min_SNR: float, default: 2.5
                trace SNR threshold. Traces with SNR above this will get accepted

            min_cnn_thr: float, default: 0.9
                CNN classifier threshold. Components with score higher than this will get accepted

            rval_lowest: float, default: -1
                minimum required space correlation. Components with correlation below this will get rejected

            rval_thr: float, default: 0.8
                space correlation threshold. Components with correlation higher than this will get accepted

            use_cnn: bool, default: True
                flag for using the CNN classifier.

            use_ecc:
                (undocumented)

            max_ecc:
                (undocumented)

          CNMFParams.online (these control the Online/OnACID mode):
            N_samples_exceptionality: int, default: np.ceil(decay_time*fr),
                Number of frames over which trace SNR is computed (usually length of a typical transient)

            batch_update_suff_stat: bool, default: False
                Whether to update sufficient statistics in batch mode

            dist_shape_update: bool, default: False,
                update shapes in a distributed fashion

            ds_factor: int, default: 1,
                spatial downsampling factor for faster processing (if > 1)

            epochs: int, default: 1,
                number of times to go over data

            expected_comps: int, default: 500
                number of expected components (for memory allocation purposes)

            full_XXt: bool, default: False
                save the full residual sufficient statistic matrix for updating W in 1p.
                If set to False, a list of submatrices is saved (typically faster).
            
            init_batch: int, default: 200,
                length of mini batch used for initialization

            init_method: 'bare'|'cnmf'|'seeded', default: 'bare',
                initialization method

            iters_shape: int, default: 5
                Number of block-coordinate decent iterations for each shape update

            max_comp_update_shape: int, default: np.inf
                Maximum number of spatial components to be updated at each time

            max_num_added: int, default: 5
                Maximum number of new components to be added in each frame

            max_shifts_online: int, default: 10,
                Maximum shifts for motion correction during online processing

            min_SNR: float, default: 2.5
                Trace SNR threshold for accepting a new component

            min_num_trial: int, default: 5
                Number of mew possible components for each frame

            minibatch_shape: int, default: 100
                Number of frames stored in rolling buffer

            minibatch_suff_stat: int, default: 5
                mini batch size for updating sufficient statistics

            motion_correct: bool, default: True
                Whether to perform motion correction during online processing

            movie_name_online: str, default: 'online_movie.avi'
                Name of saved movie (appended in the data directory)

            normalize: bool, default: False
                Whether to normalize each frame prior to online processing

            n_refit: int, default: 0
                Number of additional iterations for computing traces

            num_times_comp_updated: int, default: np.inf
                (undocumented)

            opencv_codec: str, default: 'H264'
                FourCC video codec for saving movie. Check http://www.fourcc.org/codecs.php

            path_to_model: str, default: os.path.join(caiman_datadir(), 'model', 'cnn_model_online.h5')
                Path to online CNN classifier

            ring_CNN:
                Whether to use a ring CNN model (XXX due to bugs, this flag may not work and may never have worked)

            rval_thr: float, default: 0.8
                space correlation threshold for accepting a new component

            save_online_movie: bool, default: False
                Whether to save the results movie

            show_movie: bool, default: False
                Whether to display movie of online processing

            simultaneously: bool, default: False
                Whether to demix and deconvolve simultaneously

            sniper_mode: bool, default: False
                Whether to use the online CNN classifier for screening candidate components (otherwise space
                correlation is used)

            stop_detection:
                Stop detecting neurons at the last epoch (XXX what does this mean?)

            test_both: bool, default: False
                Whether to use both the CNN and space correlation for screening new components

            thresh_CNN_noisy: float, default: 0,5,
                Threshold for the online CNN classifier

            thresh_fitness_delta: float (negative)
                Derivative test for detecting traces

            thresh_fitness_raw: float (negative), default: computed from min_SNR
                Threshold value for testing trace SNR

            thresh_overlap: float, default: 0.5
                Intersection-over-Union space overlap threshold for screening new components

            update_freq: int, default: 200
                Update each shape at least once every X frames when in distributed mode

            update_num_comps: bool, default: True
                Whether to search for new components

            use_corr_img:
                Use correlation image to detect new components

            use_dense: bool, default: True
                Whether to store and represent A and b as a dense matrix

            use_peak_max: bool, default: True
                Whether to find candidate centroids using skimage's find local peaks function

            W_update_factor:
                Update W less often than shapes by a given factor (XXX does this work?)

          CNMFParams.motion (these control motion-correction):
            border_nan: bool or str, default: 'copy'
                flag for allowing NaN in the boundaries. True allows NaN, whereas 'copy' copies the value of the
                nearest data point.

            gSig_filt: int or None, default: None
                size of kernel for high pass spatial filtering in 1p data. If None no spatial filtering is performed

            is3D: bool, default: False
                flag for 3D recordings for motion correction

            max_deviation_rigid: int, default: 3
                maximum deviation in pixels between rigid shifts and shifts of individual patches

            max_shifts: (int, int), default: (6,6)
                maximum shifts per dimension in pixels.

            min_mov: float or None, default: None
                minimum value of movie. If None it get computed.

            niter_rig: int, default: 1
                number of iterations rigid motion correction.

            nonneg_movie: bool, default: True
                flag for producing a non-negative movie.

            num_frames_split: int, default: 80
                split movie every x frames for parallel processing

            num_splits_to_process_rig, default: None
                (Undocumented, changing this likely to break the code - FIXME why is this a parameter then?)

            overlaps: (int, int), default: (24, 24)
                overlap between patches in pixels in pw-rigid motion correction.

            pw_rigid: bool, default: False
                flag for performing pw-rigid motion correction.

            shifts_opencv: bool, default: True
                flag for applying shifts using cubic interpolation (otherwise FFT)

            splits_els: int, default: 14
                number of splits across time for pw-rigid registration.

            splits_rig: int, default: 14
                number of splits across time for rigid registration.

            strides: (int, int), default: (96, 96)
                how often to start a new patch in pw-rigid registration. Size of each patch will be strides + overlaps

            upsample_factor_grid" int, default: 4
                motion field upsampling factor during FFT shifts.

            use_cuda: bool, default: False
                flag for using a GPU.

            indices: tuple(slice), default: (slice(None), slice(None))
                Use that to apply motion correction only on a part of the FOV

          CNMFParams.ring_CNN (these control the ring neural networks):
            n_channels: int, default: 2
                Number of "ring" kernels

            use_bias: bool, default: False
                Flag for using bias in the convolutions

            use_add: bool, default: False
                Flag for using an additive layer

            pct: float between 0 and 1, default: 0.01
                Quantile used during training with quantile loss function

            patience: int, default: 3
                Number of epochs to wait before early stopping

            max_epochs: int, default: 100
                Maximum number of epochs to be used during training

            width: int, default: 5
                Width of "ring" kernel

            loss_fn: str, default: 'pct'
                Loss function specification ('pct' for quantile loss function,
                'mse' for mean squared error)

            lr: float, default: 1e-3
                (initial) learning rate

            lr_scheduler: function, default: None
                Learning rate scheduler function

            path_to_model: str, default: None
                Path to saved weights (if training then path to saved model weights)

            remove_activity: bool, default: False
                Flag for removing activity of last frame prior to background extraction

            reuse_model: bool, default: False
                Flag for reusing an already trained model (saved in path to model)
        """

        if float(decay_time) == float(0.0):
            raise Exception("A decay time of zero is not permitted")

        self.data = {
            'fnames': fnames,
            'dims': dims,
            'fr': fr,
            'decay_time': decay_time,
            'dxy': dxy,
            'var_name_hdf5': var_name_hdf5,
            'caiman_version': pkg_resources.get_distribution('caiman').version,
            'last_commit': None,
            'mmap_F': None,
            'mmap_C': None
        }

        self.patch = {
            'border_pix': border_pix,
            'del_duplicates': del_duplicates,
            'in_memory': True,
            'low_rank_background': low_rank_background,
            'memory_fact': memory_fact,
            'n_processes': n_processes,
            'nb_patch': nb_patch,
            'only_init': only_init_patch,
            'p_patch': 0,                 # AR order within patch
            'remove_very_bad_comps': remove_very_bad_comps,
            'rf': rf,
            'skip_refinement': False,
            'p_ssub': p_ssub,             # spatial downsampling factor
            'stride': stride,
            'p_tsub': p_tsub,             # temporal downsampling factor
        }

        self.preprocess = {
            'check_nan': check_nan,
            'compute_g': False,          # flag for estimating global time constant
            'include_noise': False,      # flag for using noise values when estimating g
            # number of autocovariance lags to be considered for time constant estimation
            'lags': 5,
            'max_num_samples_fft': 3 * 1024,
            'n_pixels_per_process': n_pixels_per_process,
            'noise_method': 'mean',      # averaging method ('mean','median','logmexp')
            'noise_range': [0.25, 0.5],  # range of normalized frequencies over which to average
            'p': p,                      # order of AR indicator dynamics
            'pixels': None,              # pixels to be excluded due to saturation
            'sn': None,                  # noise level for each pixel
        }

        self.init = {
            'K': k,                   # number of components,
            'SC_kernel': 'heat',         # kernel for graph affinity matrix
            'SC_sigma' : 1,              # std for SC kernel
            'SC_thr': 0,                 # threshold for affinity matrix
            'SC_normalize': True,        # standardize entries prior to
                                         # computing affinity matrix
            'SC_use_NN': False,          # sparsify affinity matrix by using
                                         # only nearest neighbors
            'SC_nnn': 20,                # number of nearest neighbors to use
            'alpha_snmf': alpha_snmf,
            'center_psf': center_psf,
            'gSig': gSig,
            # size of bounding box
            'gSiz': gSiz,
            'init_iter': init_iter,
            'kernel': None,           # user specified template for greedyROI
            'lambda_gnmf' :1,         # regularization weight for graph NMF
            'maxIter': 5,             # number of HALS iterations
            'max_iter_snmf': 500,
            'method_init': method_init,    # can be greedy_roi, corr_pnr sparse_nmf, local_NMF
            'min_corr': min_corr,
            'min_pnr': min_pnr,
            'nIter': 5,               # number of refinement iterations
            'nb': gnb,                # number of global background components
            # whether to pixelwise equalize the movies during initialization
            'normalize_init': normalize_init,
            # dictionary with parameters to pass to local_NMF initializer
            'options_local_NMF': options_local_NMF,
            'perc_baseline_snmf': 20,
            'ring_size_factor': ring_size_factor,
            'rolling_length': rolling_length,
            'rolling_sum': rolling_sum,
            'seed_method': 'auto',
            'sigma_smooth_snmf': (.5, .5, .5),
            'ssub': ssub,             # spatial downsampling factor
            'ssub_B': ssub_B,
            'tsub': tsub,             # temporal downsampling factor
        }

        self.spatial = {
            'block_size_spat': block_size_spat, # number of pixels to parallelize residual computation ** DECREASE IF MEMORY ISSUES
            'dist': 3,                       # expansion factor of ellipse
            'expandCore': iterate_structure(generate_binary_structure(2, 1), 2).astype(int),
            # Flag to extract connected components (might want to turn to False for dendritic imaging)
            'extract_cc': True,
            'maxthr': 0.1,                   # Max threshold
            'medw': None,                    # window of median filter
            # method for determining footprint of spatial components ('ellipse' or 'dilate')
            'method_exp': 'dilate',
            # 'nnls_L0'. Nonnegative least square with L0 penalty
            # 'lasso_lars' lasso lars function from scikit learn
            'method_ls': 'lasso_lars',
            # number of pixels to be processed by each worker
            'n_pixels_per_process': n_pixels_per_process,
            'normalize_yyt_one': True,
            'nrgthr': 0.9999,                # Energy threshold
            'num_blocks_per_run_spat': num_blocks_per_run_spat, # number of process to parallelize residual computation ** DECREASE IF MEMORY ISSUES
            'se': None,  # Morphological closing structuring element
            'ss': None,  # Binary element for determining connectivity
            'thr_method': 'nrg',             # Method of thresholding ('max' or 'nrg')
            # whether to update the background components in the spatial phase
            'update_background_components': update_background_components,
        }

        self.temporal = {
            'ITER': 2,                  # block coordinate descent iterations
            # flag for setting non-negative baseline (otherwise b >= min(y))
            'bas_nonneg': False,
            # number of pixels to process at the same time for dot product. Make it
            # smaller if memory problems
            'block_size_temp': block_size_temp, # number of pixels to parallelize residual computation ** DECREASE IF MEMORY ISSUES
            # bias correction factor (between 0 and 1, close to 1)
            'fudge_factor': .96,
            # number of autocovariance lags to be considered for time constant estimation
            'lags': 5,
            'optimize_g': False,         # flag for optimizing time constants
            'memory_efficient': False,
            # method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
            # if method cvxpy, primary and secondary (if problem unfeasible for approx
            # solution) solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
            'method_deconvolution': method_deconvolution,  # 'cvxpy', # 'oasis'
            'noise_method': 'mean',     # averaging method ('mean','median','logmexp')
            'noise_range': [.25, .5],   # range of normalized frequencies over which to average
            'num_blocks_per_run_temp': num_blocks_per_run_temp, # number of process to parallelize residual computation ** DECREASE IF MEMORY ISSUES
            'p': p,                     # order of AR indicator dynamics
            's_min': s_min,             # minimum spike threshold
            'solvers': ['ECOS', 'SCS'],
            'verbosity': False,
        }

        self.merging = {
            'do_merge': do_merge,
            'merge_thr': merge_thresh,
            'merge_parallel': False,
            'max_merge_area': max_merge_area
        }

        self.quality = {
            'SNR_lowest': 0.5,         # minimum accepted SNR value
            'cnn_lowest': 0.1,         # minimum accepted value for CNN classifier
            'gSig_range': None,        # range for gSig scale for CNN classifier
            'min_SNR': min_SNR,        # transient SNR threshold
            'min_cnn_thr': 0.9,        # threshold for CNN classifier
            'rval_lowest': -1,         # minimum accepted space correlation
            'rval_thr': rval_thr,      # space correlation threshold
            'use_cnn': True,           # use CNN based classifier
            'use_ecc': False,          # flag for eccentricity based filtering
            'max_ecc': 3
        }

        self.online = {
            'N_samples_exceptionality': N_samples_exceptionality,  # timesteps to compute SNR
            'batch_update_suff_stat': batch_update_suff_stat,
            'dist_shape_update': False,        # update shapes in a distributed way
            'ds_factor': 1,                    # spatial downsampling for faster processing
            'epochs': 1,                       # number of epochs
            'expected_comps': expected_comps,  # number of expected components
            'full_XXt': False,                 # store entire XXt matrix (as opposed to a list of sub-matrices) 
            'init_batch': 200,                 # length of mini batch for initialization
            'init_method': 'bare',             # initialization method for first batch,
            'iters_shape': iters_shape,        # number of block-CD iterations
            'max_comp_update_shape': max_comp_update_shape,
            'max_num_added': max_num_added,    # maximum number of new components for each frame
            'max_shifts_online': 10,           # maximum shifts during motion correction
            'min_SNR': min_SNR,                # minimum SNR for accepting a new trace
            'min_num_trial': min_num_trial,    # number of mew possible components for each frame
            'minibatch_shape': minibatch_shape,  # number of frames in each minibatch
            'minibatch_suff_stat': minibatch_suff_stat,
            'motion_correct': True,            # flag for motion correction
            'movie_name_online': 'online_movie.mp4',  # filename of saved movie (appended to directory where data is located)
            'normalize': False,                # normalize frame
            'n_refit': n_refit,                # Additional iterations to simultaneously refit
            # path to CNN model for testing new comps
            'num_times_comp_updated': num_times_comp_updated,
            'opencv_codec': 'H264',            # FourCC video codec for saving movie. Check http://www.fourcc.org/codecs.php
            'path_to_model': os.path.join(caiman_datadir(), 'model',
                                          'cnn_model_online.h5'),
            'ring_CNN': False,                 # flag for using a ring CNN background model 
            'rval_thr': rval_thr,              # space correlation threshold
            'save_online_movie': False,        # flag for saving online movie
            'show_movie': False,               # display movie online
            'simultaneously': simultaneously,  # demix and deconvolve simultaneously
            'sniper_mode': sniper_mode,        # flag for using CNN
            'stop_detection': False,           # flag for stop detecting new neurons at the last epoch 
            'test_both': test_both,            # flag for using both CNN and space correlation
            'thresh_CNN_noisy': thresh_CNN_noisy,  # threshold for online CNN classifier
            'thresh_fitness_delta': thresh_fitness_delta,
            'thresh_fitness_raw': thresh_fitness_raw,    # threshold for trace SNR (computed below)
            'thresh_overlap': thresh_overlap,
            'update_freq': update_freq,            # update every shape at least once every update_freq steps
            'update_num_comps': update_num_comps,  # flag for searching for new components
            'use_corr_img': use_corr_img,      # flag for using correlation image to detect new components
            'use_dense': use_dense,            # flag for representation and storing of A and b
            'use_peak_max': use_peak_max,      # flag for finding candidate centroids
            'W_update_factor': 1,              # update W less often than shapes by a given factor 
        }

        self.motion = {
            'border_nan': 'copy',               # flag for allowing NaN in the boundaries
            'gSig_filt': None,                  # size of kernel for high pass spatial filtering in 1p data
            'is3D': False,                      # flag for 3D recordings for motion correction
            'max_deviation_rigid': 3,           # maximum deviation between rigid and non-rigid
            'max_shifts': (6, 6),               # maximum shifts per dimension (in pixels)
            'min_mov': None,                    # minimum value of movie
            'niter_rig': 1,                     # number of iterations rigid motion correction
            'nonneg_movie': True,               # flag for producing a non-negative movie
            'num_frames_split': 80,             # split across time every x frames
            'num_splits_to_process_els': None,  # Unused, will be removed in a future version of Caiman
            'num_splits_to_process_rig': None,  # DO NOT MODIFY
            'overlaps': (32, 32),               # overlap between patches in pw-rigid motion correction
            'pw_rigid': False,                  # flag for performing pw-rigid motion correction
            'shifts_opencv': True,              # flag for applying shifts using cubic interpolation (otherwise FFT)
            'splits_els': 14,                   # number of splits across time for pw-rigid registration
            'splits_rig': 14,                   # number of splits across time for rigid registration
            'strides': (96, 96),                # how often to start a new patch in pw-rigid registration
            'upsample_factor_grid': 4,          # motion field upsampling factor during FFT shifts
            'use_cuda': False,                  # flag for using a GPU
            'indices': (slice(None), slice(None))  # part of FOV to be corrected
        }

        self.ring_CNN = {
            'n_channels' : 2,                   # number of "ring" kernels   
            'use_bias' : False,                 # use bias in the convolutions
            'use_add' : False,                  # use an additive layer
            'pct' : 0.01,                       # quantile loss specification
            'patience' : 3,                     # patience for early stopping
            'max_epochs': 100,                  # maximum number of epochs
            'width': 5,                         # width of "ring" kernel
            'loss_fn': 'pct',                   # loss function
            'lr': 1e-3,                         # (initial) learning rate
            'lr_scheduler': None,               # learning rate scheduler function
            'path_to_model': None,              # path to saved weights
            'remove_activity': False,           # remove activity of last frame prior to background extraction
            'reuse_model': False                # reuse an already trained model
        }

        if params_from_file is not None:
            self.change_params_from_jsonfile(params_from_file)
        self.change_params(params_dict)


    def check_consistency(self):
        """ Populates the params object with some dataset dependent values
        and ensures that certain constraints are satisfied.
        """
        self.data['last_commit'] = '-'.join(caiman.utils.utils.get_caiman_version())
        if self.data['dims'] is None and self.data['fnames'] is not None:
            self.data['dims'] = caiman.base.movies.get_file_size(self.data['fnames'], var_name_hdf5=self.data['var_name_hdf5'])[0]
        if self.data['fnames'] is not None:
            if isinstance(self.data['fnames'], str):
                self.data['fnames'] = [self.data['fnames']]
            T = caiman.base.movies.get_file_size(self.data['fnames'], var_name_hdf5=self.data['var_name_hdf5'])[1]
            if len(self.data['fnames']) > 1:
                T = T[0]
            num_splits = max(T//max(self.motion['num_frames_split'], 10), 1)
            self.motion['splits_els'] = num_splits
            self.motion['splits_rig'] = num_splits
            if isinstance(self.data['fnames'][0],tuple):
                self.online['movie_name_online'] = os.path.join(os.path.dirname(self.data['fnames'][0][0]), self.online['movie_name_online'])
            else:
                self.online['movie_name_online'] = os.path.join(os.path.dirname(self.data['fnames'][0]), self.online['movie_name_online'])
        if self.online['N_samples_exceptionality'] is None:
            self.online['N_samples_exceptionality'] = np.ceil(self.data['fr'] * self.data['decay_time']).astype('int')
        if self.online['thresh_fitness_raw'] is None:
            self.online['thresh_fitness_raw'] = scipy.special.log_ndtr(
                -self.online['min_SNR']) * self.online['N_samples_exceptionality']
        self.online['max_shifts_online'] = (np.array(self.online['max_shifts_online']) / self.online['ds_factor']).astype(int)
        if self.init['gSig'] is None:
            self.init['gSig'] = [-1, -1]
        if self.init['gSiz'] is None:
            self.init['gSiz'] = [2*gs + 1 for gs in self.init['gSig']]
        self.init['gSiz'] = tuple([gs + 1 if gs % 2 == 0 else gs for gs in self.init['gSiz']])
        if self.patch['rf'] is not None:
            if np.any(np.array(self.patch['rf']) <= self.init['gSiz'][0]):
                logging.warning(f"Changing rf from {self.patch['rf']} to {2 * self.init['gSiz'][0]} because the constraint rf > gSiz was not satisfied.")
#        if self.motion['gSig_filt'] is None:
#            self.motion['gSig_filt'] = self.init['gSig']
        if self.init['nb'] <= 0 and (self.patch['nb_patch'] != self.init['nb'] or
                                     self.patch['low_rank_background'] is not None):
            logging.warning(f"gnb={self.init['nb']}, hence setting keys nb_patch and low_rank_background in group patch automatically.")
            self.set('patch', {'nb_patch': self.init['nb'], 'low_rank_background': None})
        if self.init['nb'] == -1 and self.spatial['update_background_components']:
            logging.warning("gnb=-1, hence setting key update_background_components " +
                            "in group spatial automatically to False.")
            self.set('spatial', {'update_background_components': False})
        if self.init['method_init'] == 'corr_pnr' and self.init['ring_size_factor'] is not None \
            and self.init['normalize_init']:
            logging.warning("using CNMF-E's ringmodel for background hence setting key " +
                            "normalize_init in group init automatically to False.")
            self.set('init', {'normalize_init': False})
        if self.motion['is3D']:
            for a in ('indices', 'max_shifts', 'strides', 'overlaps'):
                if len(self.motion[a]) != 3:
                    if self.motion[a][0] == self.motion[a][1]:
                        self.motion[a] = (self.motion[a][0],) * 3
                        logging.warning("is3D=True, hence setting key " + a +
                            " automatically to " + str(self.motion[a]))
                    else:
                        raise ValueError(a + ' has to be a tuple of length 3 for volumetric 3D data')
        for key in ('max_num_added', 'min_num_trial'):
            if (self.online[key] == 0 and self.online['update_num_comps']):
                self.set('online', {'update_num_comps': False})
                logging.warning(key + "=0, hence setting key update_num_comps " +
                                "in group online automatically to False.")

    def set(self, group:str, val_dict:dict, set_if_not_exists:bool=False, verbose=False) -> None:
        """ Add key-value pairs to a group. Existing key-value pairs will be overwritten
            if specified in val_dict, but not deleted.

        Args:
            group: The name of the group
            val_dict: A dictionary with key-value pairs to be set for the group
            warn_unused: 
            set_if_not_exists: Whether to set a key-value pair in a group if the key does not currently exist in the group. (DEPRECATED)
        """

        if set_if_not_exists:
            logging.warning("The set_if_not_exists flag for CNMFParams.set() is deprecated and will be removed in a future version of Caiman")
            # can't easily catch if it's passed but set to False, but that wouldn't do anything because of the default,
            # and if they get that error it's at least really easy to fix - just remove the flag
            # we don't want to support this because it makes the structure of the object unpredictable except at runtime

        if not hasattr(self, group):
            raise KeyError(f'No group in CNMFParams named {group}')

        d = getattr(self, group)
        for k, v in val_dict.items():
            if k not in d and not set_if_not_exists:
                if verbose:
                    logging.warning(
                        f"{group}/{k} not set: invalid target in CNMFParams object")
            else:
                try:
                    if np.any(d[k] != v):
                        logging.info(f"Changing key {k} in group {group} from {d[k]} to {v}")
                except ValueError: # d[k] and v also differ if above comparison fails, e.g. lists of different length
                    logging.info(f"Changing key {k} in group {group} from {d[k]} to {v}")
                d[k] = v

    def get(self, group, key):
        """ Get a value for a given group and key. Raises an exception if no such group/key combination exists.

        Args:
            group: The name of the group.
            key: The key for the property in the group of interest.

        Returns: The value for the group/key combination.
        """

        if not hasattr(self, group):
            raise KeyError(f'No group in CNMFParams named {group}')

        d = getattr(self, group)
        if key not in d:
            raise KeyError(f'No key {key} in group {group}')

        return d[key]

    def get_group(self, group):
        """ Get the dictionary of key-value pairs for a group.

        Args:
            group: The name of the group.
        """

        if not hasattr(self, group):
            raise KeyError(f'No group in CNMFParams named {group}')

        return getattr(self, group)

    def __eq__(self, other):

        if not isinstance(other, CNMFParams):
            return False

        parent_dict1 = self.to_dict()
        parent_dict2 = other.to_dict()

        key_diff = np.setdiff1d(parent_dict1.keys(), parent_dict2.keys())
        if len(key_diff) > 0:
            return False

        for k1, child_dict1 in parent_dict1.items():
            child_dict2 = parent_dict2[k1]
            added, removed, modified, same = dict_compare(child_dict1, child_dict2)
            if len(added) != 0 or len(removed) != 0 or len(modified) != 0 or len(same) != len(child_dict1):
                return False

        return True

    def to_dict(self) -> dict:
        """Returns the params class as a dictionary with subdictionaries for each
        category."""
        return {
               'data': self.data,
               'init': self.init,
               'merging': self.merging,
               'motion': self.motion,
               'online': self.online,
               'patch': self.patch,
               'preprocess': self.preprocess,
               'ring_CNN': self.ring_CNN,
               'quality': self.quality,
               'spatial': self.spatial,
               'temporal': self.temporal
               }

    def to_json(self) -> str:
        """ Reversibly serialise CNMFParams to json """
        dictdata = self.to_dict()
        # now we need to tweak dictdata to convert ndarrays to something json can represent
        class NumpyEncoder(json.JSONEncoder): # Custom json encoder that handles ndarrays better
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, slice):
                    return list([obj.start, obj.stop, obj.step])
                return json.JSONEncoder.default(self, obj)
        return json.dumps(dictdata, cls=NumpyEncoder)


    def to_jsonfile(self, targfn:str) -> None:
        """ Reversibly serialise CNMFParams to a json file """
        with open(targfn, 'w') as targfh:
            targfh.write(self.to_json())

    def __repr__(self) -> str:
        formatted_outputs = [
            f'{group_name}:\n\n{pformat(group_dict)}'
            for group_name, group_dict in self.to_dict().items()
        ]

        return 'CNMFParams:\n\n' + '\n\n'.join(formatted_outputs)

    def change_params(self, params_dict, allow_legacy:bool=True, warn_unused:bool=True, verbose:bool=False) -> None:
        """ Method for updating the params object by providing a dictionary.

        Args:
            params_dict: dictionary with parameters to be changed
            verbose: If true, will complain if the params dictionary is not complete
            allow_legacy: If True, throw a deprecation warning and then attempt to
                          handle unconsumed keys using the older copy-it-everywhere logic.
                          We will eventually remove this option and the corresponding code.
            warn_unused: If True, emit warnings when the params dict has fields in it that
                         were never used in populating the Params object. You really should not
                         set this to False. Fix your code.
        """
        # When we're ready to remove allow_legacy, this code will get a lot simpler

        consumed = {} # Keep track of what parameters in params_dict were used to set something in params (just for legacy API)
        for paramkey in params_dict:
            if paramkey in list(self.__dict__.keys()): # Proper pathed part
                cat_handle = getattr(self, paramkey)
                for k, v in params_dict[paramkey].items():
                    if k not in cat_handle and warn_unused:
                        # For regular/pathed API, we can notice right away if the user gave us something that won't update the object
                        logging.warning(f"In setting CNMFParams, provided key {paramkey}/{k} was not consumed. This is a bug!")
                    else:
                        cat_handle[k] = v 
            # BEGIN code that we will remove in some future version of caiman
            elif allow_legacy:
                for category in list(self.__dict__.keys()):
                    cat_handle = getattr(self, category) # Thankfully a read-write handle
                    if paramkey in cat_handle: # Is it known?
                        legacy_used = True
                        consumed[paramkey] = True
                        cat_handle[paramkey] = params_dict[paramkey] # Do the update
                if legacy_used:
                    logging.warning(f"In setting CNMFParams, non-pathed parameters were used; this is deprecated. allow_legacy will default to False, and then will be removed in future versions of Caiman")
        # END
        if warn_unused:
            for toplevel_k in params_dict:
                if toplevel_k not in consumed and toplevel_k not in list(self.__dict__.keys()): # When we remove legacy behaviour, this logic will simplify and fold into above
                    logging.warning(f"In setting CNMFParams, provided toplevel key {toplevel_k} was unused. This is a bug!")
        self.check_consistency()

    def change_params_from_json(self, jsonstring:str, verbose:bool=False) -> None:
        """ Same as change_params, except it takes json as input """
        to_load = json.loads(jsonstring)
        self.change_params(to_load, verbose=verbose)

    def change_params_from_jsonfile(self, json_fn:str, verbose:bool=False) -> None:
        """ Same as change_params, except it takes a json file as input; pass the filename """
        with open(json_fn, 'r') as json_fh:
            to_load = json.load(json_fh)
        self.change_params(to_load, verbose=verbose)

