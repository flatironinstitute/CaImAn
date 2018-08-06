import logging
import os

import numpy as np
import scipy

from ...paths import caiman_datadir
from .utilities import dict_compare, get_file_size

class CNMFParams(object):

    def __init__(self, fnames=None, dims=None, dxy=(1, 1),
                 border_pix=0, del_duplicates=False, low_rank_background=True,
                 memory_fact=1, n_processes=1, nb_patch=1, p_ssub=2, p_tsub=2,
                 remove_very_bad_comps=False, rf=None, stride=None,
                 check_nan=True, n_pixels_per_process=None,
                 k=30, alpha_snmf=10e2, center_psf=False, gSig=[5, 5], gSiz=None,
                 init_iter=2, method_init='greedy_roi', min_corr=.85,
                 min_pnr=20, gnb=1, normalize_init=True, options_local_NMF=None,
                 ring_size_factor=1.5, rolling_length=100, rolling_sum=True,
                 ssub=2, ssub_B=2, tsub=2,
                 block_size=5000, num_blocks_per_run=20, update_background_components=True,
                 method_deconvolution='oasis', p=2, s_min=None,
                 do_merge=True, merge_thresh=0.8,
                 decay_time=0.4, fr=30, min_SNR=2.5, rval_thr=0.8,
                 N_samples_exceptionality=None, batch_update_suff_stat=False,
                 expected_comps=500, max_comp_update_shape=np.inf, max_num_added=3,
                 min_num_trial=2, minibatch_shape=100, minibatch_suff_stat=5,
                 n_refit=0, num_times_comp_updated=np.inf, simultaneously=False,
                 sniper_mode=False, test_both=False, thresh_CNN_noisy=0.5,
                 thresh_fitness_delta=-50, thresh_fitness_raw=None, thresh_overlap=0.5,
                 update_num_comps=True, use_dense=True, use_peak_max=False,
                 only_init_patch=False, params_dict={},
                 ):
        """Class for setting the processing parameters. All parameters for CNMF, online-CNMF, quality testing,
        and motion correction can be set here and then used in the various processing pipeline steps.
        The prefered way to set parameters is by using the set function, where a subclass is determined and a
        dictionary is passed. The whole dictionary can also be initialized at once by passing a dictionary params_dict
        when initializing the CNMFParams object. Direct setting of the positional arguments in CNMFParams is only
        present for backwards compatibility reasons and should not be used if possible.

        Parameters/Attributes
        ----------

        Any parameter that is not set get a default value specified
        by the dictionary default options
        DATA PARAMETERS (CNMFParams.data) #####

            fnames: list[str]
                list of complete paths to files that need to be processed

            dims: (int, int), default: computed from fnames
                dimensions of the FOV in pixels

            fr: float, default: 30
                imaging rate in frames per second

            decay_time: float, default: 0.4
                length of typical transient in seconds

            dxy: (float, float)
                spatial resolution of FOV in pixels per um

        PATCH PARAMS (CNMFParams.patch)######

            rf: int or None, default: None
                Half-size of patch in pixels. If None, no patches are constructed and the whole FOV is processed jointly

            stride: int or None, default: None
                Overlap between neighboring patches in pixels.

            nb_patch: int, default: 1
                Number of (local) background components per patch

            border_pix: int, default: 0
                Number of pixels to exclude around each border.

            low_rank_background: bool, default: True
                Whether to update the background using a low rank approximation.
                If False all the nonzero elements of the background components are updated using hals
                (to be used with one background per patch)

            del_duplicates: bool, default: False
                Delete duplicate components in the overlaping regions between neighboring patches. If False,
                then merging is used.

            only_init: bool, default: False
                whether to run only the initialization

            skip_refinement: bool, default: False
                Whether to skip refinement of components (deprecated?)

            remove_very_bad_comps: bool, default: True
                Whether to remove (very) bad quality components during patch processing

            ssub: float, default: 2
                Spatial downsampling factor

            tsub: float, default: 2
                Temporal downsampling factor

            memory_fact: float, default: 1
                unitless number for increasing the amount of available memory

            n_processes: int
                Number of processes used for processing patches in parallel

            in_memory: bool, default: True
                Whether to load patches in memory

        PRE-PROCESS PARAMS (CNMFParams.preprocess) #############

            sn: np.array or None, default: None
                noise level for each pixel

            noise_range: [float, float], default: [.25, .5]
                range of normalized frequencies over which to compute the PSD for noise determination

            noise_method: 'mean'|'median'|'logmexp', default: 'mean'
                PSD averaging method for computing the noise std

            max_num_samples_fft: int, default: 3*1024
                Chunk size for computing the PSD of the data (for memory considerations)

            n_pixels_per_process: int, default: 1000
                Number of pixels to be allocated to each process

            compute_g': bool, default: False
                whether to estimate global time constant

            p: int, default: 2
                 order of AR indicator dynamics

            lags: int, default: 5
                number of lags to be considered for time constant estimation

            include_noise: bool, default: False
                    flag for using noise values when estimating g

            pixels: list, default: None
                 pixels to be excluded due to saturation

            check_nan: bool, default: True
                whether to check for NaNs

        INIT PARAMS (CNMFParams.init)###############

            K: int, default: 30
                number of components to be found (per patch or whole FOV depending on whether rf=None)

            gSig: [int, int], default: [5, 5]
                radius of average neurons (in pixels)

            gSiz: [int, int], default: [int(round((x * 2) + 1)) for x in gSig],
                half-size of bounding box for each neuron

            center_psf: bool, default: False
                whether to use 1p data processing mode. Set to true for 1p

            ssub: float, default: 2
                spatial downsampling factor

            tsub: float, default: 2
                temporal downsampling factor

            nb: int, default: 1
                number of background components

            maxIter: int, default: 5
                number of HALS iterations during initialization

            method: 'greedy_roi'|'greedy_pnr'|'sparse_NMF'|'local_NMF' default: 'greedy_roi'
                initialization method. use 'greedy_pnr' for 1p processing and 'sparse_NMF' for dendritic processing.

            min_corr: float, default: 0.85
                minimum value of correlation image for determining a candidate component during greedy_pnr

            min_pnr: float, default: 20
                minimum value of psnr image for determining a candidate component during greedy_pnr

            ring_size_factor: float, default: 1.5
                radius of ring (*gSig) for computing background during greedy_pnr

            ssub_B: float, default: 2
                downsampling factor for background during greedy_pnr

            init_iter: int, default: 2
                number of iterations during greedy_pnr (1p) initialization

            nIter: int, default: 5
                number of rank-1 refinement iterations during greedy_roi initialization

            rolling_sum: bool, default: True
                use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi

            rolling_length: int, default: 100
                width of rolling window for rolling sum option

            kernel: np.array or None, default: None
                user specified template for greedyROI

            max_iter_snmf : int, default: 500
                maximum number of iterations for sparse NMF initialization

            alpha_snmf: float, default: 10e2
                sparse NMF sparsity regularization weight

            sigma_smooth_snmf : (float, float, float), default: (.5,.5,.5)
                std of Gaussian kernel for smoothing data in sparse_NMF

            perc_baseline_snmf: float, default: 20
                percentile to be removed from the data in sparse_NMF prior to decomposition

            normalize_init: bool, default: True
                whether to equalize the movies during initialization

            options_local_NMF: dict
                dictionary with parameters to pass to local_NMF initializer

        SPATIAL PARAMS (CNMFParams.spatial) ##########

            method: 'dilate'|'ellipse', default: 'dilate'
                method for expanding footprint of spatial components

            dist: float, default: 3
                expansion factor of ellipse

            expandCore: morphological element, default: None(?)
                morphological element for expanding footprints under dilate

            nb: int, default: 1
                number of global background components

            n_pixels_per_process: int, default: 1000
                number of pixels to be processed by each worker

            thr_method: 'nrg'|'max', default: 'max'
                thresholding method

            maxthr: float, default: 0.1
                Max threshold

            nrgthr: float, default: 0.9999
                Energy threshold

            extract_cc: bool, default: True
                whether to extract connected components during thresholding
                (might want to turn to False for dendritic imaging)

            medw: (int, int) default: None
                window of median filter (set to (3,)*len(dims) in cnmf.fit)

            se: np.array or None, default: None
                 Morphological closing structuring element (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

            ss: np.array or None, default: None
                Binary element for determining connectivity (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

            update_background_components: bool, default: True
                whether to update the spatial background components

            method_ls: 'lasso_lars'|'nnls_L0'|'lasso_lars_old', default: 'lasso_lars'
                'nnls_L0'. Nonnegative least square with L0 penalty
                'lasso_lars' lasso lars function from scikit learn
                'lasso_lars_old' lasso lars from old implementation, will be deprecated

            block_size : int, default: 5000
                Number of pixels to process at the same time for dot product. Reduce if you face memory problems

            num_blocks_per_run: int, default: 20
                Parallelization of A'*Y operation

            normalize_yyt_one: bool, default: True
                Whether to normalize the C and A matrices so that diag(C*C.T) = 1 during update spatial

        TEMPORAL PARAMS (CNMFParams.temporal)###########

            ITER: int, default: 2
                block coordinate descent iterations

            method: 'oasis'|'cvxpy'|'oasis', default: 'oasis'
                method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
                if method cvxpy, primary and secondary (if problem unfeasible for approx solution)

            solvers: 'ECOS'|'SCS', default: ['ECOS', 'SCS']
                 solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'

            p: 0|1|2, default: 2
                order of AR indicator dynamics

            memory_efficient: False

            bas_nonneg: bool, default: True
                whether to set a non-negative baseline (otherwise b >= min(y))

            noise_range: [float, float], default: [.25, .5]
                range of normalized frequencies over which to compute the PSD for noise determination

            noise_method: 'mean'|'median'|'logmexp', default: 'mean'
                PSD averaging method for computing the noise std

            lags: int, default: 5
                number of autocovariance lags to be considered for time constant estimation

            fudge_factor: float (close but smaller than 1) default: .96
                bias correction factor for discrete time constants

            nb: int, default: 1
                number of global background components

            verbosity: bool, default: False
                whether to be verbose

            block_size : int, default: 5000
                Number of pixels to process at the same time for dot product. Reduce if you face memory problems

            num_blocks_per_run: int, default: 20
                Parallelization of A'*Y operation

            s_min: float or None, default: None
                Minimum spike threshold amplitude (computed in the code if used).

        MERGE PARAMS (CNMFParams.merge)#####
            do_merge: bool, default: True
                Whether or not to merge

            thr: float, default: 0.8
                Trace correlation threshold for merging two components.

        QUALITY EVALUATION PARAMETERS (CNMFParams.quality)###########

            min_SNR: float, default: 2.5
                trace SNR threshold. Traces with SNR above this will get accepted

            SNR_lowest: float, default: 0.5
                minimum required trace SNR. Traces with SNR below this will get rejected

            rval_thr: float, default: 0.8
                space correlation threshold. Components with correlation higher than this will get accepted

            rval_lowest: float, default: -1
                minimum required space correlation. Components with correlation below this will get rejected

            use_cnn: bool, default: True
                flag for using the CNN classifier.

            min_cnn_thr: float, default: 0.9
                CNN classifier threshold. Components with score higher than this will get accepted

            cnn_lowest: float, default: 0.1
                minimum required CNN threshold. Components with score lower than this will get rejected.

            gSig_range: list or integers, default: None
                gSig scale values for CNN classifier. In not None, multiple values are tested in the CNN classifier.

        ONLINE CNMF (ONACID) PARAMETERS (CNMFParams.online)#####

            N_samples_exceptionality: int, default: np.ceil(decay_time*fr),
                Number of frames over which trace SNR is computed (usually length of a typical transient)

            batch_update_suff_stat: bool, default: False
                Whether to update sufficient statistics in batch mode

            ds_factor: int, default: 1,
                spatial downsampling factor for faster processing (if > 1)

            epochs: int, default: 1,
                number of times to go over data

            expected_comps: int, default: 500
                number of expected components (for memory allocation purposes)

            init_batch: int, default: 200,
                length of mini batch used for initialization

            init_method: 'bare'|'cnmf'|'seeded', default: 'bare',
                initialization method

            max_comp_update_shape: int, default: np.inf
                Maximum number of spatial components to be updated at each time

            max_num_added: int, default: 3
                Maximum number of new components to be added in each frame

            max_shifts: int, default: 10,
                Maximum shifts for motion correction during online processing

             min_SNR: float, default: 2.5
                Trace SNR threshold for accepting a new component

            min_num_trial: int, default: 3
                Number of mew possible components for each frame

            minibatch_shape: int, default: 100
                Number of frames stored in rolling buffer

            minibatch_suff_stat: int, default: 5
                mini batch size for updating sufficient statistics

            motion_correct: bool, default: True
                Whether to perform motion correction during online processing

            normalize: bool, default: False
                Whether to normalize each frame prior to online processing

            n_refit: int, default: 0
                Number of additional iterations for computing traces

            num_times_comp_updated: int, default: np.inf

            path_to_model: str, default: os.path.join(caiman_datadir(), 'model', 'cnn_model_online.h5')
                Path to online CNN classifier

            rval_thr: float, default: 0.8
                space correlation threshold for accepting a new component

            show_movie: bool, default: False
                Whether to display movie of online processing

            simultaneously: bool, default: False
                Whether to demix and deconvolve simultaneously

            sniper_mode: bool, default: False
                Whether to use the online CNN classifier for screening candidate components (otherwise space
                correlation is used)

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

            update_num_comps: bool, default: True
                Whether to search for new components

            use_dense: bool, default: True
                Whether to store and represent A and b as a dense matrix

            use_peak_max: bool, default: False
                Whether to find candidate centroids using skimage's find local peaks function

        MOTION CORRECTION PARAMETERS (CNMFParams.motion)####

            border_nan: bool or str, default: 'copy'
                flag for allowing NaN in the boundaries. True allows NaN, whereas 'copy' copies the value of the
                nearest data point.

            gSig_filt: int or None, default: None
                size of kernel for high pass spatial filtering in 1p data. If None no spatial filtering is performed

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

            num_splits_to_process_els, default: [7, None]
            num_splits_to_process_rig, default: None

            overlaps: (int, int), default: (24, 24)
                overlap between patches in pixels in pw-rigid motion correction.

            pw_rigid: bool, default: False
                flag for performing pw-rigid motion correction.

            shifts_opencv: bool, default: True
                flag for applying shifts using cubic interpolation (otherwise FFT)

            splits_els: int, default: 14
                number of splits across time for pw-rigid registration

            splits_rig: int, default: 14
                number of splits across time for rigid registration

            strides: (int, int), default: (96, 96)
                how often to start a new patch in pw-rigid registration. Size of each patch will be strides + overlaps

            upsample_factor_grid" int, default: 4
                motion field upsampling factor during FFT shifts.

            use_cuda: bool, default: False
                flag for using a GPU.
        """

        self.data = {
            'fnames': fnames,
            'dims': dims,
            'fr': fr,
            'decay_time': decay_time,
            'dxy': dxy
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
            'remove_very_bad_comps': remove_very_bad_comps,
            'rf': rf,
            'skip_refinement': False,
            'ssub': p_ssub,             # spatial downsampling factor
            'stride': stride,
            'tsub': p_tsub,             # temporal downsampling factor
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
            'alpha_snmf': alpha_snmf,
            'center_psf': center_psf,
            'gSig': gSig,
            # size of bounding box
            'gSiz': gSiz,
            'init_iter': init_iter,
            'kernel': None,           # user specified template for greedyROI
            'maxIter': 5,             # number of HALS iterations
            'max_iter_snmf': 500,
            'method': method_init,    # can be greedy_roi or sparse_nmf, local_NMF
            'min_corr': min_corr,
            'min_pnr': min_pnr,
            'nIter': 5,               # number of refinement iterations
            'nb': gnb,                # number of global background components
            # whether to pixelwise equalize the movies during initialization
            'normalize_init': normalize_init,
            # dictionary with parameters to pass to local_NMF initializaer
            'options_local_NMF': options_local_NMF,
            'perc_baseline_snmf': 20,
            'ring_size_factor': ring_size_factor,
            'rolling_length': rolling_length,
            'rolling_sum': rolling_sum,
            'sigma_smooth_snmf': (.5, .5, .5),
            'ssub': ssub,             # spatial downsampling factor
            'ssub_B': ssub_B,
            'tsub': tsub,             # temporal downsampling factor
        }

        self.spatial = {
            'block_size': block_size,
            'dist': 3,                       # expansion factor of ellipse
            'expandCore': None,
            # Flag to extract connected components (might want to turn to False for dendritic imaging)
            'extract_cc': True,
            'maxthr': 0.1,                   # Max threshold
            'medw': None,                    # window of median filter
            # method for determining footprint of spatial components ('ellipse' or 'dilate')
            'method': 'dilate',
            # 'nnls_L0'. Nonnegative least square with L0 penalty
            # 'lasso_lars' lasso lars function from scikit learn
            # 'lasso_lars_old' lasso lars from old implementation, will be deprecated
            'method_ls': 'lasso_lars',
            # number of pixels to be processed by each worker
            'n_pixels_per_process': n_pixels_per_process,
            'nb': gnb,                        # number of background components
            'normalize_yyt_one': True,
            'nrgthr': 0.9999,                # Energy threshold
            'num_blocks_per_run': num_blocks_per_run,
            'se': None,                      # Morphological closing structuring element
            'ss': None,                      # Binary element for determining connectivity
            'thr_method': 'max',             # Method of thresholding ('max' or 'nrg')
            # whether to update the background components in the spatial phase
            'update_background_components': update_background_components,
        }

        self.temporal = {
            'ITER': 2,                  # block coordinate descent iterations
            # flag for setting non-negative baseline (otherwise b >= min(y))
            'bas_nonneg': False,
            # number of pixels to process at the same time for dot product. Make it
            # smaller if memory problems
            'block_size': block_size,
            # bias correction factor (between 0 and 1, close to 1)
            'fudge_factor': .96,
            # number of autocovariance lags to be considered for time constant estimation
            'lags': 5,
            'memory_efficient': False,
            # method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
            # if method cvxpy, primary and secondary (if problem unfeasible for approx
            # solution) solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
            'method': method_deconvolution,  # 'cvxpy', # 'oasis'
            'nb': gnb,                   # number of background components
            'noise_method': 'mean',     # averaging method ('mean','median','logmexp')
            'noise_range': [.25, .5],   # range of normalized frequencies over which to average
            'num_blocks_per_run': num_blocks_per_run,
            'p': p,                     # order of AR indicator dynamics
            's_min': s_min,             # minimum spike threshold
            'solvers': ['ECOS', 'SCS'],
            'verbosity': False,
        }

        self.merging = {
            'do_merge': do_merge,
            'merge_thr': merge_thresh,
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
        }

        self.online = {
            'N_samples_exceptionality': N_samples_exceptionality,  # timesteps to compute SNR
            'batch_update_suff_stat': batch_update_suff_stat,
            'ds_factor': 1,                    # spatial downsampling for faster processing
            'epochs': 1,                       # number of epochs
            'expected_comps': expected_comps,  # number of expected components
            'init_batch': 200,                 # length of mini batch for initialization
            'init_method': 'bare',             # initialization method for first batch,
            'max_comp_update_shape': max_comp_update_shape,
            'max_num_added': max_num_added,    # maximum number of new components for each frame
            'max_shifts': 10,                  # maximum shifts during motion correction
            'min_SNR': min_SNR,                # minimum SNR for accepting a new trace
            'min_num_trial': min_num_trial,    # number of mew possible components for each frame
            'minibatch_shape': minibatch_shape,  # number of frames in each minibatch
            'minibatch_suff_stat': minibatch_suff_stat,
            'motion_correct': True,            # flag for motion correction
            'normalize': False,                # normalize frame
            'n_refit': n_refit,                # Additional iterations to simultaneously refit
            # path to CNN model for testing new comps
            'num_times_comp_updated': num_times_comp_updated,
            'path_to_model': os.path.join(caiman_datadir(), 'model',
                                          'cnn_model_online.h5'),
            'rval_thr': rval_thr,              # space correlation threshold
            'show_movie': False,               # display movie online
            'simultaneously': simultaneously,  # demix and deconvolve simultaneously
            'sniper_mode': sniper_mode,        # flag for using CNN
            'test_both': test_both,            # flag for using both CNN and space correlation
            'thresh_CNN_noisy': thresh_CNN_noisy,  # threshold for online CNN classifier
            'thresh_fitness_delta': thresh_fitness_delta,
            'thresh_fitness_raw': thresh_fitness_raw,    # threshold for trace SNR (computed below)
            'thresh_overlap': thresh_overlap,
            'update_num_comps': update_num_comps,  # flag for searching for new components
            'use_dense': use_dense,            # flag for representation and storing of A and b
            'use_peak_max': use_peak_max,      # flag for finding candidate centroids
        }
        
        self.motion = {
            'border_nan': 'copy',                 # flag for allowing NaN in the boundaries
            'gSig_filt': None,                  # size of kernel for high pass spatial filtering in 1p data
            'max_deviation_rigid': 3,           # maximum deviation between rigid and non-rigid
            'max_shifts': (6, 6),               # maximum shifts per dimension (in pixels)
            'min_mov': None,                    # minimum value of movie
            'niter_rig': 1,                     # number of iterations rigid motion correction
            'nonneg_movie': True,               # flag for producing a non-negative movie
            'num_frames_split': 80,             # split across time every x frames
            'num_splits_to_process_els': [7, None],
            'num_splits_to_process_rig': None,
            'overlaps': (32, 32),               # overlap between patches in pw-rigid motion correction
            'pw_rigid': False,                  # flag for performing pw-rigid motion correction
            'shifts_opencv': True,              # flag for applying shifts using cubic interpolation (otherwise FFT)
            'splits_els': 14,                   # number of splits across time for pw-rigid registration
            'splits_rig': 14,                   # number of splits across time for rigid registration
            'strides': (96, 96),                # how often to start a new patch in pw-rigid registration
            'upsample_factor_grid': 4,          # motion field upsampling factor during FFT shifts
            'use_cuda': False                   # flag for using a GPU
        }
        
        self.change_params(params_dict)
        if self.data['dims'] is None and self.data['fnames'] is not None:
            self.data['dims'] = get_file_size(self.data['fnames'])[0]
        if self.data['fnames'] is not None:
            T = get_file_size(self.data['fnames'])[1]
            if len(self.data['fnames'] > 1):
                T = T[0]
            num_splits = max(T//self.motion['num_frames_split'])
            self.motion['splits_els'] = num_splits
            self.motion['splits_rig'] = num_splits
        if self.online['N_samples_exceptionality'] is None:
            self.online['N_samples_exceptionality'] = np.ceil(self.data['fr'] * self.data['decay_time']).astype('int')
        if self.online['thresh_fitness_raw'] is None:
            self.online['thresh_fitness_raw'] = scipy.special.log_ndtr(
                -self.online['min_SNR']) * self.online['N_samples_exceptionality']
        self.online['max_shifts'] = (np.array(self.online['max_shifts']) / self.online['ds_factor']).astype(int)
        if self.init['gSig'] is None:
            self.init['gSig'] = [-1, -1]
        if self.init['gSiz'] is None:
            self.init['gSiz'] = [2*gs + 1 for gs in self.init['gSig']]

        if gnb <= 0:
            logging.warning("gnb={}, hence setting keys nb_patch and low_rank_background ".format(gnb) +
                            "in group patch automatically.")
            self.set('patch', {'nb_patch': gnb, 'low_rank_background': None})


    def set(self, group, val_dict, set_if_not_exists=False):
        """ Add key-value pairs to a group. Existing key-value pairs will be overwritten
            if specified in val_dict, but not deleted.

        Args:
            group: The name of the group.
            val_dict: A dictionary with key-value pairs to be set for the group.
            set_if_not_exists: Whether to set a key-value pair in a group
                               if the key does not currently exist in the group.
        """

        if not hasattr(self, group):
            raise KeyError('No group in CNMFParams named {}'.format(group))

        d = getattr(self, group)
        for k, v in val_dict.items():
            if k not in d and not set_if_not_exists:
                logging.warning(
                    "NOT setting value of key {} in group {}, because no prior key existed...".format(k, group))
            else:
                if np.any(d[k] != v):
                    logging.warning(
                        "Changing key {} in group {} from {} to {}".format(k, group, d[k], v))
                d[k] = v

    def get(self, group, key):
        """ Get a value for a given group and key. Raises an exception if no such group/key combination exists.

        Args:
            group: The name of the group.
            key: The key for the property in the group of interest.

        Returns: The value for the group/key combination.
        """

        if not hasattr(self, group):
            raise KeyError('No group in CNMFParams named {}'.format(group))

        d = getattr(self, group)
        if key not in d:
            raise KeyError('No key {} in group {}'.format(key, group))

        return d[key]

    def get_group(self, group):
        """ Get the dictionary of key-value pairs for a group.

        Args:
            group: The name of the group.
        """

        if not hasattr(self, group):
            raise KeyError('No group in CNMFParams named {}'.format(group))

        return getattr(self, group)

    def __eq__(self, other):

        if type(other) != CNMFParams:
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

    def to_dict(self):
        return {'data': self.data, 'spatial_params': self.spatial, 'temporal_params': self.temporal,
                'init_params': self.init, 'preprocess_params': self.preprocess,
                'patch_params': self.patch, 'online': self.online, 'quality': self.quality,
                'merging': self.merging, 'motion': self.motion
                }

    def change_params(self, params_dict):
        for gr in list(self.__dict__.keys()):
            self.set(gr, params_dict)
        return self