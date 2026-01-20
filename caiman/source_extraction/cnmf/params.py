#!/usr/bin/env python

import copy
from functools import cache
import importlib.metadata
import json
import logging
import math
import msgspec
from msgspec import Struct, StructMeta, field
import numpy as np
import os
from pprint import pformat
import scipy
from scipy.ndimage import generate_binary_structure, iterate_structure
from tabulate import tabulate
from types import MappingProxyType
from typing import (Optional, Any, Type, Union, Literal,
                    Mapping, Iterable, TypeVar, overload, cast)

import caiman.base.movies
import caiman.utils.utils
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf.utilities import all_same


# Definition of JSON serialization scheme, with known types (for msgspec)

def enc_hook(obj: Any) -> Any:
    """
    Extend msgspec encoding to work with additional types
    Based on caiman.source_extraction.cnmf.params.CNMFParams.to_json.NumpyEncoder,
    so it should produce json files compatible with the caiman functions.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, slice):
        return [obj.start, obj.stop, obj.step]
    else:
        raise NotImplementedError(f'Encoding objects of type {type(obj)} is not supported')

def dec_hook(decl_type: Type, obj: Any) -> Any:
    """Extend msgspec decoding to work with additional types"""
    if decl_type is np.ndarray:
        return np.array(obj)
    elif issubclass(decl_type, np.integer) or issubclass(decl_type, np.floating):
        return decl_type(obj)
    elif decl_type is slice:
        start, stop, step = obj
        return slice(start, stop, step)
    else:
        raise NotImplementedError(f'Decoding objects of type {decl_type} is not supported')


class GroupParamsMeta(StructMeta):
    def __new__(mcls, name, bases, namespace, **config):
        """Change some defaults - see https://jcristharif.com/msgspec/structs.html#metaclasses"""
        config.setdefault('forbid_unknown_fields', True)  # raise error if trying to read or set param with wrong name
        config.setdefault('eq', False)  # we use our own equality method that works with ndarrays
        return super().__new__(mcls, name, bases, namespace, **config)

class GroupParams(Struct, metaclass=GroupParamsMeta):
    """
    Struct that can also be used as a mapping, to be used for subfields of CNMFParams,
    which have historically been dicts.
    (fields can be gotten and set using [] syntax, but not deleted.)
    """
    Self = TypeVar('Self', bound='GroupParams')

    def __setitem__(self, key: str, item):
        setattr(self, key, item)
    
    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
    
    def __len__(self) -> int:
        return len(self.__struct_fields__)
    
    def __iter__(self):
        return iter(self.__struct_fields__)
    
    def __contains__(self, key: str):
        return key in self.__struct_fields__

    def keys(self):
        return self.__struct_fields__

    def values(self):
        yield from (getattr(self, k) for k in self.__struct_fields__)
    
    def items(self):
        yield from zip(self.keys(), self.values())

    def get(self, key: str, default=None, /):
        if key in self.__struct_fields__:
            return getattr(self, key)
        return default
    
    def copy(self):
        return copy.copy(self)
    
    def reversed(self):
        return reversed(self.__struct_fields__)
    
    def get_differing_params(self: Self, other: Self) -> Iterable[tuple[str, Any, Any]]:
        """
        Returns an iterable of params that are not considered equal
        Each return value is a tuple: (name, this_value, other_value)
        """
        for field in self.__struct_fields__:
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            if not all_same(self_val, other_val):
                yield field, self_val, other_val

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return not any(self.get_differing_params(other))
        else:
            return NotImplemented
    
    def __ne__(self, other) -> bool:
        if isinstance(other, type(self)):
            return any(self.get_differing_params(other))
        else:
            return NotImplemented

    @classmethod
    @cache
    def typemap(cls) -> dict[str, Any]:
        """Just a mapping from field names to types"""
        return {finfo.name: finfo.type for finfo in msgspec.structs.fields(cls)}

    def update(self, changes: dict[str, Any]):
        for key, val in changes.items():
            setattr(self, key, val)
    
    def update_checked(self, changes: dict[str, Any], verbose=True):
        """
        Apply each update to fields in changes, attempting to use msgspec.convert
        to convert to the correct types first. Extra keys that do not correspond
        to fields cause an error. If verbose is true, log a warning if conversion
        fails, meaning that the value might not match the expected type.
        """
        logger = logging.getLogger('caiman')

        # collect changes before applying any of them
        converted_changes: dict[str, Any] = {}

        for key, val in changes.items():
            if key not in self.__struct_fields__:
                raise KeyError(f"Cannot set unknown key '{key}' on {self.__class__.__name__}")

            ftype = self.typemap()[key]

            # try converting to type
            try:
                # have to convert to builtin first in case it is already something that
                # isn't considered a builtin by msgspec, like slice
                val_base = msgspec.to_builtins(val, enc_hook=enc_hook)
                val = msgspec.convert(val_base, type=ftype, dec_hook=dec_hook)
            except msgspec.ValidationError:
                if verbose:
                    logger.warning(
                        f'Field {key} of {self.__class__.__name__} could not be converted to expected type {ftype.__name__}. '
                        'The field is set anyway since strict=False, but the value may not survive a JSON round-trip.')
            converted_changes[key] = val
        self.update(converted_changes)
    
    def normalize_to_schema(self, verbose=True):
        """Try to convert all current values to the specified types"""
        self.update_checked(msgspec.structs.asdict(self), verbose=verbose)


# register as a mapping type
Mapping.register(GroupParams)  # type: ignore


# Parameter group definitions (see docstring of CNMFParams for full documentation)

class DataParams(GroupParams):
    """Parameters for features of the data and other misc settings"""
    fnames: Optional[list[str]] = None
    dims: Optional[tuple[int, ...]] = None  # None = read from fnames
    fr: float = 30.
    decay_time: float = 0.4
    dxy: tuple[float, float] = (1., 1.)     # resolution, unit: pixels/um
    var_name_hdf5: str = 'mov'
    caiman_version: str = importlib.metadata.version('caiman')
    last_commit: str = '-'.join(caiman.utils.utils.get_caiman_version())

    def __post_init__(self):
        if float(self.decay_time) == 0.:
            raise Exception("A decay time of zero is not permitted")


class PatchParams(GroupParams):
    """Parameters for how the data is divided into patches"""
    border_pix: int = 0
    del_duplicates: bool = False
    in_memory: bool = True
    low_rank_background: Optional[bool] = True
    memory_fact: float = 1.
    n_processes: int = 1
    nb_patch: int = 1
    only_init: bool = True
    p_patch: int = 0                        # AR order within patch
    remove_very_bad_comps: bool = False
    rf: Union[int, list[int], None] = None
    skip_refinement: bool = False
    p_ssub: float = 2.                      # spatial downsampling factor
    stride: Optional[int] = None
    p_tsub: float = 2.                      # temporal downsampling factor


class PreprocessParams(GroupParams):
    """Parameters for data preprocessing steps"""
    check_nan: bool = True
    compute_g: bool = False                 # flag for estimating global time constant
    include_noise: bool = False             # flag for using noise values when estimating g
    # number of autocovariance lags to be considered for time constant estimation
    lags: int = 5
    max_num_samples_fft: int = 3 * 1024
    n_pixels_per_process: Optional[int] = None
    noise_method: Literal['mean', 'median', 'logmexp'] = 'mean'  # averaging method
    # range of normalized frequencies over which to average
    noise_range: list[float] = field(default_factory=lambda: [0.25, 0.5])
    p: int = 2                              # order of AR indicator dynamics
    pixels: Optional[list[int]] = None      # pixels to be excluded due to saturation
    sn: Optional[np.ndarray] = None         # noise level for each pixel


class InitParams(GroupParams):
    """Parameters that control how CNMF should be initialized"""
    K: int = 30                             # number of components
    SC_kernel: Literal['heat', 'cos', 'binary'] = 'heat'  # kernel for graph affinity matrix
    SC_sigma: float = 1.                    # std for SC kernel
    SC_thr: float = 0.                      # threshold for affinity matrix
    SC_normalize: bool = True               # standardize entries prior to computing affinity matrix
    SC_use_NN: bool = False                 # sparsify affinity matrix by using only nearest neighbors
    SC_nnn: int = 20                        # number of nearest neighbors to use
    alpha_snmf: float = 0.5
    center_psf: bool = False
    gSig: list[int] = field(default_factory=lambda: [5, 5])
    gSiz: Optional[list[int]] = None
    # init method used in calls to NMF if geedy_roi method for component initialisation is used (offline or online)
    greedyroi_nmf_init_method: str = 'nndsvdar'
    # max_iter used in calls to NMF if greedy_roi method for component initialisation is used (online or offline)
    greedyroi_nmf_max_iter: int = 200
    init_iter: int = 2
    kernel: Optional[np.ndarray] = None     # user specified template for greedyROI
    lambda_gnmf: float = 1.                 # regularization weight for graph NMF
    snmf_l1_ratio: float = 0.               # L1 ratio, used by sparse nmf mode only
    maxIter: int = 5                        # number of HALS iterations
    max_iter_snmf: int = 500
    method_init: str = 'greedy_roi'         # can be greedy_roi, corr_pnr, sparse_nmf, compressed_nmf, graph_nmf
    min_corr: float = 0.85
    min_pnr: float = 20.
    nIter: int = 5                          # number of refinement iterations
    nb: int = 1                             # number of global background components
    normalize_init: bool = True             # whether to pixelwise equalize the movies during initialization
    options_local_NMF: Optional[dict] = None  # unused - local_NMF is removed
    perc_baseline_snmf: float = 20.
    ring_size_factor: float = 1.5
    rolling_length: int = 100
    rolling_sum: bool = True
    seed_method: Literal['auto', 'manual', 'semi'] = 'auto'
    sigma_smooth_snmf: tuple[float, float, float] = (0.5, 0.5, 0.5)
    ssub: float = 2.                        # spatial downsampling factor
    ssub_B: float = 2.
    tsub: float = 2.                        # temporal downsampling factor


def default_expandcore() -> np.ndarray:
    """
    Generates the default morphological element used for footprint expansion
    with the dilate method, which is a 5x5 matrix that is true where taxicab
    distance from the center is <= 2 and false elsewhere.
    """
    s1 = generate_binary_structure(2, 1)
    s2 = iterate_structure(s1, 2)
    return s2.astype(int)  # type: ignore


class SpatialParams(GroupParams):
    """Params that control how the algorithms handle spatial components"""
    dist: float = 3.                        # expansion factor of ellipse
    expandCore: np.ndarray = field(default_factory=default_expandcore)
    # Flag to extract connected components (might want to turn to False for dendritic imaging)
    extract_cc: bool = True
    maxthr: float = 0.1                     # Max threshold
    medw: Optional[tuple[int, ...]] = None  # window of median filter
    # method for determining footprint of spatial components
    method_exp: Literal['ellipse', 'dilate'] = 'dilate'
    # 'nnls_L0'. Nonnegative least square with L0 penalty
    # 'lasso_lars' lasso lars function from scikit learn
    method_ls: Literal['nnls_L0', 'lasso_lars'] = 'lasso_lars'
    # number of pixels to be processed by each worker
    n_pixels_per_process: Optional[int] = None
    nb: int = 1                             # number of background components
    normalize_yyt_one: bool = True
    nrgthr: float = 0.9999                  # Energy threshold
    # number of process to parallelize residual computation ** DECREASE IF MEMORY ISSUES
    num_blocks_per_run_spat: int = 20
    se: Optional[np.ndarray] = None         # Morphological closing structuring element
    ss: Optional[np.ndarray] = None         # Binary element for determining connectivity
    thr_method: Literal['max', 'nrg'] = 'nrg'  # Method of thresholding ('max' or 'nrg')
    # whether to update the background components in the spatial phase
    update_background_components: bool = True


class TemporalParams(GroupParams):
    """Params that control how the algorithms handle temporal components"""
    ITER: int = 2                           # block coordinate descent iterations
    # flag for setting non-negative baseline (otherwise b >= min(y))
    bas_nonneg: bool = False
    # number of pixels to parallelize residual computation ** DECREASE IF MEMORY ISSUES
    block_size_temp: int = 5000
    # bias correction factor (between 0 and 1, close to 1)
    fudge_factor: float = 0.96
    # number of autocovariance lags to be considered for time constant estimation
    lags: int = 5
    optimize_g: bool = False                # flag for optimizing time constants
    # method for solving the constrained deconvolution problem
    # if method cvxpy, primary and secondary (if problem unfeasible for approx
    # solution) solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
    method_deconvolution: Literal['cvx', 'cvxpy', 'oasis'] = 'oasis'
    nb: int = 1                             # number of background components
    noise_method: Literal['mean', 'median', 'logmexp'] = 'mean'  # averaging method
    # range of normalized frequencies over which to average
    noise_range: list[float] = field(default_factory=lambda: [.25, .5])
    # number of process to parallelize residual computation ** DECREASE IF MEMORY ISSUES
    num_blocks_per_run_temp: int = 20
    p: int = 2                              # order of AR indicator dynamics
    s_min: Optional[float] = None           # minimum spike threshold
    solvers: list[Literal['ECOS', 'SCS', 'CVXOPT']] = field(default_factory=lambda: ['ECOS', 'SCS'])
    verbosity: bool = False


class MergingParams(GroupParams):
    """Params that control how components are merged"""
    do_merge: bool = True
    merge_thr: float = 0.8
    merge_parallel: bool = False


class QualityParams(GroupParams):
    """Params that control how the quality of traces is evaluated"""
    SNR_lowest: float = 0.5         # minimum accepted SNR value
    cnn_lowest: float = 0.1         # minimum accepted value for CNN classifier
    gSig_range: Optional[list[int]] = None  # range for gSig scale for CNN classifier
    min_SNR: float = 2.5            # transient SNR threshold
    min_cnn_thr: float = 0.9        # threshold for CNN classifier
    rval_lowest: float = -1.        # minimum accepted space correlation
    rval_thr: float = 0.8           # space correlation threshold
    use_cnn: bool = True            # use CNN based classifier
    use_ecc: bool = False           # flag for eccentricity based filtering (2D only)
    max_ecc: float = 3.


class OnlineParams(GroupParams):
    """Params that control the online/OnACID mode"""
    N_samples_exceptionality: Optional[int] = None  # timesteps to compute SNR
    batch_update_suff_stat: bool = False
    dist_shape_update: bool = False       # update shapes in a distributed way
    ds_factor: int = 1                    # spatial downsampling for faster processing
    epochs: int = 1                       # number of epochs
    expected_comps: int = 500             # number of expected components
    full_XXt: bool = False                # store entire XXt matrix (as opposed to a list of sub-matrices) 
    init_batch: int = 200                 # length of mini batch for initialization
    init_method: Literal['bare', 'cnmf', 'seeded'] = 'bare'  # initialization method for first batch
    iters_shape: int = 5                 # number of block-CD iterations
    max_comp_update_shape: Union[int, float] = np.inf
    max_num_added: int = 5               # maximum number of new components for each frame
    max_shifts_online: int = 10          # maximum shifts during motion correction
    min_SNR: float = 2.5                 # minimum SNR for accepting a new trace
    min_num_trial: int = 5               # number of mew possible components for each frame
    minibatch_shape: int = 100           # number of frames in each minibatch
    minibatch_suff_stat: int = 5
    motion_correct: bool = True          # flag for motion correction
    movie_name_online: str = 'online_movie.mp4'  # filename of saved movie (appended to directory where data is located)
    normalize: bool = False              # normalize frame
    n_refit: int = 0                     # Additional iterations to simultaneously refit
    num_times_comp_updated: Union[int, float] = np.inf
    opencv_codec: str = 'H264'           # FourCC video codec for saving movie. Check http://www.fourcc.org/codecs.php
    # path to CNN model for testing new comps
    path_to_model: str = os.path.join(caiman_datadir(), 'model', 'cnn_model_online.pkl')
    ring_CNN: bool = False               # flag for using a ring CNN background model 
    rval_thr: float = 0.8                # space correlation threshold
    save_online_movie: bool = False      # flag for saving online movie
    show_movie: bool = False             # display movie online
    simultaneously: bool = False         # demix and deconvolve simultaneously
    sniper_mode: bool = False            # flag for using CNN
    stop_detection: bool = False         # flag for stop detecting new neurons at the last epoch 
    test_both: bool = False              # flag for using both CNN and space correlation
    thresh_CNN_noisy: float = 0.5        # threshold for online CNN classifier
    thresh_fitness_delta: float = -50.
    thresh_fitness_raw: Optional[float] = None  # threshold for trace SNR (computed below)
    thresh_overlap: float = 0.5
    update_freq: int = 200               # update every shape at least once every update_freq steps
    update_num_comps: bool = True        # flag for searching for new components
    use_corr_img: bool = False           # flag for using correlation image to detect new components
    use_dense: bool = True               # flag for representation and storing of A and b
    use_peak_max: bool = True            # flag for finding candidate centroids
    W_update_factor: int = 1             # update W less often than shapes by a given factor 


class MotionParams(GroupParams):
    """Params that control motion correction"""
    # flag for allowing NaN in the boundaries
    #  - True: keep nans
    #  - False: replace with 0s
    #  - 'min': replace with minimum value in the frame
    #  - 'copy': copy edge values
    border_nan: Union[bool, Literal['min', 'copy']] = 'copy'
    gSig_filt: Optional[int] = None     # size of kernel for high pass spatial filtering in 1p data
    is3D: bool = False                  # flag for 3D recordings for motion correction
    max_deviation_rigid: int = 3        # maximum deviation between rigid and non-rigid
    max_shifts: tuple[int, ...] = (6,6) # maximum shifts per dimension (in pixels)
    min_mov: Optional[float] = None     # minimum value of movie
    niter_rig: int = 1                  # number of iterations rigid motion correction
    nonneg_movie: bool = True           # flag for producing a non-negative movie
    num_frames_split: int = 80          # split across time every x frames (approximately)
    num_splits_to_process_els: None = None  # Unused, will be removed in a future version of Caiman
    num_splits_to_process_rig: None = None  # DO NOT MODIFY
    overlaps: tuple[int, ...] = (32,32) # overlap between patches in pw-rigid motion correction
    pw_rigid: bool = False              # flag for performing pw-rigid motion correction
    shifts_interpolate: bool = False    # interpolate shifts based on patch locations instead of resizing
    shifts_opencv: bool = True          # flag for applying shifts using cubic interpolation (otherwise FFT)
    splits_els: int = 14                # number of splits across time for pw-rigid registration (usually overridden by code)
    splits_rig: int = 14                # number of splits across time for rigid    registration (usually overridden by code)
    strides: tuple[int, ...] = (96, 96) # how often to start a new patch in pw-rigid registration
    upsample_factor_grid: int = 4       # motion field upsampling factor during FFT shifts
    use_cuda: bool = False              # flag for using a GPU
    indices: tuple[slice, ...] = (slice(None), slice(None))  # part of FOV to be corrected


class RingCNNParams(GroupParams):
    """Params that control the ring neural networks used for 1P background estimation"""
    n_channels: int = 2                 # number of "ring" kernels   
    use_bias: bool = False              # use bias in the convolutions
    use_add: bool = False               # use an additive layer
    pct: float = 0.01                   # quantile loss specification
    patience: int = 3                   # patience for early stopping
    max_epochs: int = 100               # maximum number of epochs
    width: int = 5                      # width of "ring" kernel
    loss_fn: str = 'pct'                # loss function
    lr: float = 1e-3                    # (initial) learning rate
    lr_scheduler: Optional[tuple[float, ...]] = None  # learning rate scheduler function arguments
    path_to_model: Optional[str] = None # path to saved weights
    remove_activity: bool = False       # remove activity of last frame prior to background extraction
    reuse_model: bool = False           # reuse an already trained model



class CNMFParams:
    """Class for setting and changing the various parameters."""

    class AllParamsStruct(Struct, forbid_unknown_fields=True):
        """data-only class holding the actual CNMF params (also used for deserialization)"""
        data: DataParams = field(default_factory=DataParams)
        patch: PatchParams = field(default_factory=PatchParams)
        preprocess: PreprocessParams = field(default_factory=PreprocessParams)
        init: InitParams = field(default_factory=InitParams)
        spatial: SpatialParams = field(default_factory=SpatialParams)
        temporal: TemporalParams = field(default_factory=TemporalParams)
        merging: MergingParams = field(default_factory=MergingParams)
        quality: QualityParams = field(default_factory=QualityParams)
        online: OnlineParams = field(default_factory=OnlineParams)
        motion: MotionParams = field(default_factory=MotionParams)
        ring_CNN: RingCNNParams = field(default_factory=RingCNNParams)

    # Mapping from valid keyword arguments of init to the names of the parameter group(s)
    # that should accept it, or tuples (group, name_for_group) if the parameter
    # needs to be renamed when passing it to the group. This is a way of still
    # allowing the same keyword arguments without duplicating the default values
    # within the parameter list of __init__. We don't allow all sub-parameter names
    # because this is a deprecated interface and it should be possible to make new 
    # sub-parameters with conflicting names that are only compatible with the 
    # new nested parameter syntax.
    # It's important that the dict() syntax is used with the kwarg names as the 
    # kwargs so that any accidental duplicate names are caught as a syntax error.
    _groups_for_flat_param = MappingProxyType(dict(  # MappingProxyType makes it immutable
        fnames='data', dims='data', fr='data', decay_time='data', dxy='data',
        var_name_hdf5='data',
        border_pix='patch', del_duplicates='patch', low_rank_background='patch', memory_fact='patch',
        n_processes='patch', nb_patch='patch', only_init='patch', only_init_patch=('patch', 'only_init'),
        remove_very_bad_comps='patch', rf='patch', p_ssub='patch', stride='patch', p_tsub='patch',
        check_nan='preprocess',
        K='init', k=('init', 'K'), alpha_snmf='init', center_psf='init', gSig='init', gSiz='init',
        init_iter='init', method_init='init', min_corr='init', min_pnr='init', normalize_init='init',
        options_local_NMF='init', ring_size_factor='init', rolling_length='init', rolling_sum='init',
        ssub='init', ssub_B='init', tsub='init',
        num_blocks_per_run_spat='spatial', update_background_components='spatial',
        block_size_temp='temporal', method_deconvolution='temporal', num_blocks_per_run_temp='temporal',
        s_min='temporal',
        do_merge='merging', merge_thresh='merging',
        N_samples_exceptionality='online', batch_update_suff_stat='online', expected_comps='online',
        iters_shape='online', max_comp_updated_shape='online', max_num_added='online', min_num_trial='online',
        minibatch_shape='online', minibatch_suff_stat='online', n_refit='online',
        num_times_comp_updated='online', simultaneously='online', sniper_mode='online', test_both='online',
        thresh_CNN_noisy='online', thresh_fitness_delta='online', thresh_fitness_raw='online',
        thresh_overlap='online', update_freq='online', update_num_comps='online', use_corr_img='online',
        use_dense='online', use_peak_max='online',

        # parameters shared between multiple subgroups
        n_pixels_per_process=['preprocess', 'spatial'],
        p=['preprocess', 'temporal'],
        nb=['init', 'spatial', 'temporal'],
        gnb=[('init', 'nb'), ('spatial', 'nb'), ('temporal', 'nb')],
        min_SNR=['quality', 'online'],
        rval_thr=['quality', 'online'],
        max_merge_area=[]  # keep for backwards compat. I guess
    ))
    
    def __init__(self, params_from_file: Optional[str] = None, params_dict: Optional[dict] = None,
                 _params_struct: Optional['CNMFParams.AllParamsStruct'] = None, **flat_params):
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

            remove_very_bad_comps: bool, default: False
                Whether to remove (very) bad quality components during patch processing

            rf: int or list or None, default: None
                Half-size of patch in pixels. If None, no patches are constructed and the whole FOV is processed jointly.
                If list, it should be a list of two elements corresponding to the height and width of patches

            skip_refinement: bool, default: False
                Whether to skip refinement of components

            p_ssub: float, default: 2
                Spatial downsampling factor

            stride: int or None, default: None
                Overlap between neighboring patches in pixels.

            p_tsub: float, default: 2
                Temporal downsampling factor

          CNMFParams.preprocess (these control preprocessing steps for the data):
            check_nan: bool, default: True
                whether to check for NaNs

            compute_g: bool, default: False
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

            sn: np.ndarray or None, default: None
                noise level for each pixel

          CNMFParams.init (these control how CNMF should be initialised):
            K: int, default: 30
                number of components to be found (per patch or whole FOV depending on whether rf=None)

            SC_kernel: {'heat', 'cos', 'binary'}, default: 'heat'
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

            gSig: list of int, default: [5, 5]
                radius of average neurons (in pixels)

            gSiz: list of int, default: [int(round((x * 2) + 1)) for x in gSig],
                half-size of bounding box for each neuron

            greedyroi_nmf_init_method: str
                When greedyROI is used, this is provided to sklearn's NMF() as the init method. Usually nndsvdar (the default)
                is fine, but in some cases random or some other init is preferable; see the sklearn docs for your choices

            greedyroi_nmf_max_iter: int
                When greedyROI is used, this is provided to sklearn's NMF() as the max number of iterations; the ideal value
                of this partly depends on the init method. See the sklearn docs for guidance.

            init_iter: int, default: 2
                number of iterations during corr_pnr (1p) initialization

            kernel: np.ndarray or None, default: None
                user specified template for greedyROI

            lambda_gnmf: float, default: 1.
                regularization weight for graph NMF

            snmf_l1_ratio: float, default: 0.
                L1 ratio, used by sparse NMF mode only

            maxIter: int, default: 5
                number of HALS iterations during initialization

            max_iter_snmf : int, default: 500
                maximum number of iterations for sparse NMF initialization

            method_init: 'greedy_roi'|'corr_pnr'|'sparse_nmf'|'compressed_nmf'|'graph_nmf' default: 'greedy_roi'
                initialization method. use 'corr_pnr' for 1p processing and 'sparse_nmf' for dendritic processing.

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
                dictionary with parameters to pass to local_NMF initializer.
                Now unused because the local_NMF method was removed.

            perc_baseline_snmf: float, default: 20
                percentile to be removed from the data in sparse_nmf prior to decomposition

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
            dist: float, default: 3
                expansion factor of ellipse

            expandCore: np.ndarray
                morphological element for expanding footprints under dilate
                default is a diamond generated by 

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

            nb: int, default: 1
                number of global background components. Do not set this directly; modify it in init.

            normalize_yyt_one: bool, default: True
                Whether to normalize the C and A matrices so that diag(C*C.T) = 1 during update spatial

            nrgthr: float, default: 0.9999
                Energy threshold

            num_blocks_per_run_spat: int, default: 20
                Parallelization of A'*Y operation

            se: np.ndarray or None, default: None
                 Morphological closing structuring element (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

            ss: np.ndarray or None, default: None
                Binary element for determining connectivity (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

            thr_method: 'nrg'|'max', default: 'nrg'
                thresholding method

            update_background_components: bool, default: True
                whether to update the spatial background components


          CNMFParams.temporal (these control how the algorithms handle temporal components):
            ITER: int, default: 2
                block coordinate descent iterations

            bas_nonneg: bool, default: False
                whether to set a non-negative baseline (otherwise b >= min(y))

            block_size_temp : int, default: 5000
                Number of pixels to process at the same time for dot product. Reduce if you face memory problems

            fudge_factor: float (close but smaller than 1) default: .96
                bias correction factor for discrete time constants

            lags: int, default: 5
                number of autocovariance lags to be considered for time constant estimation

            optimize_g: bool, default: False
                flag for optimizing time constants

            method_deconvolution: 'cvx'|'cvxpy'|'oasis', default: 'oasis'
                method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
                if method cvxpy, primary and secondary (if problem unfeasible for approx solution)

            nb: int, default: 1
                number of global background components. Do not set this directly; modify it in init.

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

            solvers: list of 'ECOS'|'SCS'|'CVXOPT', default: ['ECOS', 'SCS']
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
                length of mini batch used for initialization (must not exceed frame count on first file)

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

            movie_name_online: str, default: 'online_movie.mp4'
                Name of saved movie (appended in the data directory)

            normalize: bool, default: False
                Whether to normalize each frame prior to online processing

            n_refit: int, default: 0
                Number of additional iterations for computing traces

            num_times_comp_updated: int, default: np.inf
                (undocumented)

            opencv_codec: str, default: 'H264'
                FourCC video codec for saving movie. Check http://www.fourcc.org/codecs.php

            path_to_model: str, default: os.path.join(caiman_datadir(), 'model', 'cnn_model_online.pkl')
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

            thresh_CNN_noisy: float, default: 0.5,
                Threshold for the online CNN classifier

            thresh_fitness_delta: float (negative), default: -50
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

            max_shifts: tuple of ints, default: (6,6)
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

            overlaps: tuple of ints, default: (24, 24)
                overlap between patches in pixels in pw-rigid motion correction.

            pw_rigid: bool, default: False
                flag for performing pw-rigid motion correction.

            shifts_interpolate: bool, default: False
                use patch locations to interpolate shifts rather than just upscaling to size of image (for pw_rigid only)

            shifts_opencv: bool, default: True
                flag for applying shifts using cubic interpolation (otherwise FFT)

            splits_els: int, default: 14
                number of splits across time for pw-rigid registration.

            splits_rig: int, default: 14
                number of splits across time for rigid registration.

            strides: tuple of int, default: (96, 96)
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

            lr_scheduler: tuple of float or None, default: None
                Learning rate scheduler function. If provided, it should be a tuple
                with 0 or more positional arguments to nn_models.rate_scheduler.
                The arguments, in order, are factor, epoch_length, and samples_length.

            path_to_model: str, default: None
                Path to saved weights (if training then path to saved model weights)

            remove_activity: bool, default: False
                Flag for removing activity of last frame prior to background extraction

            reuse_model: bool, default: False
                Flag for reusing an already trained model (saved in path to model)
        """
        logger = logging.getLogger('caiman')

        if _params_struct is not None:
            self._params = _params_struct
        else:
            # each group is created with defaults
            self._params = self.AllParamsStruct()

        # update with individual passed-in params (deprecated)
        for key, val in flat_params.items():
            try:
                groups = self._groups_for_flat_param[key]
            except KeyError:
                raise TypeError(f"{__class__.__name__}() got an unexpected keyword argument '{key}'")

            for group in groups if isinstance(groups, list) else (groups,):
                if isinstance(group, tuple):
                    # rename key for this group
                    this_key = group[1]
                    group = group[0]
                else:
                    this_key = key
                getattr(self, group)[this_key] = val

        if params_from_file is not None:
            if params_dict is None:
                # we don't actually know if any non-default individual params were passed in
                logger.warning('It is faster to use CNMFParams.from_jsonfile() if only setting params from JSON')
            self.change_params_from_jsonfile(params_from_file)

        if params_dict is not None:
            self.change_params(params_dict)
        
        if params_from_file is None and params_dict is None:
            # make sure we do this test at least once
            self.check_consistency()

    @property
    def groups(self) -> tuple[str, ...]:
        return self._params.__struct_fields__
        

    def check_consistency(self):
        """ Populates the params object with some dataset dependent values
        and ensures that certain constraints are satisfied.
        """
        logger = logging.getLogger("caiman")
        self.data.last_commit = '-'.join(caiman.utils.utils.get_caiman_version())

        if self.data.fnames is not None:
            if self.data.dims is None:
                self.data.dims = caiman.base.movies.get_file_size(self.data.fnames, var_name_hdf5=self.data.var_name_hdf5)[0]

            # fix type of fnames to list[str]
            if isinstance(self.data.fnames, str):
                # pack single fname in a list
                self.data.fnames = [self.data.fnames]
            elif isinstance(self.data.fnames, bytes):  # also includes np.bytes_ as a subclass
                # convert reloaded data fnames from byte-encoded to string
                self.data.fnames = [self.data.fnames.decode('utf-8')]
            else:
                for i, fname in enumerate(self.data.fnames):
                    if isinstance(fname, bytes):
                        self.data.fnames[i] = fname.decode('utf-8')

            # infer number of mcorr splits from frames and num_frames_split
            T = caiman.base.movies.get_file_size(self.data.fnames, var_name_hdf5=self.data.var_name_hdf5)[1]
            if not isinstance(T, int):  # tuple returned if there are multiple files
                T = cast(int, T[0])  # TODO maybe allow different num_splits per file, or use max?

            num_splits = max(T//max(self.motion.num_frames_split, 10), 1)
            self.motion.splits_els = num_splits
            self.motion.splits_rig = num_splits

            # if movie_name_online is a relative path, resolve relative to input data directory
            self.online.movie_name_online = os.path.join(os.path.dirname(self.data.fnames[0]), self.online.movie_name_online)

        # set defaults that depend on other parameters
        if self.online.N_samples_exceptionality is None:
            self.online.N_samples_exceptionality = math.ceil(self.data.fr * self.data.decay_time)

        if self.online.thresh_fitness_raw is None:
            self.online.thresh_fitness_raw = scipy.special.log_ndtr(-self.online.min_SNR) * self.online.N_samples_exceptionality

        if self.init.gSig is None:
            self.init.gSig = [-1, -1]
        if self.init.gSiz is None:
            self.init.gSiz = [2*gs + 1 for gs in self.init.gSig]
        self.init.gSiz = [gs + 1 if gs % 2 == 0 else gs for gs in self.init.gSiz]  # ensure each entry is odd

        if self.init.nb <= 0 and (self.patch.nb_patch != self.init.nb or self.patch.low_rank_background is not None):
            logger.warning(f"nb={self.init.nb}, hence setting keys nb_patch and low_rank_background in group patch automatically.")
            self.set('patch', {'nb_patch': self.init['nb'], 'low_rank_background': None})

        if self.init.nb == -1 and self.spatial.update_background_components:
            logger.warning("nb=-1, hence setting key update_background_components " +
                           "in group spatial automatically to False.")
            self.set('spatial', {'update_background_components': False})

        if self.init.method_init == 'corr_pnr' and self.init.ring_size_factor is not None \
            and self.init.normalize_init:
            logger.warning("using CNMF-E's ringmodel for background hence setting key " +
                           "normalize_init in group init automatically to False.")
            self.set('init', {'normalize_init': False})

        if self.motion.is3D:
            for a in ('indices', 'max_shifts', 'strides', 'overlaps'):
                if len(self.motion[a]) != 3:
                    if self.motion[a][0] == self.motion[a][1]:
                        self.motion[a] = (self.motion[a][0],) * 3
                        logger.warning(f"is3D=True, hence setting key {a} to {self.motion[a]}")
                    else:
                        raise ValueError(f'{a} must be a tuple of length 3 for volumetric 3D data')

        for key in ('max_num_added', 'min_num_trial'):
            if (self.online[key] == 0 and self.online.update_num_comps):
                self.set('online', {'update_num_comps': False})
                logger.warning(f"{key}=0, hence setting key online.update_num_comps to False.")

        # FIXME The authoritative value is stored in the init field. This should later be refactored out
        #     into a general section, once we're passing around the CNMFParams object rather than splatting it out
        #     from **get_group
        self.spatial.nb  = self.init.nb
        self.temporal.nb = self.init.nb

    def set(self, group:str, val_dict:dict, set_if_not_exists:bool=False, verbose=False) -> None:
        """ Add key-value pairs to a group. Existing key-value pairs will be overwritten
            if specified in val_dict, but not deleted.

        Args:
            group: The name of the group
            val_dict: A dictionary with key-value pairs to be set for the group
            warn_unused: 
            set_if_not_exists: Whether to set a key-value pair in a group if the key does not currently exist in the group. (DEPRECATED)

        This is not intended for general use and does not run consistency checks on the CNMFParams object afterwards
        (or do any triggered actions on certain values being set like filenames). Usually the change_params() method is more appropriate.
        A future version of caiman may make this method private.
        """

        logger = logging.getLogger("caiman")
        if set_if_not_exists:
            logger.warning("The set_if_not_exists flag for CNMFParams.set() is deprecated and will be removed in a future version of Caiman")
            # can't easily catch if it's passed but set to False, but that wouldn't do anything because of the default,
            # and if they get that error it's at least really easy to fix - just remove the flag
            # we don't want to support this because it makes the structure of the object unpredictable except at runtime

        if group not in self.groups:
            raise KeyError(f'No group in CNMFParams named {group}')

        d: GroupParams = getattr(self._params, group)
        for k, v in val_dict.items():
            if k not in d and not set_if_not_exists:
                if verbose:
                    logger.warning(
                        f"{group}/{k} not set: invalid target in CNMFParams object")
            else:
                if not all_same(d[k], v):
                    logger.info(f"Changing key {k} in group {group} from {d[k]} to {v}")
                d[k] = v

    def get(self, group, key):
        """ Get a value for a given group and key. Raises an exception if no such group/key combination exists.

        Args:
            group: The name of the group.
            key: The key for the property in the group of interest.

        Returns: The value for the group/key combination.
        """  
        if group not in self.groups:
            raise KeyError(f'No group in CNMFParams named {group}')

        d: GroupParams = getattr(self._params, group)
        if key not in d:
            raise KeyError(f'No key {key} in group {group}')

        return d[key]


    # read access to each group - define overloads to pass through specific type information
    @overload
    def get_group(self, group: Literal['data']) -> DataParams:
        ...
    @overload
    def get_group(self, group: Literal['patch']) -> PatchParams:
        ...
    @overload
    def get_group(self, group: Literal['preprocess']) -> PreprocessParams:
        ...
    @overload
    def get_group(self, group: Literal['init']) -> InitParams:
        ...
    @overload
    def get_group(self, group: Literal['spatial']) -> SpatialParams:
        ...
    @overload
    def get_group(self, group: Literal['temporal']) -> TemporalParams:
        ...
    @overload
    def get_group(self, group: Literal['merging']) -> MergingParams:
        ...
    @overload
    def get_group(self, group: Literal['quality']) -> QualityParams:
        ...
    @overload
    def get_group(self, group: Literal['online']) -> OnlineParams:
        ...
    @overload
    def get_group(self, group: Literal['motion']) -> MotionParams:
        ...
    @overload
    def get_group(self, group: Literal['ring_CNN']) -> RingCNNParams:
        ...
    @overload
    def get_group(self, group: str) -> GroupParams:
        ...

    def get_group(self, group: str):
        """ Get the dictionary of key-value pairs for a group.

        Args:
            group: The name of the group.
        """
        if group in self.groups:
            return getattr(self._params, group)
        raise KeyError(f'No group in CNMFParams named {group}')
    
    # allow direct field access to group params (getattr fallback) with same overload typing as get_group
    __getattr__ = get_group

    def __eq__(self, other):
        if not isinstance(other, CNMFParams):
            return NotImplemented

        parent_dict1 = self.to_dict()
        parent_dict2 = other.to_dict()
        return parent_dict1 == parent_dict2  # uses __eq__ method defined on Subparams
    
    def get_differing_params(self, other: 'CNMFParams') -> Iterable[tuple[str, Any, Any]]:
        for groupname in self.groups:
            this_group = self.get_group(groupname)
            other_group = other.get_group(groupname)
            for (name, self_val, other_val) in this_group.get_differing_params(other_group):
                yield groupname + '.' + name, self_val, other_val


    def to_dict(self) -> dict[str, GroupParams]:
        """Returns the params class as a dictionary with subdictionaries for each
        category."""
        return msgspec.structs.asdict(self._params)
    

    def normalize_all_to_schema(self, verbose=True):
        """Try to convert each field of each group to the expected type"""
        for group in self.groups:
            self.get_group(group).normalize_to_schema(verbose=verbose)


    def to_json(self, verify=True) -> str:
        """ 
        Reversibly serialise CNMFParams to json. If verify is true, test that it can be
        deserialized correctly (meaning that all values match the original; it is
        possible that this happens even if they don't all match the schema).
        """
        logger = logging.getLogger('caiman')

        # normalize first to get the best chance of reconstruction
        self.normalize_all_to_schema(verbose=False)
        encoded = msgspec.to_builtins(self._params, enc_hook=enc_hook)
        jsonstring = json.dumps(encoded)  # use json library for dumping b/c it allows nans and infs

        if verify:
            logger.debug('Testing reconstruction from JSON')
            recon_obj = CNMFParams.from_json(jsonstring)

            mismatched = list(self.get_differing_params(recon_obj))
            if len(mismatched) > 0:
                # format a table of mismatched parameters
                headers = ('Param name', 'Current value', 'Reconstructed value', 'Expected type')
                table_rows = []
                for mismatch in mismatched:
                    group, param = mismatch[0].split('.')
                    param_type = self.get_group(group).typemap()[param]
                    if isinstance(param_type, type):
                        typename = param_type.__name__
                    else:
                        typename = str(param_type)
                    table_rows.append(mismatch + (typename,))
                
                mismatch_table = tabulate(table_rows, headers=headers)
                logger.warning(
                    'The following parameter(s) were not reconstructed correctly from JSON. '
                    'If this is an issue, please set each parameter to a value of the correct type.\n\n'
                     + mismatch_table + '\n')
            else:
                logger.debug('Reconstruction was successful.')

        return jsonstring


    def to_jsonfile(self, targfn: str, verify=True) -> None:
        """ Reversibly serialise CNMFParams to a json file """
        with open(targfn, 'w') as targfh:
            targfh.write(self.to_json(verify=verify))

    def __repr__(self) -> str:
        formatted_outputs = [
            f'{group_name}:\n\n{pformat(self.get_group(group_name))}' for group_name in self.groups
        ]

        return 'CNMFParams:\n\n' + '\n\n'.join(formatted_outputs)

    def change_params(self, params_dict, allow_legacy:bool=True, warn_unused:bool=True, verbose=True) -> None:
        """ Method for updating the params object by providing a dictionary.

        Args:
            params_dict: dictionary with parameters to be changed
            verbose: If true, will complain about types that don't match the schema.
            allow_legacy: If True, throw a deprecation warning and then attempt to
                          handle unconsumed keys using the older copy-it-everywhere logic.
                          We will eventually remove this option and the corresponding code.
            warn_unused: If True, emit warnings when the params dict has fields in it that
                         were never used in populating the Params object. You really should not
                         set this to False. Fix your code.
        """
        logger = logging.getLogger("caiman")
        # When we're ready to remove allow_legacy, this code will get a lot simpler

        # First collect updates in the nested format (and remove those that don't match any real param)
        nested_updates = {key: {} for key in self.groups}

        consumed = {} # Keep track of what parameters in params_dict were used to set something in params (just for legacy API)
        legacy_used = False # So we don't nag people multiple times in the same call
        for paramkey in params_dict:
            if paramkey in self.groups and isinstance(params_dict[paramkey], dict): # Handle proper pathed part. Latter half of the conditional is because of scoped keys with the same name as categories, because we apparently have those. ring_CNN is an example.
                curr_group = self.get_group(paramkey)
                for k, v in params_dict[paramkey].items():
                    if k == 'nb' and paramkey != 'init':
                        # Special casing to handle a misdesign in CNMFParams where some keys must have the same value in different
                        # sections.
                        if verbose:
                            logger.warning("The 'nb' parameter can only be set in the init part of CNMFParams. Attempts to set it elsewhere are ignored")
                        continue

                    if k not in curr_group and warn_unused:
                        # For regular/pathed API, we can notice right away if the user gave us something that won't update the object
                        logger.warning(f"In setting CNMFParams, provided key {paramkey}/{k} was not consumed. This is a bug!")
                    else:
                        nested_updates[paramkey][k] = v 
            # BEGIN code that we will remove in some future version of caiman
            elif allow_legacy:
                if paramkey in self._groups_for_flat_param:  # Known which group(s) to update
                    legacy_used = True
                    groups = self._groups_for_flat_param[paramkey]
                    for group in groups if isinstance(groups, list) else (groups,):
                        if isinstance(group, tuple):
                            # rename key for this group
                            this_key = group[1]
                            group = group[0]
                        else:
                            this_key = paramkey
                        nested_updates[this_key] = params_dict[paramkey]
                else:
                    logger.warning(f"Parameter {paramkey} is not supported as a flat parameter; attempting to "
                                   "change anyway, but name collisions are possible.")
                    for group in list(self.__dict__.keys()):
                        cat_handle = nested_updates[group] # Thankfully a read-write handle
                        if paramkey in cat_handle: # Is it known?
                            legacy_used = True
                            consumed[paramkey] = True
                            cat_handle[paramkey] = params_dict[paramkey] # Do the update

        if legacy_used:
            logger.warning(f"In setting CNMFParams, non-pathed parameters were used; this is deprecated. In some future version of Caiman, allow_legacy will default to False (and eventually will be removed)")

        # END
        if warn_unused:
            for toplevel_k in params_dict:
                if toplevel_k not in consumed and toplevel_k not in self.groups: # When we remove legacy behaviour, this logic will simplify and fold into above
                    logger.warning(f"In setting CNMFParams, provided toplevel key {toplevel_k} was unused. This is a bug!")

        # now update each group, attempting to convert each value
        for group in self.groups:
            if nested_updates[group]:
                group_params: GroupParams = getattr(self._params, group)
                group_params.update_checked(nested_updates[group], verbose=verbose)

        self.check_consistency()

    def change_params_from_json(self, jsonstring: str, verbose: bool = False) -> None:
        """ Same as change_params, except it takes json as input """
        # to avoid setting non-specified fields of the input params to defaults, we don't want
        # to decode entire subparameter group structs here; instead
        # defer converting types using msgspec until after change_params
        input_dict = json.loads(jsonstring)
        self.change_params(input_dict, verbose=verbose)

    def change_params_from_jsonfile(self, json_fn: str, verbose: bool = False) -> None:
        """ Same as change_params, except it takes a json file as input; pass the filename """
        with open(json_fn, 'r') as json_fh:
            jsonstring = json_fh.read()
        self.change_params_from_json(jsonstring, verbose=verbose)


    @classmethod
    def from_json(cls, jsonstring: str):
        """
        Try to directly deserialize a CNMFParams object from json, but fall back to
        change_params_from_json if conversion fails (e.g. if the JSON is in flat format).
        Unlike with change_params_from_json, since we're starting with a new object, we
        don't have to worry about keeping old values (instead of defaults) for params that
        are missing from the JSON.
        """
        logger = logging.getLogger('caiman')
        raw_dict = json.loads(jsonstring)
        try:
            params_struct = msgspec.convert(raw_dict, type=cls.AllParamsStruct, dec_hook=dec_hook)
            return cls(_params_struct=params_struct)
        except msgspec.ValidationError:
            logger.info('Could not load full params structure directly; falling back to updating each individually')
            new_obj = cls()
            new_obj.change_params_from_json(jsonstring)
            return new_obj

    @classmethod
    def from_jsonfile(cls, json_fn: str):
        with open(json_fn, 'r') as json_fh:
            jsonstring = json_fh.read()
        return cls.from_json(jsonstring)
        