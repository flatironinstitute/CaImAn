#!/usr/bin/env python

from copy import copy, deepcopy
from dataclasses import fields, InitVar
from functools import cache, cached_property
import importlib.metadata
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
from pprint import pformat
from pydantic import (
    ConfigDict, TypeAdapter, BeforeValidator, AfterValidator, InstanceOf,
    PlainValidator, PlainSerializer, ValidationError, ValidationInfo,
    WithJsonSchema, Field, field_validator, computed_field, model_validator)
from pydantic.dataclasses import dataclass 
from pydantic.fields import FieldInfo
from pydantic.json_schema import SkipJsonSchema, PydanticJsonSchemaWarning
from pydantic_core import ArgsKwargs
import scipy.special
from scipy.ndimage import generate_binary_structure, iterate_structure
from tabulate import tabulate
from typing import (Optional, Any, Union, Literal, Annotated, Callable,
                    Mapping, Iterator, TypeVar, ClassVar, cast, Type)
import warnings

import caiman.base.movies
import caiman.utils.utils
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf import utilities

try:
    from pydantic import ValidateAs
except ImportError:
    # polyfill for pydantic < 2.12
    _FromTypeT = TypeVar('_FromTypeT')
    def ValidateAs(from_type: type[_FromTypeT], /, instantiation_hook: Callable[[_FromTypeT], Any]) -> Any:
        def validate_as_validator(obj: Any) -> Any:
            ta = TypeAdapter(from_type)
            validated = ta.validate_python(obj)
            return instantiation_hook(validated)
        return PlainValidator(validate_as_validator)        


# deal with 'NoneType', b'NoneType' strings
def interpret_string_none(obj: Any) -> Any:
    if (isinstance(obj, str) and obj in ['None', 'NoneType']
        or isinstance(obj, bytes) and obj in [b'None', b'NoneType']):
        return None
    return obj

SafeNone = Annotated[None, BeforeValidator(interpret_string_none)]
SafeAny = Annotated[Any, BeforeValidator(interpret_string_none)]

T = TypeVar('T')
SafeOptional = Union[SafeNone, T]


# validation/serialization of types not supported by pydantic out of the box
NDArray = Annotated[
    InstanceOf[np.ndarray],  # after applying np.asarray, just check that it is the right type
    BeforeValidator(np.asarray),
    PlainSerializer(lambda x: x.tolist()),
    WithJsonSchema({})
]


Slice = Annotated[
    Union[  # these are the same base types (slice) but with different validators
        InstanceOf[slice],  # accept existing slices as is
        # anything convertible to a len-3 tuple, with 'NoneType' conversion, can be a slice
        Annotated[slice, ValidateAs(tuple[SafeAny, SafeAny, SafeAny], lambda tup: slice(*tup))]],
    PlainSerializer(lambda sl: (sl.start, sl.stop, sl.step)),
    WithJsonSchema(TypeAdapter(tuple[Any, Any, Any]).json_schema())
]


# string pre-processing to use for string literals
LiteralType = TypeVar('LiteralType', bound=str)
LitStr = Annotated[LiteralType, BeforeValidator(TypeAdapter(str).validate_python)]


# automatically package string in list, for fnames
ItemType = TypeVar('ItemType')
AutoListStr = Union[list[str],  # first try parsing as the list of the desired type
                    Annotated[list[str], ValidateAs(str, lambda x: [x])]  # otherwise pack in list
]


# for gSiz, potentially other values that have to be odd integers
OddInt = Annotated[int, AfterValidator(lambda x: x + 1 if x % 2 == 0 else x)]


# ("Self" type for python < 3.11)
GPSelf = TypeVar('GPSelf', bound='GroupParams')

@dataclass(kw_only=True, eq=False, frozen=True)
class GroupParams(Mapping):
    """
    Struct that can also be used as a non-mutable mapping, to be used
    for subfields of CNMFParams, which have historically been dicts.

    Aliases and computed fields are used for parameters that are computed
    from other fields (potentially elsewhere in CNMFParams) if an explicit
    value is not provided. This allows these parameters to continue to be updated
    when the params it depends on are changed. The field containing the user-provided
    value (or None if none was provided) is prefixed with an underscore, but has
    the un-prefixed name as an alias; this allows the name to be used in the constructor,
    change_params, etc. This alias can also used to serialize the user-provided value
    when the round_trip option is True (e.g., when saving to JSON, or when copying
    the object using replace). When accessing the parameter using .<name> attribute or
    ['name'] mapping syntax, a property with the same name computes the actual value
    to use if none has been provided.
    """
    __pydantic_config__ = ConfigDict(extra='forbid', serialize_by_alias=True)
    __pydantic_fields__: ClassVar[Mapping[str, FieldInfo]]  # automatic, just declaring for typing purposes

    group_name: ClassVar[str]  # name of the attribute on CNMFParams

    # back-reference to help with some computed fields
    _full_params: 'SkipJsonSchema[Optional[CNMFParams]]' = Field(default=None, init=False, exclude=True, repr=False)

    
    @classmethod
    @cache
    def input_params(cls) -> set[str]:
        """Param names that can be used in constructor etc. (excludes purely computed fields)"""
        ta = TypeAdapter(cls)
        # we don't care if some defaults aren't serializable
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=PydanticJsonSchemaWarning, message='Default value')
            return set(ta.json_schema(mode='validation')['properties'].keys())


    @model_validator(mode='before')
    @classmethod
    def check_for_extra_fields(cls, data: Any, info: ValidationInfo) -> Any:
        logger = logging.getLogger('caiman')

        warn_unused = True
        if info.context is not None and 'warn_unused' in info.context:
            warn_unused = info.context['warn_unused']

        if isinstance(data, ArgsKwargs):  # from constructor
            if len(data.args) > 0:
                # Shouldn't happen, but I think kw_only may be buggy
                raise TypeError(f'{cls.__name__} does not take positional arguments.')
            
            if data.kwargs is None:
                return data
            
            input_dict = data.kwargs
        elif isinstance(data, dict):  # from validate_* methods
            input_dict = data
        else:
            return data
        
        # check for and remove extra fields
        argnames = tuple(input_dict)
        for argname in argnames:
            if argname not in cls.input_params():
                del input_dict[argname]
                if argname in cls.params():
                    logger.warning(f'The parameter {cls.group_name}/{argname} was ignored because it is '
                                   'computed from other parameters and cannot be set directly. ')
                elif warn_unused:
                    logger.warning(
                        f'When creating {cls.group_name} params, provided key {argname} was not consumed. '
                        'This is a bug!')
        return data


    @field_validator('*', mode='wrap')
    @classmethod
    def validation_wrapper(cls, value: Any, handler, info: ValidationInfo) -> Any:
        """
        Function that wraps validation on every field.
        This avoids raising a validation error when fields can't be converted, instead logging a warning.
        """
        try:
            return handler(value)
        except ValidationError:
            logger = logging.getLogger('caiman')
            assert info.field_name is not None, 'Should have field name in model field validator'

            field_info = cls.__pydantic_fields__[info.field_name]
            expected_type = field_info.annotation
            name = field_info.alias or info.field_name

            logger.warning(
                f'The value {repr(value)} provided for {cls.group_name}.{name} could not be converted '
                f'to the expected type {expected_type} and may not be valid.')
            
            return value


    def replace(self: GPSelf, warn_unused=True, **changes) -> GPSelf:
        """Create a GroupParams object with the given fields replaced"""
        ta = TypeAdapter(type(self))
        # use round_trip=True to serialize underlying fields rather than computed properties
        param_dict = ta.dump_python(self, round_trip=True)
        param_dict.update(changes)
        context = {'warn_unused': warn_unused}
        new_obj = ta.validate_python(param_dict, context=context)        
        return new_obj

    # support copy.replace (for 3.13 and above)
    __replace__ = replace

    # don't copy _full_params through copy or deepcopy
    def _copy_no_full(self: GPSelf, deep: bool, memo=None) -> GPSelf:
        attrs = self.__dict__.copy()
        del attrs['_full_params']
        if deep:
            attrs = deepcopy(attrs, memo=memo)
        my_copy = type(self)()
        my_copy.__dict__.update(attrs)
        return my_copy
    
    def __copy__(self):
        return self._copy_no_full(deep=False)
    
    def __deepcopy__(self, memo):
        return self._copy_no_full(deep=True, memo=memo)

    
    def get_differing_params(self: GPSelf, other: GPSelf) -> Iterator[tuple[str, Any, Any]]:
        """
        Returns an iterable of params that are not considered equal
        Each return value is a tuple: (name, this_value, other_value)
        """
        # here since we care about equality of the underlying fields, use
        # __pydantic_fields__ which includes the underscore-prefixed names
        for field, info in type(self).__pydantic_fields__.items():
            if info.exclude or info.init_var:
                continue

            self_val = getattr(self, field)
            other_val = getattr(other, field)
            if not utilities.all_same(self_val, other_val):
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


    #---- read-only mapping interface (for algorithms that use the parameters, includes computed fields) ----#

    # define things that don't require parameter values as classmethods

    @classmethod
    @cache
    def params(cls) -> set[str]:
        """Parameters available to read from this group"""
        # Use the JSON schema to ensure we respect excluded fields, etc
        ta = TypeAdapter(cls)
        # we don't care if some defaults aren't serializable
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=PydanticJsonSchemaWarning, message='Default value')
            return set(ta.json_schema(mode='serialization')['properties'].keys())

    @classmethod
    def __iter__(cls) -> Iterator[str]:
        yield from cls.params()

    @classmethod
    @cache
    def __len__(cls) -> int:
        return len(cls.params())


    def __getitem__(self, key: str) -> Any:
        if key in self.params():
            return getattr(self, key)
        raise KeyError(key)

    def copy(self) -> dict[str, Any]:
        """Implement dict.copy - make a copy of the data as a (mutable) dict"""
        # It's safe to assign to a copy, so just make it a (shallow-copied) dict
        ta = TypeAdapter(type(self))
        # user round_trip=False to serialize computed properties
        return ta.dump_python(self, round_trip=False)


# Parameter group definitions (see docstring of CNMFParams for full documentation)

@dataclass(kw_only=True, eq=False, frozen=True)
class DataParams(GroupParams):
    """Parameters for features of the data and other misc settings"""
    group_name = 'data'

    fnames: SafeOptional[AutoListStr] = None
    fr: float = Field(default=30., gt=0)
    decay_time: float = Field(default=0.4, gt=0)
    dxy: tuple[float, float] = (1., 1.)     # resolution, unit: pixels/um
    var_name_hdf5: str = 'mov'
    caiman_version: str = importlib.metadata.version('caiman')
    last_commit: str = '-'.join(caiman.utils.utils.get_caiman_version())

    @cached_property
    def first_file_size(self) -> Optional[tuple[tuple[int, ...], int]]:
        """get dims and T of first file, as in caiman.base.movies.get_file_size"""
        logger = logging.getLogger('caiman')

        if self.fnames is not None and len(self.fnames) > 0:
            try:
                dims, T = caiman.base.movies.get_file_size(self.fnames[0], var_name_hdf5=self.var_name_hdf5)
                return dims, cast(int, T)
            except FileNotFoundError:
                logger.warning('The first movie path in fnames was not found; cannot use dims.')


    @model_validator(mode='after')
    def refresh_file_size(self):
        """
        Make sure file size & things that depend on it are updated at least when
        the data params are changed (not foolproof but better than before)
        """
        # this is how you clear cache for a cached_property
        try:
            object.__delattr__(self, 'x')
        except AttributeError:
            pass
        return self

    @computed_field
    @property
    def dims(self) -> Optional[tuple[int, ...]]:
        sz = self.first_file_size
        if sz is not None:
            return sz[0]



@dataclass(kw_only=True, eq=False, frozen=True)
class PatchParams(GroupParams):
    """Parameters for how the data is divided into patches"""
    group_name = 'patch'

    border_pix: int = 0
    del_duplicates: bool = False
    in_memory: bool = True
    low_rank_background: SafeOptional[bool] = True
    memory_fact: float = 1.
    n_processes: int = 1
    nb_patch: int = 1
    only_init: bool = True
    p_patch: int = 0                        # AR order within patch
    remove_very_bad_comps: bool = False
    rf: Union[int, list[int], SafeNone] = None
    skip_refinement: bool = False
    p_ssub: float = 2.                      # spatial downsampling factor
    stride: SafeOptional[int] = None
    p_tsub: float = 2.                      # temporal downsampling factor


@dataclass(kw_only=True, eq=False, frozen=True)
class PreprocessParams(GroupParams):
    """Parameters for data preprocessing steps"""
    group_name = 'preprocess'

    check_nan: bool = True
    compute_g: bool = False                 # flag for estimating global time constant
    include_noise: bool = False             # flag for using noise values when estimating g
    # number of autocovariance lags to be considered for time constant estimation
    lags: int = 5
    max_num_samples_fft: int = 3 * 1024
    n_pixels_per_process: SafeOptional[int] = None
    noise_method: LitStr[Literal['mean', 'median', 'logmexp']] = 'mean'  # averaging method
    # range of normalized frequencies over which to average
    noise_range: list[float] = Field(default_factory=lambda: [0.25, 0.5])
    p: int = 2                              # order of AR indicator dynamics
    pixels: SafeOptional[list[int]] = None  # pixels to be excluded due to saturation
    sn: SafeOptional[NDArray] = None        # noise level for each pixel


@dataclass(kw_only=True, eq=False, frozen=True)
class InitParams(GroupParams):
    """Parameters that control how CNMF should be initialized"""
    group_name = 'init'

    K: SafeOptional[int] = 30               # number of components
    SC_kernel: LitStr[Literal['heat', 'cos', 'binary']] = 'heat'  # kernel for graph affinity matrix
    SC_sigma: float = 1.                    # std for SC kernel
    SC_thr: float = 0.                      # threshold for affinity matrix
    SC_normalize: bool = True               # standardize entries prior to computing affinity matrix
    SC_use_NN: bool = False                 # sparsify affinity matrix by using only nearest neighbors
    SC_nnn: int = 20                        # number of nearest neighbors to use
    alpha_snmf: float = 0.5
    center_psf: bool = False
    # this sets the default to [5, 5], but automatically converts None to [-1, -1]
    gSig: Annotated[list[int], 
                    BeforeValidator(interpret_string_none),
                    BeforeValidator(lambda val: [-1, -1] if val is None else val)
                    ] = Field(default_factory=lambda: [5, 5])
    # default based on gSiz computed below
    _gSiz: SafeOptional[list[OddInt]] = Field(default=None, alias='gSiz')
    # init method used in calls to NMF if geedy_roi method for component initialisation is used (offline or online)
    greedyroi_nmf_init_method: str = 'nndsvdar'
    # max_iter used in calls to NMF if greedy_roi method for component initialisation is used (online or offline)
    greedyroi_nmf_max_iter: int = 200
    init_iter: int = 2
    kernel: SafeOptional[NDArray] = None    # user specified template for greedyROI
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
    options_local_NMF: SafeOptional[dict] = None  # unused - local_NMF is removed
    perc_baseline_snmf: float = 20.
    ring_size_factor: float = 1.5
    rolling_length: int = 100
    rolling_sum: bool = True
    seed_method: LitStr[Literal['auto', 'manual', 'semi']] = 'auto'
    sigma_smooth_snmf: tuple[float, float, float] = (0.5, 0.5, 0.5)
    ssub: int = 2                        # spatial downsampling factor
    ssub_B: int = 2
    tsub: int = 2                        # temporal downsampling factor

    @property
    def gSiz(self) -> list[OddInt]:
        if self._gSiz is not None:
            return self._gSiz
        return [2*gs + 1 for gs in self.gSig]

    @model_validator(mode='after')
    def check_K_method(self):
        """Log an error if K is incompatible with the initialization method"""
        accept_none_K_methods = ['corr_pnr', 'sparse_nmf', 'graph_nmf']
        if self.K is None and self.method_init not in accept_none_K_methods:
            logger = logging.getLogger('caiman')
            logger.error(f'Parameter init.K cannot be set to None for the initialization method {self.method_init}.')
        return self


def default_expandcore() -> np.ndarray:
    """
    Generates the default morphological element used for footprint expansion
    with the dilate method, which is a 5x5 matrix that is true where taxicab
    distance from the center is <= 2 and false elsewhere.
    """
    s1 = generate_binary_structure(2, 1)
    s2 = iterate_structure(s1, 2)
    return s2.astype(int)  # type: ignore

@dataclass(kw_only=True, eq=False, frozen=True)
class SpatialParams(GroupParams):
    """Params that control how the algorithms handle spatial components"""
    group_name = 'spatial'

    dist: float = 3.                        # expansion factor of ellipse
    expandCore: NDArray = Field(default_factory=default_expandcore)
    # Flag to extract connected components (might want to turn to False for dendritic imaging)
    extract_cc: bool = True
    maxthr: float = 0.1                     # Max threshold
    medw: SafeOptional[tuple[int, ...]] = None  # window of median filter
    # method for determining footprint of spatial components
    method_exp: LitStr[Literal['ellipse', 'dilate']] = 'dilate'
    # 'nnls_L0'. Nonnegative least square with L0 penalty
    # 'lasso_lars' lasso lars function from scikit learn
    method_ls: LitStr[Literal['nnls_L0', 'lasso_lars']] = 'lasso_lars'
    # number of pixels to be processed by each worker
    n_pixels_per_process: SafeOptional[int] = None
    normalize_yyt_one: bool = True
    nrgthr: float = 0.9999                  # Energy threshold
    # number of process to parallelize residual computation ** DECREASE IF MEMORY ISSUES
    num_blocks_per_run_spat: int = 20
    se: SafeOptional[NDArray] = None        # Morphological closing structuring element
    ss: SafeOptional[NDArray] = None        # Binary element for determining connectivity
    thr_method: LitStr[Literal['max', 'nrg']] = 'nrg'  # Method of thresholding ('max' or 'nrg')
    # whether to update the background components in the spatial phase
    update_background_components: bool = True

    @computed_field
    @property
    def nb(self) -> int:
        if self._full_params is None:
            raise ValueError('Cannot access nb without reference to full params')
        return self._full_params.init.nb


@dataclass(kw_only=True, eq=False, frozen=True)
class TemporalParams(GroupParams):
    """Params that control how the algorithms handle temporal components"""
    group_name = 'temporal'

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
    method_deconvolution: LitStr[Literal['cvx', 'cvxpy', 'oasis']] = 'oasis'
    noise_method: LitStr[Literal['mean', 'median', 'logmexp']] = 'mean'  # averaging method
    # range of normalized frequencies over which to average
    noise_range: list[float] = Field(default_factory=lambda: [.25, .5])
    # number of process to parallelize residual computation ** DECREASE IF MEMORY ISSUES
    num_blocks_per_run_temp: int = 20
    p: int = 2                              # order of AR indicator dynamics
    s_min: SafeOptional[float] = None       # minimum spike threshold
    solvers: list[LitStr[Literal['ECOS', 'SCS', 'CVXOPT']]] = Field(default_factory=lambda: ['ECOS', 'SCS'])
    verbosity: bool = False

    @computed_field
    @property
    def nb(self) -> int:
        if self._full_params is None:
            raise ValueError('Cannot access nb without reference to full params')
        return self._full_params.init.nb


@dataclass(kw_only=True, eq=False, frozen=True)
class MergingParams(GroupParams):
    """Params that control how components are merged"""
    group_name = 'merging'

    do_merge: bool = True
    merge_thr: float = 0.8
    merge_parallel: bool = False


@dataclass(kw_only=True, eq=False, frozen=True)
class QualityParams(GroupParams):
    """Params that control how the quality of traces is evaluated"""
    group_name = 'quality'

    SNR_lowest: float = 0.5         # minimum accepted SNR value
    cnn_lowest: float = 0.1         # minimum accepted value for CNN classifier
    gSig_range: SafeOptional[list[int]] = None  # range for gSig scale for CNN classifier
    min_SNR: float = 2.5            # transient SNR threshold
    min_cnn_thr: float = 0.9        # threshold for CNN classifier
    rval_lowest: float = -1.        # minimum accepted space correlation
    rval_thr: float = 0.8           # space correlation threshold
    use_cnn: bool = True            # use CNN based classifier
    use_ecc: bool = False           # flag for eccentricity based filtering (2D only)
    max_ecc: float = 3.


@dataclass(kw_only=True, eq=False, frozen=True)
class OnlineParams(GroupParams):
    """Params that control the online/OnACID mode"""
    group_name = 'online'

    # timesteps to compute SNR (default computed below)
    _N_samples_exceptionality: SafeOptional[int] = Field(default=None, alias='N_samples_exceptionality')
    batch_update_suff_stat: bool = False
    dist_shape_update: bool = False       # update shapes in a distributed way
    ds_factor: int = 1                    # spatial downsampling for faster processing
    epochs: int = 1                       # number of epochs
    expected_comps: int = 500             # number of expected components
    full_XXt: bool = False                # store entire XXt matrix (as opposed to a list of sub-matrices) 
    init_batch: int = 200                 # length of mini batch for initialization
    init_method: LitStr[Literal['bare', 'cnmf', 'seeded']] = 'bare'  # initialization method for first batch
    iters_shape: int = 5                 # number of block-CD iterations
    max_comp_update_shape: Union[int, float] = np.inf
    max_num_added: int = 5               # maximum number of new components for each frame
    max_shifts_online: int = 10          # maximum shifts during motion correction
    min_SNR: float = 2.5                 # minimum SNR for accepting a new trace
    min_num_trial: int = 5               # number of mew possible components for each frame
    minibatch_shape: int = 100           # number of frames in each minibatch
    minibatch_suff_stat: int = 5
    motion_correct: bool = True          # flag for motion correction
    # filename of saved movie (appended to directory where data is located)
    _movie_name_online: str = Field(default='online_movie.mp4', alias='movie_name_online')
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
    # threshold for trace SNR (default computed below)
    _thresh_fitness_raw: SafeOptional[float] = Field(default=None, alias='thresh_fitness_raw')
    thresh_overlap: float = 0.5
    update_freq: int = 200               # update every shape at least once every update_freq steps
    update_num_comps: bool = True        # flag for searching for new components
    use_corr_img: bool = False           # flag for using correlation image to detect new components
    use_dense: bool = True               # flag for representation and storing of A and b
    use_peak_max: bool = True            # flag for finding candidate centroids
    W_update_factor: int = 1             # update W less often than shapes by a given factor

    @computed_field
    @property
    def N_samples_exceptionality(self) -> int:
        """compute N_samples_exceptionality from other params if None"""
        if self._N_samples_exceptionality is not None:
            return self._N_samples_exceptionality
    
        if self._full_params is None:
            raise RuntimeError('Cannot compute N_samples_exceptionality without reference to full params')
        
        fr = self._full_params.data.fr
        decay_time = self._full_params.data.decay_time
        return math.ceil(fr * decay_time)

    @computed_field
    @property
    def thresh_fitness_raw(self) -> float:
        """computes thresh_fitness_raw from other params if None"""
        if self._thresh_fitness_raw is not None:
            return self._thresh_fitness_raw
        return scipy.special.log_ndtr(-self.min_SNR) * self.N_samples_exceptionality
    
    @computed_field
    @property
    def movie_name_online(self) -> str:
        """Make movie_name_online relative to first movie path if it is available"""
        if os.path.isabs(self._movie_name_online) or self._full_params is None:
            return self._movie_name_online

        fnames = self._full_params.data.fnames
        if fnames is None or len(fnames) == 0:
            return self._movie_name_online
        
        return os.path.join(os.path.dirname(fnames[0]), self._movie_name_online)


@dataclass(kw_only=True, eq=False, frozen=True)
class MotionParams(GroupParams):
    """Params that control motion correction"""
    group_name = 'motion'

    # flag for allowing NaN in the boundaries
    #  - True: keep nans
    #  - False: replace with 0s
    #  - 'min': replace with minimum value in the frame
    #  - 'copy': copy edge values
    border_nan: Union[bool, LitStr[Literal['min', 'copy']]] = 'copy'
    gSig_filt: SafeOptional[tuple[int, ...]] = None # size of kernel for high pass spatial filtering in 1p data
    is3D: bool = False                  # flag for 3D recordings for motion correction
    max_deviation_rigid: int = 3        # maximum deviation between rigid and non-rigid
    max_shifts: tuple[int, ...] = (6,6) # maximum shifts per dimension (in pixels)
    min_mov: SafeOptional[float] = None # minimum value of movie
    niter_rig: int = 1                  # number of iterations rigid motion correction
    nonneg_movie: bool = True           # flag for producing a non-negative movie
    num_frames_split: int = 80          # split across time every x frames (approximately)
    overlaps: tuple[int, ...] = (32,32) # overlap between patches in pw-rigid motion correction
    pw_rigid: bool = False              # flag for performing pw-rigid motion correction
    shifts_interpolate: bool = False    # interpolate shifts based on patch locations instead of resizing
    shifts_opencv: bool = True          # flag for applying shifts using cubic interpolation (otherwise FFT)
    strides: tuple[int, ...] = (96, 96) # how often to start a new patch in pw-rigid registration
    upsample_factor_grid: int = 4       # motion field upsampling factor during FFT shifts
    use_cuda: bool = False              # flag for using a GPU
    indices: tuple[Slice, ...] = (slice(None), slice(None))  # part of FOV to be corrected

    @computed_field
    @property
    def num_splits_to_process_els(self) -> None:
        """Unused, will be removed in a future version of Caiman"""
        return None
    
    @computed_field
    @property
    def num_splits_to_process_rig(self) -> None:
        """Unused, will be removed in a future version of Caiman"""
        return None
    
    def _compute_splits_from_data(self) -> Optional[int]:
        """Compute splits_els and splits_rig values to use from data"""
        if self._full_params is not None:
            sz = self._full_params.data.first_file_size
            if sz is not None:
                # TODO maybe allow different num_splits per file, or use max?
                T_first = sz[1]
                return max(T_first // max(self.num_frames_split, 10), 1)
    
    @computed_field
    @property
    def splits_els(self) -> int:
        """number of splits across time for pw-rigid registration"""
        splits_from_data = self._compute_splits_from_data()
        if splits_from_data is not None:
            return splits_from_data
        return 14

    @computed_field
    @property
    def splits_rig(self) -> int:
        """number of splits across time for rigid registration"""
        splits_from_data = self._compute_splits_from_data()
        if splits_from_data is not None:
            return splits_from_data
        return 14


@dataclass(kw_only=True, eq=False, frozen=True)
class RingCNNParams(GroupParams):
    """Params that control the ring neural networks used for 1P background estimation"""
    group_name = 'ring_CNN'

    n_channels: int = 2                 # number of "ring" kernels   
    use_bias: bool = False              # use bias in the convolutions
    use_add: bool = False               # use an additive layer
    pct: float = 0.01                   # quantile loss specification
    patience: int = 3                   # patience for early stopping
    max_epochs: int = 100               # maximum number of epochs
    width: int = 5                      # width of "ring" kernel
    loss_fn: str = 'pct'                # loss function
    lr: float = 1e-3                    # (initial) learning rate
    lr_scheduler: SafeOptional[tuple[float, ...]] = None  # learning rate scheduler function arguments
    path_to_model: SafeOptional[str] = None # path to saved weights
    remove_activity: bool = False       # remove activity of last frame prior to background extraction
    reuse_model: bool = False           # reuse an already trained model


@dataclass(kw_only=True, frozen=True)
class CNMFParams:
    """
    Class for setting the processing parameters. All parameters for CNMF, online-CNMF, quality testing,
    and motion correction can be set here and then used in the various processing pipeline steps.

    The constructor supports setting params through 3 methods, in order of precedence (e.g., params
    set through method A override those set through method B). Note that attempting to override a
    group's parameters with an existing GroupParams object raises an error (a dict can be used instead).

        A) From individual group parameter objects passed to arguments matching the name of the group, as in:
            CNMFParams(data=DataParams(fnames=['example.tif']), motion=MotionParams(max_shifts=10))
            This method allows for static type checking of each parameter value.
           If preferred, raw dictionaries can also be passed instead of GroupParams objects,
            as in: CNMFParams(data={'fnames': ['example.tif']}, motion={'max_shifts': 10})
        B) From a nested dictionary through the params_dict parameter, as in:
            CNMFParams(params_dict={'data': {'fnames': ['example.tif']}, 'motion': {'max_shifts': 10}})
        C) From a JSON file through the params_from_file parameter.

    After construction, parameters can be changed from a nested dict using change_params() or from a
    JSON file using change_params_from_jsonfile(). These are the preferred methods for updating params
    because they automatically call check_consistency() to enforce consistency between different params.
    
    All other means of changing parameters are deprecated (including other constructor arguments)
    and will be removed in some future version of Caiman (whether they give a deprecation warning or not). 

    Args (keyword only):
        params_from_file
            name of a json file used to initialise the object
        params_dict
            a dictionary used to initialise the object
        <groupname> (e.g., data, init, preprocess...)
            a dictionary or object used to override parameters for just this group
        
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
                Whether to remove components with very low values of component quality directly on the patch.
                This might create some minor imprecisions, but can be important for performance because of bottlenecks
                caused by handling many components (we have seen over 2000) that will need to be processed.

            rf: int or list or None, default: None
                Half-size of patch in pixels. If None, no patches are constructed and the whole FOV is processed jointly.
                If list, it should be a list of two elements corresponding to the height and width of patches

            skip_refinement: bool, default: False
                If true it only performs one iteration of update spatial update temporal instead of two
                TODO: why is this in the patch section?

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
            K: int or None, default: 30
                number of components to be found (per patch or whole FOV depending on whether rf=None)
                None is only supported for the following methods: 'corr_pnr', 'sparse_nmf', and 'graph_nmf'.

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

            ssub: int, default: 2
                spatial downsampling factor

            ssub_B: int, default: 2
                downsampling factor for background during corr_pnr

            tsub: int, default: 2
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

            gSig_filt: tuple of ints or None, default: None
                size of kernel for high pass spatial filtering in 1p data. If None no spatial filtering is performed.
                Only the first element is used in practice (the kernel is circular).

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
    __pydantic_config__ = ConfigDict(extra='forbid')
    __pydantic_fields__: ClassVar[Mapping[str, FieldInfo]]  # automatic, just declaring for typing purposes

    # mapping of alternate names of flat params (previously used in constructor) to their canonical names
    flat_param_renames: ClassVar[dict[str, str]] = {
        'only_init_patch': 'only_init',
        'k': 'K',
        'gnb': 'nb',
    }

    # init-only params - these are the normal arguments to the constructor
    params_from_file: InitVar[Union[str, Path, None]] = None
    params_dict: InitVar[Optional[dict[str, Any]]] = None

    # group fields
    data: DataParams = Field(default_factory=DataParams)
    patch: PatchParams = Field(default_factory=PatchParams)
    preprocess: PreprocessParams = Field(default_factory=PreprocessParams)
    init: InitParams = Field(default_factory=InitParams)
    spatial: SpatialParams = Field(default_factory=SpatialParams)
    temporal: TemporalParams = Field(default_factory=TemporalParams)
    merging: MergingParams = Field(default_factory=MergingParams)
    quality: QualityParams = Field(default_factory=QualityParams)
    online: OnlineParams = Field(default_factory=OnlineParams)
    motion: MotionParams = Field(default_factory=MotionParams)
    ring_CNN: RingCNNParams = Field(default_factory=RingCNNParams)


    @cached_property
    def groups(self) -> list[str]:
        return [f.name for f in fields(self)]
    
    @classmethod
    @cache
    def get_group_types(cls) -> dict[str, Type[GroupParams]]:
        """Get name and type of each group params object. Depends on these being the only fields."""
        groups: dict[str, Type[GroupParams]] = {}
        for info in fields(cls):
            assert isinstance(info.type, type) and issubclass(info.type, GroupParams), \
                'Each field should be a GroupParams subclass'
            groups[info.name] = info.type
        return groups
    

    @model_validator(mode='before')
    @classmethod
    def _combine_parameters(cls, data: Any) -> Any:
        """
        Combine nested and/or flat parameters from JSON, params_dict, and/or direct arguments
        to a single neseted dict and pass this on to the dataclass constructor.
        This avoids multiple rounds of validation and check_consistency.
        """
        if isinstance(data, ArgsKwargs):  # from constructor  
            if len(data.args) > 0:
                # Shouldn't happen, but I think kw_only may be buggy
                raise TypeError('CNMFParams() does not take positional arguments.')
            
            if data.kwargs is None:
                return data
            
            input_dict = data.kwargs
        elif isinstance(data, dict):  # from validate_* method of TypeAdapter
            input_dict = data
        else:
            return data
        
        # Order: First JSON, then params_dict, finally individual arguments
        # No support for combining full GroupParams objects with JSON or params_dict
        # (will work for params_dict if the top-level keys don't overlap)
        kwargs = input_dict.copy()
        new_kwargs: dict[str, Any] = {}

        if (params_from_file := kwargs.pop('params_from_file', None)) is not None:
            with open(params_from_file, 'r') as fh:
                loaded_data = json.load(fh)
            
            if not isinstance(loaded_data, dict):
                raise ValueError('Params loaded from JSON must be a dict')
            
            cls.update_nested_params(nested_params=new_kwargs, new_params=loaded_data)
        
        if (params_dict := kwargs.pop('params_dict', None)) is not None:
            if not isinstance(params_dict, dict):
                raise ValueError('params_dict must be a dict')

            cls.update_nested_params(nested_params=new_kwargs, new_params=params_dict)
        
        # add group params and flat params passed as keyword arguments
        cls.update_nested_params(nested_params=new_kwargs, new_params=kwargs)
    
        return new_kwargs

    
    @model_validator(mode='after')
    def _check_post_validation(self):
        # add reference to self in each GroupParams object, bypassing frozen
        for field in fields(self):
            object.__setattr__(getattr(self, field.name), '_full_params', self)

        self.check_consistency()
        return self

    def check_consistency(self):
        """ Populates the params object with some dataset dependent values
        and ensures that certain constraints are satisfied.
        """
        logger = logging.getLogger("caiman")

        data_updates = {}
        data_updates['last_commit'] = '-'.join(caiman.utils.utils.get_caiman_version())

        if self.init.method_init == 'corr_pnr' and self.init.ring_size_factor is not None:
            if self.init.normalize_init:
                logger.warning("using CNMF-E's ringmodel for background hence setting key " +
                               "normalize_init in group init automatically to False.")
                self.set('init', {'normalize_init': False}, warn=False, verbose=False)
            
            # Set structuring element to the no-op value (previously done in initialization.greedyROI_corr)
            ndim = len(self.data.dims) if self.data.dims is not None else 2
            null_se = np.ones((1,) * ndim, dtype=np.uint8)
            if not utilities.all_same(self.spatial.se, null_se):
                logger.warning("using CNMF-E's ringmodel for background hence setting key "
                               "se in group spatial automatically to null element.")
                self.set('spatial', {'se': null_se}, warn=False, verbose=False)
       
        # -- end init params --

        if self.init.nb <= 0 and (self.patch.nb_patch != self.init.nb or self.patch.low_rank_background is not None):
            logger.warning(f"nb={self.init.nb}, hence setting keys nb_patch and low_rank_background in group patch automatically.")
            self.set('patch', {'nb_patch': self.init.nb, 'low_rank_background': None}, warn=False, verbose=False)

        if self.init.nb == -1 and self.spatial.update_background_components:
            logger.warning("nb=-1, hence setting key update_background_components " +
                           "in group spatial automatically to False.")
            self.set('spatial', {'update_background_components': False}, warn=False, verbose=False)

        if self.motion.is3D:
            motion_updates = {}
            for a in ('indices', 'max_shifts', 'strides', 'overlaps'):
                if len(self.motion[a]) != 3:
                    if self.motion[a][0] == self.motion[a][1]:
                        motion_updates[a] = (self.motion[a][0],) * 3
                        logger.warning(f"is3D=True, hence setting key {a} to {self.motion[a]}")
                    else:
                        raise ValueError(f'{a} must be a tuple of length 3 for volumetric 3D data')
            if motion_updates:
                self.set('motion', motion_updates, warn=False, verbose=False)

        for key in ('max_num_added', 'min_num_trial'):
            if (self.online[key] == 0 and self.online.update_num_comps):
                logger.warning(f"{key}=0, hence setting key online.update_num_comps to False.")
                self.set('online', {'update_num_comps': False}, warn=False, verbose=False)
                break
    

    def set(self, group: str, val_dict: dict, verbose=True, warn=True) -> None:
        """ Add key-value pairs to a group. Existing key-value pairs will be overwritten
            if specified in val_dict, but not deleted.

        Args:
            group: The name of the group
            val_dict: A dictionary with key-value pairs to be set for the group
            warn_unused: 

        This is not intended for general use and does not run consistency checks on the CNMFParams object afterwards
        (or do any triggered actions on certain values being set like filenames). Usually the change_params() method is more appropriate.
        A future version of caiman may make this method private.
        """
        logger = logging.getLogger("caiman")
        if warn:
            # this is the only way to change a param without running consistency checks...
            logger.warning("CNMFParams.set() is dangerous! Use CNMFParams.change_params() instead.")

        d = self.get_group(group)
        updates = {}

        for k, v in val_dict.items():
            if k not in d:
                if verbose:
                    logger.warning(
                        f"{group}/{k} not set: invalid target in CNMFParams object")
            else:
                if verbose and not utilities.all_same(d[k], v):
                    logger.info(f"Changing key {k} in group {group} from {d[k]} to {v}")
                updates[k] = v
        
        # apply changes, bypassing frozen
        if updates:
            d_new = d.replace(**updates)
            object.__setattr__(d_new, '_full_params', self)
            object.__setattr__(self, group, d_new)


    def get(self, group, key):
        """ Get a value for a given group and key. Raises an exception if no such group/key combination exists.

        Args:
            group: The name of the group.
            key: The key for the property in the group of interest.

        Returns: The value for the group/key combination.
        """  
        d = self.get_group(group)
        if key not in d:
            raise KeyError(f'No key {key} in group {group}')

        return d[key]


    def get_group(self, group: str) -> GroupParams:
        """ Get the dictionary of key-value pairs for a group.

        Args:
            group: The name of the group.
        """
        if group in self.groups:
            return getattr(self, group)
        raise KeyError(f'No group in CNMFParams named {group}')
    
    
    def get_differing_params(self, other: 'CNMFParams') -> Iterator[tuple[str, Any, Any]]:
        for groupname in self.groups:
            this_group = self.get_group(groupname)
            other_group = other.get_group(groupname)
            for (name, self_val, other_val) in this_group.get_differing_params(other_group):
                yield groupname + '.' + name, self_val, other_val


    def to_dict(self) -> dict[str, GroupParams]:
        """Returns the params class as a dictionary with subdictionaries for each
        category."""
        return {group: self.get_group(group) for group in self.groups}
    

    def to_json(self, verify=True) -> str:
        """ 
        Reversibly serialise CNMFParams to json. If verify is true, test that it can be
        deserialized correctly (meaning that all values match the original; it is
        possible that this happens even if they don't all match the schema).
        """
        logger = logging.getLogger('caiman')

        ta = TypeAdapter(type(self))
        encoded = ta.dump_python(self, mode='json', round_trip=True)
        jsonstring = json.dumps(encoded)  # use json library for dumping b/c it allows nans and infs

        if verify:
            logger.debug('Testing reconstruction from JSON')
            recon_obj = ta.validate_json(jsonstring)

            mismatched = list(self.get_differing_params(recon_obj))
            if len(mismatched) > 0:
                # format a table of mismatched parameters
                headers = ('Param name', 'Current value', 'Reconstructed value', 'Expected type')
                table_rows = []
                for mismatch in mismatched:
                    group, param = mismatch[0].split('.')
                    param_type = self.get_group(group).__pydantic_fields__[param].annotation
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


    def to_jsonfile(self, targfn: Union[str, Path], verify=True) -> None:
        """ Reversibly serialise CNMFParams to a json file """
        with open(targfn, 'w') as targfh:
            targfh.write(self.to_json(verify=verify))

    def __repr__(self) -> str:
        formatted_outputs = [
            f'{group_name}:\n\n{pformat(self.get_group(group_name))}' for group_name in self.groups
        ]

        return 'CNMFParams:\n\n' + '\n\n'.join(formatted_outputs)


    @classmethod
    def update_nested_params(cls, nested_params: dict[str, dict], new_params: Mapping[str, Any],
                             allow_legacy=True, warn_unused=True) -> dict[str, dict]:
        """
        Update a nested params dict with a dict potentially containing both flat and nested params.
        Keys are processed in order, later ones overriding earlier ones,
        but a warning is logged for each override. Does not check nested keys for validity.


        Pre-constructed objects of GroupParams subtypes are accepted under the group top-level keys,
        but only if no params have previously been processed from that group (including existing keys in
        nested_dict_in) because we don't know which params are user-specified vs. defaults.
        """
        logger = logging.getLogger('caiman')
        groups = cls.get_group_types()

        legacy_used = False
        for paramkey, paramval in new_params.items():
             # Handle proper pathed part. Latter half of the conditional is because of scoped keys with the same name as categories, because we apparently have those. ring_CNN is an example.
            if paramkey in groups and isinstance(paramval, (dict, GroupParams)):
                if paramkey not in nested_params:
                    if len(paramval) > 0:  # leave missing otherwise
                        if isinstance(paramval, GroupParams):
                            # avoid directly converting to dict which uses computed values
                            # and can fail if _full_params is unavailable
                            nested_params[paramkey] = TypeAdapter(type(paramval)).dump_python(paramval, round_trip=True)
                        else:
                            nested_params[paramkey] = paramval
                            
                elif isinstance(paramval, GroupParams):
                    raise ValueError(f'Cannot override other params for the {paramkey} group with a {type(paramval).__name__} object')
                else: # updating existing dict with a dict
                    overridden_keys = nested_params[paramkey].keys() & paramval.keys()
                    for subkey in overridden_keys:
                        logger.warning(f'Top-level parameter {subkey} was overridden by nested parameter {paramkey}/{subkey} - was this intended?') 
                    nested_params[paramkey].update(paramval)  # don't check every subkey here, they will be checked in GroupParams validator
            
            # BEGIN code that we will remove in some future version of caiman
            elif allow_legacy:
                paramkey_orig = paramkey
                if paramkey in cls.flat_param_renames:
                    # search for the renamed name (luckily there are not any that use different names for different groups)
                    paramkey = cls.flat_param_renames[paramkey]

                found = False
                for group, group_class in groups.items():
                    if paramkey in group_class.params(): # Is it known?
                        found = True
                        if group not in nested_params:
                            nested_params[group] = {paramkey: paramval}
                        else:
                            if paramkey in nested_params[group]:  # works for dicts or GroupParams
                                logger.warning(f'Parameter {group}/{paramkey} was overridden by top-level parameter {paramkey_orig} - was this intended?')

                            nested_params[group][paramkey] = paramval                
                if found:
                    legacy_used = True
                elif warn_unused:
                    logger.warning(f"In setting CNMFParams, provided toplevel key {paramkey_orig} was not consumed. This is a bug!")
            # END
            else:
                raise ValueError(f'Key {paramkey} does not match a parameter group, or the value is not a dictionary.')
        
        if legacy_used:
            logger.warning("In setting CNMFParams, non-pathed parameters were used; this is deprecated. "
                           "In some future version of Caiman, allow_legacy will default to False (and eventually will be removed).")

        return nested_params


    def change_params(self, params_dict: dict[str, Any], allow_legacy=True, warn_unused=True, verbose=False) -> None:
        """ Method for updating the params object by providing a dictionary.

        Args:
            params_dict: dictionary with parameters to be changed. Values may be in raw format
                         read directly from JSON; they will be converted based on each field's declared type.
            allow_legacy: If True, throw a deprecation warning and then attempt to
                          handle unconsumed keys using the older copy-it-everywhere logic.
                          We will eventually remove this option and the corresponding code.
            warn_unused: If True, emit warnings when the params dict has fields in it that
                         were never used in populating the Params object. You really should not
                         set this to False. Fix your code.
        """
        # First collect updates in the nested format (and remove those that don't match any real param)
        # Start with an empty dict for each group, to prevent passing GroupParams objects (since that would
        # confusingly override all params for that group)
        nested_params = {group: {} for group in self.groups}
        self.update_nested_params(
            nested_params=nested_params, new_params=params_dict, allow_legacy=allow_legacy, warn_unused=warn_unused)

        # now update each group, attempting to convert each value
        for group, group_updates in nested_params.items():
            if group_updates:
                # update group, bypassing frozen
                group_params = self.get_group(group)
                new_group_params = group_params.replace(warn_unused=warn_unused, **group_updates)
                object.__setattr__(new_group_params, '_full_params', self)
                object.__setattr__(self, group, new_group_params)

        self.check_consistency()

    def change_params_from_json(self, jsonstring: str, verbose: bool = False) -> None:
        """ Same as change_params, except it takes json as input """
        input_dict = json.loads(jsonstring)
        self.change_params(input_dict, verbose=verbose)

    def change_params_from_jsonfile(self, json_fn: str, verbose: bool = False) -> None:
        """ Same as change_params, except it takes a json file as input; pass the filename """
        with open(json_fn, 'r') as json_fh:
            jsonstring = json_fh.read()
        self.change_params_from_json(jsonstring, verbose=verbose)
