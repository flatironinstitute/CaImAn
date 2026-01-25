#!/usr/bin/env python
"""Test CNMFParams object functionality"""

from dataclasses import replace, FrozenInstanceError, asdict
import logging
import math
import numpy as np
import numpy.testing as npt
import os
import pytest
import scipy.special
from tabulate import tabulate
from typing import Any, cast

from caiman.base import movies
import caiman.utils.utils
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf import params


def test_validation(caplog):
    """Test GroupParams type validators"""
    temporal_params = params.TemporalParams(solvers=[b'CVXOPT', 'SCS'], noise_range=(0.25, 0.5))  # type: ignore
    assert temporal_params.solvers == ['CVXOPT', 'SCS'], 'Bytes should be converted to string'
    assert temporal_params.noise_range == [0.25, 0.5], 'Tuple should be converted to list'
    assert len(caplog.records) == 0, 'Coercing these types should not cause a warning'

    # dataclasses.replace should be the same thing, but with an update
    modified_params = replace(temporal_params, noise_method=b'logmexp')
    assert modified_params.noise_method == 'logmexp', 'dataclasses.replace should update and validate'
    unmodified_params = replace(modified_params, noise_method=temporal_params.noise_method)
    assert unmodified_params == temporal_params, 'Should be the same after changing back'

    # we should get a warning if we try to update nb since it's "shared" (this should never be done this way in practice)
    caplog.clear()
    modified_params = replace(temporal_params, nb=2)
    assert len(caplog.records) == 1 and caplog.records[0].levelname == "WARNING" and \
         "can only be set in" in caplog.records[0].message, 'Should warn appropriately when setting nb'
    
    # test automatically wrapping scalar filename in list
    data_params = params.DataParams(fnames='abc')  # type: ignore
    assert data_params.fnames == ['abc'], 'Scalar fnames should be wrapped in a list.'

    # test frozenness
    with pytest.raises(FrozenInstanceError):
        data_params.dxy = [1., 1.]  # type: ignore

    # __getitem__
    assert data_params.var_name_hdf5 == data_params['var_name_hdf5'], '__getitem__ should work on GroupParams'
    
    
def test_params_serialization_eq(caplog):
    """Ensure we can create CNMFParams and the object is unchanged after roundtripping with JSON"""
    params_orig = params.CNMFParams()
    params_json = params_orig.to_json()
    params_recon = params.CNMFParams.from_json(params_json)
    assert params_orig == params_recon, \
         'Default params object should be equal after roundtripping with JSON. Differing parameters: \n\n' + \
         tabulate(params_orig.get_differing_params(params_recon), headers=['Name', 'Expected', 'Actual']) + '\n\n'

    assert len(caplog.records) == 0, 'Converting to and from JSON should not cause a warning'


def test_flat_constructor():
    """Test constructing CNMFParams with flat parameter names"""
    params_default = params.CNMFParams()

    changes = dict(
        var_name_hdf5='movie',   # typical one
        k=20,                    # renamed one
        p=3,                     # shared one 
        gnb=2                    # shared and renamed
    )

    params_obj = params.CNMFParams(**changes)  # type: ignore

    assert params_obj.data.var_name_hdf5 == changes['var_name_hdf5'], 'Normal flat param should be set'
    object.__setattr__(params_obj.data, 'var_name_hdf5', params_default.data.var_name_hdf5)

    assert params_obj.init.K == changes['k'], 'Renamed flat param should be set'
    object.__setattr__(params_obj.init, 'K', params_default.init.K)

    assert params_obj.preprocess.p == params_obj.temporal.p == changes['p'], \
        'Shared flat param should be set on both groups'
    object.__setattr__(params_obj.preprocess, 'p', params_default.preprocess.p)
    object.__setattr__(params_obj.temporal, 'p', params_default.temporal.p)

    assert params_obj.init.nb == params_obj.spatial.nb == params_obj.temporal.nb \
        == changes['gnb'], 'Shared and renamed param should be set on all groups'
    object.__setattr__(params_obj.init, 'nb', params_default.init.nb)
    object.__setattr__(params_obj.spatial, 'nb', params_default.spatial.nb)
    object.__setattr__(params_obj.temporal, 'nb', params_default.temporal.nb)

    assert params_obj == params_default, 'These should be the only changes. Differing parameters: \n\n' + \
         tabulate(params_default.get_differing_params(params_obj), headers=['Name', 'Expected', 'Actual']) + '\n\n'


def check_params_equal_expected(params_obj: params.CNMFParams, expected_params: dict[str, dict[str, tuple[Any, str]]],
                                cause='was not correct.'):
    """Validate the given CNMFParams object against a nested params dict"""
    for groupname, gt_group in expected_params.items():
        group = params_obj.get_group(groupname)
        for field, (gt_val, elaboration) in gt_group.items():
            npt.assert_array_equal(gt_val, group[field], f'Field {groupname}.{field} ' + cause + ' ' + elaboration)


def test_dict_constructor():
    pass


def test_json_constructor():
    pass


def test_object_constructor():
    pass


def test_change_params(caplog):
    """Test change_params method"""
    pass
    # params_orig = params.CNMFParams()


def test_check_consistency(caplog):
    # make params to test corrections performed by check_consistency
    demo_movie_path = os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')
    dims, T = movies.get_file_size(demo_movie_path)
    T = cast(int, T)

    input_params = {
        'data': {
            'fnames': [demo_movie_path],
            'fr': 15.5,
            'decay_time': 0.3
        },
        'init': {
            'gSig': None,
            'gSiz': None,
            # to trigger ring model check
            'method_init': 'corr_pnr',
            'ring_size_factor': 1.5,
            'normalize_init': True,
            'nb': -1
        },
        'patch': {
            'nb_patch': 1,
            'low_rank_background': None
        },
        'online': {
            'movie_name_online': 'onlineMovie.tif',
            'N_samples_exceptionality': None,
            'thresh_fitness_raw': None,
            'min_SNR': 2.5,
            'max_num_added': 0,
            'update_num_comps': True
        },
        'motion': {
            'num_frames_split': 80,
            'is3D': True,
            'indices': (slice(None), slice(None)),
            'max_shifts': (5, 5, 6),
            'strides': (96, 96),
            'overlaps': (32, 32)
        }
    }

    num_splits = max(T // max(input_params['motion']['num_frames_split'], 10), 1)
    nsamp_exc = math.ceil(input_params['data']['fr'] * input_params['data']['decay_time'])

    # corrected version of above
    expected_params = {
        'data': {
            'last_commit': ('-'.join(caiman.utils.utils.get_caiman_version()),
                            'Should be set to current CaImAn version.'),
            'dims': (dims, 'Should be set to dimensions of given movie')
        },
        'init': {
            'gSig': ([-1, -1], 'Should be set to [-1, -1] by default'),
            'gSiz': ([-1, -1], 'Should equal 2*gSig+1 by default'),
            'normalize_init': (False, 'Should turn off normalize_init with corr_pnr method')
        },
        'patch': {
            'nb_patch': (-1, 'Should be set to nb when nb < 0'),
        },
        'spatial': {
            'update_background_components': (False, 'Should be set to False when nb == -1'),
            'nb': (-1, 'Should be set based on init.nb'),
            'se': (np.ones((1,) * len(dims), dtype=np.uint8), 'Should be set due to corr_pnr method')
        },
        'temporal': {
            'nb': (-1, 'Should be set based on init.nb')
        },
        'online': {
            'movie_name_online': (os.path.join(os.path.dirname(demo_movie_path), input_params['online']['movie_name_online']),
                                  'Relative path should be resolved from the directory of fnames[0].'),
            'N_samples_exceptionality': (nsamp_exc, 'Should be set based on fr and decay_time'),
            'thresh_fitness_raw': (scipy.special.log_ndtr(-input_params['online']['min_SNR']) * nsamp_exc,
                                   'Should be set based on SNR threshold and N_samples_exceptionality'),
            'update_num_comps': (False, 'Should be set to False when online.max_num_added == 0')
        },
        'motion': {
            'splits_els': (num_splits, 'Should be set based on number of frames and num_frames_split'),
            'splits_rig': (num_splits, 'Should be set based on number of frames and num_frames_split'),
            'indices': ((slice(None), slice(None), slice(None)), 'Should be expanded to 3D'),
            'max_shifts': (input_params['motion']['max_shifts'], 'Should be left alone when already 3D'),
            'strides': ((96, 96, 96), 'Should be expanded to 3D'),
            'overlaps': ((32, 32, 32), 'Should be expanded to 3D')
        }
    }

    with caplog.at_level(logging.ERROR):  # ignore messages about automatically changed params
        params_obj = params.CNMFParams(params_dict=input_params)
    check_params_equal_expected(params_obj, expected_params, 'was not updated in check_consistency')
    
    # another gSig/gSiz case
    with caplog.at_level(logging.ERROR):
        params_obj.change_params({'init': {'gSiz': [6, 5]}})
    check_params_equal_expected(params_obj, {'init': {'gSiz': ([7, 5], 'Should be changed so each entry is odd')}})

    # another combination for patch
    with caplog.at_level(logging.ERROR):
        params_obj.change_params({
            'init': {'nb': -2},
            'patch': {'nb_patch': -2, 'low_rank_background': True},
            'spatial': {'update_background_components': True}
        })

    check_params_equal_expected(params_obj, {
        'patch': {'low_rank_background': (None, 'Should be set to None when nb < 0')},
        'spatial': {'update_background_components': (True, 'Should not be changed unless nb == -1')}
    })

    with caplog.at_level(logging.ERROR):
        params_obj.change_params({
            'online': {'min_num_trial': 0, 'update_num_comps': True}
        })
    
    check_params_equal_expected(params_obj, {
        'online': {'update_num_comps': (False, 'Should be set to False when online.min_num_trial == 0')}
    })


if __name__ == '__main__':
    retcode = pytest.main(['--verbose', __file__])
    exit(retcode)