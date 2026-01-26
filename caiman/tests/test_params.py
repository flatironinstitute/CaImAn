#!/usr/bin/env python
"""Test CNMFParams object functionality"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
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


def tabulate_differing_params(expected: params.CNMFParams, actual: params.CNMFParams) -> str:
    return '\n\n' + tabulate(expected.get_differing_params(actual), headers=['Name', 'Expected', 'Actual']) + '\n\n'


def test_validation(caplog):
    """Test GroupParams type validators"""
    temporal_params = params.TemporalParams(solvers=[b'CVXOPT', 'SCS'], noise_range=(0.25, 0.5))  # type: ignore
    assert temporal_params.solvers == ['CVXOPT', 'SCS'], 'Bytes should be converted to string'
    assert temporal_params.noise_range == [0.25, 0.5], 'Tuple should be converted to list'
    assert len(caplog.records) == 0, 'Coercing these types should not cause a warning'

    # replace should be the same thing, but with an update
    modified_params = temporal_params.replace(noise_method=b'logmexp')
    assert modified_params.noise_method == 'logmexp', 'dataclasses.replace should update and validate'
    unmodified_params = modified_params.replace(noise_method=temporal_params.noise_method)
    assert unmodified_params == temporal_params, 'Should be the same after changing back'

    # we should get a warning if we try to update nb since it's "shared" (this should never be done this way in practice)
    caplog.clear()
    modified_params = temporal_params.replace(nb=2)
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

    # validation error handling
    caplog.clear()
    modified_params = temporal_params.replace(fudge_factor='foobar')
    assert len(caplog.records) == 1 and caplog.records[0].levelname == "WARNING" and \
        "could not be converted" in caplog.records[0].message, 'Should warn appropriately when changing a value to the wrong type'
    assert modified_params.fudge_factor == 'foobar', 'Should allow change even if there is a validation error'
    
    
def test_params_serialization_eq(caplog):
    """Ensure we can create CNMFParams and the object is unchanged after roundtripping with JSON"""
    params_orig = params.CNMFParams()
    params_json = params_orig.to_json()
    params_recon = params.CNMFParams.from_json(params_json)
    assert params_orig == params_recon, \
        'Default params object should be equal after roundtripping with JSON. Differing parameters: ' + \
        tabulate_differing_params(params_orig, params_recon)

    assert len(caplog.records) == 0, 'Converting to and from JSON should not cause a warning'


# example nested dict with some various parameters to test constructors with
params_dict = {
    'data': {
        'var_name_hdf5': 'movie',
    },
    'init': {
        'K': 20,
        'nb': 2  # automatically updates spatial.nb and temporal.nb
    },
    'preprocess': {
        'p': 3
    },
    'temporal': {
        'p': 3
    }
}

# flat params version of the same parameters
params_dict_flat = {
    'var_name_hdf5': params_dict['data']['var_name_hdf5'],
    'k': params_dict['init']['K'],
    'gnb': params_dict['init']['nb'],
    'p': params_dict['preprocess']['p']
}


def test_change_params_flat():
    """
    Test that change_params method changes params as expected 
    (result is used to validate each constructor)
    """
    params_orig = params.CNMFParams()
    params_changed = deepcopy(params_orig)
    params_changed.change_params(params_dict_flat, verbose=False)
    
    assert params_changed.data.var_name_hdf5 == params_dict_flat['var_name_hdf5'], 'Normal flat param should be set'
    object.__setattr__(params_changed.data, 'var_name_hdf5', params_orig.data.var_name_hdf5)

    assert params_changed.init.K == params_dict_flat['k'], 'Renamed flat param should be set'
    object.__setattr__(params_changed.init, 'K', params_orig.init.K)

    assert params_changed.preprocess.p == params_changed.temporal.p == params_dict_flat['p'], \
        'Shared flat param should be set on both groups'
    object.__setattr__(params_changed.preprocess, 'p', params_orig.preprocess.p)
    object.__setattr__(params_changed.temporal, 'p', params_orig.temporal.p)

    assert params_changed.init.nb == params_changed.spatial.nb == params_changed.temporal.nb \
        == params_dict_flat['gnb'], 'Shared and renamed param should be set on all groups'
    object.__setattr__(params_changed.init, 'nb', params_orig.init.nb)
    object.__setattr__(params_changed.spatial, 'nb', params_orig.spatial.nb)
    object.__setattr__(params_changed.temporal, 'nb', params_orig.temporal.nb)

    assert params_changed == params_orig, 'These should be the only changes. Differing parameters: ' + \
        tabulate_differing_params(params_orig, params_changed)


def test_change_params_nested():
    params_changed_flat = params.CNMFParams()
    params_changed_flat.change_params(params_dict_flat, verbose=False) 

    params_changed_nested = params.CNMFParams()
    params_changed_nested.change_params(params_dict)

    assert params_changed_flat == params_changed_nested, \
        'Equivalent flat and nested params changes should result in equal CNMFParams objects. Differences: ' + \
        tabulate_differing_params(params_changed_flat, params_changed_nested)


def test_flat_constructor():
    """Test constructing CNMFParams with flat parameter names"""
    params_changed_flat = params.CNMFParams()
    params_changed_flat.change_params(params_dict_flat, verbose=False) 
    params_constr_flat = params.CNMFParams(**params_dict_flat)
    assert params_changed_flat == params_constr_flat, \
        'Constructing directly with flat params should work the same as change_params. Differences: ' + \
        tabulate_differing_params(params_changed_flat, params_constr_flat)


def test_dict_constructor():
    """Test constructing with params_dict"""
    params_changed = params.CNMFParams()
    params_changed.change_params(params_dict)
    params_constr_dict = params.CNMFParams(params_dict=params_dict)
    assert params_changed == params_constr_dict, \
        'Constructing directly with dict should work the same as change_params. Differences: ' + \
        tabulate_differing_params(params_changed, params_constr_dict)


def test_json_constructor(tmp_path):
    """Test constructing from a JSON file"""
    json_path = tmp_path / 'test_params.json'
    with open(json_path, 'w') as fh:
        json.dump(params_dict, fh)
    
    params_changed = params.CNMFParams()
    params_changed.change_params(params_dict)
    params_from_json = params.CNMFParams(params_from_file=json_path)
    assert params_changed == params_from_json, \
        'Constructing from JSON should work the same as change_params. Differences: ' + \
        tabulate_differing_params(params_changed, params_from_json)


def test_object_constructor():
    """Test constructing from individual GroupParams objects"""
    data = params.DataParams(**params_dict['data'])
    init = params.InitParams(**params_dict['init'])
    preprocess = params.PreprocessParams(**params_dict['preprocess'])
    temporal = params.TemporalParams(**params_dict['temporal'])

    params_changed = params.CNMFParams()
    params_changed.change_params(params_dict)
    params_from_objs = params.CNMFParams(data=data, init=init, preprocess=preprocess, temporal=temporal)
    assert params_changed == params_from_objs, \
        'Constructing from sub-objects should work the same as change_params. Differences: ' + \
        tabulate_differing_params(params_changed, params_from_objs)


def test_multi_dict_constructor():
    """Test constructing from multiple dicts of group params"""
    params_changed = params.CNMFParams()
    params_changed.change_params(params_dict)
    params_from_dicts = params.CNMFParams(**params_dict)
    assert params_changed == params_from_dicts, \
        'Constructing from dicts for each group should work the same as change_params. Differences: ' + \
        tabulate_differing_params(params_changed, params_from_dicts)


def test_json_roundtrip(tmp_path):
    """Test that saving and restoring whole object to/from JSON is successful"""
    json_path = tmp_path / 'full_params.json'
    params_orig = params.CNMFParams()
    params_orig.to_jsonfile(str(json_path), verify=False)
    params_recon  = params.CNMFParams.from_jsonfile(json_path)
    assert params_orig == params_recon, \
        'Full object should be equal after saving and restoring from JSON. Differences: ' + \
        tabulate_differing_params(params_orig, params_recon)


def check_params_equal_expected(params_obj: params.CNMFParams, expected_params: dict[str, dict[str, tuple[Any, str]]],
                                cause='was not correct.'):
    """Validate the given CNMFParams object against a nested params dict (not a test)"""
    for groupname, gt_group in expected_params.items():
        group = params_obj.get_group(groupname)
        for field, (gt_val, elaboration) in gt_group.items():
            npt.assert_array_equal(group[field], gt_val, f'Field {groupname}.{field} ' + cause + ' ' + elaboration)


def test_check_consistency():
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

    params_obj = params.CNMFParams(params_dict=input_params)
    check_params_equal_expected(params_obj, expected_params, 'was not updated in check_consistency.')
    
    # another gSig/gSiz case
    params_obj.change_params({'init': {'gSiz': [6, 5]}})
    check_params_equal_expected(params_obj, {'init': {'gSiz': ([7, 5], 'Should be changed so each entry is odd')}})

    # another combination for patch
    params_obj.change_params({
        'init': {'nb': -2},
        'patch': {'nb_patch': -2, 'low_rank_background': True},
        'spatial': {'update_background_components': True}
    })

    check_params_equal_expected(params_obj, {
        'patch': {'low_rank_background': (None, 'Should be set to None when nb < 0')},
        'spatial': {'update_background_components': (True, 'Should not be changed unless nb == -1')}
    })

    params_obj.change_params({
        'online': {'min_num_trial': 0, 'update_num_comps': True}
    })
    
    check_params_equal_expected(params_obj, {
        'online': {'update_num_comps': (False, 'Should be set to False when online.min_num_trial == 0')}
    })


if __name__ == '__main__':
    retcode = pytest.main(['--verbose', __file__])
    exit(retcode)