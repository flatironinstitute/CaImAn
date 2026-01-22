#!/usr/bin/env python
"""Test CNMFParams object functionality"""

from dataclasses import replace, FrozenInstanceError
import numpy as np
import pytest
from tabulate import tabulate

from caiman.source_extraction.cnmf import params
from caiman.source_extraction.cnmf.utilities import all_same


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
    
    # test wrap_scalar - tricky because this validator has to be called inside the global field one
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


# some dummy nested params to use, including some values that need to be normalized
nested_params = {
    'data': {
        'dxy': [1.2, 1.4],
    }    
}

# def test_update(caplog):
#     """Test update_params"""
#     params_orig = params.CNMFParams()


if __name__ == '__main__':
    retcode = pytest.main(['--verbose', __file__])
    exit(retcode)