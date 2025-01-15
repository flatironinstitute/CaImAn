#!/usr/bin/env python

import numpy as np
import numpy.testing as npt
import os
from caiman.utils import utils
from caiman.paths import get_tempdir


def _recursively_assert_array_equal(a, b):
    """Get around array_equal not ignoring nans for nested objects"""
    if isinstance(a, dict):
        if not isinstance(b, dict):
            raise AssertionError('Values have different types')
        if len(a) != len(b):
            raise AssertionError('Dicts have different sizes')

        for key in a:
            if key not in b:
                raise AssertionError(f'Dicts have different keys ({key} not found)')
            _recursively_assert_array_equal(a[key], b[key])
    else:
        npt.assert_array_equal(a, b)


def test_save_and_load_dict_to_hdf5():
    filename = os.path.join(get_tempdir(), 'test_hdf5.hdf5')
    dict_to_save = {
        'int_scalar': 1,
        'int_vector': np.array([1, 2], dtype=int),
        'int_matrix': np.array([[1, 2], [3, 4]], dtype=int),
        'float32': np.array([[1., 2.], [3., 4.]], dtype='float32'),
        'float32_w_nans': np.array([[1., 2.], [3., np.nan]], dtype='float32'),
        'float64_w_nans': np.array([[1., 2.], [3., np.nan]], dtype='float64'),
        'dict': {
            'nested_float': np.array([1.0, 2.0])
        },
        'string': 'foobar',
        'bool': True,
        'dxy': (1.0, 2.0)  # specific key that should be saved as a tuple
    }
    # test no validation error on save
    utils.save_dict_to_hdf5(dict_to_save, filename)

    # test that the same data gets loaded
    loaded = utils.load_dict_from_hdf5(filename)
    _recursively_assert_array_equal(dict_to_save, loaded)


if __name__ == '__main__':
    test_save_and_load_dict_to_hdf5()