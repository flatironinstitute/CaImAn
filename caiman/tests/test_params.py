#!/usr/bin/env python
"""Test CNMFParams object functionality"""

from copy import deepcopy
import msgspec
import numpy as np
from typing import Annotated

from caiman.source_extraction.cnmf import params
from caiman.source_extraction.cnmf.utilities import all_same


def test_custom_serialization():
    """Test custom enc_hook and dec_hook"""
    class TestObj(msgspec.Struct):
        """Object with a variety of types that should be encodable/decodable with custom hooks"""
        a_list: list[int] = msgspec.field(default_factory=lambda: [1, 2, 3])
        b_tuple_np: tuple[np.float64, ...] = tuple(np.array([1., 2., np.nan]))
        c_ndarray: np.ndarray = msgspec.field(default_factory=lambda: np.array("abc"))
        d_slice: slice = slice(0, 10, 2)
        e_int_np: np.int8 = np.int8(10)

    x = TestObj()
    x_builtins = msgspec.to_builtins(x, enc_hook=params.enc_hook)

    try:
        x_recon = msgspec.convert(x_builtins, type=TestObj, dec_hook=params.dec_hook)
    except msgspec.ValidationError as e:
        raise AssertionError('Test object could not be reconstructed') from e

    # test the types and values are the same
    for field_info in msgspec.structs.fields(TestObj):
        orig_val = getattr(x, field_info.name)
        recon_val = getattr(x_recon, field_info.name)
        assert type(orig_val) == type(recon_val), \
            f'Reconstucting field {field_info.name} converted type {type(orig_val)} to type {type(recon_val)}'
        assert all_same(orig_val, recon_val), \
            f'Reconstructing field {field_info.name} converted {orig_val} to {recon_val}'


# def test_group_params():
#     """Test GroupParams methods"""
#     # should be possible to assign wrong type to a parameter, but normalize_to_schema should fix it
#     data_params = params.DataParams(fnames=['string path', b'bytes path'])  # type: ignore
#     orig_params = deepcopy(data_params)
#     data_params.normalize_to_schema()
#     assert data_params.fnames == ['string path', 'bytes_path']

    
    

# def test_params_serialization_eq(caplog):
#     """Ensure we can create CNMFParams and the object is unchanged after roundtripping with JSON"""
#     params_orig = params.CNMFParams()
    