#!/usr/bin/env python

import os

import numpy as np
import numpy.testing as npt
import caiman as cm
from caiman.utils import sbx_utils
from caiman.paths import caiman_datadir

testdata_path = os.path.join(caiman_datadir(), 'testdata')

def subinds_to_ix(subinds, array_shape):
    """Helper to avoid advanced slicing"""
    fixed_subinds = [range(s)[inds] if isinstance(inds, slice) else inds
                     for inds, s in zip(subinds, array_shape)]
    return np.ix_(*fixed_subinds)


def test_load_2d():
    file_2d = os.path.join(testdata_path, '2d_sbx.sbx')
    data_2d = sbx_utils.sbxread(file_2d)
    meta_2d = sbx_utils.sbx_meta_data(file_2d)
    
    assert data_2d.ndim == 3, 'Loaded 2D data has wrong dimensionality'
    assert data_2d.shape == (10, 512, 796), 'Loaded 2D data has wrong shape'
    assert data_2d.shape == (meta_2d['num_frames'], *meta_2d['frame_size']), 'Shape in metadata does not match loaded data'
    npt.assert_array_equal(data_2d[0, 0, :10], [712, 931, 1048, 825, 1383, 882, 601, 798, 1022, 966], 'Loaded 2D data has wrong values')

    data_2d_movie = cm.load(file_2d)
    assert data_2d_movie.ndim == data_2d.ndim, 'Movie loaded with cm.load has wrong dimensionality'
    assert data_2d_movie.shape == data_2d.shape, 'Movie loaded with cm.load has wrong shape'
    npt.assert_array_almost_equal(data_2d_movie, data_2d, err_msg='Movie loaded with cm.load has wrong values')


def test_load_3d():
    file_3d = os.path.join(testdata_path, '3d_sbx_1.sbx')
    data_3d = sbx_utils.sbxread(file_3d)
    meta_3d = sbx_utils.sbx_meta_data(file_3d)
    
    assert data_3d.ndim == 4, 'Loaded 3D data has wrong dimensionality'
    assert data_3d.shape == (2, 512, 796, 4), 'Loaded 3D data has wrong shape'
    assert data_3d.shape == (meta_3d['num_frames'], *meta_3d['frame_size'], meta_3d['num_planes']), 'Shape in metadata does not match loaded data'
    npt.assert_array_equal(data_3d[0, 0, :10, 0], [2167, 2525, 1713, 1747, 1887, 1741, 1873, 1244, 1747, 1637], 'Loaded 2D data has wrong values')

    data_3d_movie = cm.load(file_3d, is3D=True)
    assert data_3d_movie.ndim == data_3d.ndim, 'Movie loaded with cm.load has wrong dimensionality'
    assert data_3d_movie.shape == data_3d.shape, 'Movie loaded with cm.load has wrong shape'
    npt.assert_array_almost_equal(data_3d_movie, data_3d, err_msg='Movie loaded with cm.load has wrong values')


def test_load_subind():
    file_2d = os.path.join(testdata_path, '2d_sbx.sbx')
    data_2d_full = cm.load(file_2d)

    # Just frame subset
    data_2d_head3 = cm.load(file_2d, subindices=slice(0, 3))
    npt.assert_array_equal(data_2d_full[:3], data_2d_head3)
    data_2d_ds2 = cm.load(file_2d, subindices=slice(0, None, 2))
    npt.assert_array_equal(data_2d_full[::2], data_2d_ds2)
    arb_subind = [0, 3, 5, 8]
    data_2d_arb = cm.load(file_2d, subindices=arb_subind)
    npt.assert_array_equal(data_2d_full[arb_subind], data_2d_arb)

    # Subset on multiple dimensions
    subind_t_y = (slice(0, 3), [0, *range(2, 512)])
    data_2d_t_y = cm.load(file_2d, subindices=subind_t_y)
    npt.assert_array_equal(data_2d_full[subinds_to_ix(subind_t_y, data_2d_full.shape)], data_2d_t_y)
    subind_t_y_x = subind_t_y + (slice(0, None, 2),)
    data_2d_t_y_x = cm.load(file_2d, subindices=subind_t_y_x)
    npt.assert_array_equal(data_2d_full[subinds_to_ix(subind_t_y_x, data_2d_full.shape)], data_2d_t_y_x)

    # Check 3D file
    file_3d = os.path.join(testdata_path, '3d_sbx_1.sbx')
    data_3d_full = cm.load(file_3d, is3D=True)
    data_3d_first = cm.load(file_3d, is3D=True, subindices=slice(0, 1))
    npt.assert_array_equal(data_3d_full[:1], data_3d_first)
    subind_t_y_x_z = (slice(0, 1), [0, *range(2, 512)], slice(0, None, 2), [0, 2])
    data_3d_t_y_x_z = cm.load(file_3d, is3D=True, subindices=subind_t_y_x_z)
    npt.assert_array_equal(data_3d_full[subinds_to_ix(subind_t_y_x_z, data_3d_full.shape)], data_3d_t_y_x_z)


def test_sbx_to_tif():
    tif_file = os.path.join(caiman_datadir(), 'temp', 'from_sbx.tif')
    try:
        file_2d = os.path.join(testdata_path, '2d_sbx.sbx')
        data_2d_from_sbx = cm.load(file_2d)
        sbx_utils.sbx_to_tif(file_2d, fileout=tif_file)
        data_2d_from_tif = cm.load(tif_file)
        npt.assert_array_almost_equal(data_2d_from_sbx, data_2d_from_tif,
                                      err_msg='Data do not match when loaded from .sbx vs. .tif')
        
        file_3d = os.path.join(testdata_path, '3d_sbx_1.sbx')
        data_3d_from_sbx = cm.load(file_3d, is3D=True)
        sbx_utils.sbx_to_tif(file_3d, fileout=tif_file)
        data_3d_from_tif = cm.load(tif_file, is3D=True)
        npt.assert_array_almost_equal(data_3d_from_sbx, data_3d_from_tif,
                                      err_msg='3D data do not match when loaded from .sbx vs. .tif')
        
        # with subindices
        subinds = (slice(0, None, 2), [0, 1, 3], slice(None))
        sbx_utils.sbx_to_tif(file_2d, fileout=tif_file, subindices=subinds)
        sub_data_from_tif = cm.load(tif_file)
        npt.assert_array_almost_equal(data_2d_from_sbx[subinds_to_ix(subinds, data_2d_from_sbx.shape)], sub_data_from_tif)

    finally:
        # cleanup
        if os.path.isfile(tif_file):
            os.remove(tif_file)


def test_sbx_chain_to_tif():
    tif_file = os.path.join(caiman_datadir(), 'temp', 'from_sbx.tif')
    try:
        file_3d_1 = os.path.join(testdata_path, '3d_sbx_1.sbx')
        data_3d_1 = sbx_utils.sbxread(file_3d_1)
        file_3d_2 = os.path.join(testdata_path, '3d_sbx_2.sbx')
        data_3d_2 = sbx_utils.sbxread(file_3d_2)

        # normal chain
        sbx_utils.sbx_chain_to_tif([file_3d_1, file_3d_2], fileout=tif_file)
        data_chain_tif = cm.load(tif_file, is3D=True)
        data_chain_gt = np.concatenate([data_3d_1, data_3d_2], axis=0)
        npt.assert_array_almost_equal(data_chain_tif, data_chain_gt,
                                      err_msg='Tif from chain does not match expected')

        # matching subindices
        sbx_utils.sbx_chain_to_tif([file_3d_1, file_3d_2], fileout=tif_file,
                                   subindices=(slice(None), slice(0, None, 2)))
        data_chain_tif = cm.load(tif_file, is3D=True)
        data_chain_gt = data_chain_gt[:, ::2]
        npt.assert_array_almost_equal(data_chain_tif, data_chain_gt,
                                      err_msg='Tif from chain with subindices does not match expected')
        
        # non-matching subindices with compatible shapes
        subinds_1 = (slice(None), [0, 1, 3], slice(0, None, 2), [0, 2])
        subinds_2 = (slice(1, None), [-4, -2, -1], slice(1, None, 2), [1, 3])
        sbx_utils.sbx_chain_to_tif([file_3d_1, file_3d_2], fileout=tif_file,
                                    subindices=[subinds_1, subinds_2])
        data_chain_tif = cm.load(tif_file, is3D=True)
        data_chain_gt = np.concatenate([data_3d_1[subinds_to_ix(subinds_1, data_3d_1.shape)],
                                        data_3d_2[subinds_to_ix(subinds_2, data_3d_2.shape)]], axis=0)
        npt.assert_array_almost_equal(data_chain_tif, data_chain_gt,
                                      err_msg='Tif from chain with non-matching subindices does not match expected')

    finally:
        # cleanup
        if os.path.isfile(tif_file):
            os.remove(tif_file)
    