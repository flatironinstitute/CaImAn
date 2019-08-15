import pathlib

import numpy as np
import nose

from caiman import mmapping
from caiman.paths import caiman_datadir

twod_fname = pathlib.Path(caiman_datadir()) / 'testdata/memmap__d1_10_d2_11_d3_1_order_F_frames_12_.mmap'
threed_fname = pathlib.Path(caiman_datadir()) / 'testdata/memmap__d1_10_d2_11_d3_13_order_F_frames_12_.mmap'

def test_load_raises_wrong_ext():
    fname = "a.mmapp"
    try:
        mmapping.load_memmap(fname)
    except ValueError:
        assert True
    else:
        assert False


def test_load_raises_multiple_ext():
    fname = "a.mmap.mma"
    try:
        mmapping.load_memmap(fname)
    except ValueError:
        assert True
    else:
        assert False


def setup_2d_mmap():
    np.memmap(twod_fname, mode='w+', dtype=np.float32, shape=(12, 10, 11, 13), order='F')

def teardown_2d_mmap():
    twod_fname.unlink()

def setup_3d_mmap():
    np.memmap(threed_fname, mode='w+', dtype=np.float32, shape=(12, 10, 11, 13), order='F')

def teardown_3d_mmap():
    threed_fname.unlink()


@nose.with_setup(setup_2d_mmap, teardown_2d_mmap)
def test_load_successful_2d():
    fname = pathlib.Path(caiman_datadir()) / 'testdata' / twod_fname
    Yr, (d1, d2), T = mmapping.load_memmap(str(fname))
    assert (d1, d2) == (10, 11)
    assert T == 12
    assert isinstance(Yr, np.memmap)

@nose.with_setup(setup_3d_mmap, teardown_3d_mmap)
def test_load_successful_3d():
    fname = pathlib.Path(caiman_datadir()) / 'testdata' / threed_fname
    Yr, (d1, d2, d3), T = mmapping.load_memmap(str(fname))
    assert (d1, d2, d3) == (10, 11, 13)
    assert T == 12
    assert isinstance(Yr, np.memmap)


