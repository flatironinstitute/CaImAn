import pathlib

import numpy as np
import nose

from caiman import mmapping
from caiman.paths import caiman_datadir


TWO_D_FNAME = (
    pathlib.Path(caiman_datadir())
    / "testdata/memmap__d1_10_d2_11_d3_1_order_F_frames_12_.mmap"
)
THREE_D_FNAME = (
    pathlib.Path(caiman_datadir())
    / "testdata/memmap__d1_10_d2_11_d3_13_order_F_frames_12_.mmap"
)


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
    np.memmap(
        TWO_D_FNAME, mode="w+", dtype=np.float32, shape=(12, 10, 11, 13), order="F"
    )


def teardown_2d_mmap():
    TWO_D_FNAME.unlink()


def setup_3d_mmap():
    np.memmap(
        THREE_D_FNAME, mode="w+", dtype=np.float32, shape=(12, 10, 11, 13), order="F"
    )


def teardown_3d_mmap():
    THREE_D_FNAME.unlink()


@nose.with_setup(setup_2d_mmap, teardown_2d_mmap)
def test_load_successful_2d():
    fname = pathlib.Path(caiman_datadir()) / "testdata" / TWO_D_FNAME
    Yr, (d1, d2), T = mmapping.load_memmap(str(fname))
    assert (d1, d2) == (10, 11)
    assert T == 12
    assert isinstance(Yr, np.memmap)


@nose.with_setup(setup_3d_mmap, teardown_3d_mmap)
def test_load_successful_3d():
    fname = pathlib.Path(caiman_datadir()) / "testdata" / THREE_D_FNAME
    Yr, (d1, d2, d3), T = mmapping.load_memmap(str(fname))
    assert (d1, d2, d3) == (10, 11, 13)
    assert T == 12
    assert isinstance(Yr, np.memmap)
