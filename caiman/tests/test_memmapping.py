import pathlib

import numpy as np
import pytest

from caiman import mmapping
from caiman.paths import caiman_datadir


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


@pytest.fixture(scope="function")
def three_d_mmap_fname():
    THREE_D_FNAME = (
        pathlib.Path(caiman_datadir())
        / "testdata/memmap__d1_10_d2_11_d3_13_order_F_frames_12_.mmap"
    )
    np.memmap(
        THREE_D_FNAME, mode="w+", dtype=np.float32, shape=(12, 10, 11, 13), order="F"
    )
    try:
        yield THREE_D_FNAME
    finally:
        THREE_D_FNAME.unlink()


@pytest.fixture(scope="function")
def two_d_mmap_fname():
    TWO_D_FNAME = (
        pathlib.Path(caiman_datadir())
        / "testdata/memmap__d1_10_d2_11_d3_1_order_F_frames_12_.mmap"
    )
    np.memmap(
        TWO_D_FNAME, mode="w+", dtype=np.float32, shape=(12, 10, 11, 13), order="F"
    )
    try:
        yield TWO_D_FNAME
    finally:
        TWO_D_FNAME.unlink()


def test_load_successful_2d(two_d_mmap_fname):
    fname = two_d_mmap_fname
    Yr, (d1, d2), T = mmapping.load_memmap(str(fname))
    assert (d1, d2) == (10, 11)
    assert T == 12
    assert isinstance(Yr, np.memmap)


def test_load_successful_3d(three_d_mmap_fname):
    fname = three_d_mmap_fname
    Yr, (d1, d2, d3), T = mmapping.load_memmap(str(fname))
    assert (d1, d2, d3) == (10, 11, 13)
    assert T == 12
    assert isinstance(Yr, np.memmap)
