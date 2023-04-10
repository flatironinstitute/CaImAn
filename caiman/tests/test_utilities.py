#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from scipy import ndimage as ndi
from caiman.source_extraction.cnmf import utilities


def _test_gaussian_filter(D):
    """test agreement with scipy.ndimage"""
    img = np.random.rand(*(10,)*D)
    gSig = (5, 3, 2, 4)[:D]
    for mode in ('nearest', 'reflect', 'constant', 'mirror'):
        for truncate in (2, 4):
            sc = ndi.gaussian_filter(img, gSig, mode=mode, truncate=truncate, cval=2)
            cv = utilities.gaussian_filter(img, gSig, mode=mode, truncate=truncate, cval=2)
            npt.assert_allclose(sc, cv)


def test__test_gaussian_filter_1d():
    _test_gaussian_filter(1)


def test__test_gaussian_filter():
    _test_gaussian_filter(2)


def test__test_gaussian_filter_3d():
    _test_gaussian_filter(3)


def test__test_gaussian_filter_4d():
    _test_gaussian_filter(4)


def test_gaussian_filter_output():
    """test array in which to place the output"""
    for D in (1, 2, 3):
        img = np.random.rand(*(10,)*D)
        out = np.zeros_like(img)
        tmp = utilities.gaussian_filter(img, (5, 3, 2, 4)[:D], output=out)
        npt.assert_array_equal(out, tmp)
        tmp = utilities.gaussian_filter(img, (5, 3, 2, 4)[:D], output=out, mode='constant', cval=2)
        npt.assert_array_equal(out, tmp)
