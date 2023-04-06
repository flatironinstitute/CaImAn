#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from skimage.data import lfw_subset
from caiman.motion_correction import *

def gen_data(true_shifts=np.array([2, 4])):
    if len(true_shifts)==2:
        img = lfw_subset()[25].astype(np.float32)
        templ = img[4:20, 4:24]
        frame = img[4-true_shifts[0]:20-true_shifts[0], 4-true_shifts[1]:24-true_shifts[1]]
        nans = np.ones((16, 20), dtype=bool)
        nans[:-true_shifts[0],:-true_shifts[1]] = False
    elif len(true_shifts)==3:
        img = lfw_subset()[np.array([0,10,25,40,50])].transpose(1,2,0)
        templ = img[4:20, 4:24, 1:5]
        frame = img[4-true_shifts[0]:20-true_shifts[0],
                    4-true_shifts[1]:24-true_shifts[1],
                    1-true_shifts[2]:5-true_shifts[2]]
        nans = np.ones((16, 20, 4), dtype=bool)
        nans[:-true_shifts[0],:-true_shifts[1],:-true_shifts[2]] = False
    return frame, templ, nans


def _test_register_n_apply(D):
    true_shifts = np.array([2, 4, 1])[:D]
    frame, templ, nans = gen_data(true_shifts)
    reg = register_translation if D==2 else register_translation_3d
    shifts, src_freq, phasediff = reg(frame, templ)
    npt.assert_allclose(shifts, true_shifts)
    frame_cor = apply_shifts_dft(src_freq, -shifts, phasediff, is_freq=True, border_nan=True)
    npt.assert_array_equal(np.isnan(frame_cor), nans)
    npt.assert_allclose(frame_cor[~nans], templ[~nans], 1e-6)

def test_register_n_apply():
    _test_register_n_apply(2)

def test_register_n_apply_3d():
    _test_register_n_apply(3)


def _test_tile_and_correct(D):
    true_shifts = np.array([2, 4, 1])[:D]
    frame, templ, _ = gen_data(true_shifts)
    tac = tile_and_correct if D==2 else tile_and_correct_3d
    frame_cor, shifts = tac(frame, templ, None, None, (4,)*D, max_deviation_rigid=0)[:2]
    npt.assert_allclose(shifts, -true_shifts, .2)
    nans = np.isnan(frame_cor)
    npt.assert_allclose(np.corrcoef(frame_cor[~nans].ravel(), templ[~nans].ravel())[0,1], 1, .06)

def test_tile_and_correct():
    _test_tile_and_correct(2)

def test_tile_and_correct_3d():
    _test_tile_and_correct(3)


def _test_iteration(fast):
    true_shifts = np.array([2, 4])
    frame, templ, nans = gen_data(true_shifts)
    if fast:
        frame_cor, shifts = motion_correct_iteration_fast(frame, templ, 4, 4)
    else:
        frame_cor, _, shifts, _ = motion_correct_iteration(frame, templ, 1, 4, 4)
    npt.assert_allclose(shifts, -true_shifts)
    npt.assert_allclose(np.corrcoef(frame_cor[~nans].ravel(), templ[~nans].ravel())[0,1], 1)

def test_motion_correct_iteration():
    _test_iteration(False)

def test_motion_correct_iteration_fast():
    _test_iteration(True)
