#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.data import lfw_subset
import caiman as cm
from caiman.motion_correction import *


def gen_frame_n_templ(true_shifts=np.array([2, 4])):
    if len(true_shifts) == 2:
        img = lfw_subset()[25].astype(np.float32)
        templ = img[4:20, 4:24]
        frame = img[4-true_shifts[0]:20-true_shifts[0], 4-true_shifts[1]:24-true_shifts[1]]
        nans = np.ones((16, 20), dtype=bool)
        nans[:-true_shifts[0], :-true_shifts[1]] = False
    elif len(true_shifts) == 3:
        img = lfw_subset()[np.array([0, 10, 25, 40, 50])].transpose(1, 2, 0)
        templ = img[4:20, 4:24, 1:5]
        frame = img[4-true_shifts[0]:20-true_shifts[0],
                    4-true_shifts[1]:24-true_shifts[1],
                    1-true_shifts[2]:5-true_shifts[2]]
        nans = np.ones((16, 20, 4), dtype=bool)
        nans[:-true_shifts[0], :-true_shifts[1], :-true_shifts[2]] = False
    return frame, templ, nans


def _test_register_n_apply(D):
    true_shifts = np.array([2, 4, 1])[:D]
    frame, templ, nans = gen_frame_n_templ(true_shifts)
    reg = register_translation if D == 2 else register_translation_3d
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
    frame, templ, _ = gen_frame_n_templ(true_shifts)
    tac = tile_and_correct if D == 2 else tile_and_correct_3d
    frame_cor, shifts = tac(frame, templ, None, None, (4,)*D, max_deviation_rigid=0)[:2]
    npt.assert_allclose(shifts, -true_shifts, .2)
    nans = np.isnan(frame_cor)
    npt.assert_allclose(np.corrcoef(frame_cor[~nans].ravel(), templ[~nans].ravel())[0, 1], 1, .06)


def test_tile_and_correct():
    _test_tile_and_correct(2)


def test_tile_and_correct_3d():
    _test_tile_and_correct(3)


def _test_iteration(fast):
    true_shifts = np.array([2, 4])
    frame, templ, nans = gen_frame_n_templ(true_shifts)
    if fast:
        frame_cor, shifts = motion_correct_iteration_fast(frame, templ, 4, 4)
    else:
        frame_cor, _, shifts, _ = motion_correct_iteration(frame, templ, 1, 4, 4)
    npt.assert_allclose(shifts, -true_shifts)
    npt.assert_allclose(np.corrcoef(frame_cor[~nans].ravel(), templ[~nans].ravel())[0, 1], 1)


def test_motion_correct_iteration():
    _test_iteration(False)


def test_motion_correct_iteration_fast():
    _test_iteration(True)


def gen_data(D=2, noise=.01, T=300, framerate=30, firerate=2., motion=True):
    N = 4                                                      # number of neurons
    dims = [(40, 50), (12, 14, 16)][D - 2]                     # size of image
    sig = (2, 2, 2)[:D]                                        # neurons size
    bkgrd = 1                                                  # fluorescence baseline
    gamma = .9                                                 # calcium decay time constant
    np.random.seed(5)
    centers = np.asarray([[np.random.randint(4, x - 4) for x in dims] for i in range(N)])

    S = np.random.rand(N, T) < firerate / float(framerate)
    S[:, 0] = 0
    C = S.astype(np.float32)
    for i in range(1, T):
        C[:, i] += gamma * C[:, i - 1]

    if motion:
        sig_m = np.array(sig)
        shifts = -np.transpose([np.convolve(np.random.randn(T-10),
                                            np.ones(11)/11*s) for s in sig_m])
    else:
        sig_m = np.zeros(D, dtype=int)
        shifts = None

    A = np.zeros(tuple(np.array(dims) + sig_m * 4) + (N,), dtype=np.float32)
    for i in range(N):
        A[tuple(centers[i] + sig_m*2) + (i,)] = 1.
    A = gaussian_filter(A, sig + (0,))
    A /= np.sqrt(np.sum(A.reshape((-1, N), order='F')**2, 0))

    Yr = A.reshape((-1, N), order='F').dot(C)
    Yr += noise * np.random.randn(*Yr.shape)
    Y = bkgrd + Yr.T.reshape((-1,) + tuple(np.array(dims) + sig_m * 4), order='F')
    if motion:
        Y = np.array([cm.motion_correction.apply_shifts_dft(
            img, sh, 0, is_freq=False, border_nan='copy')
            for img, sh in zip(Y, -shifts)], dtype=np.float32)
        for d in range(D):
            Y = np.take(Y, range(2*sig_m[d], Y.shape[d+1]-2*sig_m[d]), d+1)
            A = np.take(A, range(2*sig_m[d], A.shape[d]-2*sig_m[d]), d)
    return Y, C, S, A.reshape((-1, N), order='F'), centers, dims, shifts


def _test_motion_correct_rigid(D):
    Y, C, S, A, centers, dims, shifts = gen_data(D)
    fname = 'testMovie.tif'
    cm.movie(Y).save(fname)
    params_dict = {'max_shifts': (4, 4),   # maximum allowed rigid shifts (in pixels)
                   'pw_rigid': False,      # flag for performing non-rigid motion correction
                   'border_nan': True,
                   'is3D': D == 3}
    opts = cm.source_extraction.cnmf.params.CNMFParams(params_dict=params_dict)
    mc = MotionCorrect(fname, dview=None, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    npt.assert_(np.corrcoef(shifts.T, np.transpose(mc.shifts_rig))[:D, D:].diagonal().mean() > .8)
    Y_cor = cm.load(mc.mmap_file, is3D=(D == 3))
    nans = np.isnan(Y_cor)
    npt.assert_(np.corrcoef(Y_cor[~nans], Y[~nans])[0, 1] > .8)


def test_motion_correct_rigid():
    _test_motion_correct_rigid(2)


def test_motion_correct_rigid_3d():
    _test_motion_correct_rigid(3)
