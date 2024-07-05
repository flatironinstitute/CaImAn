#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import caiman.source_extraction.cnmf.params
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.visualization import get_contours


#%%
def gen_data(D=3, noise=.5, T=300, framerate=30, firerate=2.):
    N = 4                                                              # number of neurons
    dims = [(20, 30), (12, 14, 16)][D - 2]                             # size of image
    sig = (2, 2, 2)[:D]                                                # neurons size
    bkgrd = 10                                                         # fluorescence baseline
    gamma = .9                                                         # calcium decay time constant
    np.random.seed(5)
    centers = np.asarray([[np.random.randint(4, x - 4) for x in dims] for i in range(N)])
    trueA = np.zeros(dims + (N,), dtype=np.float32)
    trueS = np.random.rand(N, T) < firerate / float(framerate)
    trueS[:, 0] = 0
    trueC = trueS.astype(np.float32)
    for i in range(1, T):
        trueC[:, i] += gamma * trueC[:, i - 1]
    for i in range(N):
        trueA[tuple(centers[i]) + (i,)] = 1.
    tmp = np.zeros(dims)
    tmp[tuple(d // 2 for d in dims)] = 1.
    z = np.linalg.norm(gaussian_filter(tmp, sig).ravel())
    trueA = 10 * gaussian_filter(trueA, sig + (0,)) / z
    Yr = bkgrd + noise * np.random.randn(*(np.prod(dims), T)) + \
        trueA.reshape((-1, 4), order='F').dot(trueC)
    return Yr, trueC, trueS, trueA, centers, dims


def pipeline(D):
    #%% GENERATE GROUND TRUTH DATA
    Yr, trueC, trueS, trueA, centers, dims = gen_data(D)
    N, T = trueC.shape
    # INIT
    params = caiman.source_extraction.cnmf.params.CNMFParams(dims=dims,
                                                             k=4,
                                                             gSig=[2, 2, 2][:D],
                                                             p=1,
                                                             n_pixels_per_process=np.prod(dims),
                                                             block_size_temp=np.prod(dims))
    params.spatial['thr_method'] = 'nrg'
    params.spatial['extract_cc'] = False
    cnm = cnmf.CNMF(2, params=params)
    # FIT
    images = np.reshape(Yr.T, (T,) + dims, order='F')
    cnm = cnm.fit(images)

    # VERIFY HIGH CORRELATION WITH GROUND TRUTH
    sorting = [np.argmax([np.corrcoef(tc, c)[0, 1] for tc in trueC]) for c in cnm.estimates.C]
    # verifying the temporal components
    corr = [np.corrcoef(trueC[sorting[i]], cnm.estimates.C[i])[0, 1] for i in range(N)]
    npt.assert_allclose(corr, 1, .05)
    # verifying the spatial components
    corr = [
        np.corrcoef(np.reshape(trueA, (-1, 4), order='F')[:, sorting[i]],
                    cnm.estimates.A.toarray()[:, i])[0, 1] for i in range(N)
    ]
    npt.assert_allclose(corr, 1, .05)

    # Check that get_contours works regardless of swap_dim
    coor_normal = get_contours(cnm.estimates.A, dims, swap_dim=False)
    coor_swapped = get_contours(cnm.estimates.A, dims[::-1], swap_dim=True)
    for c_normal, c_swapped in zip(coor_normal, coor_swapped):
        if D == 3:
            for plane_coor_normal, plane_coor_swapped in zip(c_normal['coordinates'], c_swapped['coordinates']):
                compare_contour_coords(plane_coor_normal, plane_coor_swapped[:, ::-1])
        else:
            compare_contour_coords(c_normal['coordinates'], c_swapped['coordinates'][:, ::-1])

        npt.assert_allclose(c_normal['CoM'], c_swapped['CoM'][::-1])

def compare_contour_coords(coords1: np.ndarray, coords2: np.ndarray):
    """
    Compare 2 matrices of contour coordinates that should be the same, but may be calculated in a different order/
    from different starting points. 

    The first point of each contour component is repeated, and this may be a different point depending on orientation.
    To get around this, compare differences instead (have to take absolute value b/c direction may be opposite).
    Also sort coordinates b/c starting point is unimportant & depends on orientation
    """
    diffs_sorted = []
    for coords in [coords1, coords2]:
        abs_diffs = np.abs(np.diff(coords, axis=0))
        sort_order = np.lexsort(abs_diffs.T)
        diffs_sorted.append(abs_diffs[sort_order, :])
    npt.assert_allclose(diffs_sorted[0], diffs_sorted[1])


def test_2D():
    pipeline(2)


def test_3D():
    pipeline(3)
