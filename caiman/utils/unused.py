#!/usr/bin/env python

#######################
# unused functions
#
# These are not presently used in the Caiman codebase, but are
# being kept for being potentially useful (and either not trivial to
# reimplement, or themselves a nontrivial idea).
#
# Code in this file should:
# 1) Not be considered for purposes of "don't break APIs in use". The
#    person who would re-integrate the code has the responsibility for
#    adapting it to current conventions
# 2) Never be called without moving it back out. This file should never be
#    imported.
# 3) Be considered one step from deletion, if someone decides that the git
#    history is good enough.
# 4) Note which file it came from
#
# Please prefer to delete code unless there's some reason to keep it

# From caiman/cluster.py
def get_patches_from_image(img, shapes, overlaps):
    # todo todocument
    d1, d2 = np.shape(img)
    rf = np.divide(shapes, 2)
    _, coords_2d = extract_patch_coordinates(d1, d2, rf=rf, stride=overlaps)
    imgs = np.empty(coords_2d.shape[:2], dtype=img.dtype)

    for idx_0, count_0 in enumerate(coords_2d):
        for idx_1, count_1 in enumerate(count_0):
            imgs[idx_0, idx_1] = img[count_1[0], count_1[1]]

    return imgs, coords_2d

# From caiman/components_evaluation.py
def estimate_noise_mode(traces, robust_std=False, use_mode_fast=False, return_all=False):
    """ estimate the noise in the traces under assumption that signals are sparse and only positive. The last dimension should be time.

    """
    # todo todocument
    if use_mode_fast:
        md = mode_robust_fast(traces, axis=1)
    else:
        md = mode_robust(traces, axis=1)

    ff1 = traces - md[:, None]
    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, _ in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1**2, 1) / Ns)

    if return_all:
        return md, sd_r
    else:
        return sd_r
