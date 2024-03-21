#!/usr/bin/env python

"""
Functions related to optical flow
"""

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.io import loadmat
from sklearn.decomposition import NMF
import time

import caiman

try:
    cv2.setNumThreads(0)
except:
    pass


def select_roi(img: np.ndarray, n_rois: int = 1) -> list:
    """
    Create a mask from a convex polygon enclosed between selected points

    Args:
        img: 2D ndarray
            image used to select the points for the mask
        n_rois: int
            number of rois to select

    Returns:
        mask: list
            each element is an the mask considered a ROIs
    """

    masks = []
    for _ in range(n_rois):
        fig = plt.figure()
        plt.imshow(img, cmap=matplotlib.cm.gray)
        pts = fig.ginput(0, timeout=0)
        mask = np.zeros(np.shape(img), dtype=np.int32)
        pts = np.asarray(pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, (1, 1, 1), lineType=cv2.LINE_AA)
        masks.append(mask)
        plt.close()

    return masks


def to_polar(x, y):
    mag, ang = cv2.cartToPolar(x, y)
    return mag, ang


def get_nonzero_subarray(arr, mask):
    x, y = mask.nonzero()
    return arr.toarray()[x.min():x.max() + 1, y.min():y.max() + 1]


def extract_motor_components_OF(m,
                                n_components,
                                mask=None,
                                resize_fact: float = .5,
                                only_magnitude: bool = False,
                                max_iter: int = 1000,
                                verbose: bool = False,
                                method_factorization: str = 'nmf',
                                max_iter_DL=-30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # todo todocument
    if mask is not None:
        mask = coo_matrix(np.array(mask).squeeze())
        ms = [get_nonzero_subarray(mask.multiply(fr), mask) for fr in m]
        ms = np.dstack(ms)
        ms = caiman.movie(ms.transpose([2, 0, 1]))

    else:
        ms = m
    of_or = compute_optical_flow(ms, do_show=False, polar_coord=False)
    of_or = np.concatenate([
        caiman.movie(of_or[0]).resize(resize_fact, resize_fact, 1)[np.newaxis, :, :, :],
        caiman.movie(of_or[1]).resize(resize_fact, resize_fact, 1)[np.newaxis, :, :, :]
    ],
                           axis=0)

    if only_magnitude:
        of = np.sqrt(of[0]**2 + of[1]**2)
    else:
        if method_factorization == 'nmf':
            offset_removed = np.min(of_or)
            of = of_or - offset_removed
        else:
            of = of_or

    spatial_filter_, time_trace_, _ = extract_components(of,
                                                         n_components=n_components,
                                                         verbose=verbose,
                                                         normalize_std=False,
                                                         max_iter=max_iter,
                                                         method_factorization=method_factorization,
                                                         max_iter_DL=max_iter_DL)

    return spatial_filter_, time_trace_, of_or


def extract_magnitude_and_angle_from_OF(spatial_filter_,
                                        time_trace_,
                                        of_or,
                                        num_std_mag_for_angle=.6,
                                        sav_filter_size=3,
                                        only_magnitude=False) -> tuple[list, list, list, list]:
    # todo todocument

    mags = []
    dircts = []
    dircts_thresh = []
    n_components = len(spatial_filter_)
    spatial_masks_thr = []
    for ncmp in range(n_components):
        spatial_filter = spatial_filter_[ncmp]
        time_trace = time_trace_[ncmp]
        if only_magnitude:
            mag = scipy.signal.medfilt(time_trace, kernel_size=[1, 1]).T
            mag = scipy.signal.savgol_filter(mag.squeeze(), sav_filter_size, 1)
            dirct = None
        else:
            x, y = scipy.signal.medfilt(time_trace, kernel_size=[1, 1]).T
            x = scipy.signal.savgol_filter(x.squeeze(), sav_filter_size, 1)
            y = scipy.signal.savgol_filter(y.squeeze(), sav_filter_size, 1)
            mag, dirct = to_polar(x - caiman.components_evaluation.mode_robust(x),
                                  y - caiman.components_evaluation.mode_robust(y))
            dirct = scipy.signal.medfilt(dirct.squeeze(), kernel_size=1).T

        # normalize to pixel units
        spatial_mask = spatial_filter
        spatial_mask[spatial_mask < (np.nanpercentile(spatial_mask[0], 99.9) * .9)] = np.nan
        ofmk = of_or * spatial_mask[None, None, :, :]
        range_ = np.std(np.nanpercentile(np.sqrt(ofmk[0]**2 + ofmk[1]**2), 95, (1, 2)))
        mag = (mag / np.std(mag)) * range_
        mag = scipy.signal.medfilt(mag.squeeze(), kernel_size=1)
        dirct_orig = dirct.copy()
        if not only_magnitude:
            dirct[mag < num_std_mag_for_angle * np.nanstd(mag)] = np.nan

        mags.append(mag)
        dircts.append(dirct_orig)
        dircts_thresh.append(dirct)
        spatial_masks_thr.append(spatial_mask)

    return mags, dircts, dircts_thresh, spatial_masks_thr


def compute_optical_flow(m,
                         mask=None,
                         polar_coord: bool = True,
                         do_show: bool = False,
                         do_write: bool = False,
                         file_name=None,
                         gain_of=None,
                         frate: float = 30.0,
                         pyr_scale: float = .1,
                         levels: int = 3,
                         winsize: int = 25,
                         iterations: int = 3,
                         poly_n=7,
                         poly_sigma=1.5):
    """
    This function compute the optical flow of behavioral movies using the opencv cv2.calcOpticalFlowFarneback function

    Args:
        m: 3D ndarray:
            input movie

        mask: 2D ndarray
            mask selecting relevant pixels

        polar_coord: boolean
            whether to return the coordinate in polar coordinates (or cartesian)

        do_show: bool
            show flow movie

        do_write: bool
            save flow movie

        frate: double
            frame rate saved movie

        parameters_opencv_function: cv2.calcOpticalFlowFarneback
            pyr_scale,levels,winsize,iterations,poly_n,poly_sigma

    Returns:
        mov_tot: 4D ndarray containing the movies of the two coordinates

    Raises:
        Exception 'You need to provide file name (.avi) when saving video'

    """
    prvs = np.uint8(m[0])
    frame1 = cv2.cvtColor(prvs, cv2.COLOR_GRAY2RGB)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    if do_show:
        cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)

    if mask is not None:
        data = mask.astype(np.int32)
    else:
        data = 1

    T, d1, d2 = m.shape
    mov_tot = np.zeros([2, T, d1, d2])

    if do_write:
        if file_name is not None:
            video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frate, (d2 * 2, d1), 1)
        else:
            raise Exception('You need to provide file name (.avi) when saving video')

    for counter, next_ in enumerate(m):
        if counter % 100 == 0:
            print(counter)
        frame2 = cv2.cvtColor(np.uint8(next_), cv2.COLOR_GRAY2RGB)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, pyr_scale, levels, winsize, iterations, poly_n,
                                            poly_sigma, 0)

        if polar_coord:
            coord_1, coord_2 = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        else:
            coord_1, coord_2 = flow[:, :, 0], flow[:, :, 1]

        coord_1 *= data
        coord_2 *= data

        if do_show or do_write:
            if polar_coord:
                hsv[..., 0] = coord_2 * 180 / np.pi / 2
            else:
                hsv[..., 0] = cv2.normalize(coord_2, None, 0, 255, cv2.NORM_MINMAX)
            if gain_of is None:
                hsv[..., 2] = cv2.normalize(coord_1, None, 0, 255, cv2.NORM_MINMAX)
            else:
                hsv[..., 2] = gain_of * coord_1
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            frame_tot = np.concatenate([rgb, frame2], axis=1)

        if do_write:
            video.write(frame_tot)

        if do_show:
            cv2.imshow('frame2', frame_tot)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        mov_tot[0, counter] = coord_1
        mov_tot[1, counter] = coord_2

        prvs = next_.copy()

    if do_write:
        video.release()

    if do_show:
        cv2.destroyAllWindows()

    return mov_tot

# NMF

def extract_components(mov_tot,
                       n_components: int = 6,
                       normalize_std: bool = True,
                       max_iter_DL=-30,
                       method_factorization: str = 'nmf',
                       **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From optical flow images can extract spatial and temporal components

    Args:
        mov_tot: ndarray (can be 3 or 4D)
            contains the optical flow values, either in cartesian or polar, either one (3D) or both (4D coordinates)
            the input is generated by the compute_optical_flow function

        n_components: int
            number of components to look for

        normalize_std: bool
            whether to normalize each oof the optical flow components

        normalize_output_traces: boolean
            whether to normalize the behavioral traces so that they match the units in the movie

    Returns:
        spatial_filter: ndarray
            set of spatial inferred filters

        time_trace: ndarray
            set of time components

        norm_fact: ndarray
            used notmalization factors
    """

    if mov_tot.ndim == 4:
        if normalize_std:
            norm_fact = np.nanstd(mov_tot, axis=(1, 2, 3))
            mov_tot = mov_tot / norm_fact[:, np.newaxis, np.newaxis, np.newaxis]
        else:
            norm_fact = np.array([1., 1.])
        c, T, d1, d2 = np.shape(mov_tot)

    else:
        norm_fact = 1
        T, d1, d2 = np.shape(mov_tot)
        c = 1

    tt = time.time()
    newm = np.reshape(mov_tot, (c * T, d1 * d2))

    if method_factorization == 'nmf':
        nmf = NMF(n_components=n_components, **kwargs)

        time_trace = nmf.fit_transform(newm)
        spatial_filter = nmf.components_
        spatial_filter = np.concatenate([np.reshape(sp, (d1, d2))[np.newaxis, :, :] for sp in spatial_filter], axis=0)

    elif method_factorization == 'dict_learn':
        import spams
        newm = np.asfortranarray(newm, dtype=np.float32)
        time_trace = spams.trainDL(newm, K=n_components, mode=0, lambda1=1, posAlpha=True, iter=max_iter_DL)

        spatial_filter = spams.lasso(newm,
                                     D=time_trace,
                                     return_reg_path=False,
                                     lambda1=0.01,
                                     mode=spams.spams_wrap.PENALTY,
                                     pos=True)

        spatial_filter = np.concatenate([np.reshape(sp, (d1, d2))[np.newaxis, :, :] for sp in spatial_filter.toarray()],
                                        axis=0)

    time_trace = [np.reshape(ttr, (c, T)).T for ttr in time_trace.T]

    el_t = time.time() - tt
    print(el_t)
    return spatial_filter, time_trace, norm_fact


def plot_components(sp_filt, t_trace) -> None:
    # todo: todocument
    plt.figure()
    count = 0
    for comp, tr in zip(sp_filt, t_trace):
        count += 1
        plt.subplot(6, 2, count)
        plt.imshow(comp)
        count += 1
        plt.subplot(6, 2, count)
        plt.plot(tr)
