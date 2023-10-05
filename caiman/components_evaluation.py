#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:12:34 2016

@author: agiovann
"""

import cv2
import itertools
import logging
import numpy as np
import os
import peakutils
import tensorflow as tf
import scipy
from scipy.sparse import csc_matrix
from scipy.stats import norm
from typing import Any, Union
import warnings

from caiman.paths import caiman_datadir
from .utils.stats import mode_robust, mode_robust_fast
from .utils.utils import load_graph

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    profile
except:

    def profile(a):
        return a


@profile
def compute_event_exceptionality(traces: np.ndarray,
                                 robust_std: bool = False,
                                 N: int = 5,
                                 use_mode_fast: bool = False,
                                 sigma_factor: float = 3.) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """
    Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Args:
        Y: ndarray
            movie x,y,t

        A: scipy sparse array
            spatial components

        traces: ndarray
            Fluorescence traces

        N: int
            N number of consecutive events (N must be greater than 0)

        sigma_factor: float
            multiplicative factor for noise estimate (added for backwards compatibility)

    Returns:
        fitness: ndarray
            value estimate of the quality of components (the lesser the better)

        erfc: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise

        noise_est: ndarray
            the components ordered according to the fitness
    """
    if N == 0:
        # Without this, numpy ranged syntax does not work correctly, and also N=0 is conceptually incoherent
        raise Exception("FATAL: N=0 is not a valid value for compute_event_exceptionality()")

    T = np.shape(traces)[-1]
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

    # compute z value
    z = (traces - md[:, None]) / (sigma_factor * sd_r[:, None])

    # probability of observing values larger or equal to z given normal
    # distribution with mean md and std sd_r
    #erf = 1 - norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    #erf = np.log(erf)
    # compute with this numerically stable function
    erf = scipy.special.log_ndtr(-z)

    # moving sum
    erfc = np.cumsum(erf, 1)
    erfc[:, N:] -= erfc[:, :-N]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    return fitness, erfc, sd_r, md

def compute_eccentricity(A, dims, order='F'):
    """computes eccentricity of components (currently only for 2D)"""
    ecc = []
    for i in range(A.shape[-1]):
        a = A[:,i].reshape(dims, order=order).toarray()
        M = cv2.moments(a)
        cov = np.array([[M['mu20'], M['mu11']], [M['mu11'], M['mu02']]])/(M['m00'] + np.finfo(np.float32).eps)
        eigs = np.sort(np.linalg.eigvals(cov))
        ecc.append(np.sqrt(eigs[1]/(eigs[0] + np.finfo(np.float32).eps)))
    return np.array(ecc)

#%%
def find_activity_intervals(C, Npeaks: int = 5, tB=-3, tA=10, thres: float = 0.3) -> list:
    # todo todocument
    K, T = np.shape(C)
    L:list = []
    for i in range(K):
        if np.sum(np.abs(np.diff(C[i, :]))) == 0:
            L.append([])
            logging.debug('empty component at:' + str(i))
            continue
        indexes = peakutils.indexes(C[i, :], thres=thres)
        srt_ind = indexes[np.argsort(C[i, indexes])][::-1]
        srt_ind = srt_ind[:Npeaks]
        L.append(srt_ind)

    LOC = []
    for i in range(K):
        if len(L[i]) > 0:
            interval = np.kron(L[i], np.ones(int(np.round(tA - tB)), dtype=int)) + \
                np.kron(np.ones(len(L[i]), dtype=int), np.arange(tB, tA))
            interval[interval < 0] = 0
            interval[interval > T - 1] = T - 1
            LOC.append(np.array(list(set(interval))))
        else:
            LOC.append(None)

    return LOC


#%%
def classify_components_ep(Y, A, C, b, f, Athresh=0.1, Npeaks=5, tB=-3, tA=10, thres=0.3) -> tuple[np.ndarray, list]:
    """Computes the space correlation values between the detected spatial
    footprints and the original data when background and neighboring component
    activity has been removed.
    Args:
        Y: ndarray
            movie x,y,t

        A: scipy sparse array
            spatial components

        C: ndarray
            Fluorescence traces

        b: ndarrray
            Spatial background components

        f: ndarrray
            Temporal background components

        Athresh: float
            Degree of spatial overlap for a neighboring component to be
            considered overlapping

        Npeaks: int
            Number of peaks to consider for computing spatial correlation

        tB: int
            Number of frames to include before peak

        tA: int
            Number of frames to include after peak

        thres: float
            threshold value for computing distinct peaks

    Returns:
        rval: ndarray
            Space correlation values

        significant_samples: list
            Frames that were used for computing correlation values
    """

    K, _ = np.shape(C)
    A = csc_matrix(A)
    AA = (A.T * A).toarray()
    nA = np.sqrt(np.array(A.power(2).sum(0)))
    AA /= np.outer(nA, nA.T)
    AA -= np.eye(K)

    LOC = find_activity_intervals(C, Npeaks=Npeaks, tB=tB, tA=tA, thres=thres)
    rval = np.zeros(K)

    significant_samples:list[Any] = []
    for i in range(K):
        if (i + 1) % 200 == 0:         # Show status periodically
            logging.info('Components evaluated:' + str(i))
        if LOC[i] is not None:
            atemp = A[:, i].toarray().flatten()
            atemp[np.isnan(atemp)] = np.nanmean(atemp)
            ovlp_cmp = np.where(AA[:, i] > Athresh)[0]
            indexes = set(LOC[i])
            for _, j in enumerate(ovlp_cmp):
                if LOC[j] is not None:
                    indexes = indexes - set(LOC[j])

            if len(indexes) == 0:
                indexes = set(LOC[i])
                logging.warning('Component {0} is only active '.format(i) +
                                'jointly with neighboring components. Space ' +
                                'correlation calculation might be unreliable.')

            indexes = np.array(list(indexes)).astype(int)
            px = np.where(atemp > 0)[0]
            if px.size < 3:
                logging.warning('Component {0} is almost empty. '.format(i) + 'Space correlation is set to 0.')
                rval[i] = 0
                significant_samples.append({0})
            else:
                ysqr = np.array(Y[px, :])
                ysqr[np.isnan(ysqr)] = np.nanmean(ysqr)
                mY = np.mean(ysqr[:, indexes], axis=-1)
                significant_samples.append(indexes)
                rval[i] = scipy.stats.pearsonr(mY, atemp[px])[0]

        else:
            rval[i] = 0
            significant_samples.append(0)

    return rval, significant_samples


#%%


def evaluate_components_CNN(A,
                            dims,
                            gSig,
                            model_name: str = os.path.join(caiman_datadir(), 'model', 'cnn_model'),
                            patch_size: int = 50,
                            loaded_model=None,
                            isGPU: bool = False) -> tuple[Any, np.array]:
    """ evaluate component quality using a CNN network

        if isGPU is false, and the environment variable 'CAIMAN_ALLOW_GPU' is not set,
        then this code will try not to use a GPU. Otherwise it will use one if it finds it.
    """

    # TODO: Find a less ugly way to do this
    if not isGPU and 'CAIMAN_ALLOW_GPU' not in os.environ:
        print("GPU run not requested, disabling use of GPUs")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        from tensorflow.keras.models import model_from_json
        use_keras = True
        logging.info('Using Keras')
    except (ModuleNotFoundError):
        use_keras = False
        logging.info('Using Tensorflow')

    if loaded_model is None:
        if use_keras:
            if os.path.isfile(os.path.join(caiman_datadir(), model_name + ".json")):
                model_file = os.path.join(caiman_datadir(), model_name + ".json")
                model_weights = os.path.join(caiman_datadir(), model_name + ".h5")
            elif os.path.isfile(model_name + ".json"):
                model_file = model_name + ".json"
                model_weights = model_name + ".h5"
            else:
                raise FileNotFoundError(f"File for requested model {model_name} not found")
            with open(model_file, 'r') as json_file:
                print(f"USING MODEL (keras API): {model_file}")
                loaded_model_json = json_file.read()

            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_name + '.h5')
        else:
            if os.path.isfile(os.path.join(caiman_datadir(), model_name + ".h5.pb")):
                model_file = os.path.join(caiman_datadir(), model_name + ".h5.pb")
            elif os.path.isfile(model_name + ".h5.pb"):
                model_file = model_name + ".h5.pb"
            else:
                raise FileNotFoundError(f"File for requested model {model_name} not found")
            print(f"USING MODEL (tensorflow API): {model_file}")
            loaded_model = load_graph(model_file)

        logging.debug("Loaded model from disk")

    half_crop = np.minimum(gSig[0] * 4 + 1, patch_size), np.minimum(gSig[1] * 4 + 1, patch_size)
    dims = np.array(dims)
    coms = [scipy.ndimage.center_of_mass(mm.toarray().reshape(dims, order='F')) for mm in A.tocsc().T]
    coms = np.maximum(coms, half_crop)
    coms = np.array([np.minimum(cms, dims - half_crop) for cms in coms]).astype(int)
    crop_imgs = [
        mm.toarray().reshape(dims, order='F')[com[0] - half_crop[0]:com[0] + half_crop[0], com[1] -
                                              half_crop[1]:com[1] + half_crop[1]] for mm, com in zip(A.tocsc().T, coms)
    ]
    final_crops = np.array([cv2.resize(im / np.linalg.norm(im), (patch_size, patch_size)) for im in crop_imgs])
    if use_keras:
        predictions = loaded_model.predict(final_crops[:, :, :, np.newaxis], batch_size=32, verbose=1)
    else:
        tf_in = loaded_model.get_tensor_by_name('prefix/conv2d_20_input:0')
        tf_out = loaded_model.get_tensor_by_name('prefix/output_node0:0')
        with tf.Session(graph=loaded_model) as sess:
            predictions = sess.run(tf_out, feed_dict={tf_in: final_crops[:, :, :, np.newaxis]})
            sess.close()

    return predictions, final_crops


#%%


def evaluate_components(Y: np.ndarray,
                        traces: np.ndarray,
                        A,
                        C,
                        b,
                        f,
                        final_frate,
                        remove_baseline: bool = True,
                        N: int = 5,
                        robust_std: bool = False,
                        Athresh: float = 0.1,
                        Npeaks: int = 5,
                        thresh_C: float = 0.3,
                        sigma_factor: float = 3.) -> tuple[Any, Any, Any, Any, Any, Any]:
    """ Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode.
    The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.
    The algorithm also measures the reliability of the spatial mask by comparing the filters in A
     with the average of the movies over samples where exceptional events happen, after  removing (if possible)
    frames when neighboring neurons were active

    Args:
        Y: ndarray
            movie x,y,t

        traces: ndarray
            Fluorescence traces

        A,C,b,f: various types
            outputs of cnmf

        final_frate: (undocumented)

        remove_baseline: bool
            whether to remove the baseline in a rolling fashion *(8 percentile)

        N: int
            N number of consecutive events probability multiplied


        Athresh: float
            threshold on overlap of A (between 0 and 1)

        Npeaks: int
            Number of local maxima to consider

        thresh_C: float
            fraction of the maximum of C that is used as minimum peak height

        sigma_factor: float
            multiplicative factor for noise

    Returns:
        idx_components: ndarray
            the components ordered according to the fitness

        fitness_raw: ndarray
            value estimate of the quality of components (the lesser the better) on the raw trace

        fitness_delta: ndarray
            value estimate of the quality of components (the lesser the better) on diff(trace)

        erfc_raw: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise on the raw trace

        erfc_raw: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise on diff(trace)

        r_values: list
            float values representing correlation between component and spatial mask obtained by averaging important points

        significant_samples: ndarray
            indexes of samples used to obtain the spatial mask by average
    """

    tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
    tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
    logging.info('tB:' + str(tB) + ',tA:' + str(tA))
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]

    Yr = np.reshape(Y, (np.prod(dims), T), order='F')

    logging.debug('Computing event exceptionality delta')
    fitness_delta, erfc_delta, _, _ = compute_event_exceptionality(np.diff(traces, axis=1),
                                                                   robust_std=robust_std,
                                                                   N=N,
                                                                   sigma_factor=sigma_factor)

    logging.debug('Removing Baseline')
    if remove_baseline:
        num_samps_bl = np.minimum(np.shape(traces)[-1]// 5, 800)
        slow_baseline = False
        if slow_baseline:

            traces = traces - \
                scipy.ndimage.percentile_filter(
                    traces, 8, size=[1, num_samps_bl])

        else:                                                                                # fast baseline removal
            downsampfact = num_samps_bl
            elm_missing = int(np.ceil(T * 1.0 / downsampfact) * downsampfact - T)
            padbefore = int(np.floor(elm_missing / 2.))
            padafter = int(np.ceil(elm_missing / 2.))
            tr_tmp = np.pad(traces.T, ((padbefore, padafter), (0, 0)), mode='reflect')
            numFramesNew, num_traces = np.shape(tr_tmp)
                                                                                             #% compute baseline quickly
            logging.debug("binning data ...")
            tr_BL = np.reshape(tr_tmp, (downsampfact, numFramesNew // downsampfact, num_traces), order='F')
            tr_BL = np.percentile(tr_BL, 8, axis=0)
            logging.debug("interpolating data ...")
            logging.debug(tr_BL.shape)
            tr_BL = scipy.ndimage.zoom(np.array(tr_BL, dtype=np.float32), [downsampfact, 1],
                                       order=3,
                                       mode='constant',
                                       cval=0.0,
                                       prefilter=True)
            if padafter == 0:
                traces -= tr_BL.T
            else:
                traces -= tr_BL[padbefore:-padafter].T

    logging.debug('Computing event exceptionality')
    fitness_raw, erfc_raw, _, _ = compute_event_exceptionality(traces,
                                                               robust_std=robust_std,
                                                               N=N,
                                                               sigma_factor=sigma_factor)

    logging.debug('Evaluating spatial footprint')
    # compute the overlap between spatial and movie average across samples with significant events
    r_values, significant_samples = classify_components_ep(Yr,
                                                           A,
                                                           C,
                                                           b,
                                                           f,
                                                           Athresh=Athresh,
                                                           Npeaks=Npeaks,
                                                           tB=tB,
                                                           tA=tA,
                                                           thres=thresh_C)

    return fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples


def grouper(n: int, iterable, fillvalue: bool = None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def evaluate_components_placeholder(params):
    import caiman as cm
    fname, traces, A, C, b, f, final_frate, remove_baseline, N, robust_std, Athresh, Npeaks, thresh_C = params
    Yr, dims, T = cm.load_memmap(fname)
    Y = np.reshape(Yr, dims + (T,), order='F')
    fitness_raw, fitness_delta, _, _, r_values, significant_samples = \
        evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=remove_baseline,
                            N=N, robust_std=robust_std, Athresh=Athresh, Npeaks=Npeaks, thresh_C=thresh_C)

    return fitness_raw, fitness_delta, [], [], r_values, significant_samples


def estimate_components_quality_auto(Y,
                                     A,
                                     C,
                                     b,
                                     f,
                                     YrA,
                                     frate,
                                     decay_time,
                                     gSig,
                                     dims,
                                     dview=None,
                                     min_SNR=2,
                                     r_values_min=0.9,
                                     r_values_lowest=-1,
                                     Npeaks=10,
                                     use_cnn=True,
                                     thresh_cnn_min=0.95,
                                     thresh_cnn_lowest=0.1,
                                     thresh_fitness_delta=-20.,
                                     min_SNR_reject=0.5,
                                     gSig_range=None) -> tuple[np.array, np.array, float, float, float]:
    ''' estimates the quality of component automatically

    Args:
        Y, A, C, b, f, YrA:
            from CNMF

        frate:
            frame rate in Hz

        decay_time:
            decay time of transients/indocator

        gSig:
            same as CNMF parameter

        gSig_range: list
            list of possible neuronal sizes

        dims:
            same as CNMF parameter

        dview:
            same as CNMF parameter

        min_SNR:
            adaptive way to set threshold (will be equal to min_SNR)

        r_values_min:
            all r values above this are accepted (spatial consistency metric)

        r_values_lowest:
            all r values above this are rejected (spatial consistency metric)

        use_cnn:
            whether to use CNN to filter components (not for 1 photon data)

        thresh_cnn_min:
            all samples with probabilities larger than this are accepted

        thresh_cnn_lowest:
            all samples with probabilities smaller than this are rejected

        min_SNR_reject:
            adaptive way to set threshold (like min_SNR but used to discard components with std lower than this value)

    Returns:
        idx_components: list
            list of components that pass the tests

        idx_components_bad: list
            list of components that fail the tests

        comp_SNR: float
            peak-SNR over the length of a transient for each component

        r_values: float
            space correlation values

        cnn_values: float
            prediction values from the CNN classifier
    '''

    # number of timesteps to consider when testing new neuron candidates
    N_samples = np.ceil(frate * decay_time).astype(int)
    # inclusion probability of noise transient
    thresh_fitness_raw = scipy.special.log_ndtr(-min_SNR) * N_samples

    # threshold on time variability
    fitness_min = scipy.special.log_ndtr(-min_SNR) * N_samples
    # components with SNR lower than 0.5 will be rejected
    thresh_fitness_raw_reject = scipy.special.log_ndtr(-min_SNR_reject) * N_samples

    traces = C + YrA
    _, _, fitness_raw, _, r_values = estimate_components_quality(      # type: ignore # mypy cannot reason about return_all
        traces,
        Y,
        A,
        C,
        b,
        f,
        final_frate=frate,
        Npeaks=Npeaks,
        r_values_min=r_values_min,
        fitness_min=fitness_min,
        fitness_delta_min=thresh_fitness_delta,
        return_all=True,
        dview=dview,
        num_traces_per_group=50,
        N=N_samples)

    comp_SNR = -norm.ppf(np.exp(fitness_raw / N_samples))

    idx_components, idx_components_bad, cnn_values = select_components_from_metrics(A, dims, gSig, r_values, comp_SNR,
                                                                                    r_values_min, r_values_lowest,
                                                                                    min_SNR, min_SNR_reject,
                                                                                    thresh_cnn_min, thresh_cnn_lowest,
                                                                                    use_cnn, gSig_range)

    return idx_components, idx_components_bad, comp_SNR, r_values, cnn_values


def select_components_from_metrics(A,
                                   dims,
                                   gSig,
                                   r_values,
                                   comp_SNR,
                                   r_values_min=0.8,
                                   r_values_lowest=-1,
                                   min_SNR=2.5,
                                   min_SNR_reject=0.5,
                                   thresh_cnn_min=0.8,
                                   thresh_cnn_lowest=0.1,
                                   use_cnn=True,
                                   gSig_range=None,
                                   neuron_class=1,
                                   predictions=None,
                                   **kwargs) -> tuple[np.array, np.array, Any]:
    '''Selects components based on pre-computed metrics. For each metric
    space correlation, trace SNR, and CNN classifier both an upper and a lower
    thresholds are considered. A component is accepted if and only if it
    exceeds the upper threshold for at least one of the metrics and the lower
    threshold for all metrics. If the CNN classifier values are not provided,
    or a different scale parameter is used, the values are computed from within
    the script.
    '''

    # TODO Refactoring note: kwargs is unused and should be refactored carefully out

    idx_components_r = np.where(r_values >= r_values_min)[0]
    idx_components_raw = np.where(comp_SNR > min_SNR)[0]

    idx_components: Any = []   # changes type over the function

    if use_cnn:
        # normally 1
        if gSig_range is None:
            if predictions is None:
                predictions, _ = evaluate_components_CNN(A, dims, gSig)
                predictions = predictions[:, neuron_class]
        else:
            predictions = np.zeros(len(r_values))
            for size_range in gSig_range:
                predictions = np.maximum(predictions, evaluate_components_CNN(A, dims, size_range)[0][:, neuron_class])

        idx_components_cnn = np.where(predictions >= thresh_cnn_min)[0]
        bad_comps = np.where((r_values <= r_values_lowest) | (comp_SNR <= min_SNR_reject) |
                             (predictions <= thresh_cnn_lowest))[0]
        idx_components = np.union1d(idx_components, idx_components_cnn)
        cnn_values = predictions
    else:
        bad_comps = np.where((r_values <= r_values_lowest) | (comp_SNR <= min_SNR_reject))[0]
        cnn_values = []

    idx_components = np.union1d(idx_components, idx_components_r)
    idx_components = np.union1d(idx_components, idx_components_raw)
    idx_components = np.setdiff1d(idx_components, bad_comps)
    idx_components_bad = np.setdiff1d(list(range(len(r_values))), idx_components)

    return idx_components.astype(int), idx_components_bad.astype(int), cnn_values


def estimate_components_quality(traces,
                                Y,
                                A,
                                C,
                                b,
                                f,
                                final_frate=30,
                                Npeaks=10,
                                r_values_min=.95,
                                fitness_min=-100,
                                fitness_delta_min=-100,
                                return_all: bool = False,
                                N=5,
                                remove_baseline=True,
                                dview=None,
                                robust_std=False,
                                Athresh=0.1,
                                thresh_C=0.3,
                                num_traces_per_group=20) -> tuple[np.ndarray, ...]:
    """ Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode.
    The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.
    The algorithm also measures the reliability of the spatial mask by comparing the filters in A
     with the average of the movies over samples where exceptional events happen, after  removing (if possible)
    frames when neighboring neurons were active

    Args:
        Y: ndarray
            movie x,y,t

        A,C,b,f: various types
            outputs of cnmf

        traces: ndarray
            Fluorescence traces

        N: int
            N number of consecutive events probability multiplied

        Npeaks: int

        r_values_min: list
            minimum correlation between component and spatial mask obtained by averaging important points

        fitness_min: ndarray
            minimum acceptable quality of components (the lesser the better) on the raw trace

        fitness_delta_min: ndarray
            minimum acceptable the quality of components (the lesser the better) on diff(trace)

        thresh_C: float
            fraction of the maximum of C that is used as minimum peak height

    Returns:
        idx_components: ndarray
            the components ordered according to the fitness

        idx_components_bad: ndarray
            the components ordered according to the fitness

        fitness_raw: ndarray
            value estimate of the quality of components (the lesser the better) on the raw trace

        fitness_delta: ndarray
            value estimate of the quality of components (the lesser the better) on diff(trace)

        r_values: list
            float values representing correlation between component and spatial mask obtained by averaging important points

    """
    # TODO: Consider always returning it all and let the caller ignore what it does not want

    if 'memmap' not in str(type(Y)):
        logging.warning('NOT MEMORY MAPPED. FALLING BACK ON SINGLE CORE IMPLEMENTATION')
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, _ = \
            evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=remove_baseline,
                                N=N, robust_std=robust_std, Athresh=Athresh,
                                Npeaks=Npeaks, thresh_C=thresh_C)

    else:      # memory mapped case
        fitness_raw = []
        fitness_delta = []
        erfc_raw = []
        erfc_delta = []
        r_values = []
        Ncomp = A.shape[-1]

        if Ncomp > 0:
            groups = grouper(num_traces_per_group, range(Ncomp))
            params = []
            for g in groups:
                idx = list(g)
                # idx = list(filter(None.__ne__, idx))
                idx = list(filter(lambda a: a is not None, idx))
                params.append([
                    Y.filename, traces[idx],
                    A.tocsc()[:, idx], C[idx], b, f, final_frate, remove_baseline, N, robust_std, Athresh, Npeaks,
                    thresh_C
                ])

            if dview is None:
                res = map(evaluate_components_placeholder, params)
            else:
                logging.info('Component evaluation in parallel')
                if 'multiprocessing' in str(type(dview)):
                    res = dview.map_async(evaluate_components_placeholder, params).get(4294967)
                else:
                    res = dview.map_sync(evaluate_components_placeholder, params)

            for r_ in res:
                fitness_raw__, fitness_delta__, erfc_raw__, erfc_delta__, r_values__, _ = r_
                fitness_raw = np.concatenate([fitness_raw, fitness_raw__])
                fitness_delta = np.concatenate([fitness_delta, fitness_delta__])
                r_values = np.concatenate([r_values, r_values__])

                if len(erfc_raw) == 0:
                    erfc_raw = erfc_raw__
                    erfc_delta = erfc_delta__
                else:
                    erfc_raw = np.concatenate([erfc_raw, erfc_raw__], axis=0)
                    erfc_delta = np.concatenate([erfc_delta, erfc_delta__], axis=0)
        else:
            warnings.warn("There were no components to evaluate. Check your parameter settings.")

    idx_components_r = np.where(np.array(r_values) >= r_values_min)[0]         # threshold on space consistency
    idx_components_raw = np.where(np.array(fitness_raw) < fitness_min)[0]      # threshold on time variability
                                                                               # threshold on time variability (if nonsparse activity)
    idx_components_delta = np.where(np.array(fitness_delta) < fitness_delta_min)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    if return_all:
        return idx_components, idx_components_bad, np.array(fitness_raw), np.array(fitness_delta), np.array(r_values)
    else:
        return idx_components, idx_components_bad
