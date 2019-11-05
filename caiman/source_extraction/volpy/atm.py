# -*- coding: utf-8 -*-
"""
@author: Johannes Friedrich and Takashi Kawashima
"""
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import sys
from scipy.sparse.linalg import lsqr
from scipy.stats import ttest_1samp
import caiman as cm


def volspike(pars):

    fnames = pars[0]
    sampleRate = pars[1]
    cellN = pars[2]
    ROI1_image = pars[3]
    weights_init = pars[4]
    args = pars[5]

    print('Now processing cell number {0}'.format(cellN))

    output = {}

    use_NMF = True

    Yr, dims, T = cm.load_memmap(fnames)
    if ROI1_image.shape == dims:
        img = np.reshape(Yr.T, [T] + list(dims), order='F')
    elif ROI1_image.shape == dims[::-1]:
        img = np.reshape(Yr.T, [T] + list(dims), order='F').transpose([0, 2, 1])
    else:
        print('size of ROI and video does not agree')

    mean_img = img.mean(0)
    output['meanIM'] = mean_img

    cell_inds = np.where(pars[3])

    ROI2_image = binary_fill_holes(binary_dilation(ROI1_image, iterations=5))
    ROI_candidates = np.where(ROI2_image > 0)

    minx = max(ROI_candidates[1].min() - 20, 0)
    maxx = min(ROI_candidates[1].max() + 20, mean_img.shape[1])
    miny = max(ROI_candidates[0].min() - 20, 0)
    maxy = min(ROI_candidates[0].max() + 20, mean_img.shape[0])

    img_snippet = mean_img[miny:maxy, minx:maxx].copy()
    back = (np.sort(img_snippet.flatten())[0:int(img_snippet.size / 20)]).mean()

    target_widths = np.zeros((10,))
    for i in range(10):
        target_widths[i] = 0.5 * (i + 1)

    first_timecourse = img[:, cell_inds[0], cell_inds[1]].mean(axis=1) - back
    norm_tcourse1 = first_timecourse / \
        np.array(pd.Series(first_timecourse).rolling(
            window=150, min_periods=75, center=True).quantile(0.8))

    norm_tcourse1 = 2 - norm_tcourse1  # flip

    (sub_thresh1, high_freq1, spiketimes1, spiketrain1, spikesizes1, super_times1,
        super_sizes1, kernel1, upsampled_kernel1, tlimit1, threshold1) = denoise_spikes(
        norm_tcourse1, superfactor=10, threshs=(.35, .5, .6))

    if isinstance(spiketimes1, int):
        print("%d spikes found" % spiketimes1)
    else:
        print("%d spikes found" % len(spiketimes1))

    ###############################################################################
    # checking whether found spikes are genuine

    spike_tcourse1 = np.zeros((len(norm_tcourse1),))

    not_optimize = 0
    if isinstance(spiketimes1, int):
        not_optimize = 1
        spike_tcourse1[spiketimes1] = 1
    elif spiketimes1.any():
        if (len(spiketimes1) < 20):
            not_optimize = 1
        spike_tcourse1[spiketimes1] = 1

    npix = len(ROI_candidates[0])
    weight_init = np.zeros((npix,))
    weight_init[ROI1_image[ROI_candidates] == 1] = 0.5

    ###############################################################################
    # optimize pixel weights

    if (isinstance(spiketimes1, int) or (not_optimize == 1)):

        print('not active cell')

        ans = np.array([(first_timecourse, np.array(first_timecourse.shape),
                         norm_tcourse1, np.zeros(norm_tcourse1.shape),
                         spike_tcourse1, np.zeros(spike_tcourse1.shape),
                         tlimit1, 0, threshold1, 0, spiketimes1, super_times1,
                         kernel1, ROI_candidates[0], ROI_candidates[1],
                         weight_init, np.zeros(weight_init.shape), np.zeros((21, 3)), 0., 0., 0)],
                       dtype=[('raw_tcourse1', np.ndarray),
                              ('raw_tcourse2', np.ndarray),
                              ('norm_tcourse1', np.ndarray),
                              ('norm_tcourse2', np.ndarray),
                              ('spike_tcourse1', np.ndarray),
                              ('spike_tcourse2', np.ndarray),
                              ('tlimit1', np.int),
                              ('tlimit2', np.int),
                              ('threshold1', np.int),
                              ('threshold2', np.int),
                              ('spiketime', np.ndarray),
                              ('super_spiketime', np.ndarray),
                              ('spike_kernel', np.ndarray),
                              ('ROI_Y', np.ndarray),
                              ('ROI_X', np.ndarray),
                              ('Weight_init', np.ndarray),
                              ('Weight_final', np.ndarray),
                              ('Learning_curve', np.ndarray),
                              ('SN_initial', np.float),
                              ('SN_final', np.float),
                              ('active', np.int)])

    else:
        print('optimizing ROI')

        tcourse_raw = img[:, ROI_candidates[0], ROI_candidates[1]].astype('float64')

        kernel = np.ones((51,)) / 51
        divider = np.convolve(np.ones((tcourse_raw.shape[0],)), kernel, mode='same')
        tcourse_detrend = np.zeros(tcourse_raw.shape)
        for i in range(npix):
            tcourse_detrend[:, i] = np.convolve(tcourse_raw[:, i], kernel, mode='same') / divider

        tcourse_zeroed = tcourse_raw - tcourse_detrend

        SN = np.zeros((2,))

        noise_inds = np.where(binary_dilation(spike_tcourse1, iterations=10) == 0)[0]

        learn_speed = 0.2

        W = weight_init.copy()

        peak_M = tcourse_zeroed[spike_tcourse1 > 0, :]
        noise_M = tcourse_zeroed[noise_inds, :]

        if use_NMF:
            W = -peak_M.mean(0)
            SN[0] = abs(np.dot(peak_M, W).mean() / np.dot(noise_M, W).std())
            for _ in range(5):
                # update trace
                noisy_trace = -tcourse_zeroed.dot(W)
                spikeshape = get_kernel(noisy_trace, spiketimes1)
                spikesizes = get_spikesizes(noisy_trace, spiketimes1, spikeshape)
                spiketrain = get_spiketrain(spiketimes1, spikesizes, len(noisy_trace))
                denoised_trace = np.convolve(spiketrain, spikeshape, 'same')
                # update weights
                W = np.maximum(-denoised_trace.dot(tcourse_zeroed), 0)  # maybe also sparsify?
                W /= np.sqrt(W.dot(W))
            Losses = np.array(np.nan)
        else:
            peak_dot = np.dot(peak_M, W).mean()
            noise_dot = np.dot(noise_M, W)
            noise_dot1 = noise_dot.mean()
            noise_dot2 = (np.dot(noise_M, W)**2).mean()
            peak_mean = peak_M.mean(axis=0)
            noise_mean = noise_M.mean(axis=0)

            L1 = (peak_dot - noise_dot1)**2
            L2 = (noise_dot1)**2
            L3 = noise_dot2

            V = L3 - L2

            N = (W**2).sum()
            L2_alpha = 0.2  # 0.2

            Losses = np.zeros((21, 3))
            Losses[0, 0] = (L1 / V)
            Losses[0, 1] = L2_alpha * N
            Losses[0, 2] = ((L1 / V) - L2_alpha * N)
            SN[0] = abs(np.dot(peak_M, W).mean() / np.dot(noise_M, W).std())

            for i in range(20):

                dW1 = 2 * (peak_dot - noise_dot1) * (peak_mean - noise_mean)
                dW2 = 2 * noise_dot1 * noise_mean
                dW3 = 2 * np.mean(noise_dot[:, None] * noise_M, axis=0)

                new_W = W + (((dW1 / V) - (L1 * (dW3 - dW2)) / (V**2)) -
                             2 * L2_alpha * W) * learn_speed
                new_W[new_W < 0] = 0

                peak_dot = np.dot(peak_M, new_W).mean()
                noise_dot = np.dot(noise_M, new_W)
                noise_dot1 = noise_dot.mean(axis=0)
                noise_dot2 = (noise_dot**2).mean()

                L1 = (peak_dot - noise_dot1)**2
                L2 = (noise_dot1)**2
                L3 = noise_dot2
                V = L3 - L2
                N = (W**2).sum()
                new_L = (L1 / V) - L2_alpha * N
                print("%d, %d" % ((L1 / V), L2_alpha * N))

                if new_L < Losses[i, 2]:
                    break
                else:
                    W = new_W
                    Losses[i + 1, 0] = (L1 / V)
                    Losses[i + 1, 1] = L2_alpha * N
                    Losses[i + 1, 2] = new_L

        SN[1] = abs(np.dot(peak_M, W).mean() / np.dot(noise_M, W).std())

        second_timecourse = np.dot(tcourse_raw, W) / (W.sum()) - back
        norm_tcourse2 = second_timecourse / \
            np.array(pd.Series(second_timecourse).rolling(
                window=150, min_periods=75, center=True).quantile(0.8))

        norm_tcourse2 = 2 - norm_tcourse2

        (sub_thresh2, high_freq2, spiketimes2, spiketrain2, spikesizes2, super_times2,
            super_sizes2, kernel2, upsampled_kernel2, tlimit2, threshold2) = denoise_spikes(
            norm_tcourse2, superfactor=10, threshs=(.35, .5, .6))

        spike_tcourse2 = np.zeros((len(norm_tcourse2),))
        if isinstance(spiketimes2, int):
            spike_tcourse2[spiketimes2] = 1
        elif spiketimes2.any():
            spike_tcourse2[spiketimes2] = 1

        not_active = 0
        if isinstance(spiketimes2, int):
            not_active = 1
            print("%d spikes found" % 1)
        else:
            if (len(spiketimes2) < 20):
                not_active = 1
            print("%d spikes found" % len(spiketimes2))

        if (not_active == 1):
            ans = np.array([(first_timecourse, np.array(first_timecourse.shape),
                             norm_tcourse1, np.zeros(norm_tcourse1.shape),
                             spike_tcourse1, np.zeros(spike_tcourse1.shape),
                             tlimit1, 0, threshold1, 0, spiketimes1, super_times1,
                             kernel1, ROI_candidates[0], ROI_candidates[1],
                             weight_init, np.zeros(weight_init.shape), np.zeros((21, 3)), 0., 0., 0)],
                           dtype=[('raw_tcourse1', np.ndarray),
                                  ('raw_tcourse2', np.ndarray),
                                  ('norm_tcourse1', np.ndarray),
                                  ('norm_tcourse2', np.ndarray),
                                  ('spike_tcourse1', np.ndarray),
                                  ('spike_tcourse2', np.ndarray),
                                  ('tlimit1', np.int),
                                  ('tlimit2', np.int),
                                  ('threshold1', np.int),
                                  ('threshold2', np.int),
                                  ('spiketime', np.ndarray),
                                  ('super_spiketime', np.ndarray),
                                  ('spike_kernel', np.ndarray),
                                  ('ROI_Y', np.ndarray),
                                  ('ROI_X', np.ndarray),
                                  ('Weight_init', np.ndarray),
                                  ('Weight_final', np.ndarray),
                                  ('Learning_curve', np.ndarray),
                                  ('SN_initial', np.float),
                                  ('SN_final', np.float),
                                  ('active', np.int)])

        else:
            ans = np.array([(first_timecourse, second_timecourse, norm_tcourse1,
                             norm_tcourse2, spike_tcourse1, spike_tcourse2,
                             tlimit1, tlimit2, threshold1, threshold2, spiketimes2, super_times2,
                             kernel2, ROI_candidates[0], ROI_candidates[1],
                             weight_init, W, Losses, SN[0], SN[1], 1)],
                           dtype=[('raw_tcourse1', np.ndarray),
                                  ('raw_tcourse2', np.ndarray),
                                  ('norm_tcourse1', np.ndarray),
                                  ('norm_tcourse2', np.ndarray),
                                  ('spike_tcourse1', np.ndarray),
                                  ('spike_tcourse2', np.ndarray),
                                  ('tlimit1', np.int),
                                  ('tlimit2', np.int),
                                  ('threshold1', np.int),
                                  ('threshold2', np.int),
                                  ('spiketime', np.ndarray),
                                  ('super_spiketime', np.ndarray),
                                  ('spike_kernel', np.ndarray),
                                  ('ROI_Y', np.ndarray),
                                  ('ROI_X', np.ndarray),
                                  ('Weight_init', np.ndarray),
                                  ('Weight_final', np.ndarray),
                                  ('Learning_curve', np.ndarray),
                                  ('SN_initial', np.float),
                                  ('SN_final', np.float),
                                  ('active', np.int)])

    # output
    output['spikeTimes'] = ans['super_spiketime']
    output['yFilt'] = ans['norm_tcourse2']
    output['spatialFilter'] = None
    output['cellN'] = cellN
    output['templates'] = ans['spike_kernel']
    output['snr'] = ans['SN_final']
    output['num_spikes'] = ans['super_spiketime'].shape[0]
    output['passedLocalityTest'] = None
    output['low_spk'] = None
    output['weights'] = ans['Weight_final']

    return output


def denoise_spikes(trace, superfactor=10, threshs=(.4, .6, .75)):

    # Originally written by Johannes Friedrich @ Flatiron Institute
    # Modified by Takashi Kawashima @ HHMI Janelia

    # calculate beforehand the matrix for calculating pre-spike ramp gradients

    regressor = np.hstack((np.array([[1], [1], [1]]), np.array([[-1], [-0], [1]])))
    inverse_matrix = np.dot(np.linalg.inv(np.dot(regressor.T, regressor)), regressor.T)

    for iters in range(3):
        sub_thresh1 = trace if iters == 0 else trace - \
            np.convolve(spiketrain, kernel, 'same')  # subtract spikes
        sub_thresh2 = butter_filter(sub_thresh1, 'low')  # filter subthreshold part
        high_freq = trace - sub_thresh2  # high frequency part: spikes and noise

        high_freq_med = np.array(pd.Series(high_freq).rolling(
            window=9000, min_periods=4500, center=True).median())
        high_freq_std = np.array(pd.Series(high_freq).rolling(
            window=9000, min_periods=4500, center=True).std())

        trace_med = np.array(pd.Series(sub_thresh1).rolling(
            window=9000, min_periods=4500, center=True).median())
        trace_std = np.array(pd.Series(sub_thresh1).rolling(
            window=9000, min_periods=4500, center=True).std())

        if iters == 0:

            # adapt threshold for each neurons based on spike shape integrity

            threshold_sets = (2.5, 3., 3.5)
            th_scores = np.ones((len(threshold_sets),))
            th_tlimits = np.zeros((len(threshold_sets),))

            for th in range(len(threshold_sets)):

                tlimit = len(trace)

                thre = threshold_sets[th]

                # check validity of spike kernel by detecting pre-spike ramp
                # if the shape of first 50 spikes are suspicious, it will return an empty array
                # this will stop erroneous detection of noise after there is no spike anymore

                def test_spikeshape(time, tcourse, tcourse_med, tcourse_std, regress_matrix):

                    time = time[(time - 4) >= 0]

                    spike_matrix = np.zeros((len(time), 3))
                    for t in range(3):
                        spike_matrix[:, t] = tcourse[time - 3 + t]

                    spike_matrix -= spike_matrix.mean(axis=1)[:, None]

                    gradient = np.dot(regress_matrix, spike_matrix.T)[1, :]

                    s, p = ttest_1samp(gradient, 0)

                    return (s, p)

                spiketimes = get_spiketimes(high_freq, high_freq_med + thre * high_freq_std,
                                            trace, trace_med + thre * trace_std, tlimit)

                spikebins = 50
                spikenrep = (len(spiketimes) // spikebins) + int((len(spiketimes) % spikebins) > 0)

                tlimit = 0
                for n in range(spikenrep):
                    spike_inds = np.arange(
                        spikebins * n, min(spikebins * (n + 1), len(spiketimes)))
                    slen = len(spike_inds)
                    spike_t = spiketimes[spike_inds]

                    (s, p) = test_spikeshape(spike_t, trace, trace_med, trace_std, inverse_matrix)
                    if n == 0:
                        th_scores[th] = p

                    if (p < 0.05) and (s > 0):

                        tlimit = min(spike_t[-1] + 15, len(trace))

                    elif n > 0:
                        for j in range(slen):
                            endt = min(spikebins * (n + 1), len(spiketimes)) - j
                            spike_inds = np.arange((endt - 50), endt)
                            spike_t = spiketimes[spike_inds]

                            (s, p) = test_spikeshape(spike_t, trace,
                                                     trace_med, trace_std, inverse_matrix)

                            if (p < 0.05) and (s > 0):
                                tlimit = min(spike_t[-1] + 15, len(trace))
                                break

                        break

                    else:
                        th_scores[th] = 1
                        break

                th_tlimits[th] = tlimit

            best_inds = np.where(th_scores < 0.05)[0]
            if best_inds.size > 0:
                best_thre = threshold_sets[best_inds[0]]
                best_tlimit = int(th_tlimits[best_inds[0]])
            else:
                best_thre = threshold_sets[0]
                best_tlimit = 0

            if best_tlimit == 0:
                spiketimes = np.zeros((0,))
                break

            spiketimes = get_spiketimes(high_freq, high_freq_med + best_thre * high_freq_std,
                                        trace, trace_med + best_thre * trace_std, best_tlimit)

        if spiketimes.size == 0:
            break

        kernel = get_kernel(high_freq, spiketimes)

        # lower threshold, now picking up spikes not merely based on threshold but spike shape

        spiketimes = get_spiketimes(high_freq, high_freq_med + (best_thre - 0.5) * high_freq_std,
                                    trace, trace_med + (best_thre - 0.5) * trace_std, best_tlimit)
        spikesizes = get_spikesizes(high_freq, spiketimes, kernel)
        spiketrain = get_spiketrain(spiketimes, spikesizes, len(trace))

        # iteratively remove too small spikes 'slowly' increasing threshold
        for thresh in threshs:
            while np.sum(spikesizes < thresh):
                spiketimes = np.where(spiketrain > thresh)[0]
                spikesizes = get_spikesizes(high_freq, spiketimes, kernel)
                spiketrain = get_spiketrain(spiketimes, spikesizes, len(trace))

    if spiketimes.size > 0:
        # refine frame_rate result to obtain super-resolution
        upsampled_kernel = upsample_kernel(kernel, superfactor=superfactor, interpolation='linear')
        super_times, super_sizes = superresolve(
            high_freq, spiketimes, spikesizes, upsampled_kernel, superfactor)

        return (sub_thresh2, high_freq, spiketimes, spiketrain, spikesizes,
                super_times, super_sizes, kernel, upsampled_kernel, best_tlimit, best_thre)
    else:
        return (sub_thresh2, high_freq, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def butter_filter(data, btype='high', cutoff=10., fs=300, order=5):

    # Originally written by Johannes Friedrich @ Flatiron Institute
    # Modified by Takashi Kawashima @ HHMI Janelia

    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def get_spiketimes(trace1, thresh1, trace2, thresh2, tlimit):

    # Originally written by Johannes Friedrich @ Flatiron Institute
    # Modified by Takashi Kawashima @ HHMI Janelia
    '''determine spike times based on threshold'''
    times = np.where((trace1[:tlimit] > thresh1[:tlimit]) &
                     (trace2[:tlimit] > thresh2[:tlimit]))[0]

    # group neigbours together
    if (times.size > 0):
        ls = [[times[0]]]
        for t in times[1:]:
            if t == ls[-1][-1] + 1:
                ls[-1].append(t)
            else:
                ls.append([t])
        # take local maximum if neighbors are above threshold
        times = np.array([l[np.argmax(trace1[l])] for l in ls])
    return times


def get_kernel(trace, spiketimes, spikesizes=None, tau=31, superfactor=1, b=False):

    # Originally written by Johannes Friedrich @ Flatiron Institute
    # Modified by Takashi Kawashima @ HHMI Janelia
    '''determine kernel via regression
    resolution of spike times must be some integer divided by superfactor
    '''
    th = tau // 2
    t = len(trace)
    s = np.zeros((superfactor, t + th))
    for k in range(superfactor):
        tmp = (spiketimes * superfactor + k) % superfactor == 0
        s[k, (spiketimes[tmp] + k / float(superfactor)).astype(int)
          ] = 1 if spikesizes is None else spikesizes[tmp]
    ss = np.zeros((tau * superfactor, t + th))
    for i in range(tau):
        ss[i * superfactor:(i + 1) * superfactor, i:] = s[:, :t + th - i]
    ssm = ss - ss.mean() if b else ss

    symm = ssm.dot(ssm.T)

    if np.linalg.cond(symm) < 1 / sys.float_info.epsilon:
        invm = np.linalg.inv(symm)
    else:
        noise = np.random.rand(symm.shape[0], symm.shape[1]) / 10000
        symm += noise
        invm = np.linalg.inv(symm)

    return invm.dot(ssm.dot(np.hstack([np.zeros(th), trace])))


def get_spikesizes(trace, spiketimes, kernel):

    # Originally written by Johannes Friedrich @ Flatiron Institute
    # Modified by Takashi Kawashima @ HHMI Janelia
    '''determine spike sizes via regression'''
    tau = len(kernel)
    th = tau // 2

    trace = trace.astype(np.float32)

    ans = np.zeros((len(spiketimes),)).astype(np.float32)

    spikebin = 200
    binnum = int(len(spiketimes) / spikebin) + 1

    for i in range(binnum):
        binsize = min(len(spiketimes) - (spikebin * i), spikebin)
        spike_range = np.arange((spikebin * i), (spikebin * i + binsize)).astype(int)
        if binsize > 0:

            spike_min = spiketimes[spike_range[0]]
            spike_max = spiketimes[spike_range[-1]]
            if spike_min > th:
                trace_pre = trace[spike_min - th:spike_min]
            else:
                trace_pre = np.zeros(th)

            if spike_max < (len(trace) - (tau - th)):
                trace_post = trace[spike_max:spike_max + tau - th]
            else:
                trace_post = np.zeros(tau - th)

            trace_tmp = trace[spike_min:spike_max]

            tmp = np.zeros((binsize, len(trace_tmp) + tau), dtype=np.float32)
            for j, t in enumerate(spiketimes[spike_range] - spike_min):
                tmp[j, t:t + tau] = kernel.astype(np.float32)

            ans[spike_range] = lsqr(tmp.T, np.hstack([trace_pre, trace_tmp, trace_post]))[0]

    return ans


def get_spiketrain(spiketimes, spikesizes, T):

    # Originally written by Johannes Friedrich @ Flatiron Institute
    # Modified by Takashi Kawashima @ HHMI Janelia

    s = np.zeros(T)
    s[spiketimes] = spikesizes
    return s


def upsample_kernel(kernel, superfactor=10, interpolation='linear'):

    # Originally written by Johannes Friedrich @ Flatiron Institute
    # Modified by Takashi Kawashima @ HHMI Janelia

    tau = len(kernel)
    k = interp1d(range(len(kernel)), kernel, kind=interpolation,
                 assume_sorted=True, fill_value='extrapolate')
    return k(np.arange(-1, tau + 1, 1. / superfactor))


# upsampled_k[grid-delta] is kernel for spike at time t+delta/superfactor instead t
def superresolve(high_freq, spiketimes, spikesizes, upsampled_k, superfactor=10):

    # Originally written by Johannes Friedrich @ Flatiron Institute
    # Modified by Takashi Kawashima @ HHMI Janelia

    tau = int(len(upsampled_k) / superfactor - 2)
    th = int(tau // 2)
    grid = superfactor * np.arange(1, tau + 1).astype(int)
    kk = [upsampled_k[grid - delta].dot(upsampled_k[grid - delta])
          for delta in range(1 - superfactor, superfactor)]  # precompute
    N = len(spiketimes)
    super_times = np.zeros(N)
    super_sizes = np.zeros(N)
    for i in range(N):
        t = spiketimes[i]
        int_t = int(t + .5)
        snippet = high_freq[max(0, int_t - th):int_t + tau - th].copy()
        if t < th:  # very early spike -> shortened snippet
            snippet = np.concatenate((np.zeros(th - int_t), snippet))
        elif t + tau - th > len(high_freq):  # very late spike -> shortened snippet
            zeropad = tau - len(snippet)
            snippet = np.concatenate((snippet, np.zeros(zeropad)))
        # remove contributions of other spikes to the snippet
        if i:
            tpre = spiketimes[i - 1]
            int_tpre = int(tpre + .5)
            if (tpre > t - tau) and ((int_t - int_tpre) > 0):
                delta = int(superfactor * ((tpre - int_tpre) - (t - int_t)))
                snippet[:int_tpre - int_t] -= spikesizes[i - 1] * \
                    upsampled_k[grid - delta][int_t - int_tpre:]
        if i < N - 1:
            tpost = spiketimes[i + 1]
            int_tpost = int(tpost + .5)
            if (tpost < t + tau) and ((int_tpost - int_t) > 0):
                delta = int(superfactor * ((tpost - int_tpost) - (t - int_t)))
                snippet[int_tpost - int_t:] -= spikesizes[i + 1] * \
                    upsampled_k[grid - delta][:int_t - int_tpost]
        # find best spike time and size
        ls = []
        for delta in range(1 - superfactor, superfactor):
            q = (snippet - upsampled_k[grid - delta] *
                 (upsampled_k[grid - delta].dot(snippet) / kk[delta + superfactor - 1]))
            if t < th:  # very early spike -> shortened snippet
                q = q[th - int_t:]
            elif t + tau - th > len(high_freq):  # very late spike -> shortened snippet
                q = q[:-zeropad]
            ls.append(q.dot(q))
        delta = np.argmin(ls) - superfactor + 1
        super_times[i] = t + delta / float(superfactor)
        super_sizes[i] = (upsampled_k[grid - delta].dot(snippet) / kk[delta + superfactor - 1])

    return super_times, super_sizes
