#!/usr/bin/env python

import logging

import numpy as np
import numpy.testing as npt
import pytest
from scipy import signal

import caiman
from caiman.source_extraction.volpy import atm, spikepursuit
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY

# One spike as a fraction of its peak, which sits at index SPIKE_PEAK. atm only accepts spikes
# that rise over the samples before the peak, hence the ramp.
SPIKE_SHAPE = np.array([0.1, 0.3, 0.6, 1.0, 0.5, 0.2])
SPIKE_PEAK = 3


def gen_voltage_data(dims=(40, 40), T=5000, fr=400, n_spikes=40, seed=7):
    """Synthetic voltage-imaging movie: one cell (a small disk in the centre) firing
    brief positive spikes over a noisy baseline with slow photobleaching.

    T must comfortably exceed 1000 frames: whitened_matched_filter() estimates the
    noise spectrum with a fixed 1000-sample Welch window after censoring the
    samples around candidate spikes."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:dims[0], :dims[1]]
    cy, cx = dims[0] // 2, dims[1] // 2
    roi = (yy - cy)**2 + (xx - cx)**2 <= 3**2

    baseline = 100.0
    bleach = np.linspace(0, -5, T)[:, None, None]
    movie = baseline + bleach + rng.normal(0, 2.0, size=(T,) + dims)

    spike_frames = np.sort(rng.choice(np.arange(20, T - 20), size=n_spikes, replace=False))
    movie[spike_frames] += 30.0 * roi[None, :, :]
    return movie.astype(np.float32), roi, spike_frames, fr


def gen_voltron_like_data(seed=11):
    """Synthetic movie resembling a negative-going indicator such as Voltron: three bright cells
    whose fluorescence dips by 5% on each spike. The cells are far enough apart that their
    context regions do not overlap, and the third one never fires.

    The spikes are kept small on purpose. With much larger ones, adaptive_thresh() can put the
    threshold below every peak and the results change from run to run. T covers 20 s because
    atm needs at least 15 s of data for its rolling thresholds."""
    T, fr, dims = 6000, 300, (48, 144)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:dims[0], :dims[1]]
    movie = 100.0 + 2.0 * rng.standard_normal(size=(T,) + dims, dtype=np.float32)
    movie += np.linspace(0, -3, T, dtype=np.float32)[:, None, None]

    rois, spike_frames = [], []
    for cx, n_spikes in ((24, 60), (72, 60), (120, 0)):
        roi = (yy - 24)**2 + (xx - cx)**2 <= 3**2
        # at most one spike per 60-frame slot, so spikes never overlap
        slots = rng.permutation(np.arange(100, T - 100, 60))[:n_spikes]
        frames = np.sort(slots + rng.integers(0, 20, size=n_spikes))
        trace = np.ones(T, dtype=np.float32)
        for frame in frames:
            start = frame - SPIKE_PEAK
            trace[start:start + len(SPIKE_SHAPE)] -= 0.05 * SPIKE_SHAPE
        movie[:, roi] += 100.0 * trace[:, None]
        rois.append(roi)
        spike_frames.append(frames)
    return movie, np.array(rois), spike_frames, fr


def match_spikes(truth, detected, tolerance=2):
    """Match detections to true spike frames one-to-one, so that one detection cannot count for
    two spikes. The tolerance allows small timing shifts from filtering.

    Returns (recall, precision)."""
    truth = np.sort(np.asarray(truth))
    detected = np.sort(np.asarray(detected))
    i = j = hits = 0
    while i < len(truth) and j < len(detected):
        if detected[j] < truth[i] - tolerance:
            j += 1
        elif detected[j] > truth[i] + tolerance:
            i += 1
        else:
            hits += 1
            i += 1
            j += 1
    recall = hits / len(truth) if len(truth) else 0.0
    precision = hits / len(detected) if len(detected) else 0.0
    return recall, precision


@pytest.fixture(scope="module")
def single_cell_movie(tmp_path_factory):
    movie, roi, spike_frames, fr = gen_voltage_data()
    base_name = str(tmp_path_factory.mktemp('volpy') / 'test-volpy')
    fname = caiman.save_memmap([movie], base_name=base_name, order='C')
    return fname, movie.shape, roi, spike_frames, fr


def single_cell_opts(single_cell_movie, **kwargs):
    fname, _, roi, _, fr = single_cell_movie
    # The synthetic spikes are positive; do not invert them.
    return volparams(fnames=fname, fr=fr, index=[0], ROIs=roi[np.newaxis], flip_signal=False, **kwargs)


@pytest.fixture(scope="module")
def volspike_output(single_cell_movie):
    """Run the single-cell pipeline once for structure and detection checks."""
    fname, shape, roi, spike_frames, fr = single_cell_movie
    opts = single_cell_opts(single_cell_movie)
    output = spikepursuit.volspike([fname, fr, 0, roi, None, opts.volspike])
    return output, shape, roi, spike_frames, opts.volspike


def test_volspike_structure(volspike_output):
    """Check the result structure after the footprint and DFT fixes."""
    output, (T, d1, d2), roi, _, _ = volspike_output

    assert output['cell_n'] == 0
    assert output['weights'].shape == (d1, d2)
    assert output['t'].shape == (T,)
    assert output['context_coord'].shape == (2, 2)
    assert np.isfinite(output['t']).all()
    assert np.isfinite(output['weights']).all()

    # the context box must contain the ROI
    (x0, y0), (x1, y1) = output['context_coord']
    ys, xs = np.nonzero(roi)
    assert x0 <= ys.min() and ys.max() <= x1
    assert y0 <= xs.min() and xs.max() <= y1


def test_volspike_detection(volspike_output):
    """Check recovery without accepting excessive false detections."""
    output, (T, _, _), _, spike_frames, args = volspike_output

    # strong, clean spikes should be found comfortably above min_spikes
    assert not output['low_spikes']
    assert output['spikes'].size >= args['min_spikes']
    detected = np.asarray(output['spikes'])
    assert np.all((detected >= 0) & (detected < T))
    assert np.unique(detected).size == detected.size

    recall, precision = match_spikes(spike_frames, detected)
    # Provisional floors for this strong synthetic signal, not scientific
    # performance guarantees. Check stability in the supported environments.
    assert recall > 0.5, (recall, precision)
    assert precision > 0.5, (recall, precision)


def test_whitened_matched_filter_runs():
    """whitened_matched_filter() raised IndexError under OpenCV 5.

    cv2.dft() returns (n, 2) for a 1-D input under OpenCV 5 but (n, 1, 2) for an
    (n, 1) input; the indexing here expects the latter. This is a fast, targeted
    guard that needs no movie on disk.
    """
    rng = np.random.default_rng(3)
    T, fr = 5000, 400
    data = rng.normal(0, 1.0, T)
    locs = np.sort(rng.choice(np.arange(100, T - 100), size=30, replace=False))
    window_length = int(fr * 0.02)     # volparams template_size default
    window = np.arange(-window_length, window_length, dtype=int)
    data[locs[:, np.newaxis] + window] += 5.0

    filtered = spikepursuit.whitened_matched_filter(data, locs, window)

    assert filtered.shape == (T,)
    assert np.all(np.isfinite(filtered))


def test_volspike_outputs(volspike_output):
    output, (T, _, _), _, _, args = volspike_output

    for key in ('t', 'ts', 't_rec', 't_sub', 'F0', 'dFF'):
        assert output[key].shape == (T,), key
        assert np.isfinite(output[key]).all(), key
    # one spike count for the first pass over the ROI average, then one per iteration
    assert len(output['num_spikes']) == args['n_iter'] + 1
    assert output['num_spikes'][-1] == output['spikes'].size
    npt.assert_allclose(output['dFF'], output['t'] / output['F0'])


@pytest.mark.parametrize('option, value', [('weight_update', 'NMF'), ('threshold_method', 'simple'), ('n_iter', 1)])
def test_volspike_options(single_cell_movie, option, value):
    fname, _, roi, spike_frames, fr = single_cell_movie
    args = single_cell_opts(single_cell_movie, **{option: value}).volspike

    output = spikepursuit.volspike([fname, fr, 0, roi, None, args])

    assert not output['low_spikes']
    assert len(output['num_spikes']) == args['n_iter'] + 1
    recall, precision = match_spikes(spike_frames, output['spikes'])
    assert recall > 0.5, (recall, precision)
    assert precision > 0.5, (recall, precision)


def test_volspike_rejects_roi_of_wrong_shape(single_cell_movie):
    fname, _, roi, _, fr = single_cell_movie
    args = single_cell_opts(single_cell_movie).volspike
    with pytest.raises(Exception, match='Dimensions of movie and ROIs do not accord'):
        spikepursuit.volspike([fname, fr, 0, roi[:, :30], None, args])


def test_volspike_needs_background_pixels(single_cell_movie):
    # with a context region this small, the censored area around the cell covers all of it
    fname, _, roi, _, fr = single_cell_movie
    args = single_cell_opts(single_cell_movie, context_size=5).volspike
    with pytest.raises(Exception, match='Too few pixels'):
        spikepursuit.volspike([fname, fr, 0, roi, None, args])


def test_volspike_flags_too_few_spikes(single_cell_movie):
    """The cell fires 40 times. Asking for at least 60 spikes lowers the threshold until 60 peaks
    pass, and sets low_spikes."""
    fname, _, roi, spike_frames, fr = single_cell_movie
    args = single_cell_opts(single_cell_movie, min_spikes=60).volspike

    output = spikepursuit.volspike([fname, fr, 0, roi, None, args])

    assert output['low_spikes']
    assert output['spikes'].size == 60
    assert match_spikes(spike_frames, output['spikes'])[0] > 0.9


def test_volparams_change_params(caplog):
    opts = volparams(fr=400, context_size=21)
    assert opts.data['fr'] == 400
    assert opts.volspike['context_size'] == 21

    with caplog.at_level(logging.WARNING):
        opts.change_params({'clip': 20, 'max_shifts': (3, 3), 'not_a_parameter': 1})

    # each key lands in whichever group owns it
    assert opts.volspike['clip'] == 20
    assert opts.motion['max_shifts'] == (3, 3)
    assert 'No parameter not_a_parameter found!' in caplog.text


@pytest.fixture(scope="module")
def noisy_trace():
    """Unit-variance noise with 39 well separated, three-sample positive spikes."""
    rng = np.random.default_rng(5)
    data = rng.normal(0, 1, 6000)
    spike_frames = np.arange(150, 5850, 150)
    for offset, height in ((-1, 4.0), (0, 12.0), (1, 5.0)):
        data[spike_frames + offset] += height
    return data, spike_frames


def test_simple_thresh(noisy_trace):
    data, spike_frames = noisy_trace
    pks = data[signal.find_peaks(data)[0]]

    # the threshold is a multiple of the noise level, which is estimated from the negative samples
    thresh, low_spikes = spikepursuit.simple_thresh(data, pks, clip=0, threshold=3.5, min_spikes=10)
    assert thresh == pytest.approx(3.5, rel=0.1)
    assert not low_spikes
    n_above = np.sum(pks > thresh)
    assert n_above >= spike_frames.size

    # more peaks than clip: the threshold rises to keep only the `clip` largest
    thresh, low_spikes = spikepursuit.simple_thresh(data, pks, clip=20, threshold=3.5, min_spikes=10)
    assert np.sum(pks > thresh) == 20
    assert not low_spikes

    # fewer peaks than min_spikes: the threshold drops to take the `min_spikes` largest
    thresh, low_spikes = spikepursuit.simple_thresh(data, pks, clip=0, threshold=3.5, min_spikes=n_above + 50)
    assert np.sum(pks > thresh) == n_above + 50
    assert low_spikes


def test_adaptive_thresh():
    rng = np.random.default_rng(1)
    noise_peaks = rng.normal(1.0, 0.3, 3000)
    spike_peaks = rng.normal(8.0, 0.5, 60)
    pks = np.concatenate([noise_peaks, spike_peaks])

    # the threshold falls in the gap between the two populations
    result = spikepursuit.adaptive_thresh(pks, clip=0, pnorm=0.5, min_spikes=10)
    thresh, false_pos_rate, detection_rate, low_spikes = result
    assert noise_peaks.max() < thresh < spike_peaks.min()
    assert false_pos_rate < 0.05
    assert detection_rate > 0.9
    assert not low_spikes

    thresh, _, _, low_spikes = spikepursuit.adaptive_thresh(pks, clip=30, pnorm=0.5, min_spikes=10)
    assert np.sum(pks > thresh) == 30
    assert not low_spikes

    thresh, _, _, low_spikes = spikepursuit.adaptive_thresh(pks, clip=0, pnorm=0.5, min_spikes=200)
    assert np.sum(pks > thresh) == 200
    assert low_spikes


@pytest.mark.parametrize('threshold_method', ['adaptive_threshold', 'simple'])
def test_denoise_spikes(noisy_trace, threshold_method):
    data, spike_frames = noisy_trace
    window_length = 8

    datafilt, spikes, t_rec, templates, low_spikes, thresh = spikepursuit.denoise_spikes(
        data, window_length, fr=400, threshold_method=threshold_method, do_plot=False)

    assert datafilt.shape == data.shape
    assert templates.shape == (2 * window_length + 1,)
    assert not low_spikes
    recall, precision = match_spikes(spike_frames, spikes)
    assert recall > 0.9, (recall, precision)
    assert precision > 0.9, (recall, precision)
    # the reconstruction is the template placed at each detected spike and zero elsewhere
    near_spikes = (spikes[:, np.newaxis] + np.arange(-window_length, window_length + 1)).clip(0, data.size - 1)
    far_from_spikes = np.ones(data.size, dtype=bool)
    far_from_spikes[near_spikes] = False
    assert not t_rec[far_from_spikes].any()
    assert t_rec[spikes].min() > 0


@pytest.fixture(scope="module")
def voltron_like_movie(tmp_path_factory):
    movie, rois, spike_frames, fr = gen_voltron_like_data()
    base_name = str(tmp_path_factory.mktemp('volpy') / 'test-voltron')
    fname = caiman.save_memmap([movie], base_name=base_name, order='C')
    return fname, movie.shape, rois, spike_frames, fr


def voltron_opts(voltron_like_movie, index=(0, 1, 2), **kwargs):
    fname, _, rois, _, fr = voltron_like_movie
    # flip_signal keeps its default (True), which is what a negative-going indicator needs
    return volparams(fnames=fname, fr=fr, index=list(index), ROIs=rois, **kwargs)


@pytest.fixture(scope="module")
def voltron_estimates(voltron_like_movie):
    """Run VOLPY once, serially, on all three cells with the default (spikepursuit) method."""
    vpy = VOLPY(n_processes=1, dview=None, params=voltron_opts(voltron_like_movie))
    vpy.fit(n_processes=1, dview=None)
    return vpy.estimates


def test_volpy_fit_estimates_structure(voltron_estimates, voltron_like_movie):
    """fit() stacks the per-cell outputs, one entry per requested cell."""
    _, (T, d1, d2), rois, _, _ = voltron_like_movie
    estimates = voltron_estimates
    n_cells = len(rois)

    npt.assert_array_equal(estimates['cell_n'], np.arange(n_cells))
    for key in ('t', 'ts', 't_rec', 't_sub', 'F0', 'dFF'):
        assert estimates[key].shape == (n_cells, T), key
        assert np.isfinite(estimates[key]).all(), key
    assert estimates['weights'].shape == (n_cells, d1, d2)
    assert len(estimates['spikes']) == n_cells

    # Each ROI is a disk of radius 3 and the default context_size is 35, so every context region
    # extends 17 pixels beyond its ROI. This pins the dilation that selects the region.
    expected_context = [[[4, 4], [44, 44]], [[4, 52], [44, 92]], [[4, 100], [44, 140]]]
    npt.assert_array_equal(estimates['context_coord'], expected_context)
    for weights, ((x0, y0), (x1, y1)) in zip(estimates['weights'], expected_context):
        outside = np.ones((d1, d2), dtype=bool)
        outside[x0:x1 + 1, y0:y1 + 1] = False
        assert not weights[outside].any()


def test_volpy_fit_finds_each_cells_own_spikes(voltron_estimates, voltron_like_movie):
    _, _, rois, spike_frames, _ = voltron_like_movie
    estimates = voltron_estimates

    for cell, other in ((0, 1), (1, 0)):
        recall, precision = match_spikes(spike_frames[cell], estimates['spikes'][cell])
        assert recall > 0.8, (cell, recall, precision)
        assert precision > 0.8, (cell, recall, precision)
        # ... and not the spikes of the other cell
        assert match_spikes(spike_frames[other], estimates['spikes'][cell])[0] < 0.2
        assert not estimates['low_spikes'][cell]
        assert estimates['locality'][cell]
        assert estimates['snr'][cell] > 5
        # the largest spatial weight lies on the cell
        assert rois[cell][np.unravel_index(np.argmax(estimates['weights'][cell]), rois[cell].shape)]


def test_volpy_fit_silent_cell_has_low_snr(voltron_estimates):
    """A cell that never fires still gets some noise peaks above the adaptive threshold, so
    low_spikes does not flag it. Its signal-to-noise ratio is far lower than the active cells'."""
    snr = voltron_estimates['snr']
    assert snr[2] < 0.5 * min(snr[0], snr[1]), snr


def test_volpy_fit_subset_of_cells(voltron_like_movie, voltron_estimates):
    """`index` selects which ROIs are processed."""
    _, _, _, spike_frames, _ = voltron_like_movie
    vpy = VOLPY(n_processes=1, dview=None, params=voltron_opts(voltron_like_movie, index=[1]))

    assert vpy.fit(n_processes=1, dview=None) is vpy

    npt.assert_array_equal(vpy.estimates['cell_n'], [1])
    npt.assert_array_equal(vpy.estimates['context_coord'][0], voltron_estimates['context_coord'][1])
    recall, precision = match_spikes(spike_frames[1], vpy.estimates['spikes'][0])
    assert recall > 0.8, (recall, precision)
    assert precision > 0.8, (recall, precision)


def test_volspike_even_context_size(voltron_like_movie):
    """An even context_size goes through the padding branch of _shift_footprint_old()."""
    fname, _, rois, spike_frames, fr = voltron_like_movie
    args = voltron_opts(voltron_like_movie, context_size=34).volspike

    output = spikepursuit.volspike([fname, fr, 1, rois[1], None, args])

    # Dilating the 7 x 7 ROI with a 34 x 34 footprint gives a 40 x 40 region. The ROI covers
    # rows 21 to 27 and columns 69 to 75.
    (x0, y0), (x1, y1) = output['context_coord']
    assert (x1 - x0 + 1, y1 - y0 + 1) == (40, 40)
    assert x0 < 21 and x1 > 27 and y0 < 69 and y1 > 75
    recall, precision = match_spikes(spike_frames[1], output['spikes'])
    assert recall > 0.8, (recall, precision)
    assert precision > 0.8, (recall, precision)


def test_volpy_fit_reuses_weights(voltron_like_movie, voltron_estimates):
    """Weights from an earlier run, cropped to each context region as the demo does, seed the next run."""
    _, _, _, spike_frames, _ = voltron_like_movie
    cells = [0, 1]
    weights = []
    for cell in cells:
        (x0, y0), (x1, y1) = voltron_estimates['context_coord'][cell]
        weights.append(voltron_estimates['weights'][cell][x0:x1 + 1, y0:y1 + 1])
    vpy = VOLPY(n_processes=1, dview=None, params=voltron_opts(voltron_like_movie, index=cells, weights=weights))

    vpy.fit(n_processes=1, dview=None)

    for cell in cells:
        npt.assert_array_equal(vpy.estimates['rawROI'][cell]['weights'], weights[cell])
        recall, precision = match_spikes(spike_frames[cell], vpy.estimates['spikes'][cell])
        assert recall > 0.8, (cell, recall, precision)
        assert precision > 0.8, (cell, recall, precision)


def test_volpy_fit_in_parallel(voltron_like_movie, voltron_estimates):
    """A multiprocessing pool gives the same answer as the serial run. Three cells over two
    processes also exercises the batching of cells."""
    _, dview, n_processes = caiman.cluster.setup_cluster(backend='multiprocessing',
                                                         n_processes=2,
                                                         ignore_preexisting=True)
    try:
        vpy = VOLPY(n_processes=n_processes, dview=dview, params=voltron_opts(voltron_like_movie))
        vpy.fit(n_processes=n_processes, dview=dview)
    finally:
        caiman.stop_server(dview=dview)

    npt.assert_array_equal(vpy.estimates['cell_n'], voltron_estimates['cell_n'])
    npt.assert_array_equal(vpy.estimates['context_coord'], voltron_estimates['context_coord'])
    for cell in (0, 1):
        recall, precision = match_spikes(voltron_estimates['spikes'][cell], vpy.estimates['spikes'][cell])
        assert recall > 0.9, (cell, recall, precision)
        assert precision > 0.9, (cell, recall, precision)


def test_atm_get_spiketimes():
    trace = np.zeros(50)
    trace[[10, 11, 12]] = [2, 5, 3]
    trace[30] = 4
    trace[45] = 6
    thresh = np.ones(50)

    # a run of samples above threshold counts once, at its maximum; nothing beyond tlimit counts
    npt.assert_array_equal(atm.get_spiketimes(trace, thresh, trace, thresh, tlimit=40), [11, 30])

    # both traces have to exceed their thresholds
    other = np.zeros(50)
    other[30] = 4
    npt.assert_array_equal(atm.get_spiketimes(trace, thresh, other, thresh, tlimit=50), [30])


def test_atm_get_spiketrain():
    train = atm.get_spiketrain(np.array([2, 5]), np.array([1.5, 0.5]), 8)
    npt.assert_array_equal(train, [0, 0, 1.5, 0, 0, 0.5, 0, 0])


@pytest.fixture(scope="module")
def kernel_trace():
    """A noise-free trace made of one known kernel, centred on each spike and scaled by its size."""
    window_length = 5
    kernel = np.array([0, 0.1, 0.3, 0.6, 0.8, 1.0, 0.6, 0.3, 0.15, 0.05, 0])
    spiketimes = np.array([30, 80, 141, 200, 260, 333])
    sizes = np.array([1.0, 0.5, 2.0, 1.5, 0.8, 1.2])
    trace = np.zeros(400)
    for time, size in zip(spiketimes, sizes):
        trace[time - window_length:time + window_length + 1] += size * kernel
    return trace, spiketimes, sizes, kernel, window_length


def test_atm_get_kernel_and_spikesizes(kernel_trace):
    trace, spiketimes, sizes, kernel, window_length = kernel_trace
    npt.assert_allclose(atm.get_kernel(trace, spiketimes, window_length, spikesizes=sizes), kernel, atol=1e-3)
    npt.assert_allclose(atm.get_spikesizes(trace, spiketimes, kernel), sizes, atol=1e-3)


def test_atm_superresolve(kernel_trace):
    trace, spiketimes, sizes, kernel, _ = kernel_trace
    superfactor = 4
    upsampled = atm.upsample_kernel(kernel, superfactor=superfactor)

    # spikes that sit exactly on a frame stay there
    times, estimated_sizes = atm.superresolve(trace, spiketimes, sizes, upsampled, superfactor)
    npt.assert_array_equal(times, spiketimes)
    npt.assert_allclose(estimated_sizes, sizes, atol=1e-6)

    # Sampling the upsampled kernel half a frame early gives a spike half a frame late.
    # upsampled[grid] is the kernel itself, as upsample_kernel pads it by one frame on each side.
    grid = superfactor * np.arange(1, kernel.size + 1)
    late = np.zeros(100)
    late[50 - kernel.size // 2:50 + kernel.size // 2 + 1] = upsampled[grid - superfactor // 2]
    times, _ = atm.superresolve(late, np.array([50]), np.array([1.0]), upsampled, superfactor)
    npt.assert_array_equal(times, [50.5])


def test_atm_denoise_spikes():
    rng = np.random.default_rng(2)
    fr, T = 300, 6000
    window_length = fr // 20
    spike_frames = np.arange(200, T - 200, 90)
    trace = 1 + rng.normal(0, 0.005, T)
    for frame in spike_frames:
        start = frame - SPIKE_PEAK
        trace[start:start + len(SPIKE_SHAPE)] += 0.3 * SPIKE_SHAPE

    result = atm.denoise_spikes(trace, fr, window_length, superfactor=10)

    sub_thresh, high_freq, spiketimes, spiketrain, spikesizes, super_times, super_sizes, kernel = result[:8]
    assert sub_thresh.shape == high_freq.shape == spiketrain.shape == (T,)
    assert kernel.shape == (2 * window_length + 1,)
    recall, precision = match_spikes(spike_frames, spiketimes, tolerance=1)
    assert recall > 0.9, (recall, precision)
    assert precision > 0.9, (recall, precision)
    assert spikesizes.shape == super_times.shape == super_sizes.shape == spiketimes.shape
    # the refined times stay within a frame of the frame-resolution ones
    assert np.abs(super_times - spiketimes).max() < 1


# atm only accepts these two weight updates, not the default 'ridge'
@pytest.mark.parametrize('weight_update', ['NMF', 'maxSNR'])
def test_atm_volspike(voltron_like_movie, weight_update):
    fname, (T, _, _), rois, spike_frames, fr = voltron_like_movie
    args = voltron_opts(voltron_like_movie, method='atm', weight_update=weight_update).volspike

    output = atm.volspike([fname, fr, 0, rois[0], None, args])

    assert output['cellN'] == 0
    assert output['yFilt'].shape == (T,)
    assert not output['low_spk']
    times = output['spikeTimes']
    assert output['num_spikes'] == times.size
    # spike times are refined to a fraction of a frame
    assert times.dtype.kind == 'f'
    recall, precision = match_spikes(spike_frames[0], np.round(times))
    assert recall > 0.8, (recall, precision)
    assert precision > 0.8, (recall, precision)
    assert match_spikes(spike_frames[1], np.round(times))[0] < 0.2


def test_volpy_fit_atm(voltron_like_movie):
    """VOLPY dispatches to atm and stacks its outputs. Unlike spikepursuit, atm flags the silent cell."""
    _, (T, d1, d2), _, spike_frames, _ = voltron_like_movie
    vpy = VOLPY(n_processes=1, dview=None, params=voltron_opts(voltron_like_movie, method='atm', weight_update='NMF'))

    estimates = vpy.fit(n_processes=1, dview=None).estimates

    npt.assert_array_equal(estimates['cellN'], [0, 1, 2])
    npt.assert_array_equal(estimates['num_spikes'] > 0, [True, True, False])
    npt.assert_array_equal(estimates['low_spk'], [False, False, True])
    assert estimates['meanIM'].shape == (3, d1, d2)
    assert estimates['yFilt'].shape == (3, T)
    for cell in (0, 1):
        recall, precision = match_spikes(spike_frames[cell], np.round(estimates['spikeTimes'][cell]))
        assert recall > 0.8, (cell, recall, precision)
        assert precision > 0.8, (cell, recall, precision)


def test_reconstructed_movie(voltron_like_movie, voltron_estimates):
    # imported here because volpy.utils pulls in torch, which nothing else in this file needs
    from caiman.source_extraction.volpy.utils import reconstructed_movie
    fname, (_, d1, d2), _, _, _ = voltron_like_movie
    # reconstructed_movie() clips the weights it is given in place, so pass a copy
    estimates = dict(voltron_estimates, weights=voltron_estimates['weights'].copy())

    movie = reconstructed_movie(estimates, fnames=fname, idx=[0, 1], scope=(0, 300), flip_signal=True)

    # the input, baseline-subtracted and reconstructed movies side by side, each scaled to [0, 1]
    assert movie.shape == (300, d1, 3 * d2)
    assert np.isfinite(movie).all()
    assert movie.min() == 0
    assert movie.max() == 1
