#!/usr/bin/env python

import numpy as np
import pytest

import caiman
from caiman.source_extraction.volpy import spikepursuit
from caiman.source_extraction.volpy.volparams import volparams


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


@pytest.fixture(scope="module")
def volspike_output(tmp_path_factory):
    """Run the single-cell pipeline once for structure and detection checks."""
    movie, roi, spike_frames, fr = gen_voltage_data()
    base_name = str(tmp_path_factory.mktemp('volpy') / 'test-volpy')
    fname = caiman.save_memmap([movie], base_name=base_name, order='C')
    # The synthetic spikes are positive; do not invert them.
    opts = volparams(fnames=fname, fr=fr, index=[0], ROIs=roi[np.newaxis], flip_signal=False)
    output = spikepursuit.volspike([fname, fr, 0, roi, None, opts.volspike])
    return output, movie.shape, roi, spike_frames, opts.volspike


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

    # One-to-one matching prevents one detection counting as multiple hits.
    # A two-frame tolerance allows small timing shifts from filtering.
    truth = np.sort(spike_frames)
    predicted = np.sort(detected)
    i = j = hits = 0
    while i < len(truth) and j < len(predicted):
        if predicted[j] < truth[i] - 2:
            j += 1
        elif predicted[j] > truth[i] + 2:
            i += 1
        else:
            hits += 1
            i += 1
            j += 1
    recall = hits / len(truth)
    precision = hits / len(predicted)
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
