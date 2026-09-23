#!/usr/bin/env python

import numpy as np
import pytest
import matplotlib.pyplot as plt

from caiman.source_extraction.volpy import spikepursuit, utils as volpy_utils


def test_volspike_builds_two_dimensional_context_footprint(monkeypatch):
    class ContextFootprintObserved(Exception):
        pass

    def inspect_context_footprint(image, footprint):
        assert image.shape == (4, 4)
        assert footprint.ndim == 2
        # Even dimensions are padded by _shift_footprint_old so that the
        # footprint remains centered for skimage.morphology.dilation.
        assert footprint.shape == (5, 5)
        raise ContextFootprintObserved

    movie_matrix = np.zeros((16, 2), dtype=np.float32)
    monkeypatch.setattr(
        spikepursuit.cm,
        'load_memmap',
        lambda _: (movie_matrix, (4, 4), 2),
    )
    monkeypatch.setattr(spikepursuit, 'dilation', inspect_context_footprint)

    roi = np.zeros((4, 4), dtype=bool)
    roi[1:3, 1:3] = True
    args = {'template_size': 0.02, 'context_size': 4}

    with pytest.raises(ContextFootprintObserved):
        spikepursuit.volspike(['movie.mmap', 400, 0, roi, None, args])


def test_view_components_renders_initial_component(monkeypatch):
    frame_count = 10
    estimates = {
        'weights': np.ones((2, 4, 4), dtype=float),
        't': np.ones((2, frame_count), dtype=float),
        't_sub': np.full((2, frame_count), 0.5),
        't_rec': np.full((2, frame_count), 0.75),
        'spikes': [np.array([1, 3]), np.array([2, 4])],
        'snr': np.array([2.0, 3.0]),
        'locality': np.array([True, True]),
    }
    monkeypatch.setattr(plt, 'show', lambda: None)

    volpy_utils.view_components(
        estimates,
        np.ones((4, 4), dtype=float),
        np.array([0, 1]),
    )

    figure = plt.gcf()
    assert len(figure.axes) == 4
    assert figure.axes[1].get_title() == 'Spatial component 1'
    assert figure.axes[3].get_title() == 'Signal and spike times 1'
    plt.close(figure)
