#!/usr/bin/env python
import numpy as np
import numpy.testing as npt
import os
from caiman.base.movies import load, load_iter
from caiman.paths import caiman_datadir


def test_load_iter():
    fname = os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')
    for subindices in (None, slice(100, None, 2)):
        m = load_iter(fname, subindices=subindices)
        S = 0
        while True:
            try:
                S += np.sum(next(m))
            except StopIteration:
                break
        npt.assert_allclose(S, load(fname, subindices=subindices).sum(), rtol=1e-6)
