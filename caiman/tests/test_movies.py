#!/usr/bin/env python
import numpy as np
import numpy.testing as npt
import os
from caiman.base import movies
from caiman.paths import caiman_datadir


def test_load_iter():
    fname = os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')
    for subindices in (None, slice(100, None, 2)):
        m = movies.load_iter(fname, subindices=subindices)
        S = 0
        while True:
            try:
                S += np.sum(next(m))
            except StopIteration:
                break
        npt.assert_allclose(S, movies.load(fname, subindices=subindices).sum(), rtol=1e-6)


def test_to2D_to3D():
    def reference_to2D(mov, order: str):
        return mov.T.reshape((-1, mov.shape[0]), order=order)

    # set up small test movies
    rng = np.random.default_rng()
    T, y, x, z = 10, 6, 4, 2
    movie_3D = movies.movie(rng.random((T, y, x)))
    movie_4D = movies.movie(rng.random((T, y, x, z)))

    for movie in (movie_3D, movie_4D):
        for order in ('F', 'C'):
            shape = movie.shape
            movie_flat = movie.to2DPixelxTime(order=order)
            npt.assert_array_equal(movie_flat, reference_to2D(movie, order))
            movie_from_flat = movie_flat.to3DFromPixelxTime(shape, order)
            npt.assert_array_equal(movie_from_flat, movie)
