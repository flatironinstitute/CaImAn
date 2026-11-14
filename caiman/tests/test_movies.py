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


def test_play_plot_text(monkeypatch):
    # cv2.putText only draws on 8-bit images in OpenCV 5, while play() works on float frames.
    # Record the frames instead of opening a window so this runs headless.
    shown = []
    monkeypatch.setattr(movies.cv2, 'imshow', lambda name, frame: shown.append(frame.copy()))
    monkeypatch.setattr(movies.cv2, 'waitKey', lambda *args: -1)
    monkeypatch.setattr(movies.cv2, 'destroyAllWindows', lambda: None)

    rng = np.random.default_rng(0)
    mov = movies.movie(rng.random((3, 40, 60)).astype(np.float32), fr=30)
    mov.play(backend='opencv', plot_text=True, fr=1000, q_max=100, q_min=0)

    assert len(shown) == 3
    for frame in shown:
        assert np.issubdtype(frame.dtype, np.floating)
        # the frame number is burned in at full brightness, the rest stays in [0, 1]
        assert np.any(frame == 255)
        npt.assert_array_less(frame[frame != 255], 1 + 1e-6)
