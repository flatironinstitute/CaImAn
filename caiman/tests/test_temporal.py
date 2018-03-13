import numpy.testing as npt
import numpy as np
from caiman.source_extraction import cnmf


def test_make_G_matrix():
    g = np.array([1, 2, 3])
    T = 6
    G = cnmf.temporal.make_G_matrix(T, g)
    G = G.todense()
    true_G = np.matrix(
        [[1., 0., 0., 0., 0., 0.],
         [-1., 1., 0., 0., 0., 0.],
         [-2., -1., 1., 0., 0., 0.],
         [-3., -2., -1., 1., 0., 0.],
         [0., -3., -2., -1., 1., 0.],
         [0., 0., -3., -2., -1., 1.]])

    npt.assert_allclose(G, true_G)
