#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from caiman.source_extraction import cnmf as cnmf


def test_axcov():
    data = np.random.randn(1000)
    maxlag = 5
    C = cnmf.pre_processing.axcov(data, maxlag)
    print(C)

    npt.assert_allclose(C, np.concatenate((np.zeros(maxlag), np.array([1]), np.zeros(maxlag))), atol=1)
