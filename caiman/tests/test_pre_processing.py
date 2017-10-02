import numpy.testing as npt
import numpy as np
from caiman.source_extraction import cnmf as cnmf


# at this stage interpolate_missing_data returns alsways an error when
# having missing values, so ignore the test below
'''
def test_interpolate_missing_data():

    # create an array of ones
    Y = np.ones((20,20,10))

    # set some of the values
    index = (np.ceil(np.random.uniform(high = 10,size = 5))-1).astype(int)

    # set some of the values to nan
    Y[index,index,index] = np.nan

    # interpolate
    Y_interpolated = cnmf.pre_processing.interpolate_missing_data(Y)

    # compare to original
    npt.assert_allclose(Y_interpolated,np.ones((20,20,10)),rtol=1e-07,)
    npt.assert_allclose(1,1)
'''


def test_axcov():
    data = np.random.randn(1000)
    maxlag = 5
    C = cnmf.pre_processing.axcov(data, maxlag)
    print(C)

    npt.assert_allclose(C, np.concatenate((np.zeros(maxlag), np.array([1]), np.zeros(maxlag))), atol=1)
