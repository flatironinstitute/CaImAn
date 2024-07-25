#!/usr/bin/env python

import logging
import numpy as np
import scipy

from scipy.linalg.lapack import dpotrf, dpotrs
from scipy import fftpack

def mode_robust_fast(inputData, axis=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:

        def fnc(x):
            return mode_robust_fast(x)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        data = inputData.ravel()
        # The data need to be sorted for this to work
        data = np.sort(data)
        # Find the mode
        dataMode = _hsm(data)

    return dataMode

def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:

        def fnc(x):
            return mode_robust(x, dtype=dtype)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:

                wMin = np.inf
                N = data.size // 2 + data.size % 2

                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode

def _hsm(data):
    if data.size == 1:
        return data[0]
    elif data.size == 2:
        return data.mean()
    elif data.size == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return data[:2].mean()
        elif i2 > i1:
            return data[1:].mean()
        else:
            return data[1]
    else:

        wMin = np.inf
        N = data.size // 2 + data.size % 2

        for i in range(0, N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i

        return _hsm(data[j:j + N])


def compressive_nmf(A, L, R, r, X=None, Y=None, max_iter=100, ls=0):
    """Implements compressive NMF using an ADMM method as described in 
    Tepper and Shapiro, IEEE TSP 2015
    min_{U,V,X,Y} ||A - XY||_F^2 s.t. U = LX >= 0 and V = YR >=0
    """
    #r_ov = L.shape[1]
    m = L.shape[0]
    n = R.shape[1]
    U = np.random.rand(m, r)
    V = np.random.rand(r, n)
    Y = V.dot(R.T)
    Lam = np.zeros(U.shape)
    Phi = np.zeros(V.shape)
    l = 1
    f = 1
    x = 1
    I = np.eye(r)
    it = 0
    while it < max_iter:
        it += 1
        X = np.linalg.solve(Y.dot(Y.T) + l*I, Y.dot(A.T) + (l*U.T - Lam.T).dot(L)).T
        Y = np.linalg.solve(X.T.dot(X) + f*I, X.T.dot(A) + (f*V - Phi - ls).dot(R.T))
        LX = L.dot(X)
        U = LX + Lam/l
        U = np.where(U>0, U, 0)
        YR = Y.dot(R)
        V = YR + Phi/f
        V = np.where(V>0, V, 0)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - V)
        print(it)

    return X, Y

def mode_robust_kde(inputData, axis=None):
    """
    Extracting the dataset of the mode using kernel density estimation
    """
    if axis is not None:
        def fnc(x):
            return mode_robust_kde(x)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        bandwidth, mesh, density, cdf = kde(inputData)
        dataMode = mesh[np.argamax(density)]

    return dataMode


def df_percentile(inputData, axis=None):
    """
    Extracting the percentile of the data where the mode occurs and its value.
    Used to determine the filtering level for DF/F extraction. Note that
    computation can be inaccurate for short traces.

    If errors occur, return fallback values
    """
    logger = logging.getLogger("caiman")

    if axis is not None:

        def fnc(x):
            return df_percentile(x)

        result = np.apply_along_axis(fnc, axis, inputData)
        data_prct = result[:, 0]
        val = result[:, 1]
    else:
        # Create the function that we can use for the half-sample mode
        err = True
        err_count = 0
        max_err_count = 10
        while err and err_count < max_err_count:
            try:
                bandwidth, mesh, density, cdf = kde(inputData)
                err = False
            except ValueError:
                err_count += 1
                logger.warning(f"kde() failure {err_count}. Concatenating traces and recomputing.")
                if not isinstance(inputData, list):
                    inputData = inputData.tolist()
                inputData += inputData

        # There are three ways the above calculation can go wrong. 
        # For each case: just use median.
        # if cdf was never defined, it means kde never ran without error
        # TODO: Review if we can just look at err instead of this weird try
        try:
            cdf
        except NameError:
            logger.warning('Max err count reached. Reverting to median.')
            data_prct = 50
            val = np.median(np.array(inputData))
        else:
            data_prct = cdf[np.argmax(density)] * 100
            val = mesh[np.argmax(density)]

        if data_prct >= 100 or data_prct < 0:
            logger.warning('Invalid percentile (<0 or >100) computed. Reverting to median.')
            data_prct = 50
            val = np.median(np.array(inputData))

        if np.isnan(data_prct):
            logger.warning('NaN percentile computed. Reverting to median.')
            data_prct = 50
            val = np.median(np.array(inputData))

    return data_prct, val


"""
An implementation of the kde bandwidth selection method outlined in:
Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
Based on the implementation in Matlab by Zdravko Botev:
See: https://www.mathworks.com/matlabcentral/fileexchange/14034-kernel-density-estimator
Daniel B. Smith, PhD
Updated 1-23-2013
"""
def kde(data, N=None, MIN=None, MAX=None):

    # Parameters to set up the mesh on which to calculate
    N = 2**12 if N is None else int(2 ** np.ceil(np.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        MIN = minimum - Range / 10 if MIN is None else MIN
        MAX = maximum + Range / 10 if MAX is None else MAX

    # Range of the data
    R = MAX - MIN

    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = np.histogram(data, bins=N, range=(MIN, MAX))
    DataHist = DataHist / M
    DCTData = fftpack.dct(DataHist, norm=None)

    I = [iN * iN for iN in range(1, N)]
    SqDCTData = (DCTData[1:] / 2)**2

    # The fixed point calculation finds the bandwidth = t_star for smoothing (see paper cited above)
    guess = 0.1
    try:
       t_star = scipy.optimize.brentq(fixed_point, 0, guess, args=(M, I, SqDCTData))
    except ValueError as e:
       # 
       raise ValueError(f"Unable to find root: {str(e)}")

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData * np.exp(-np.arange(N)**2 * np.pi**2 * t_star / 2)
    # Inverse DCT to get density
    density = fftpack.idct(SmDCTData, norm=None) * N / R
    mesh = [(bins[i] + bins[i + 1]) / 2 for i in range(N)]
    bandwidth = np.sqrt(t_star) * R

    density = density / np.trapz(density, mesh)
    cdf = np.cumsum(density) * (mesh[1] - mesh[0])

    return bandwidth, mesh, density, cdf


def fixed_point(t, M, I, a2):
    # TODO: document this: 
    #   From Matlab code: this implements the function t-zeta*gamma^[l](t)
    l = 7
    I = np.float64(I)
    M = np.float64(M)
    a2 = np.float64(a2)
    f = 2 * np.pi ** (2 * l) * np.sum(I**l * a2 * np.exp(-I * np.pi**2 * t))
    for s in range(l, 1, -1):
        K0 = np.prod(range(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2)**(s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f)**(2 / (3 + 2 * s))
        f = 2 * np.pi**(2 * s) * np.sum(I**s * a2 * np.exp(-I * np.pi**2 * time))
    return t - (2 * M * np.sqrt(np.pi) * f)**(-2 / 5)


def csc_column_remove(A, ind):
    """ Removes specified columns for a scipy.sparse csc_matrix
    Args:
        A: scipy.sparse.csc_matrix
            Input matrix
        ind: iterable[int]
            list or np.array with columns to be removed
    """
    logger = logging.getLogger("caiman")
    d1, d2 = A.shape
    if 'csc_matrix' not in str(type(A)): # FIXME
        logger.warning("Original matrix not in csc_format. Converting it anyway.")
        A = scipy.sparse.csc_matrix(A)
    indptr = A.indptr
    ind_diff = np.diff(A.indptr).tolist()
    ind_sort = sorted(ind, reverse=True)
    data_list = [A.data[indptr[i]:indptr[i + 1]] for i in range(d2)]
    indices_list = [A.indices[indptr[i]:indptr[i + 1]] for i in range(d2)]
    for i in ind_sort:
        del data_list[i]
        del indices_list[i]
        del ind_diff[i]
    indptr_final = np.cumsum([0] + ind_diff)
    data_final = [item for sublist in data_list for item in sublist]
    indices_final = [item for sublist in indices_list for item in sublist]
    A = scipy.sparse.csc_matrix((data_final, indices_final, indptr_final), shape=[d1, d2 - len(ind)])
    return A


def pd_solve(a, b):
    """ Fast matrix solve for positive definite matrix a"""
    L, info = dpotrf(a)
    if info == 0:
        return dpotrs(L, b)[0]
    else:
        return np.linalg.solve(a, b)
