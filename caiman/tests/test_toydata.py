from caiman.source_extraction import cnmf as cnmf
import numpy.testing as npt
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def gen_data(D=3, noise=.5, T=300, framerate=30, firerate=2.):
    N = 4  # number of neurons
    dims = [(20, 30), (12, 14, 16)][D - 2]  # size of image
    sig = (2, 2, 2)[:D]  # neurons size
    bkgrd = 10  # fluorescence baseline
    gamma = .9  # calcium decay time constant
    np.random.seed(5)
    centers = np.asarray([[np.random.randint(4, x - 4) for x in dims] for i in range(N)])
    trueA = np.zeros(dims + (N,), dtype=np.float32)
    trueS = np.random.rand(N, T) < firerate / float(framerate)
    trueS[:, 0] = 0
    trueC = trueS.astype(np.float32)
    for i in range(1, T):
        trueC[:, i] += gamma * trueC[:, i - 1]
    for i in range(N):
        trueA[tuple(centers[i]) + (i,)] = 1.
    tmp = np.zeros(dims)
    tmp[tuple(d / 2 for d in dims)] = 1.
    z = np.linalg.norm(gaussian_filter(tmp, sig).ravel())
    trueA = 10 * gaussian_filter(trueA, sig + (0,)) / z
    Yr = bkgrd + noise * np.random.randn(*(np.prod(dims), T)) + \
        trueA.reshape((-1, 4), order='F').dot(trueC)
    return Yr, trueC, trueS, trueA, centers, dims


def pipeline(D):
    # GENERATE GROUND TRUTH DATA
    Yr, trueC, trueS, trueA, centers, dims = gen_data(D)
    N, T = trueC.shape
    Y = np.reshape(Yr, dims + (-1,), order='F')
    n_processes = 2
    K = 4  # number of neurons expected per patch
    gSig = [2, 2, 2]  # expected half size of neurons
    p = 1  # order of the autoregressive system
    options = cnmf.utilities.CNMFSetParms(Y, n_processes, p=p, gSig=gSig, K=K)
    options['spatial_params']['thr_method'] = 'nrg'
    options['spatial_params']['extract_cc'] = False
    options['temporal_params']['method'] = 'oasis'

    # PREPROCESS DATA AND INITIALIZE COMPONENTS
    Yr, sn, g, psx = cnmf.pre_processing.preprocess_data(
        Yr, dview=None, **options['preprocess_params'])
    Ain, Cin, b_in, f_in, center = cnmf.initialization.initialize_components(
        Y, **options['init_params'])

    # UPDATE SPATIAL COMPONENTS
    A, b, Cin = cnmf.spatial.update_spatial_components(
        Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])

    # UPDATE TEMPORAL COMPONENTS
    C, f, S, bl, c1, neurons_sn, g, YrA = cnmf.temporal.update_temporal_components(
        Yr, A, b, Cin, f_in, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

    # VERIFY HIGH CORRELATION WITH GROUND TRUTH
    sorting = [np.argmax([np.corrcoef(tc, c)[0, 1] for tc in trueC]) for c in C]
    # verifying the temporal components
    corr = [np.corrcoef(trueC[sorting[i]], C[i])[0, 1] for i in range(N)]
    npt.assert_allclose(corr, 1, .05)
    # verifying the spatial components
    corr = [np.corrcoef(np.reshape(trueA, (-1, 4), order='F')[:, sorting[i]],
                        A.toarray()[:, i])[0, 1] for i in range(N)]
    npt.assert_allclose(corr, 1, .05)


def test_2D():
    pipeline(2)


def test_3D():
    pipeline(3)
