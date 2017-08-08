import numpy as np
import matplotlib.pyplot as plt
import caiman as cm
from caiman.source_extraction.cnmf.initialization import greedyROI, initialize_components, greedyROI_corr
from scipy.ndimage.filters import gaussian_filter
from skimage.external.tifffile import imread


#%% function defs for simulated data and plotting
def gen_data(dims=(48, 48), N=10, sig=(3, 3), tau=1., noise=.3, T=2000,
             framerate=30, firerate=.5, seed=3, cmap=False, truncate=np.exp(-2),
             difference_of_Gaussians=True, fluctuating_bkgrd=[50, 300]):
    bkgrd = 10  # fluorescence baseline
    np.random.seed(seed)
    boundary = 4
    M = int(N * 1.5)
    centers = boundary + (np.random.rand(M, 2) *
                          (np.array(dims) - 2 * boundary)).astype('uint16')
    trueA = np.zeros(dims + (M,), dtype='float32')
    for i in range(M):
        trueA[tuple(centers[i]) + (i,)] = 1.
    if difference_of_Gaussians:
        q = .75
        for n in range(M):
            s = (.67 + .33 * np.random.rand(2)) * np.array(sig)
            tmp = gaussian_filter(trueA[:, :, n], s)
            trueA[:, :, n] = np.maximum(tmp - gaussian_filter(trueA[:, :, n], q * s) *
                                        q**2 * (.2 + .6 * np.random.rand()), 0)

    else:
        for n in range(M):
            s = [ss * (.75 + .25 * np.random.rand()) for ss in sig]
            trueA[:, :, n] = gaussian_filter(trueA[:, :, n], s)
    trueA = trueA.reshape((-1, M), order='F')
    trueA *= (trueA >= trueA.max(0) * truncate)
    trueA /= np.linalg.norm(trueA, 2, 0)
    keep = np.ones(M, dtype=bool)
    overlap = trueA.T.dot(trueA) - np.eye(M)
    while(keep.sum() > N):
        keep[np.argmax(overlap * np.outer(keep, keep)) % M] = False
    trueA = trueA[:, keep]
    trueS = np.random.rand(N, T) < firerate / float(framerate)
    trueS[:, 0] = 0
    for i in range(N // 2):
        trueS[i, :500 + i * T // N * 2 // 3] = 0
    trueC = trueS.astype('float32')
    for i in range(N):
        gamma = np.exp(-1. / (tau * framerate))  # * (.9 + .2 * np.random.rand())))
        for t in range(1, T):
            trueC[i, t] += gamma * trueC[i, t - 1]

    if fluctuating_bkgrd:
        K = np.array([[np.exp(-(i - j)**2 / 2. / fluctuating_bkgrd[0]**2)
                       for i in range(T)] for j in range(T)])
        ch = np.linalg.cholesky(K + 1e-10 * np.eye(T))
        truef = 1e-2 * ch.dot(np.random.randn(T)).astype('float32') / bkgrd
        truef -= truef.mean()
        truef += 1
        K = np.array([[np.exp(-(i - j)**2 / 2. / fluctuating_bkgrd[1]**2)
                       for i in range(dims[0])] for j in range(dims[0])])
        ch = np.linalg.cholesky(K + 1e-10 * np.eye(dims[0]))
        trueb = 3 * 1e-2 * \
            np.outer(*ch.dot(np.random.randn(dims[0], 2)).T).ravel().astype('float32')
        trueb -= trueb.mean()
        trueb += 1
    else:
        truef = np.ones(T, dtype='float32')
        trueb = np.ones(np.prod(dims), dtype='float32')
    trueb *= bkgrd
    Yr = np.outer(trueb, truef) + noise * np.random.randn(
        * (np.prod(dims), T)).astype('float32') + trueA.dot(trueC)

    if cmap:
        Y = np.reshape(Yr, dims + (T,), order='F')
        Cn = cm.local_correlations(Y)
        plt.figure(figsize=(20, 3))
        plt.plot(trueC.T)
        plt.figure(figsize=(20, 3))
        plt.plot((trueA.T.dot(Yr - bkgrd) / np.sum(trueA**2, 0).reshape(-1, 1)).T)
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.scatter(*centers[keep].T[::-1], c='g')
        plt.scatter(*centers[~keep].T[::-1], c='r')
        plt.imshow(Y[:T // 10 * 10].reshape(dims + (T // 10, 10)).mean(-1).max(-1), cmap=cmap)
        plt.title('Max')
        plt.subplot(132)
        plt.scatter(*centers[keep].T[::-1], c='g')
        plt.scatter(*centers[~keep].T[::-1], c='r')
        plt.imshow(Y.mean(-1), cmap=cmap)
        plt.title('Mean')
        plt.subplot(133)
        plt.scatter(*centers[keep].T[::-1], c='g')
        plt.scatter(*centers[~keep].T[::-1], c='r')
        plt.imshow(Cn, cmap=cmap)
        plt.title('Correlation')
        plt.show()
    return Yr, trueC, trueS, trueA, trueb, truef, centers, dims


def foo(data, centers, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.imshow(np.log(data.reshape((-1, 10) + data.shape[1:]).mean(1).max(0).T))
    plt.scatter(*np.transpose(centers), c='r')
    plt.title('log(max(.))')
    plt.subplot(122)
    plt.imshow(cm.summary_images.local_correlations_fft(data, swap_dim=False).T)
    plt.scatter(*np.transpose(centers), c='r')
    plt.title('corr')
    plt.show()

#

#

#%% simulated data
tau = 1.
gamma = np.exp(-1. / tau / 30)
noise = .2
gSig = (3.2, 3.2)
Yr, trueC, trueS, trueA, trueb, truef, centers, dims = gen_data(
    tau=tau, noise=noise, sig=gSig, seed=3)
N, T = trueC.shape
Y = Yr.reshape(dims + (-1,))

# greedy ROI
A, C, centers, b, f = greedyROI(Y, nr=N, gSig=[4, 4])
foo(Y.transpose(2, 0, 1), centers)


# greedy ROI corr
A, C, centers, b, f = greedyROI_corr(data=Y, center_psf=False, min_corr=0.8, min_pnr=10,
                                     ring_model=False, nb=1)
foo(Y.transpose(2, 0, 1), centers)

#

#

#%% 2P data

# demo data
data = imread('example_movies/demoMovie.tif')
N = 30

# greedy ROI
centers = greedyROI(data.reshape((-1, 10) + data.shape[1:]).mean(1)
                        .transpose(1, 2, 0), nr=N, gSig=[6, 6])[2]
foo(data, centers, figsize=(10, 6))

# greedy ROI corr
centers = greedyROI_corr(data=data.reshape((-1, 10) + data.shape[1:]).mean(1)
                         .transpose(1, 2, 0), g_size=15, g_sig=5,
                         center_psf=False, min_corr=0.95, min_pnr=15,
                         ring_model=False, nb=1)[2]
foo(data, centers, figsize=(10, 6))

#

#%% Weijan's data
data = imread('../public/multi-scale/180um_20fps_350umX350um.tif')
N = 40

# greedy ROI
centers = greedyROI(data.reshape((-1, 10) + data.shape[1:]).mean(1)
                        .transpose(1, 2, 0), nr=N, gSig=[8, 8])[2].astype(int)
foo(data, centers, figsize=(17, 8))

centers = initialize_components(data.transpose(1, 2, 0),
                                method='greedy_roi', K=N, ssub=2, tsub=10)[-1]
foo(data, centers, figsize=(10, 6))

# greedy ROI corr
centers = greedyROI_corr(data=data.reshape((-1, 10) + data.shape[1:]).mean(1)
                         .transpose(1, 2, 0), max_number=None, g_size=15, g_sig=5,
                         center_psf=False, min_corr=0.95, min_pnr=15,
                         seed_method='auto', deconvolve_options=None,
                         min_pixel=3, bd=1, thresh_init=2,
                         ring_size_factor=1.5, ring_model=False, nb=1)[2]
foo(data, centers, figsize=(17, 8))

# memmaping
# fname_new = cm.save_memmap(['../public/multi-scale/180um_20fps_350umX350um.tif'], base_name='Yr')
# Yr, dims, T = cm.load_memmap(fname_new)
# d1, d2 = dims

# A, C, centers, b, f = greedyROI_corr(data=Yr.reshape(d1, d2, -1), max_number=None, g_size=15, g_sig=5,
#                                      center_psf=False, min_corr=0.95, min_pnr=15,
#                                      seed_method='auto', deconvolve_options=None,
#                                      min_pixel=3, bd=1, thresh_init=2,
#                                      ring_size_factor=1.5, ring_model=False, nb=1)

#

#

#%% 1P data

# demo data
data = imread('example_movies/data_endoscope.tif')
N = 30

# greedy ROI
centers = greedyROI(data.reshape((-1, 10) + data.shape[1:]).mean(1)
                        .transpose(1, 2, 0), nr=N, gSig=[6, 6])[2]
foo(data, centers, figsize=(10, 6))

# greedy ROI corr
centers = greedyROI_corr(data=data.reshape((-1, 10) + data.shape[1:]).mean(1)
                         .transpose(1, 2, 0), max_number=None, g_size=15, g_sig=5,
                         center_psf=True, min_corr=0.9, min_pnr=10,
                         seed_method='auto', deconvolve_options=None,
                         min_pixel=3, bd=1, thresh_init=2,
                         ring_size_factor=1.5, ring_model=False, nb=1)[2]
foo(data, centers, figsize=(10, 6))

#

#

#%%  zebrafish data
data = np.load('../public/multi-scale/data_zebrafish.npy')
N = 46

# greedy ROI
centers = greedyROI(data.reshape((-1, 30) + data.shape[1:]).mean(1)
                        .transpose(1, 2, 0), nr=N, gSig=[4, 4])[2].astype(int)
foo(data, centers)

# greedy ROI corr
A, C, centers, b, f = greedyROI_corr(data=data.transpose(1, 2, 0),
                                     max_number=None, g_size=15, g_sig=5,
                                     center_psf=False, min_corr=0.95, min_pnr=15,
                                     seed_method='auto', deconvolve_options=None,
                                     min_pixel=3, bd=1, thresh_init=2,
                                     ring_size_factor=1.5, ring_model=False, nb=1)
foo(data, centers)
