""" pure utilitaries(other)

 all of other usefull functions

See Also
------------
https://docs.python.org/2/library/urllib.html

"""
#\package Caiman/utils
#\version   1.0
#\bug
#\warning
#\copyright GNU General Public License v2.0
#\date Created on Tue Jun 30 21:01:17 2015
#\author: andrea giovannucci
#\namespace utils
#\pre none


from __future__ import print_function


import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter
try:
    from urllib2 import urlopen as urlopen
except:
    from urllib.request import urlopen as urlopen
try:  # python2
    import cPickle as pickle
except ImportError:  # python3
    import pickle


def download_demo(name='Sue_2x_3000_40_-46.tif', save_folder=''):
    """download a file from the file list with the url of its location


    using urllib, you can add you own name and location in this global parameter

        Parameters:
        -----------

        name: str
            the path of the file correspondong to a file in the filelist (''Sue_2x_3000_40_-46.tif' or 'demoMovieJ.tif')

        save_folder: str
            folder inside ./example_movies to which the files will be saved. Will be created if it doesn't exist

    Raise:
    ---------
        WrongFolder Exception


    """

    #\bug
    #\warning

    file_dict = {'Sue_2x_3000_40_-46.tif': 'https://www.dropbox.com/s/09z974vkeg3t5gn/Sue_2x_3000_40_-46.tif?dl=1',
                 'demoMovieJ.tif': 'https://www.dropbox.com/s/8j1cnqubye3asmu/demoMovieJ.tif?dl=1',
                 'demo_behavior.h5': 'https://www.dropbox.com/s/53jmhc9sok35o82/movie_behavior.h5?dl=1',
                 'Tolias_mesoscope_1.hdf5': 'https://www.dropbox.com/s/t1yt35u0x72py6r/Tolias_mesoscope_1.hdf5?dl=1',
                 'Tolias_mesoscope_2.hdf5': 'https://www.dropbox.com/s/i233b485uxq8wn6/Tolias_mesoscope_2.hdf5?dl=1',
                 'Tolias_mesoscope_3.hdf5': 'https://www.dropbox.com/s/4fxiqnbg8fovnzt/Tolias_mesoscope_3.hdf5?dl=1'}
    #          ,['./example_movies/demoMovie.tif','https://www.dropbox.com/s/obmtq7305ug4dh7/demoMovie.tif?dl=1']]
    base_folder = './example_movies'
    if os.path.exists(base_folder):
        if not os.path.isdir(os.path.join(base_folder, save_folder)):
            os.makedirs(os.path.join(base_folder, save_folder))
        path_movie = os.path.join(base_folder, save_folder, name)
        if not os.path.exists(path_movie):
            url = file_dict[name]
            print("downloading " + name + "with urllib")
            f = urlopen(url)
            data = f.read()
            with open(path_movie, "wb") as code:
                code.write(data)
        else:

            print("File already downloaded")
    else:

        raise Exception('You must be in caiman folder')
#    print("downloading with requests")
#    r = requests.get(url)
#    with open("code3.tif", "wb") as code:
#        code.write(r.content)


def val_parse(v):
    """parse values from si tags into python objects if possible from si parse

     Parameters:
     -----------

     v: si tags

     returns:
     -------

    v: python object

    """

    try:

        return eval(v)

    except:

        if v == 'true':
            return True
        elif v == 'false':
            return False
        elif v == 'NaN':
            return np.nan
        elif v == 'inf' or v == 'Inf':
            return np.inf

        else:
            return v


def si_parse(imd):
    """parse image_description field embedded by scanimage from get iamge description

     Parameters:
     -----------

     imd: image description

     returns:
     -------

    imd: the parsed description

    """

    imd = imd.split('\n')
    imd = [i for i in imd if '=' in i]
    imd = [i.split('=') for i in imd]
    imd = [[ii.strip(' \r') for ii in i] for i in imd]
    imd = {i[0]: val_parse(i[1]) for i in imd}
    return imd


def get_image_description_SI(fname):
    """Given a tif file acquired with Scanimage it returns a dictionary containing the information in the image description field

     Parameters:
     -----------

     fname: name of the file

     returns:
     -------

        image_description: information of the image

    Raise:
    -----
        ('tifffile package not found, using skimage.external.tifffile')


    """

    image_descriptions = []

    try:
        # todo check this unresolved reference
        from tifffile import TiffFile

    except:

        print('tifffile package not found, using skimage.external.tifffile')
        from skimage.external.tifffile import TiffFile

    tf = TiffFile(fname)

    for idx, pag in enumerate(tf.pages):
        if idx % 1000 == 0:
            print(idx)
    #        i2cd=si_parse(pag.tags['image_description'].value)['I2CData']
        field = pag.tags['image_description'].value

        image_descriptions.append(si_parse(field))

    return image_descriptions


#%% Generate data
def gen_data(dims=(48, 48), N=10, sig=(3, 3), tau=1., noise=.3, T=2000,
             framerate=30, firerate=.5, seed=3, cmap=False, truncate=np.exp(-2),
             difference_of_Gaussians=True, fluctuating_bkgrd=[50, 300]):
    bkgrd = 10  # fluorescence baseline
    np.random.seed(seed)
    boundary = 4
    M = int(N * 1.5)
    # centers = boundary + (np.array(GeneralizedHalton(2, seed).get(M)) *
    #                       (np.array(dims) - 2 * boundary)).astype('uint16')
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
        import matplotlib.pyplot as plt
        import caiman as cm
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


#%%
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input_obj:
        obj = pickle.load(input_obj)
    return obj


def downscale(Y, ds):
    """downscaling without zero padding
    faster version of skimage.transform._warps.block_reduce(Y, ds, np.nanmean, np.nan)"""
    size = np.ceil(np.array(Y.shape) / np.array(ds, dtype=float)).astype(int) * np.array(ds)
    tmp = np.nan * np.zeros(size, dtype=Y.dtype)
    if Y.ndim == 2:
        d = Y.shape
        tmp[:d[0], :d[1]] = Y
        return np.nanmean(np.nanmean(
            tmp.reshape(size[0] / ds[0], ds[0], size[1] / ds[1], ds[1]), 1), 2)
    elif Y.ndim == 3:
        d = Y.shape
        tmp[:d[0], :d[1], :d[2]] = Y
        return np.nanmean(np.nanmean(np.nanmean(
            tmp.reshape(size[0] / ds[0], ds[0], size[1] / ds[1], ds[1],
                        size[2] / ds[2], ds[2]), 1), 2), 3)
    elif Y.ndim == 4:
        d = Y.shape
        tmp[:d[0], :d[1], :d[2], :d[3]] = Y
        return np.nanmean(np.nanmean(np.nanmean(np.nanmean(
            tmp.reshape(size[0] / ds[0], ds[0], size[1] / ds[1], ds[1],
                        size[2] / ds[2], ds[2], size[3] / ds[3], ds[3]), 1), 2), 3), 4)
    else:
        raise NotImplementedError
