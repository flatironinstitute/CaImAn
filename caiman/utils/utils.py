#!/usr/bin/env python

""" pure utilities (other)

generally useful functions for CaImAn

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

import cv2
import logging
import h5py

import numpy as np
import os
import pickle
import scipy
from scipy.ndimage.filters import gaussian_filter
from tifffile import TiffFile

try:
    cv2.setNumThreads(0)
except:
    pass

# TODO: Simplify conditional imports below
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
from ..external.cell_magic_wand import cell_magic_wand
from ..source_extraction.cnmf.spatial import threshold_components

from caiman.paths import caiman_datadir

#%%


def download_demo(name='Sue_2x_3000_40_-46.tif', save_folder=''):
    """download a file from the file list with the url of its location


    using urllib, you can add you own name and location in this global parameter

        Args:
            name: str
                the path of the file correspondong to a file in the filelist (''Sue_2x_3000_40_-46.tif' or 'demoMovieJ.tif')
    
            save_folder: str
                folder inside ./example_movies to which the files will be saved. Will be created if it doesn't exist

    Raise:
        WrongFolder Exception
    """

    #\bug
    #\warning

    file_dict = {'Sue_2x_3000_40_-46.tif': 'https://www.dropbox.com/s/09z974vkeg3t5gn/Sue_2x_3000_40_-46.tif?dl=1',
                 'demoMovieJ.tif': 'https://www.dropbox.com/s/8j1cnqubye3asmu/demoMovieJ.tif?dl=1',
                 'demo_behavior.h5': 'https://www.dropbox.com/s/53jmhc9sok35o82/movie_behavior.h5?dl=1',
                 'Tolias_mesoscope_1.hdf5': 'https://www.dropbox.com/s/t1yt35u0x72py6r/Tolias_mesoscope_1.hdf5?dl=1',
                 'Tolias_mesoscope_2.hdf5': 'https://www.dropbox.com/s/i233b485uxq8wn6/Tolias_mesoscope_2.hdf5?dl=1',
                 'Tolias_mesoscope_3.hdf5': 'https://www.dropbox.com/s/4fxiqnbg8fovnzt/Tolias_mesoscope_3.hdf5?dl=1',
                 'data_endoscope.tif': 'https://www.dropbox.com/s/dcwgwqiwpaz4qgc/data_endoscope.tif?dl=1'}
    #          ,['./example_movies/demoMovie.tif','https://www.dropbox.com/s/obmtq7305ug4dh7/demoMovie.tif?dl=1']]
    base_folder = os.path.join(caiman_datadir(), 'example_movies')
    if os.path.exists(base_folder):
        if not os.path.isdir(os.path.join(base_folder, save_folder)):
            os.makedirs(os.path.join(base_folder, save_folder))
        path_movie = os.path.join(base_folder, save_folder, name)
        if not os.path.exists(path_movie):
            url = file_dict[name]
            logging.info("downloading " + str(name) + " with urllib")
            f = urlopen(url)
            data = f.read()
            with open(path_movie, "wb") as code:
                code.write(data)
        else:
            logging.info("File " + str(name) + " already downloaded")
    else:
        raise Exception('Cannot find the example_movies folder in your caiman_datadir - did you make one with caimanmanager.py?')
    return path_movie

def val_parse(v):
    """parse values from si tags into python objects if possible from si parse

     Args:
         v: si tags

     Returns:
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

     Args:
         imd: image description

    Returns:
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

     Args:
         fname: name of the file
     Returns:
        image_description: information of the image
    """

    image_descriptions = []

    tf = TiffFile(fname)

    for idx, pag in enumerate(tf.pages):
        if idx % 1000 == 0:
            logging.debug(idx)
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
    while keep.sum() > N:
        keep[np.argmax(overlap * np.outer(keep, keep)) % M] = False
    trueA = trueA[:, keep]
    trueS = np.random.rand(N, T) < firerate / float(framerate)
    trueS[:, 0] = 0
    for i in range(N // 2):
        trueS[i, :500 + i * T // N * 2 // 3] = 0
    trueC = trueS.astype('float32')
    for i in range(N):
        # * (.9 + .2 * np.random.rand())))
        gamma = np.exp(-1. / (tau * framerate))
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
            np.outer(
                *ch.dot(np.random.randn(dims[0], 2)).T).ravel().astype('float32')
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
        plt.imshow(Y[:T // 10 * 10].reshape(dims +
                                            (T // 10, 10)).mean(-1).max(-1), cmap=cmap)
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
#%%
def apply_magic_wand(A, gSig, dims, A_thr=None, coms=None, dview=None,
                     min_frac=0.7, max_frac=1.0, roughness=2, zoom_factor=1,
                     center_range=2):
    """ Apply cell magic Wand to results of CNMF to ease matching with labels

    Args:
        A:
            output of CNMF
    
        gSig: tuple
            input of CNMF (half neuron size)
    
        A_thr:
            thresholded version of A
    
        coms:
            centers of the magic wand
    
        dview:
            for parallelization
    
        min_frac:
            fraction of minimum of gSig to take as minimum size
    
        max_frac:
            multiplier of maximum of gSig to take as maximum size

    Returns:
        masks: ndarray
            binary masks
    """

    if (A_thr is None) and (coms is None):
        import pdb
        pdb.set_trace()
        A_thr = threshold_components(
                        A.tocsc()[:], dims, medw=None, thr_method='max',
                        maxthr=0.2, nrgthr=0.99, extract_cc=True,se=None,
                        ss=None, dview=dview)>0

        coms = [scipy.ndimage.center_of_mass(mm.reshape(dims, order='F')) for
                mm in A_thr.T]

    if coms is None:
        coms = [scipy.ndimage.center_of_mass(mm.reshape(dims, order='F')) for
                mm in A_thr.T]

    min_radius = np.round(np.min(gSig)*min_frac).astype(np.int)
    max_radius = np.round(max_frac*np.max(gSig)).astype(np.int)


    params = []
    for idx in range(A.shape[-1]):
        params.append([A.tocsc()[:,idx].toarray().reshape(dims, order='F'),
            coms[idx], min_radius, max_radius, roughness, zoom_factor, center_range])

    logging.debug(len(params))

    if dview is not None:
        masks = np.array(list(dview.map(cell_magic_wand_wrapper, params)))
    else:
        masks = np.array(list(map(cell_magic_wand_wrapper, params)))

    return masks

def cell_magic_wand_wrapper(params):
      a, com, min_radius, max_radius, roughness, zoom_factor, center_range = params
      msk = cell_magic_wand(a, com, min_radius, max_radius, roughness,
                            zoom_factor, center_range)
      return msk
#%% From https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py


def save_dict_to_hdf5(dic, filename):
    ''' Save dictionary to hdf5 file
    Args:
        dic: dictionary
            input (possibly nested) dictionary
        filename: str
            file name to save the dictionary to (in hdf5 format for now)
    '''

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def load_dict_from_hdf5(filename):
    ''' Load dictionary from hdf5 file

    Args:
        filename: str
            input file to load
    Returns:
        dictionary
    '''

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_save_dict_contents_to_group(h5file, path, dic):
    '''
    Args:
        h5file: hdf5 object
            hdf5 file where to store the dictionary
        path: str
            path within the hdf5 file structure
        dic: dictionary
            dictionary to save
    '''
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")

    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")

    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)

        if key == 'g':
            logging.info(key + ' is an object type')
            item = np.array(list(item))
        if key == 'g_tot':
            item = np.asarray(item, dtype=np.float)
        if key in ['groups', 'idx_tot', 'ind_A', 'Ab_epoch','coordinates','loaded_model', 'optional_outputs','merged_ROIs']:
            logging.info(['groups', 'idx_tot', 'ind_A', 'Ab_epoch', 'coordinates', 'loaded_model', 'optional_outputs', 'merged_ROIs',
                   '** not saved'])
            continue

        if isinstance(item, list):
            item = np.array(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        if isinstance(item, (np.int64, np.float64, str, np.float, float, np.float32,int)):
            h5file[path + key] = item
            if not h5file[path + key].value == item:
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S9')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key].value, item):
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif 'sparse' in str(type(item)):
            logging.info(key + ' is sparse ****')
            h5file[path + key + '/data'] = item.tocsc().data
            h5file[path + key + '/indptr'] = item.tocsc().indptr
            h5file[path + key + '/indices'] = item.tocsc().indices
            h5file[path + key + '/shape'] = item.tocsc().shape
        # other types cannot be saved and will result in an error
        elif item is None or key == 'dview':
            h5file[path + key] = 'NoneType'
        elif key in ['dims','medw', 'sigma_smooth_snmf', 'dxy', 'max_shifts', 'strides', 'overlaps', 'gSig']:
            logging.info(key + ' is a tuple ****')
            h5file[path + key] = np.array(item)
        elif type(item).__name__ in ['CNMFParams', 'Estimates']: # parameter object
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item.__dict__)
        else:
            raise ValueError('Cannot save %s type.' % type(item))


def recursively_load_dict_contents_from_group( h5file, path):
    '''load dictionary from hdf5 object
    Args:
        h5file: hdf5 object
            object where dictionary is stored
        path: str
            path within the hdf5 file
    '''

    ans = {}
    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):
            val_set = np.nan
            if isinstance(item.value, str):
                if item.value == 'NoneType':
                    ans[key] = None
                else:
                    ans[key] = item.value
            elif key in ['dims', 'medw', 'sigma_smooth_snmf', 'dxy', 'max_shifts', 'strides', 'overlaps']:

                if type(item.value) == np.ndarray:
                    ans[key] = tuple(item.value)
                else:
                    ans[key] = item.value
            else:
                if type(item.value) == np.bool_:
                    ans[key] = bool(item.value)
                else:
                    ans[key] = item.value

        elif isinstance(item, h5py._hl.group.Group):
            if key == 'A':
                data =  item[path + key + '/data']
                indices = item[path + key + '/indices']
                indptr = item[path + key + '/indptr']
                shape = item[path + key + '/shape']
                ans[key] = scipy.sparse.csc_matrix((data[:],indices[:],
                    indptr[:]), shape[:])
            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans