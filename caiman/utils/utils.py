#!/usr/bin/env python

""" pure utilities (other)

generally useful functions for CaImAn

See Also
------------
https://docs.python.org/3/library/urllib.request.htm

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
import h5py
import multiprocessing
import inspect
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy
import subprocess
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from tifffile import TiffFile
from typing import Any, Dict, List, Tuple, Union, Iterable

try:
    cv2.setNumThreads(0)
except:
    pass

from urllib.request import urlopen

from ..external.cell_magic_wand import cell_magic_wand
from ..source_extraction.cnmf.spatial import threshold_components
from caiman.paths import caiman_datadir
import caiman.utils

#%%


def download_demo(name:str='Sue_2x_3000_40_-46.tif', save_folder:str='') -> str:
    """download a file from the file list with the url of its location


    using urllib, you can add you own name and location in this global parameter

        Args:
            name: str
                the path of the file correspondong to a file in the filelist (''Sue_2x_3000_40_-46.tif' or 'demoMovieJ.tif')
    
            save_folder: str
                folder inside ./example_movies to which the files will be saved. Will be created if it doesn't exist
        Returns:
            Path of the saved file
    Raise:
        WrongFolder Exception
    """

    #\bug
    #\warning

    file_dict = {'Sue_2x_3000_40_-46.tif': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/Sue_2x_3000_40_-46.tif',
                 'demoMovieJ.tif': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demoMovieJ.tif',
                 'demo_behavior.h5': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demo_behavior.h5',
                 'k53.tif': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/k53.tif',
                 'k53_ROIs.hdf5': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/k53_ROIs.hdf5',
                 'Tolias_mesoscope_1.hdf5': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/Tolias_mesoscope_1.hdf5',
                 'Tolias_mesoscope_2.hdf5': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/Tolias_mesoscope_2.hdf5',
                 'Tolias_mesoscope_3.hdf5': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/Tolias_mesoscope_3.hdf5',
                 'data_endoscope.tif': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/data_endoscope.tif',
                 'gmc_960_30mw_00001_red.tif': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/gmc_960_30mw_00001_red.tif',
                 'gmc_960_30mw_00001_green.tif': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/gmc_960_30mw_00001_green.tif',
                 'msCam13.avi': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/msCam13.avi',
                 'alignment.pickle': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/alignment.pickle',
                 'data_dendritic.tif': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/2014-04-05-003.tif',
                 'blood_vessel_10Hz.mat': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/blood_vessel_10Hz.mat',
                 'online_vs_offline.npz': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/online_vs_offline.npz',
                 'demo_voltage_imaging_ROIs.hdf5': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demo_voltage_imaging_ROIs.hdf5',
                 'demo_voltage_imaging.hdf5': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demo_voltage_imaging.hdf5'}
    #          ,['./example_movies/demoMovie.tif','https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demoMovie.tif']]
    base_folder = os.path.join(caiman_datadir(), 'example_movies')
    if os.path.exists(base_folder):
        if not os.path.isdir(os.path.join(base_folder, save_folder)):
            os.makedirs(os.path.join(base_folder, save_folder))
        path_movie = os.path.join(base_folder, save_folder, name)
        if not os.path.exists(path_movie):
            url = file_dict[name]
            logging.info(f"downloading {name} with urllib")
            logging.info(f"GET {url} HTTP/1.1")
            try:
                f = urlopen(url)
            except:
                logging.info(f"Trying to set user agent to download demo")
                from urllib.request import Request
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                f = urlopen(req)
                
                
            data = f.read()
            with open(path_movie, "wb") as code:
                code.write(data)
        else:
            logging.info("File " + str(name) + " already downloaded")
    else:
        raise Exception('Cannot find the example_movies folder in your caiman_datadir - did you make one with caimanmanager.py?')
    return path_movie


def download_model(name:str='mask_rcnn', save_folder:str='') -> str:
    """download a NN model from the file list with the url of its location


    using urllib, you can add you own name and location in this global parameter

        Args:
            name: str
                the path of the file correspondong to a file in the filelist
    
            save_folder: str
                folder inside caiman_data/model to which the files will be saved. Will be created if it doesn't exist
        Returns:
            Path of the saved file
    Raise:
        WrongFolder Exception
    """

    #\bug
    #\warning

    file_dict = {'mask_rcnn': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/model/mask_rcnn_neurons_0040.h5'}
    base_folder = os.path.join(caiman_datadir(), 'model')
    if os.path.exists(base_folder):
        if not os.path.isdir(os.path.join(base_folder, save_folder)):
            os.makedirs(os.path.join(base_folder, save_folder))
        path_movie = os.path.join(base_folder, save_folder, name)
        if not os.path.exists(path_movie):
            url = file_dict[name]
            logging.info(f"downloading {name} with urllib")
            logging.info(f"GET {url} HTTP/1.1")
            try:
                f = urlopen(url)
            except:
                logging.info(f"Trying to set user agent to download demo")
                from urllib.request import Request
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                f = urlopen(req)
                                
            data = f.read()
            with open(path_movie, "wb") as code:
                code.write(data)
        else:
            logging.info("File " + str(name) + " already downloaded")
    else:
        raise Exception('Cannot find the model folder in your caiman_datadir - did you make one with caimanmanager.py?')
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


def si_parse(imd:str) -> Dict:
    """parse image_description field embedded by scanimage from get image description

     Args:
         imd: image description

    Returns:
        imd: the parsed description

    """

    imddata:Any = imd.split('\n')
    imddata = [i for i in imddata if '=' in i]
    imddata = [i.split('=') for i in imddata]
    imddata = [[ii.strip(' \r') for ii in i] for i in imddata]
    imddata = {i[0]: val_parse(i[1]) for i in imddata}
    return imddata


def get_image_description_SI(fname:str) -> List:
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
            logging.debug(idx) # progress report to the user
        field = pag.tags['image_description'].value

        image_descriptions.append(si_parse(field))

    return image_descriptions


#%% Generate data
def gen_data(dims:Tuple[int,int]=(48, 48), N:int=10, sig:Tuple[int,int]=(3, 3), tau:float=1., noise:float=.3, T:int=2000,
             framerate:int=30, firerate:float=.5, seed:int=3, cmap:bool=False, truncate:float=np.exp(-2),
             difference_of_Gaussians:bool=True, fluctuating_bkgrd:List=[50, 300]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
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
    return Yr, trueC, trueS, trueA, trueb, truef, centers, dims # XXX dims is always the same as passed into the function?


#%%
def save_object(obj, filename:str) -> None:
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename:str) -> Any:
    with open(filename, 'rb') as input_obj:
        obj = pickle.load(input_obj)
    return obj
#%%
def apply_magic_wand(A, gSig, dims, A_thr=None, coms=None, dview=None,
                     min_frac=0.7, max_frac=1.0, roughness=2, zoom_factor=1,
                     center_range=2) -> np.ndarray:
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


def save_dict_to_hdf5(dic:Dict, filename:str, subdir:str='/') -> None:
    ''' Save dictionary to hdf5 file
    Args:
        dic: dictionary
            input (possibly nested) dictionary
        filename: str
            file name to save the dictionary to (in hdf5 format for now)
    '''

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, subdir, dic)

def load_dict_from_hdf5(filename:str) -> Dict:
    ''' Load dictionary from hdf5 file

    Args:
        filename: str
            input file to load
    Returns:
        dictionary
    '''

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_save_dict_contents_to_group(h5file:h5py.File, path:str, dic:Dict) -> None:
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
            if item is None:
                item = 0
            logging.info(key + ' is an object type')
            try:
                item = np.array(list(item))
            except:
                item = np.asarray(item, dtype=np.float)
        if key == 'g_tot':
            item = np.asarray(item, dtype=np.float)
        if key in ['groups', 'idx_tot', 'ind_A', 'Ab_epoch', 'coordinates',
                   'loaded_model', 'optional_outputs', 'merged_ROIs', 'tf_in',
                   'tf_out', 'empty_merged']:
            logging.info(f'Key {key} is not saved')
            continue

        if isinstance(item, (list, tuple)):
            if len(item) > 0 and all(isinstance(elem, str) for elem in item):
                item = np.string_(item)
            else:
                item = np.array(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, numpy.int32, and numpy.float64 types
        if isinstance(item, (np.int64, np.int32, np.float64, str, np.float, float, np.float32, int)):
            h5file[path + key] = item
            logging.debug(f'Saving numeric or string {key}')
            if not h5file[path + key][()] == item:
                raise ValueError(f'Error while saving numeric or string {key}: assigned value {h5file[path + key][()]} does not match intended value {item}')
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            logging.debug(f'Saving {key}')
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S32')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key][()], item):
                raise ValueError(f'Error while saving ndarray {key}')
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
        elif key in ['dims', 'medw', 'sigma_smooth_snmf', 'dxy', 'max_shifts',
                     'strides', 'overlaps', 'gSig']:
            logging.info(key + ' is a tuple ****')
            h5file[path + key] = np.array(item)
        elif type(item).__name__ in ['CNMFParams', 'Estimates']: #  parameter object
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item.__dict__)
        else:
            raise ValueError(f"Cannot save {type(item)} type for key '{key}'.")


def recursively_load_dict_contents_from_group(h5file:h5py.File, path:str) -> Dict:
    '''load dictionary from hdf5 object
    Args:
        h5file: hdf5 object
            object where dictionary is stored
        path: str
            path within the hdf5 file
    '''

    ans:Dict = {}
    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):
            val_set = np.nan
            if isinstance(item[()], str):
                if item[()] == 'NoneType':
                    ans[key] = None
                else:
                    ans[key] = item[()]

            elif key in ['dims', 'medw', 'sigma_smooth_snmf', 'dxy', 'max_shifts', 'strides', 'overlaps']:

                if isinstance(item[()], np.ndarray):
                    ans[key] = tuple(item[()])
                else:
                    ans[key] = item[()]
            else:
                if isinstance(item[()], np.bool_): # sigh
                    ans[key] = bool(item[()])
                else:
                    ans[key] = item[()]

        elif isinstance(item, h5py._hl.group.Group):
            if key in ('A', 'W', 'Ab', 'downscale_matrix', 'upscale_matrix'):
                data =  item[path + key + '/data']
                indices = item[path + key + '/indices']
                indptr = item[path + key + '/indptr']
                shape = item[path + key + '/shape']
                ans[key] = scipy.sparse.csc_matrix((data[:], indices[:],
                    indptr[:]), shape[:])
                if key in ('W', 'upscale_matrix'):
                    ans[key] = ans[key].tocsr()
            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def load_graph(frozen_graph_filename):
    """ Load a tensorflow .pb model and use it for inference"""
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a
    # graph_def into the current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            producer_op_list=None
        )
    return graph

def get_caiman_version() -> Tuple[str, str]:
    """ Get the version of CaImAn, as best we can determine"""
    # This does its best to determine the version of CaImAn. This uses the first successful
    # from these methods:
    # 'GITW' ) git rev-parse if caiman is built from "pip install -e ." and we are working
    #    out of the checkout directory (the user may have since updated without reinstall)
    # 'RELF') A release file left in the process to cut a release. Should have a single line
    #    in it whick looks like "Version:1.4"
    # 'FILE') The date of some frequently changing files, which act as a very rough
    #    approximation when no other methods are possible
    #
    # Data is returned as a tuple of method and version, with method being the 4-letter string above
    # and version being a format-dependent string

    # Attempt 'GITW'.
    # TODO:
    # A) Find a place to do it that's better than cwd
    # B) Hide the output from the terminal
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").split("\n")[0]
    except:
        rev = None
    if rev is not None:
        return 'GITW', rev

    # Attempt: 'RELF'
    relfile = os.path.join(caiman_datadir(), 'RELEASE')
    if os.path.isfile(relfile):
        with open(relfile, 'r') as sfh:
            for line in sfh:
                if ':' in line: # expect a line like "Version:1.3"
                    _, version = line.rstrip().split(':')
                    return 'RELF', version 

    # Attempt: 'FILE'
    # Right now this samples the utils directory
    modpath = os.path.dirname(inspect.getfile(caiman.utils)) # Probably something like /mnt/home/pgunn/miniconda3/envs/caiman/lib/python3.7/site-packages/caiman
    newest = 0
    for fn in os.listdir(modpath):
        last_modified = os.stat(os.path.join(modpath, fn)).st_mtime
        if last_modified > newest:
            newest = last_modified
    return 'FILE', str(int(newest))

