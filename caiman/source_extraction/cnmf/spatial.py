"""
Created on Wed Aug 05 20:38:27 2015

# -*- coding: utf-8 -*-
@author: agiovann
"""
import numpy as np
#from scipy.sparse import coo_matrix as coom
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags
from scipy.linalg import eig
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
from sklearn.decomposition import NMF
from warnings import warn
import scipy
import time
import tempfile
import os
import shutil

# try:
#    import picos
# except:
#    print 'picos not installed'

try:
    from cvxopt import matrix, spmatrix, spdiag
    from cvxopt import solvers
except:
    print 'cvxopt not installed'

import pylab as pl
from caiman.mmapping import load_memmap
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_closing
from scipy.ndimage.measurements import label
#%%

#import sys
# sys.path
# sys.path.append(path_to_folder)


def basis_denoising(y, c, boh, sn, id2_, px):
    if np.size(c) > 0:
        _, _, a, _, _ = lars_regression_noise(y, c, 1, sn)
    else:
        return (None, None, None)
    return a, px, id2_
#%% update_spatial_components (in parallel)


def update_spatial_components(Y, C=None, f=None, A_in=None, sn=None, dims=None, min_size=3, max_size=8, dist=3,normalize_yyt_one=True,
                              method='ellipse', expandCore=None, dview=None, n_pixels_per_process=128,
                              medw=(3, 3), thr_method='nrg', maxthr=0.1, nrgthr=0.9999, extract_cc=True,
                              se=np.ones((3, 3), dtype=np.int), ss=np.ones((3, 3), dtype=np.int), nb=1, method_ls='nnls_L0'):
    
    """update spatial footprints and background through Basis Pursuit Denoising 

    for each pixel i solve the problem
        [A(i,:),b(i)] = argmin sum(A(i,:))
    subject to
        || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

    for each pixel the search is limited to a few spatial components

    Parameters
    ----------
    Y: np.ndarray (2D or 3D)
        movie, raw data in 2D or 3D (pixels x time).
    C: np.ndarray
        calcium activity of each neuron.
    f: np.ndarray
        temporal profile  of background activity.
    A_in: np.ndarray
        spatial profile of background activity. If A_in is boolean then it defines the spatial support of A. 
        Otherwise it is used to determine it through determine_search_location

    dims: [optional] tuple
        x, y[, z] movie dimensions

    min_size: [optional] int

    max_size: [optional] int

    dist: [optional] int


    sn: [optional] float
        noise associated with each pixel if known

    backend [optional] str
        'ipyparallel', 'single_thread'
        single_thread:no parallelization. It can be used with small datasets.
        ipyparallel: uses ipython clusters and then send jobs to each of them
        SLURM: use the slurm scheduler

    n_pixels_per_process: [optional] int
        number of pixels to be processed by each thread


    method: [optional] string
        method used to expand the search for pixels 'ellipse' or 'dilate'

    expandCore: [optional]  scipy.ndimage.morphology
        if method is dilate this represents the kernel used for expansion

    dview: view on ipyparallel client
            you need to create an ipyparallel client and pass a view on the processors (client = Client(), dview=client[:])            

    medw, thr_method, maxthr, nrgthr, extract_cc, se, ss: [optional]
        Parameters for components post-processing. Refer to spatial.threshold_components for more details

    nb: [optional] int
        Number of background components
        
    method_ls:
        method to perform the regression for the basis pursuit denoising.
             'nnls_L0'. Nonnegative least square with L0 penalty        
             'lasso_lars' lasso lars function from scikit learn
             'lasso_lars_old' lasso lars from old implementation, will be deprecated 
        
        normalize_yyt_one: bool
            wheter to norrmalize the C and A matrices so that diag(C*C.T) are ones

    Returns
    --------
    A: np.ndarray
         new estimate of spatial footprints
    b: np.ndarray
        new estimate of spatial background
    C: np.ndarray
         temporal components (updated only when spatial components are completely removed)

    
    """
    if normalize_yyt_one:
#        cct=np.diag(C.dot(C.T))
        nr_C=np.shape(C)[0]
        d = scipy.sparse.lil_matrix((nr_C,nr_C))
        d.setdiag(np.sqrt(np.sum(C**2,1)))
        A_in=A_in*d
        C=C/np.sqrt(np.sum(C**2,1)[:,np.newaxis])   
        
        
    if expandCore is None:
        expandCore = iterate_structure(generate_binary_structure(2, 1), 2).astype(int)

    if dims is None:
        raise Exception('You need to define the input dimensions')

    if Y.ndim < 2 and not type(Y) is str:
        Y = np.atleast_2d(Y)

    if Y.shape[1] == 1:
        raise Exception('Dimension of Matrix Y must be pixels x time')

    if C is not None:
        C = np.atleast_2d(C)
        if C.shape[1] == 1:
            raise Exception('Dimension of Matrix C must be neurons x time')

    if f is not None:
        f = np.atleast_2d(f)
        if f.shape[1] == 1:
            raise Exception('Dimension of Matrix f must be background comps x time ')

    if (A_in is None) and (C is None):
        raise Exception('Either A or C need to be determined')

    if A_in is not None:
        if len(A_in.shape) == 1:
            A_in = np.atleast_2d(A_in).T

        if A_in.shape[0] == 1:
            raise Exception('Dimension of Matrix A must be pixels x neurons ')

    start_time = time.time()

    [d, T] = np.shape(Y)

    if A_in is None:
        A_in = np.ones((d, np.shape(C)[1]), dtype=bool)

    if n_pixels_per_process > d:
        raise Exception(
            'The number of pixels per process (n_pixels_per_process) is larger than the total number of pixels!! Decrease suitably.')

    if f is not None:
        nb = f.shape[0]
    else:
        if b is not None:
            nb = b.shape[1]

    if A_in.dtype == bool:
        IND = A_in.copy()
        print "spatial support for each components given by the user"
        if C is None:
            INDav = IND.astype('float32') / np.sum(IND, axis=0)
            px = (np.sum(IND, axis=1) > 0)
            model = NMF(n_components=nb, init='random', random_state=0)
            b = model.fit_transform(np.maximum(Y[~px, :], 0))
            f = model.components_.squeeze()
            #f = np.mean(Y[~px,:],axis=0)
            Y_resf = np.dot(Y, f.T)
            b = np.fmax(Y_resf.dot(np.linalg.inv(f.dot(f.T)), 0))
            #b = np.fmax(Y_resf / scipy.linalg.norm(f)**2, 0)
            C = np.fmax(csr_matrix(INDav.T).dot(Y) - np.outer(INDav.T.dot(b), f), 0)
            f = np.atleast_2d(f)

    else:
        IND = determine_search_location(
            A_in, dims, method=method, min_size=min_size, max_size=max_size, dist=dist, expandCore=expandCore, dview=dview)
        print "found spatial support for each component"
        if C is None:
            raise Exception('You need to provide estimate of C and f')

    print np.shape(A_in)

    Cf = np.vstack((C, f))  # create matrix that include background components
    nr, _ = np.shape(C)       # number of neurons

    ind2_ = [np.hstack((np.where(iid_)[0], nr + np.arange(f.shape[0])))
             if np.size(np.where(iid_)[0]) > 0 else [] for iid_ in IND]

    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        tmpf = os.environ.get('SLURM_SUBMIT_DIR')
        print 'cluster temporary folder:' + tmpf
        folder = tempfile.mkdtemp(dir=tmpf)
    else:
        folder = tempfile.mkdtemp()

    
    if dview is None:

        Y_name = Y
        C_name = Cf

    else:
        
        C_name = os.path.join(folder, 'C_temp.npy')
        np.save(C_name, Cf)
    
        if type(Y) is np.core.memmap:  # if input file is already memory mapped then find the filename
            Y_name = Y.filename
        # if not create a memory mapped version (necessary for parallelization)
        elif type(Y) is str or dview is None:
            Y_name = Y
        else:
            raise Exception('Not implemented consistently')
            Y_name = os.path.join(folder, 'Y_temp.npy')
            np.save(Y_name, Y)
            Y, _, _, _ = load_memmap(Y_name)

    # create arguments to be passed to the function. Here we are grouping
    # bunch of pixels to be processed by each thread
#    pixel_groups = [(Y_name, C_name, sn, ind2_, range(i, i + n_pixels_per_process))
# for i in range(0, np.prod(dims) - n_pixels_per_process + 1,
# n_pixels_per_process)]
    cct=np.diag(C.dot(C.T))
    rank_f=nb
    pixel_groups = []
    for i in range(0, np.prod(dims) - n_pixels_per_process + 1, n_pixels_per_process):
        pixel_groups.append([Y_name, C_name, sn, ind2_, range(i, i + n_pixels_per_process), method_ls, cct,rank_f])

    if i < np.prod(dims):
        pixel_groups.append([Y_name, C_name, sn, ind2_, range(i, np.prod(dims)), method_ls, cct,rank_f])

    A_ = np.zeros((d, nr + np.size(f, 0)))

    #serial_result = map(lars_regression_noise_ipyparallel, pixel_groups)
    if dview is not None:
        parallel_result = dview.map_sync(regression_ipyparallel, pixel_groups)
        dview.results.clear()
        for chunk in parallel_result:
            for pars in chunk:
                px, idxs_, a = pars
                A_[px, idxs_] = a
    else:
        parallel_result = map(regression_ipyparallel, pixel_groups)
        for chunk in parallel_result:
            for pars in chunk:
                px, idxs_, a = pars
                A_[px, idxs_] = a
##
#        Cf_ = [Cf[idx_, :] for idx_ in ind2_]
#
#        #% LARS regression
#        A_ = np.hstack((np.zeros((d, nr)), np.zeros((d, np.size(f, 0)))))
#        
#        for c, y, s, id2_, px in zip(Cf_, Y, sn, ind2_, range(d)):
#            if px % 1000 == 0:
#                print px
#            if np.size(c) > 0:
#                _, _, a, _, _ = lars_regression_noise_old(y, np.array(c.T), 1, sn[px]**2 * T)
#                if np.isscalar(a):
#                    A_[px, id2_] = a
#                else:
#                    A_[px, id2_] = a.T
##

    #%
    print 'Updated Spatial Components'

    A_ = threshold_components(A_, dims, dview=dview, medw=(3, 3), thr_method=thr_method, maxthr=maxthr, nrgthr=nrgthr, extract_cc=extract_cc,
                              se=se, ss=ss)

    print "threshold"
    ff = np.where(np.sum(A_, axis=0) == 0)           # remove empty components
    if np.size(ff) > 0:
        ff = ff[0]
        print('eliminating empty components!!')
        nr = nr - len(ff)
        A_ = np.delete(A_, list(ff), 1)
        C = np.delete(C, list(ff), 0)

    A_ = A_[:, :nr]
    A_ = coo_matrix(A_)

    #import pdb
    # pdb.set_trace()
    Y_resf = np.dot(Y, f.T) - A_.dot(coo_matrix(C[:nr, :]).dot(f.T))
    print "Computing A_bas"
    A_bas = np.fmax(Y_resf.dot(np.linalg.inv(f.dot(f.T))), 0)  # update baseline based on residual
    # A_bas = np.fmax(Y_resf / scipy.linalg.norm(f)**2, 0)  # update baseline based on residual
    # baseline based on residual
    b = A_bas

    print("--- %s seconds ---" % (time.time() - start_time))

    try:  # clean up
        # remove temporary file created
        print "Remove temporary file created"
        shutil.rmtree(folder)

    except:

        raise Exception("Failed to delete: " + folder)

    if A_in.dtype == bool:
        
        return A_, b, C, f
    else:
        return A_, b, C


#%%lars_regression_noise_ipyparallel
def regression_ipyparallel(pars):

    # need to import since it is run from within the server
    import numpy as np
    import sys
    from sklearn import linear_model        

    Y_name, C_name, noise_sn, idxs_C, idxs_Y,method_least_square,cct,rank_f = pars
    
    if type(Y_name) is str:
        print("Reloading Y")
        Y, _, _ = load_memmap(Y_name)
        Y = np.array(Y[idxs_Y, :])
    else:
        Y = Y_name[idxs_Y, :]
        
    if type(C_name) is str: 
        print("Reloading Y")           
        C = np.load(C_name, mmap_mode='r')
        C = np.array(C)
    else:
        C = C_name

    _, T = np.shape(C)
    #sys.stdout = open(str(os.getpid()) + ".out", "w")
    As = []
    # print "*****************:" + str(idxs_Y[0]) + ',' + str(idxs_Y[-1])
    
    for y, px in zip(Y, idxs_Y):
        # print str(time.time()-st) + ": Pixel" + str(px)
#        print px,len(idxs_C),C.shape
        c = C[idxs_C[px], :]
        idx_only_neurons=idxs_C[px]
        cct_=cct[idx_only_neurons[:-rank_f]]
        
        if np.size(c) > 0:
            sn = noise_sn[px]**2 * T
            
            if method_least_square == 'lasso_lars_old': # lasso lars from old implementation, will be deprecated 
                
                a = lars_regression_noise_old(y, c.T, 1, sn)[2]
         
            elif method_least_square == 'nnls_L0': #   Nonnegative least square with L0 penalty   
                a = nnls_L0( c.T,y,1.2*sn)
            
            elif method_least_square == 'lasso_lars': # lasso lars function from scikit learn
                #a, RSS = scipy.optimize.nnls(c.T, np.ravel(y))
#                RSS = RSS * RSS                
#                if RSS <= 2*sn:  # hard noise constraint hardly feasible                    
                lambda_lasso=.5*noise_sn[px]*np.sqrt(np.max(cct_))/T 
#                lambda_lasso=1
                clf = linear_model.LassoLars(alpha=lambda_lasso,positive=True)   
                a_lrs = clf.fit(np.array(c.T),np.ravel(y))                    
                a = a_lrs.coef_
#                else:
#                    print 'Problem infeasible'
#                    pl.cla()
#                    pl.plot(a.T.dot(c));
#                    pl.plot(y)
#                    pl.pause(3)

            else:
                raise Exception('Least Square Method not found!'+method_least_square)
            
            if not np.isscalar(a):
                a = a.T

            As.append((px, idxs_C[px], a))

    if type(Y_name) is str:
        print("deleting Y")
        del Y
    
    if type(C_name) is str:            
        del C
#    gc.collect()

    return As

#%% lars_regression_noise_parallel
# def lars_regression_noise_parallel(Y,C,A,noise_sn,idx_Y,idx_C,positive=1):
#
#    _,T=np.shape(C)
#    newY=np.array(Y[idx_Y,:])
#    newC=np.array(C)
#    for px in idx_Y:
#        #
#        c=newC[idx_C[px],:]
#
#        if np.size(c)>0:
#            y=newY[px-idx_Y[0],:]
#            sn=noise_sn[px]**2*T
#           # print y.shape,sn,c.shape
#            _, _, a, _ , _= lars_regression_noise(y, c.T, positive, sn)
#            if np.isscalar(a):
#                A[px,idx_C[px]]=a
#            else:
#                A[px,idx_C[px]]=a.T


#%% determine_search_location
def determine_search_location(A, dims, method='ellipse', min_size=3, max_size=8, dist=3,
                              expandCore=iterate_structure(generate_binary_structure(2, 1), 2).astype(int), dview=None):
    """
    restrict search location to subset of pixels

    TODO
    """
    from scipy.ndimage.morphology import grey_dilation
    from scipy.sparse import coo_matrix, issparse

    if len(dims) == 2:
        d1, d2 = dims
    elif len(dims) == 3:
        d1, d2, d3 = dims

    d, nr = np.shape(A)

    A = csc_matrix(A)

    IND = False * np.ones((d, nr))
    if method == 'ellipse':
        Coor = dict()
        if len(dims) == 2:
            Coor['x'] = np.kron(np.ones(d2), range(d1))
            Coor['y'] = np.kron(range(d2), np.ones(d1))
        elif len(dims) == 3:
            Coor['x'] = np.kron(np.ones(d3 * d2), range(d1))
            Coor['y'] = np.kron(np.kron(np.ones(d3), range(d2)), np.ones(d1))
            Coor['z'] = np.kron(range(d3), np.ones(d2 * d1))
        if not dist == np.inf:             # determine search area for each neuron
            cm = np.zeros((nr, len(dims)))        # vector for center of mass
            Vr = []    # cell(nr,1);
            IND = []       # indicator for distance

            for i, c in enumerate(['x', 'y', 'z'][:len(dims)]):
                cm[:, i] = np.dot(Coor[c], A[:, :nr].todense()) / A[:, :nr].sum(axis=0)

#            for i in range(nr):            # calculation of variance for each component and construction of ellipses
#                dist_cm = coo_matrix(np.hstack([Coor[c].reshape(-1, 1) - cm[i, k]
#                                                for k, c in enumerate(['x', 'y', 'z'][:len(dims)])]))
#                Vr.append(dist_cm.T * spdiags(A[:, i].toarray().squeeze(),
#                                              0, d, d) * dist_cm / A[:, i].sum(axis=0))
#
#                if np.sum(np.isnan(Vr)) > 0:
#                    raise Exception('You cannot pass empty (all zeros) components!')
#
#                D, V = eig(Vr[-1])
#
#                dkk = [np.min((max_size**2, np.max((min_size**2, dd.real)))) for dd in D]
#
#                # search indexes for each component
#                IND.append(np.sqrt(np.sum([(dist_cm * V[:, k])**2 / dkk[k]
#                                           for k in range(len(dkk))], 0)) <= dist)
#            IND = (np.asarray(IND)).squeeze().T
            pars = []
            for i in range(nr):
                pars.append([Coor, cm[i], A[:, i], Vr, dims, dist, max_size, min_size, d])

            if dview is None:
                res = map(contruct_ellipse_parallel, pars)
            else:
                res = dview.map_sync(contruct_ellipse_parallel, pars)

            for r in res:
                IND.append(r)

            IND = (np.asarray(IND)).squeeze().T

        else:
            IND = True * np.ones((d, nr))
    elif method == 'dilate':
        for i in range(nr):
            A_temp = np.reshape(A[:, i].toarray(), dims[::-1])  # , order='F')
            # A_temp = np.reshape(A[:, i].toarray(), (d2, d1))
            if len(expandCore) > 0:
                if len(expandCore.shape) < len(dims):  # default for 3D
                    expandCore = iterate_structure(
                        generate_binary_structure(len(dims), 1), 2).astype(int)
                A_temp = grey_dilation(A_temp, footprint=expandCore)
            else:
                A_temp = grey_dilation(A_temp, [1] * len(dims))

            IND[:, i] = np.squeeze(np.reshape(A_temp, (d, 1))) > 0
    else:
        IND = True * np.ones((d, nr))

    return IND

#%%


def contruct_ellipse_parallel(pars):

    Coor, cm, A_i, Vr, dims, dist, max_size, min_size, d = pars
    dist_cm = coo_matrix(np.hstack([Coor[c].reshape(-1, 1) - cm[k]
                                    for k, c in enumerate(['x', 'y', 'z'][:len(dims)])]))
    Vr.append(dist_cm.T * spdiags(A_i.toarray().squeeze(),
                                  0, d, d) * dist_cm / A_i.sum(axis=0))

    if np.sum(np.isnan(Vr)) > 0:
        raise Exception('You cannot pass empty (all zeros) components!')

    D, V = eig(Vr[-1])

    dkk = [np.min((max_size**2, np.max((min_size**2, dd.real)))) for dd in D]

    # search indexes for each component
    return np.sqrt(np.sum([(dist_cm * V[:, k])**2 / dkk[k] for k in range(len(dkk))], 0)) <= dist
#%% threshold_components


def threshold_components(A, dims, medw=(3, 3), thr_method='nrg', maxthr=0.1, nrgthr=0.9999, extract_cc=True,
                         se=np.ones((3, 3), dtype=np.int), ss=np.ones((3, 3), dtype=np.int), dview=None):
    '''
    Post-processing of spatial components which includes the following steps
    (i) Median filtering
    (ii) Thresholding
    (iii) Morphological closing of spatial support
    (iv) Extraction of largest connected component

    Parameters:
    A:      np.ndarray
        2d matrix with spatial components
    dims:   tuple
        dimensions of spatial components
    medw: [optional] tuple
        window of median filter
    thr_method: [optional] string
        Method of thresholding: 
            'max' sets to zero pixels that have value less than a fraction of the max value
            'nrg' keeps the pixels that contribute up to a specified fraction of the energy
    maxthr: [optional] scalar
        Threshold of max value
    nrgthr: [optional] scalar
        Threshold of energy
    extract_cc: [optional] bool
        Flag to extract connected components (might want to turn to False for dendritic imaging)
    se: [optional] np.intarray
        Morphological closing structuring element
    ss: [optinoal] np.intarray
        Binary element for determining connectivity
    '''

    if len(dims) == 3:  # default values for 3D
        if len(medw) == 2:
            medw = (3, 3, 1)
        if len(se.shape) == 2:
            se = np.ones((3, 3, 1), dtype=np.int)
        if len(ss.shape) == 2:
            ss = np.ones((3, 3, 1), dtype=np.int)

    d, nr = np.shape(A)
    Ath = np.zeros((d, nr))

    pars = []
    for i in range(nr):
        pars.append([A[:, i], i, dims, medw, d, thr_method, se, ss, maxthr, nrgthr, extract_cc])

    if dview is not None:
        res = dview.map_async(threshold_components_parallel, pars)
    else:
        res = map(threshold_components_parallel, pars)

    for r in res:
        At, i = r
        Ath[:, i] = At

#    for i in range(nr):
#
#        A_temp = np.reshape(A[:, i], dims[::-1])
#        A_temp = median_filter(A_temp, medw)
#        Asor = np.sort(np.squeeze(np.reshape(A_temp, (d, 1))))[::-1]
#        temp = np.cumsum(Asor**2)
#        ff = np.squeeze(np.where(temp < (1 - thr) * temp[-1]))
#
#        if ff.size > 0:
#            if ff.ndim == 0:
#                ind = ff
#            else:
#                ind = ff[-1]
#            A_temp[A_temp < Asor[ind]] = 0
#            BW = (A_temp >= Asor[ind])
#        else:
#            BW = (A_temp >= 0)
#
#        Ath[:, i] = np.squeeze(np.reshape(A_temp, (d, 1)))
#        BW = binary_closing(BW.astype(np.int), structure=se)
#        labeled_array, num_features = label(BW, structure=ss)
#        BW = np.reshape(BW, (d, 1))
#        labeled_array = np.squeeze(np.reshape(labeled_array, (d, 1)))
#        nrg = np.zeros((num_features, 1))
#        for j in range(num_features):
#            nrg[j] = np.sum(Ath[labeled_array == j + 1, i]**2)
#
#        indm = np.argmax(nrg)
#        Ath[labeled_array == indm + 1, i] = A[labeled_array == indm + 1, i]

    return Ath


def threshold_components_parallel(pars):

    A_i, i, dims, medw, d, thr_method, se, ss, maxthr, nrgthr, extract_cc = pars
    A_temp = np.reshape(A_i, dims[::-1])
    A_temp = median_filter(A_temp, medw)
    if thr_method == 'max':
        BW = (A_temp > maxthr * np.max(A_temp))
    elif thr_method == 'nrg':
        Asor = np.sort(np.squeeze(np.reshape(A_temp, (d, 1))))[::-1]
        temp = np.cumsum(Asor**2)
        ff = np.squeeze(np.where(temp < (1 - nrgthr) * temp[-1]))
        if ff.size > 0:
            if ff.ndim == 0:
                ind = ff
            else:
                ind = ff[-1]
            A_temp[A_temp < Asor[ind]] = 0
            BW = (A_temp >= Asor[ind])
        else:
            BW = (A_temp >= 0)

    Ath = np.squeeze(np.reshape(A_temp, (d, 1)))
    Ath2 = np.zeros((d))
    BW = binary_closing(BW.astype(np.int), structure=se)
    if extract_cc:
        labeled_array, num_features = label(BW, structure=ss)
        BW = np.reshape(BW, (d, 1))
        labeled_array = np.squeeze(np.reshape(labeled_array, (d, 1)))
        nrg = np.zeros((num_features, 1))
        for j in range(num_features):
            nrg[j] = np.sum(Ath[labeled_array == j + 1]**2)

        indm = np.argmax(nrg)
        #Ath2[labeled_array == indm + 1] = A_i[labeled_array == indm + 1]
        Ath2[labeled_array == indm + 1] = Ath[labeled_array == indm + 1]
    else:
        BW = BW.flatten()
        Ath2[BW] = Ath[BW]

    return Ath2, i

#%%
def nnls_L0(X,Yp,noise):
    """
    Nonnegative least square with L0 penalty
    min_||W_lam||_0 || Yp-W_lam*X||**2 <= noise
    
    """
    W_lam, RSS = scipy.optimize.nnls(X, np.ravel(Yp))
    RSS = RSS * RSS
    if RSS > noise:  # hard noise constraint problem infeasible
        return W_lam

    while 1:
        eliminate = []
        for i in np.where(W_lam[:-1] > 0)[0]:  # W_lam[:-1] to skip background
            mask = W_lam > 0
            mask[i] = 0
            Wtmp, tmp = scipy.optimize.nnls(X * mask, np.ravel(Yp))
            if tmp * tmp < noise:
                eliminate.append([i, tmp])
        if eliminate == []:
            return W_lam
        else:
            W_lam[eliminate[np.argmin(np.array(eliminate)[:, 1])][0]] = 0
            
            
#%% lars_regression_noise
def lars_regression_noise_old(Yp, X, positive, noise, verbose=False):
    """
     Run LARS for regression problems with LASSO penalty, with optional positivity constraints
     Author: Andrea Giovannucci. Adapted code from Eftychios Pnevmatikakis


     Input Parameters:
       Yp:          Yp[:,t] is the observed data at time t
       X:           the regresion problem is Yp=X*W + noise
       maxcomps:    maximum number of active components to allow
       positive:    a flag to enforce positivity
       noise:       the noise of the observation equation. if it is not
                    provided as an argument, the noise is computed from the
                    variance at the end point of the algorithm. The noise is
                    used in the computation of the Cp criterion.


     Output Parameters:
       Ws: weights from each iteration
       lambdas: lambda_ values at each iteration
       TODO: W_lam, lam, flag
       Cps: C_p estimates
       last_break:     last_break(m) == n means that the last break with m non-zero weights is at Ws(:,:,n)
    """
    #%%

    # verbose=true;

    # do NNLS first
    # if  hard noise constraint problem infeasible we are done, otherwise good for warm-starting
#    W_lam, RSS = scipy.optimize.nnls(X, np.ravel(Yp))
#    RSS = RSS * RSS
#    if RSS > noise:  # hard noise constraint problem infeasible
#        return 0, 0, W_lam, 0, 0
#
#    while 1:
#        eliminate = []
#        for i in np.where(W_lam[:-1] > 0)[0]:  # W_lam[:-1] to skip background
#            mask = W_lam > 0
#            mask[i] = 0
#            Wtmp, tmp = scipy.optimize.nnls(X * mask, np.ravel(Yp))
#            if tmp * tmp < noise:
#                eliminate.append([i, tmp])
#        if eliminate == []:
#            return 0, 0, W_lam, 0, 0
#        else:
#            W_lam[eliminate[np.argmin(np.array(eliminate)[:, 1])][0]] = 0

    # obsolete stuff below

    # solve hard noise constraint problem
    T = len(Yp)  # of time steps
#    A = Yp.dot(X)
#    B = X.T.dot(X)
#    dB = 1 / np.diag(B)
#    import pdb
#    pdb.set_trace()
#    z = np.sqrt(1 / dB)
#    lam = np.max(1 * np.sqrt(noise / T) * z)
#    counter = 0
#    res = X.dot(W_lam) - Yp
#    tmp = X.dot(dB)
#    aa = tmp.dot(tmp)
#    while (RSS < noise * .9999 or RSS > noise * 1.0001) and counter < 20:
#        counter += 1
#        bb = tmp.dot(res)
#        cc = RSS - noise
#        if counter > 1:
#            lam += (bb + (0 if bb * bb <= aa * cc else np.sqrt(bb * bb - aa * cc))) / aa
#        for _ in range(3):
#            for i in range(len(W_lam)):
#                W_lam[i] = np.clip(W_lam[i] + (A[i] - W_lam.dot(B)[i] - lam) * dB[i], 0, np.inf)
#        res = X.dot(W_lam) - Yp
#        RSS = res.dot(res)
#    return 1, 1, W_lam, 1, 1

    k = 1
    Yp = np.squeeze(np.asarray(Yp))

    Yp = np.expand_dims(Yp, axis=1)  # necessary for matrix multiplications

    _, T = np.shape(Yp)  # of time steps
    _, N = np.shape(X)  # of compartments

    maxcomps = N
    W = np.zeros((N, k))
    active_set = np.zeros((N, k))
    visited_set = np.zeros((N, k))
    lambdas = []
    # =np.zeros((W.shape[0],W.shape[1],maxcomps));  # Just preallocation. Ws may end with more or less than maxcomp columns
    Ws = []
    r = np.expand_dims(np.dot(X.T, Yp.flatten()), axis=1)       # N-dim vector
    M = np.dot(-X.T, X)            # N x N matrix

    #%% begin main loop
    i = 0
    flag = 0
    while 1:
        if flag == 1:
            W_lam = 0
            break
    #% calculate new gradient component if necessary
        if i > 0 and new >= 0 and visited_set[new] == 0:  # AG NOT CLEAR HERE
            visited_set[new] = 1  # % remember this direction was computed

    #% Compute full gradient of Q
        dQ = r + np.dot(M, W)

    #% Compute new W
        if i == 0:
            if positive:
                dQa = dQ
            else:
                dQa = np.abs(dQ)
            lambda_, new = np.max(dQa), np.argmax(dQa)

            if lambda_ < 0:
                print 'All negative directions!'
                break
        else:

            #% calculate vector to travel along
            avec, gamma_plus, gamma_minus = calcAvec(new, dQ, W, lambda_, active_set, M, positive)

           # % calculate time of travel and next new direction
            if new == -1:                              # % if we just dropped a direction we don't allow it to emerge
                if dropped_sign == 1:               # % with the same sign
                    gamma_plus[dropped] = np.inf
                else:
                    gamma_minus[dropped] = np.inf

            gamma_plus[active_set == 1] = np.inf  # % don't consider active components
            gamma_plus[gamma_plus <= 0] = np.inf  # % or components outside the range [0, lambda_]
            gamma_plus[gamma_plus > lambda_] = np.inf
            gp_min, gp_min_ind = np.min(gamma_plus), np.argmin(gamma_plus)

            if positive:
                gm_min = np.inf  # % don't consider new directions that would grow negative
            else:
                gamma_minus[active_set == 1] = np.inf
                gamma_minus[gamma_minus > lambda_] = np.inf
                gamma_minus[gamma_minus <= 0] = np.inf
                gm_min, gm_min_ind = np.min(gamma_minus), np.argmin(gamma_minus)

            [g_min, which] = np.min(gp_min), np.argmin(gp_min)

            if g_min == np.inf:  # % if there are no possible new components, try move to the end
                g_min = lambda_  # % This happens when all the components are already active or, if positive==1, when there are no new positive directions

            #% LARS check  (is g_min*avec too large?)
            gamma_zero = -W[active_set == 1] / np.squeeze(avec)
            gamma_zero_full = np.zeros((N, k))
            gamma_zero_full[active_set == 1] = gamma_zero
            gamma_zero_full[gamma_zero_full <= 0] = np.inf
            gz_min, gz_min_ind = np.min(gamma_zero_full), np.argmin(gamma_zero_full)

            if gz_min < g_min:
                #                print 'check_here'
                if verbose:
                    print 'DROPPING active weight:' + str(gz_min_ind)

                active_set[gz_min_ind] = 0
                dropped = gz_min_ind
                dropped_sign = np.sign(W[dropped])
                W[gz_min_ind] = 0
                avec = avec[gamma_zero != gz_min]
                g_min = gz_min
                new = -1  # new = 0;

            elif g_min < lambda_:
                if which == 0:
                    new = gp_min_ind
                    if verbose:
                        print 'new positive component:' + str(new)

                else:
                    new = gm_min_ind
                    print 'new negative component:' + str(new)

            W[active_set == 1] = W[active_set == 1] + np.dot(g_min, np.squeeze(avec))

            if positive:
                if any(W < 0):
                    # min(W);
                    flag = 1
                    #%error('negative W component');

            lambda_ = lambda_ - g_min

    #%  Update weights and lambdas

        lambdas.append(lambda_)
        Ws.append(W.copy())

    #    print Ws
        if len((Yp - np.dot(X, W)).shape) > 2:
            res = scipy.linalg.norm(np.squeeze(Yp - np.dot(X, W)), 'fro')**2
        else:
            res = scipy.linalg.norm(Yp - np.dot(X, W), 'fro')**2

    #% Check finishing conditions
        if lambda_ == 0 or (new >= 0 and np.sum(active_set) == maxcomps) or (res < noise):
            if verbose:
                print 'end. \n'
            break

        #%
        if new >= 0:
            active_set[new] = 1

        i = i + 1

    Ws_old = Ws
    # end main loop

    #%% final calculation of mus
    Ws = np.asarray(np.swapaxes(np.swapaxes(Ws_old, 0, 1), 1, 2))
    if flag == 0:
        if i > 0:
            Ws = np.squeeze(Ws[:, :, :len(lambdas)])
            w_dir = -(Ws[:, i] - Ws[:, i - 1]) / (lambdas[i] - lambdas[i - 1])
            Aw = np.dot(X, w_dir)
            y_res = np.squeeze(Yp) - np.dot(X, Ws[:, i - 1] + w_dir * lambdas[i - 1])
            ld = scipy.roots([scipy.linalg.norm(Aw)**2, -2 * np.dot(Aw.T, y_res),
                              np.dot(y_res.T, y_res) - noise])
            lam = ld[np.intersect1d(np.where(ld > lambdas[i]), np.where(ld < lambdas[i - 1]))]
            if len(lam) == 0 or np.any(lam) < 0 or np.any(~np.isreal(lam)):
                lam = np.array([lambdas[i]])

            W_lam = Ws[:, i - 1] + np.dot(w_dir, lambdas[i - 1] - lam[0])
        else:
            warn('LARS REGRESSION NOT SOLVABLE, USING NN LEAST SQUARE')
            W_lam = scipy.optimize.nnls(X, np.ravel(Yp))[0]
#            problem = picos.Problem(X,Yp)
#            W_lam = problem.add_variable('W_lam', X.shape[1])
#            problem.set_objective('min', 1|W_lam)
#            problem.add_constraint(W_lam >= 0)
#            problem.add_constraint(picos.norm(matrix(Yp.astype(np.float))-matrix(X.astype(np.float))*W_lam,2)<=np.sqrt(noise))
#            sel_solver = []
#            problem.solver_selection()
#            problem.solve(verbose=True)

    #        cvx_begin quiet
    #            variable W_lam(size(X,2));
    #            minimize(sum(W_lam));
    #            subject to
    #                W_lam >= 0;
    #                norm(Yp-X*W_lam)<= sqrt(noise);
    #        cvx_end
            lam = 10

    else:
        W_lam = 0
        Ws = 0
        lambdas = 0
        lam = 0

    return Ws, lambdas, W_lam, lam, flag

#%% auxiliary functions


def calcAvec(new, dQ, W, lambda_, active_set, M, positive):
    # TODO: comment
    r, c = np.nonzero(active_set)
#    [r,c] = find(active_set);
    Mm = -M.take(r, axis=0).take(r, axis=1)

    Mm = (Mm + Mm.T) / 2

    #% verify that there is no numerical instability
    if len(Mm) > 1:
        #        print Mm.shape
        eigMm, _ = scipy.linalg.eig(Mm)
        eigMm = np.real(eigMm)
#        check_here
    else:
        eigMm = Mm

    if any(eigMm < 0):
        np.min(eigMm)
        #%error('The matrix Mm has negative eigenvalues')
        flag = 1

    b = np.sign(W)

    if new >= 0:
        b[new] = np.sign(dQ[new])

    b = b[active_set == 1]

    if len(Mm) > 1:
        avec = np.linalg.solve(Mm, b)
    else:
        avec = b / Mm

    if positive:
        if new >= 0:
            in_ = np.sum(active_set[:new])
            if avec[in_] < 0:
                # new;
                #%error('new component of a is negative')
                flag = 1

    one_vec = np.ones(W.shape)

    dQa = np.zeros(W.shape)
    for j in range(len(r)):
        dQa = dQa + np.expand_dims(avec[j] * M[:, r[j]], axis=1)

    gamma_plus = (lambda_ - dQ) / (one_vec + dQa)
    gamma_minus = (lambda_ + dQ) / (one_vec - dQa)

    return avec, gamma_plus, gamma_minus
