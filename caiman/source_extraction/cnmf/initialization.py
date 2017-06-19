from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import numpy as np
from sklearn.decomposition import NMF, FastICA
from skimage.transform import downscale_local_mean, resize
import pylab as pl
import scipy.ndimage as nd
import scipy.sparse as spr
import scipy
from scipy.ndimage.measurements import center_of_mass
import caiman
import cv2
from scipy.ndimage.filters import correlate
#from . import utilities
#%%
def initialize_components(Y, K=30, gSig=[5, 5], gSiz=None, ssub=1, tsub=1, nIter=5, maxIter=5, nb=1,
                          kernel=None, use_hals=True, normalize_init=True, img=None, method='greedy_roi', max_iter_snmf=500, alpha_snmf=10e2, sigma_smooth_snmf=(.5, .5, .5), perc_baseline_snmf=20, options_local_NMF = None):
    """Initalize components

    This method uses a greedy approach followed by hierarchical alternative least squares (HALS) NMF.
    Optional use of spatio-temporal downsampling to boost speed.

    Parameters
    ----------
    Y: np.ndarray
         d1 x d2 [x d3] x T movie, raw data.
    K: [optional] int
        number of neurons to extract (default value: 30).
    tau: [optional] list,tuple
        standard deviation of neuron size along x and y [and z] (default value: (5,5).
    gSiz: [optional] list,tuple
        size of kernel (default 2*tau + 1).
    nIter: [optional] int
        number of iterations for shape tuning (default 5).
    maxIter: [optional] int
        number of iterations for HALS algorithm (default 5).
    ssub: [optional] int
        spatial downsampling factor recommended for large datasets (default 1, no downsampling).
    tsub: [optional] int
        temporal downsampling factor recommended for long datasets (default 1, no downsampling).
    kernel: [optional] np.ndarray
        User specified kernel for greedyROI (default None, greedy ROI searches for Gaussian shaped neurons)
    use_hals: [optional] bool
        Whether to refine components with the hals method
    normalize_init: [optional] bool
        Whether to normalize_init data before running the initialization
    img: optional [np 2d array]
        Image with which to normalize. If not present use the mean + offset
    method: str
        Initialization method 'greedy_roi' or 'sparse_nmf'
    max_iter_snmf: int
        Maximum number of sparse NMF iterations
    alpha_snmf: scalar
        Sparsity penalty

    Returns
    --------
    Ain: np.ndarray
        (d1*d2[*d3]) x K , spatial filter of each neuron.
    Cin: np.ndarray
        T x K , calcium activity of each neuron.
    center: np.ndarray
        K x 2 [or 3] , inferred center of each neuron.
    bin: np.ndarray
        (d1*d2[*d3]) x nb, initialization of spatial background.
    fin: np.ndarray
        nb x T matrix, initalization of temporal background.

    """


    if method == 'local_nmf':
        tsub_lnmf = tsub
        ssub_lnmf = ssub
        tsub = 1
        ssub = 1


    if gSiz is None:
        gSiz = 2 * np.asarray(gSig) + 1

    d, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    # rescale according to downsampling factor
    gSig = np.round(np.asarray(gSig) / ssub).astype(np.int)
    gSiz = np.round(np.asarray(gSiz) / ssub).astype(np.int)

    print('Noise Normalization')
    if normalize_init is True:
        if img is None:
            img = np.mean(Y, axis=-1)
            img += np.median(img)

        Y = old_div(Y, np.reshape(img, d + (-1,), order='F'))
        if alpha_snmf is not None: alpha_snmf /= np.mean(img)

    # spatial downsampling
    mean_val = np.mean(Y)
    if ssub != 1 or tsub != 1:
        print("Spatial Downsampling ...")
        Y_ds = downscale_local_mean(Y, tuple([ssub] * len(d) + [tsub]), cval=mean_val)
    else:
        Y_ds = Y

    print('Roi Extraction...')

    if method == 'greedy_roi':
        Ain, Cin, _, b_in, f_in = greedyROI(
            Y_ds, nr=K, gSig=gSig, gSiz=gSiz, nIter=nIter, kernel=kernel, nb=nb)

        if use_hals:
            print('Refining Components...')
            Ain, Cin, b_in, f_in = hals(Y_ds, Ain, Cin, b_in, f_in, maxIter=maxIter)


    elif method == 'sparse_nmf':

        Ain, Cin, _, b_in, f_in = sparseNMF(Y_ds, nr=K, nb=nb, max_iter_snmf=max_iter_snmf, alpha=alpha_snmf,
                                            sigma_smooth=sigma_smooth_snmf, remove_baseline=True, perc_baseline=perc_baseline_snmf)
#        print np.sum(Ain), np.sum(Cin)
#        print 'Refining Components...'
#        Ain, Cin, b_in, f_in = hals(Y_ds, Ain, Cin, b_in, f_in, maxIter=maxIter)
#        print np.sum(Ain), np.sum(Cin)
    elif method == 'pca_ica':

        Ain, Cin, _, b_in, f_in = ICA_PCA(Y_ds, nr = K, sigma_smooth=sigma_smooth_snmf,  truncate = 2, fun='logcosh',\
                                          max_iter=max_iter_snmf, tol=1e-10,remove_baseline=True, perc_baseline=perc_baseline_snmf, nb=nb)

    elif method == 'local_nmf':
        from SourceExtraction.CNMF4Dendrites import CNMF4Dendrites
        from SourceExtraction.AuxilaryFunctions import GetCentersData
        #Get initialization for components center
        print(Y_ds.transpose([2,0,1]).shape)
        if options_local_NMF is None:

             raise Exception('You need to define arguments for local NMF')


#            #Define CNMF parameters
#            mbs=[tsub_lnmf] # temporal downsampling of data in intial phase of NMF
#            ds=ssub_lnmf # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
#            TargetAreaRatio=[0.01,0.06] # target sparsity range for spatial components
#            #repeats=1 # how many repeations to run NMF algorithm
#            iters0=[5] #30 number of intial NMF iterations, in which we downsample data and add components
#            iters=20 #100 number of main NMF iterations, in which we fine tune the components on the full data
#            lam1_s=10# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
#            bkg_per=0.1 # intialize of background shape at this percentile (over time) of video
#            sig=Y_ds.shape[:-1] # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
#            MergeThreshold_activity=0.85#merge components if activity is correlated above the this threshold (and sufficiently close)
#            MergeThreshold_shapes=0.99 #merge components if activity is correlated above the this threshold (and sufficiently close)
#            Connected=True # should we constrain all spatial component to be connected?
#            SigmaMask=3  # if not [], then update masks so that they are non-zero a radius of SigmaMasks around previous non-zero support of shapes

                                   #Get initialization for components center

#            NumCent=400 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
#            cent=GetCentersData(Y_ds.transpose([2,0,1]),NumCent)
#
#            #Define CNMF parameters
#            mbs=[10] # temporal downsampling of data in intial phase of NMF
#            ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
#            TargetAreaRatio=[0.01,0.06] # target sparsity range for spatial components
#            #repeats=1 # how many repeations to run NMF algorithm
#            iters0=[5] #30 number of intial NMF iterations, in which we downsample data and add components
#            iters=20 #100 number of main NMF iterations, in which we fine tune the components on the full data
#            lam1_s=10# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
#            updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
#            addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
#            updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
#            Background_num=1 #number of background components - one of which at every repetion
#            bkg_per=0.1 # intialize of background shape at this percentile (over time) of video
#            sig=Y_ds.shape[:-1] # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
#            MergeThreshold_activity=0.85#merge components if activity is correlated above the this threshold (and sufficiently close)
#            MergeThreshold_shapes=0.99 #merge components if activity is correlated above the this threshold (and sufficiently close)
#            Connected=True # should we constrain all spatial component to be connected?
#            SigmaMask=3  # if not [], then update masks so that they are non-zero a radius of SigmaMasks around previous non-zero support of shapes

#            cnmf_obj=CNMF4Dendrites(sig=sig, verbose=True,adaptBias=True,TargetAreaRatio=TargetAreaRatio,
#                     Connected=Connected, SigmaMask=SigmaMask,bkg_per=bkg_per,iters=iters,iters0=iters0, mbs=mbs,
#                     ds=ds,lam1_s=lam1_s,MergeThreshold_activity=MergeThreshold_activity,MergeThreshold_shapes=MergeThreshold_shapes)
        else:

            NumCent=options_local_NMF.pop('NumCent', None) # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
            cent=GetCentersData(Y_ds.transpose([2,0,1]),NumCent)
            sig=Y_ds.shape[:-1] # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
            cnmf_obj=CNMF4Dendrites(sig=sig, verbose=True,adaptBias=True,**options_local_NMF)

        #Define CNMF parameters
        _, _, _ =cnmf_obj.fit(np.array(Y_ds.transpose([2,0,1]),dtype = np.float),cent)

        Ain = cnmf_obj.A
        Cin = cnmf_obj.C
        b_in = cnmf_obj.b
        f_in = cnmf_obj.f

#        Cin, _, b_in, f_in = ICA_PCA(Y_ds, nr = K, sigma_smooth=sigma_smooth_snmf,  truncate = 2, fun='logcosh',\
#                                          max_iter=max_iter_snmf, tol=1e-10,remove_baseline=True, perc_baseline=perc_baseline_snmf, nb=nb)

    else:

        print(method)
        raise Exception("Unsupported method")


    K = np.shape(Ain)[-1]
    ds = Y_ds.shape[:-1]

    Ain = np.reshape(Ain, ds + (K,), order='F')

    if len(ds) == 2:

        Ain = resize(Ain, d + (K,), order=1)

    else:  # resize only deals with 2D images, hence apply resize twice

        Ain = np.reshape([resize(a, d[1:] + (K,), order=1)
                          for a in Ain], (ds[0], d[1] * d[2], K), order='F')
        Ain = resize(Ain, (d[0], d[1] * d[2], K), order=1)

    Ain = np.reshape(Ain, (np.prod(d), K), order='F')

    #import pdb
    # pdb.set_trace()

    b_in = np.reshape(b_in, ds + (nb,), order='F')

    if len(ds) == 2:
        b_in = resize(b_in, d + (nb,), order=1)
    else:
        b_in = np.reshape([resize(b, d[1:] + (nb,), order=1)
                           for b in b_in], (ds[0], d[1] * d[2], nb), order='F')
        b_in = resize(b_in, (d[0], d[1] * d[2], nb), order=1)

    b_in = np.reshape(b_in, (np.prod(d), nb), order='F')

    Cin = resize(Cin, [K, T])

    f_in = resize(np.atleast_2d(f_in), [nb, T])
    # center = com(Ain, *d)
    center = np.asarray([center_of_mass(a.reshape(d, order='F')) for a in Ain.T])

    if normalize_init is True:
        #import pdb
        # pdb.set_trace()
        Ain = Ain * np.reshape(img, (np.prod(d), -1), order='F')

        b_in = b_in * np.reshape(img, (np.prod(d), -1), order='F')
        # b_in = np.atleast_2d(b_in * img.flatten('F')) #np.reshape(img,
        # (np.prod(d), -1),order='F')
        Y = Y * np.reshape(img, d + (-1,), order='F')


    return Ain, Cin, b_in, f_in, center

#%%
def ICA_PCA(Y_ds, nr, sigma_smooth=(.5, .5, .5),  truncate = 2, fun='logcosh', max_iter=1000, tol=1e-10,remove_baseline=True, perc_baseline=20, nb=1):
    """ Initialization using ICA and PCA. DOES NOT WORK WELL WORK IN PROGRESS"

    Parameters:
    -----------

    Returns:
    --------


    """
    m = scipy.ndimage.gaussian_filter(np.transpose(Y_ds, [2, 0, 1]), sigma=sigma_smooth, mode='nearest', truncate=truncate)
    if remove_baseline:
        bl = np.percentile(m, perc_baseline, axis=0)
        m1 = np.maximum(0, m - bl)
    else:
        bl = 0
        m1 = m

    pca_comp = nr

    T, d1, d2 = np.shape(m1)
    d = d1 * d2
    yr = np.reshape(m1, [T, d], order='F')

    [U,S,V] = scipy.sparse.linalg.svds(yr,pca_comp)
    S = np.diag(S);
#        whiteningMatrix = np.dot(scipy.linalg.inv(np.sqrt(S)),U.T)
#        dewhiteningMatrix = np.dot(U,np.sqrt(S))
    whiteningMatrix = np.dot(scipy.linalg.inv(S),U.T)
#    dewhiteningMatrix = np.dot(U,S)
    whitesig =  np.dot(whiteningMatrix,yr)
#    wsigmask=np.reshape(whitesig.T,(d1,d2,pca_comp));
    f_ica = FastICA(whiten=False, fun=fun, max_iter=max_iter, tol=tol)
    S_ = f_ica.fit_transform(whitesig.T)
    A_in = f_ica.mixing_
    A_in = np.dot(A_in,whitesig)


    masks = np.reshape(A_in.T,(d1,d2,pca_comp),order = 'F').transpose([2,0,1])


#    pl.figure()
 #   caiman.utils.visualization.matrixMontage(np.array(masks))
    masks = np.array(caiman.base.rois.extractROIsFromPCAICA(masks)[0])
#    pl.pause(3)


    if masks.size > 0:

        C_in = caiman.base.movies.movie(m1).extract_traces_from_masks(np.array(masks)).T
        A_in = np.reshape(masks,[-1,d1*d2],order = 'F').T

#        pl.figure()
#        pl.imshow(np.reshape(A_in.sum(1),[d1,d2],order = 'F'))
#        pl.pause(3)


    else:

        A_in = np.zeros([d1*d2,pca_comp])
        C_in = np.zeros([pca_comp,T])



    m1 = yr.T - A_in.dot(C_in) + np.maximum(0, bl.flatten())[:, np.newaxis]

    model = NMF(n_components=nb, init='random', random_state=0)

    b_in = model.fit_transform(np.maximum(m1, 0))
    f_in = model.components_.squeeze()

    center = caiman.base.rois.com(A_in, d1, d2)


    return A_in, C_in, center, b_in, f_in
#%%
def sparseNMF(Y_ds, nr,  max_iter_snmf=500, alpha=10e2, sigma_smooth=(.5, .5, .5), remove_baseline=True, perc_baseline=20, nb=1, truncate = 2 ):
    """
    Initilaization using sparse NMF
    Parameters
    -----------

    max_iter_snm: int
        number of iterations
    alpha_snmf:
        sparsity regularizer
    sigma_smooth_snmf:
        smoothing along z,x, and y (.5,.5,.5)
    perc_baseline_snmf:
        percentile to remove frmo movie before NMF
    nb: int
        Number of background components

    Returns:
    -------

    A: np.array
        2d array of size (# of pixels) x nr with the spatial components. Each column is
        ordered columnwise (matlab format, order='F')
    C: np.array
        2d array of size nr X T with the temporal components
    center: np.array
        2d array of size nr x 2 [ or 3] with the components centroids
    """

    m = scipy.ndimage.gaussian_filter(np.transpose(
        Y_ds, [2, 0, 1]), sigma=sigma_smooth, mode='nearest', truncate=truncate)
    if remove_baseline:
        bl = np.percentile(m, perc_baseline, axis=0)
        m1 = np.maximum(0, m - bl)
    else:
        bl = 0
        m1 = m

    mdl = NMF(n_components=nr, verbose=False, init='nndsvd', tol=1e-10,
              max_iter=max_iter_snmf, shuffle=True, alpha=alpha, l1_ratio=1)
    T, d1, d2 = np.shape(m1)
    d = d1 * d2
    yr = np.reshape(m1, [T, d], order='F')
    C = mdl.fit_transform(yr).T
    A = mdl.components_.T
    ind_good = np.where(np.logical_and((np.sum(A, 0) * np.std(C, axis=1))
                                       > 0, np.sum(A > np.mean(A), axis=0) < old_div(d, 3)))[0]
#    A_in=A[:, ind_good]
#    C_in=C[ind_good, :]
    ind_bad = np.where(np.logical_or((np.sum(A, 0) * np.std(C, axis=1))
                                     == 0, np.sum(A > np.mean(A), axis=0) > old_div(d, 3)))[0]
    A_in = np.zeros_like(A)

    C_in = np.zeros_like(C)
    A_in[:, ind_good] = A[:, ind_good]
    C_in[ind_good, :] = C[ind_good, :]
    A_in = A_in * (A_in > (.1 * np.max(A_in, axis=0))[np.newaxis, :])
    A_in[:3, ind_bad] = .0001
    C_in[ind_bad, :3] = .0001


    m1 = yr.T - A_in.dot(C_in) + np.maximum(0, bl.flatten())[:, np.newaxis]

    model = NMF(n_components=nb, init='random', random_state=0, max_iter=max_iter_snmf)

    b_in = model.fit_transform(np.maximum(m1, 0))
    f_in = model.components_.squeeze()
#    for ccount,caaa in enumerate(A.T):
#        pl.subplot(3,3,ccount+1)
#        pl.imshow(np.reshape(caaa,[d1,d2],order = 'F'))

#    pl.subplot(2,2,1)
#    pl.imshow(np.reshape(b_in,[d1,d2],order = 'F'))
#    pl.subplot(2,2,2)
#    pl.imshow(np.reshape(A_in.mean(axis = 1),[d1,d2],order = 'F'))
#    pl.subplot(2,2,3)
#    pl.imshow(np.mean(np.maximum(0, m - bl),0))
#    pl.subplot(2,2,4)
#    pl.imshow(caiman.movie(np.maximum(0, m - bl)).local_correlations(swap_dim=False))
#    pl.pause(4)

    center = caiman.base.rois.com(A_in, d1, d2)
#    for iter in range(max_iter_snmf):
#        f = np.maximum(b.T.dot(scipy.linalg.solve(scipy.linalg.norm(b).T**2,Y.T),0);
#        b = np.maximum(Y.dot(scipy.linalg.solve(scipy.linalg.norm(f).T**2,f.T),0);
#    end

    return A_in, C_in, center, b_in, f_in

#%%


def greedyROI(Y, nr=30, gSig=[5, 5], gSiz=[11, 11], nIter=5, kernel=None, nb=1):
    """
    Greedy initialization of spatial and temporal components using spatial Gaussian filtering
    Inputs:
    Y: np.array
        3d or 4d array of fluorescence data with time appearing in the last axis.
    nr: int
        number of components to be found
    gSig: scalar or list of integers
        standard deviation of Gaussian kernel along each axis
    gSiz: scalar or list of integers
        size of spatial component
    nIter: int
        number of iterations when refining estimates
    kernel: np.ndarray
        User specified kernel to be used, if present, instead of Gaussian (default None)
    nb: int
        Number of background components

    Outputs:
    A: np.array
        2d array of size (# of pixels) x nr with the spatial components. Each column is
        ordered columnwise (matlab format, order='F')
    C: np.array
        2d array of size nr X T with the temporal components
    center: np.array
        2d array of size nr x 2 [ or 3] with the components centroids

    Author: Eftychios A. Pnevmatikakis based on a matlab implementation by Yuanjun Gao
            Simons Foundation, 2015
    """
    debug_ = False
    if debug_:
        import os
        f = open('_LOG_1_' + str(os.getpid()), 'w+')
        f.write('type_rho:' + str(type(rho)) + '\n')
        f.write('rho:' + str(np.mean(rho)) + '\n')
        f.close()

    d = np.shape(Y)
    med = np.median(Y, axis=-1)
    Y = Y - med[..., np.newaxis]
    gHalf = np.array(gSiz) // 2
    gSiz = 2 * gHalf + 1

    A = np.zeros((np.prod(d[0:-1]), nr))
    C = np.zeros((nr, d[-1]))
    center = np.zeros((nr, Y.ndim - 1))

    rho = imblur(Y, sig=gSig, siz=gSiz, nDimBlur=Y.ndim - 1, kernel=kernel)

    v = np.sum(rho**2, axis=-1)

    for k in range(nr):

        ind = np.argmax(v)

        ij = np.unravel_index(ind, d[0:-1])

        for c, i in enumerate(ij):
            center[k, c] = i

        ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                 for c in range(len(ij))]

        dataTemp = np.array(Y[[slice(*a) for a in ijSig]].copy(), dtype=np.float)

        traceTemp = np.array(np.squeeze(rho[ij]), dtype=np.float)

        coef, score = finetune(dataTemp, traceTemp, nIter=nIter)

        C[k, :] = np.squeeze(score)

        dataSig = coef[..., np.newaxis] * score.reshape([1] * (Y.ndim - 1) + [-1])
        xySig = np.meshgrid(*[np.arange(s[0], s[1]) for s in ijSig], indexing='xy')
        arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze()
                        for s in xySig], dtype=np.int)
        indeces = np.ravel_multi_index(arr, d[0:-1], order='F')

        A[indeces, k] = np.reshape(coef, (1, np.size(coef)), order='C').squeeze()
        Y[[slice(*a) for a in ijSig]] -= dataSig.copy()
        if k < nr - 1:
            Mod = [[np.maximum(ij[c] - 2 * gHalf[c], 0),
                    np.minimum(ij[c] + 2 * gHalf[c] + 1, d[c])] for c in range(len(ij))]
            ModLen = [m[1] - m[0] for m in Mod]
            Lag = [ijSig[c] - Mod[c][0] for c in range(len(ij))]
            dataTemp = np.zeros(ModLen)
            dataTemp[[slice(*a) for a in Lag]] = coef
            dataTemp = imblur(dataTemp[..., np.newaxis], sig=gSig, siz=gSiz, kernel=kernel)
            rhoTEMP = dataTemp * score.reshape([1] * (Y.ndim - 1) + [-1])
            rho[[slice(*a) for a in Mod]] -= rhoTEMP.copy()
            v[[slice(*a) for a in Mod]] = np.sum(
                rho[[slice(*a) for a in Mod]]**2, axis=-1)

    res = np.reshape(Y, (np.prod(d[0:-1]), d[-1]), order='F') + med.flatten(order='F')[:, None]

    model = NMF(n_components=nb, init='random', random_state=0)

    b_in = model.fit_transform(np.maximum(res, 0))
    f_in = model.components_.squeeze()

    return A, C, center, b_in, f_in

#%%


def finetune(Y, cin, nIter=5):
    """Fine tuning of components within greedyROI using rank-1 NMF
    """
    debug_ = False

    if debug_:
        import os
        f = open('_LOG_1_' + str(os.getpid()), 'w+')
        f.write('Y:' + str(np.mean(Y)) + '\n')
        f.write('cin:' + str(np.mean(cin)) + '\n')
        f.close()

    c = cin
    for iter in range(nIter):
        a = np.maximum(np.dot(Y, c), 0)
        a = old_div(a, np.sqrt(np.sum(a**2)))
        c = np.sum(Y * a[..., np.newaxis], tuple(np.arange(Y.ndim - 1)))

    return a, c

#%%


def imblur(Y, sig=5, siz=11, nDimBlur=None, kernel=None, opencv = True):
    """Spatial filtering with a Gaussian or user defined kernel
    The parameters are specified in GreedyROI
    """
#    import cv2
    X = np.zeros(np.shape(Y))

    if kernel is None:
        if nDimBlur is None:
            nDimBlur = Y.ndim - 1
        else:
            nDimBlur = np.min((Y.ndim, nDimBlur))

        if np.isscalar(sig):
            sig = sig * np.ones(nDimBlur)

        if np.isscalar(siz):
            siz = siz * np.ones(nDimBlur)

        # xx = np.arange(-np.floor(siz[0] / 2), np.floor(siz[0] / 2) + 1)
        # yy = np.arange(-np.floor(siz[1] / 2), np.floor(siz[1] / 2) + 1)

        # hx = np.exp(-xx**2 / (2 * sig[0]**2))
        # hx /= np.sqrt(np.sum(hx**2))

        # hy = np.exp(-yy**2 / (2 * sig[1]**2))
        # hy /= np.sqrt(np.sum(hy**2))

        # temp = correlate(Y, hx[:, np.newaxis, np.newaxis], mode='constant')
        # X = correlate(temp, hy[np.newaxis, :, np.newaxis], mode='constant')

        # the for loop helps with memory
        # for t in range(np.shape(Y)[-1]):
        # temp = correlate(Y[:,:,t],hx[:,np.newaxis])#,mode='constant', cval=0.0)
        # X[:,:,t] = correlate(temp,hy[np.newaxis,:])#,mode='constant', cval=0.0)

        X = Y.copy()
        if opencv and nDimBlur == 2:
            if X.ndim > 2:
                for frame in range(X.shape[-1]):
                    X[:,:,frame] = cv2.GaussianBlur(X[:,:,frame],tuple(siz),sig[0],sig[1],cv2.BORDER_CONSTANT,0)

            else:
                X = cv2.GaussianBlur(X,tuple(siz),sig[0],sig[1],cv2.BORDER_CONSTANT,0)
        else:
            for i in range(nDimBlur):
                h = np.exp(
                    old_div(-np.arange(-np.floor(old_div(siz[i], 2)), np.floor(old_div(siz[i], 2)) + 1)**2, (2 * sig[i]**2)))
                h /= np.sqrt(h.dot(h))
                shape = [1] * len(Y.shape)
                shape[i] = -1
                X = correlate(X, h.reshape(shape), mode='constant')


    else:
        X = correlate(Y, kernel[..., np.newaxis], mode='constant')
        # for t in range(np.shape(Y)[-1]):
        #    X[:,:,t] = correlate(Y[:,:,t],kernel,mode='constant', cval=0.0)

    return X

#%%


def hals(Y, A, C, b, f, bSiz=3, maxIter=5):
    """ Hierarchical alternating least square method for solving NMF problem
    Y = A*C + b*f

    input:
       Y:      d1 X d2 [X d3] X T, raw data. It will be reshaped to (d1*d2[*d3]) X T in this
       function
       A:      (d1*d2[*d3]) X K, initial value of spatial components
       C:      K X T, initial value of temporal components
       b:      (d1*d2[*d3]) X nb, initial value of background spatial component
       f:      nb X T, initial value of background temporal component
       bSiz:   int or tuple of int
       blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will
       be convolved with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0.
       maxIter: maximum iteration of iterating HALS.

    output:
    the updated A, C, b, f

    Author: Johannes Friedrich, Andrea Giovannucci
    """

    #%% smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    nb = b.shape[1]  # number of background components
    if isinstance(bSiz, (int, float)):
        bSiz = [bSiz] * len(dims)
    ind_A = nd.filters.uniform_filter(np.reshape(A, dims + (K,), order='F'), size=bSiz + [0])
    ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels

    def HALS4activity(Yr, A, C, iters=2):
        U = A.T.dot(Yr)
        V = A.T.dot(A)
        for _ in range(iters):
            for m in range(len(U)):  # neurons and background
                C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) / V[m, m], 0, np.inf)
        return C

    def HALS4shape(Yr, A, C, iters=2):
        U = C.dot(Yr.T)
        V = C.dot(C.T)
        for _ in range(iters):
            for m in range(K):  # neurons
                ind_pixels = np.squeeze(ind_A[:, m].toarray())
                A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                           ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                            V[m, m]), 0, np.inf)
            for m in range(nb):  # background
                A[:, K + m] = np.clip(A[:, K + m] + ((U[K + m] - V[K + m].dot(A.T)) /
                                                     V[K + m, K + m]), 0, np.inf)
        return A

    Ab = np.c_[A, b]
    Cf = np.r_[C, f.reshape(nb, -1)]
    for _ in range(maxIter):
        Cf = HALS4activity(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)

    return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)
