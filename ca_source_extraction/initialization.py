import numpy as np
from sklearn.decomposition import NMF
from skimage.transform import downscale_local_mean, resize
import scipy.ndimage as nd
import scipy.sparse as spr
import scipy
from scipy.ndimage.measurements import center_of_mass
import utilities
#%%


def initialize_components(Y, K=30, gSig=[5, 5], gSiz=None, ssub=1, tsub=1, nIter=5, maxIter=5,
                          kernel=None, use_hals=True, normalize = True, img=None,method = 'greedy_roi',max_iter_snmf=500,alpha_snmf=10e2,sigma_smooth_snmf=(.5,.5,.5),perc_baseline_snmf=20):
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
    normalize: [optional] bool
        Whether to normalize data before running the initialization
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
    
        
    if gSiz is None:
        gSiz = 2 * np.asarray(gSig) + 1

    d, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    # rescale according to downsampling factor
    gSig = np.round(np.asarray(gSig) / ssub).astype(np.int)
    gSiz = np.round(np.asarray(gSiz) / ssub).astype(np.int)

    print 'Noise Normalization'
    if normalize is True:
        if img is None:
            img = np.mean(Y,axis=-1)
            img += np.median(img)
            
        Y = Y / np.reshape(img, d + (-1,),order='F')
        alpha_snmf /= np.mean(img)

    # spatial downsampling
    mean_val = np.mean(Y)
    if ssub != 1 or tsub != 1:
        print "Spatial Downsampling ..."
        Y_ds = downscale_local_mean(Y, tuple([ssub] * len(d) + [tsub]), cval=mean_val)
    else:
        Y_ds = Y

    print 'Roi Extraction...'
    if method == 'greedy_roi':
        Ain, Cin, _, b_in, f_in = greedyROI(
            Y_ds, nr=K, gSig=gSig, gSiz=gSiz, nIter=nIter, kernel=kernel)
        if use_hals:
            print 'Refining Components...'
            Ain, Cin, b_in, f_in = hals(Y_ds, Ain, Cin, b_in, f_in, maxIter=maxIter)
    elif method == 'sparse_nmf':
        Ain, Cin, _, b_in, f_in = sparseNMF( Y_ds,nr=K,  max_iter_snmf=max_iter_snmf, alpha= alpha_snmf, sigma_smooth=sigma_smooth_snmf,remove_baseline=True,perc_baseline=perc_baseline_snmf)
#        print np.sum(Ain), np.sum(Cin)        
#        print 'Refining Components...'
#        Ain, Cin, b_in, f_in = hals(Y_ds, Ain, Cin, b_in, f_in, maxIter=maxIter)            
#        print np.sum(Ain), np.sum(Cin)
    else:
        print method
        raise Exception("Unsupported method")
        

    K=np.shape(Ain)[-1]
    ds = Y_ds.shape[:-1]
    Ain = np.reshape(Ain, ds + (K,), order='F')
    
    if len(ds) == 2:

        Ain = resize(Ain, d + (K,), order=1)

    else:  # resize only deals with 2D images, hence apply resize twice

        Ain = np.reshape([resize(a, d[1:] + (K,), order=1) for a in Ain], (ds[0], d[1] * d[2], K), order='F')
        Ain = resize(Ain, (d[0], d[1] * d[2], K), order=1)

    Ain = np.reshape(Ain, (np.prod(d), K), order='F')

    b_in = np.reshape(b_in, ds, order='F')

#    b_in = resize(b_in, ds)
    b_in = resize(b_in, d)
    
    b_in = np.reshape(b_in, (-1, 1), order='F')

    Cin = resize(Cin, [K, T])
    
    f_in = resize(np.atleast_2d(f_in), [1, T])
    # center = com(Ain, *d)
    center = np.asarray([center_of_mass(a.reshape(d, order='F')) for a in Ain.T])

    if normalize is True:    
        #import pdb
        #pdb.set_trace()        
        Ain = Ain * np.reshape(img, (np.prod(d), -1),order='F')    
        b_in = b_in * np.reshape(img, (np.prod(d), -1),order='F') 
        #b_in = np.atleast_2d(b_in * img.flatten('F')) #np.reshape(img, (np.prod(d), -1),order='F')
        Y = Y * np.reshape(img, d + (-1,),order='F')
        
    
    return Ain, Cin, b_in, f_in, center

#%%
def sparseNMF(Y_ds, nr,  max_iter_snmf=500, alpha= 10e2, sigma_smooth=(.5,.5,.5),remove_baseline=True,perc_baseline=20):
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
        percentile to remove frmo movie beofre NMF
    
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
    
    m = scipy.ndimage.gaussian_filter(np.transpose(Y_ds,[2,0,1]), sigma=sigma_smooth, mode='nearest',truncate=2)
    if remove_baseline:      
        bl = np.percentile(m,perc_baseline,axis=0)
        m1 = np.maximum(0,m-bl)
    else:
        bl=0
        m1=m
    
    mdl = NMF(n_components=nr,verbose=False,init='nndsvd',tol=1e-10,max_iter=max_iter_snmf,shuffle=True,alpha=alpha,l1_ratio=1)
    T,d1,d2=np.shape(m1)
    d=d1*d2
    yr=np.reshape(m1,[T,d],order='F')    
    C=mdl.fit_transform(yr).T
    A=mdl.components_.T   
    ind_good = np.where(np.logical_and((np.sum(A,0)*np.std(C,axis=1)) > 0, np.sum(A>np.mean(A),axis=0)<d/3 ) )[0] 
#    A_in=A[:, ind_good] 
#    C_in=C[ind_good, :]    
    ind_bad = np.where(np.logical_or((np.sum(A,0)*np.std(C,axis=1)) == 0, np.sum(A>np.mean(A),axis=0)>d/3 ) )[0] 
    A_in=np.zeros_like(A)
    
    C_in=np.zeros_like(C)
    A_in[:, ind_good] = A[:, ind_good]
    C_in[ind_good, :] = C[ind_good, :]
    A_in=A_in*(A_in>(.1*np.max(A_in,axis=0))[np.newaxis,:])
    A_in[:3, ind_bad] = .0001
    C_in[ind_bad, :3] =  .0001
    
#    import pdb
#    pdb.set_trace()
    m1 = yr.T-A_in.dot(C_in)+ np.maximum(0,bl.flatten())[:,np.newaxis]
    
    
    model = NMF(n_components=1, init='random', random_state=0,max_iter=max_iter_snmf)

    b_in = model.fit_transform(np.maximum(m1, 0))
    f_in = model.components_.squeeze()
    
    center=utilities.com(A_in,d1,d2)
#    for iter in range(max_iter_snmf):
#        f = np.maximum(b.T.dot(scipy.linalg.solve(scipy.linalg.norm(b).T**2,Y.T),0);
#        b = np.maximum(Y.dot(scipy.linalg.solve(scipy.linalg.norm(f).T**2,f.T),0);        
#    end
    
    return A_in, C_in, center, b_in, f_in
    
#%%
def greedyROI(Y, nr=30, gSig=[5, 5], gSiz=[11, 11], nIter=5, kernel=None):
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
    d = np.shape(Y)
    med = np.median(Y, axis=-1)
    Y = Y - med[..., np.newaxis]
    gHalf = np.array(gSiz) / 2
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
        dataTemp = Y[map(lambda a: slice(*a), ijSig)].copy()
        traceTemp = np.squeeze(rho[ij])
        coef, score = finetune(dataTemp, traceTemp, nIter=nIter)
        C[k, :] = np.squeeze(score)
        dataSig = coef[..., np.newaxis] * score.reshape([1] * (Y.ndim - 1) + [-1])
        xySig = np.meshgrid(*[np.arange(s[0], s[1]) for s in ijSig], indexing='xy')
        arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze() for s in xySig])
        indeces = np.ravel_multi_index(arr, d[0:-1], order='F')
        A[indeces, k] = np.reshape(coef, (1, np.size(coef)), order='C').squeeze()
        Y[map(lambda a: slice(*a), ijSig)] -= dataSig.copy()
        if k < nr - 1:
            Mod = [[np.maximum(ij[c] - 2 * gHalf[c], 0),
                    np.minimum(ij[c] + 2 * gHalf[c] + 1, d[c])] for c in range(len(ij))]
            ModLen = [m[1] - m[0] for m in Mod]
            Lag = [ijSig[c] - Mod[c][0] for c in range(len(ij))]
            dataTemp = np.zeros(ModLen)
            dataTemp[map(lambda a: slice(*a), Lag)] = coef
            dataTemp = imblur(dataTemp[..., np.newaxis], sig=gSig, siz=gSiz, kernel=kernel)
            rhoTEMP = dataTemp * score.reshape([1] * (Y.ndim - 1) + [-1])
            rho[map(lambda a: slice(*a), Mod)] -= rhoTEMP.copy()
            v[map(lambda a: slice(*a), Mod)] = np.sum(
                rho[map(lambda a: slice(*a), Mod)]**2, axis=-1)

    res = np.reshape(Y, (np.prod(d[0:-1]), d[-1]), order='F') + med.flatten(order='F')[:, None]

    model = NMF(n_components=1, init='random', random_state=0)

    b_in = model.fit_transform(np.maximum(res, 0))
    f_in = model.components_.squeeze()

    return A, C, center, b_in, f_in

#%%


def finetune(Y, cin, nIter=5):
    """Fine tuning of components within greedyROI using rank-1 NMF
    """
    for iter in range(nIter):
        a = np.maximum(np.dot(Y, cin), 0)
        a = a / np.sqrt(np.sum(a**2))
        c = np.sum(Y * a[..., np.newaxis], tuple(np.arange(Y.ndim - 1)))

    return a, c

#%%


def imblur(Y, sig=5, siz=11, nDimBlur=None, kernel=None):
    """Spatial filtering with a Gaussian or user defined kernel
    The parameters are specified in GreedyROI
    """
    from scipy.ndimage.filters import correlate

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
        for i in range(nDimBlur):
            h = np.exp(-np.arange(-np.floor(siz[i] / 2), np.floor(siz[i] / 2) + 1)**2
                       / (2 * sig[i]**2))
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
       b:      (d1*d2[*d3]) X 1, initial value of background spatial component
       f:      1 X T, initial value of background temporal component
       bSiz:   int or tuple of int
       blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will 
       be convolved with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0.
       maxIter: maximum iteration of iterating HALS.

    output:
    the updated A, C, b, f

    Author: Johannes Friedrich, Andrea Giovannucci, Columbia University
       """

    #%% smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    if isinstance(bSiz, (int, long, float)):
        bSiz = [bSiz] * len(dims)
    ind_A = nd.filters.uniform_filter(np.reshape(A, dims + (K,),
                                                 order='F'), size=bSiz + [0])
    ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels

    def HALS4activity(data, S, activity, iters=2):
        A = S.dot(data)
        B = S.dot(S.T)
        for _ in range(iters):
            for mcell in range(K + 1):  # neurons and background
                activity[mcell] += (A[mcell] - np.dot(B[mcell].T, activity)) / B[mcell, mcell]
                activity[mcell][activity[mcell] < 0] = 0
        return activity

    def HALS4shape(data, S, activity, iters=2):
        C = activity.dot(data.T)
        D = activity.dot(activity.T)
        for _ in range(iters):
            for mcell in range(K):  # neurons
                ind_pixels = np.squeeze(np.asarray(ind_A[:, mcell].todense()))
                S[mcell, ind_pixels] += (C[mcell, ind_pixels] -
                                         np.dot(D[mcell], S[:, ind_pixels])) / D[mcell, mcell]
                S[mcell, ind_pixels][S[mcell, ind_pixels] < 0] = 0
            S[K] += (C[K] - np.dot(D[K], S)) / D[K, K]  # background
            S[K][S[K] < 0] = 0
        return S

    Ab = np.c_[A, b].T
    Cf = np.r_[C, f.reshape(1, -1)]
    for miter in range(maxIter):
        Cf = HALS4activity(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)

    return Ab[:-1].T, Cf[:-1], Ab[-1].reshape(-1, 1), Cf[-1].reshape(1, -1)
