import numpy as np
from sklearn.decomposition import NMF
from skimage.transform import downscale_local_mean, resize
import scipy.ndimage as nd
import scipy.sparse as spr
import scipy
from ca_source_extraction.utilities import com,local_correlations
#%%
def initialize_components(Y, K=30, gSig=[5,5], gSiz=None, ssub=1, tsub=1, nIter = 5, maxIter=5, kernel = None, use_hals=True): 
    """Initalize components 
    
    This method uses a greedy approach followed by hierarchical alternative least squares (HALS) NMF.
    Optional use of spatio-temporal downsampling to boost speed.
     
    Parameters
    ----------   
    Y: np.ndarray
         d1 x d2 x T movie, raw data.
    K: [optional] int
        number of neurons to extract (default value: 30).
    tau: [optional] list,tuple
        standard deviation of neuron size along x and y (default value: (5,5).
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
    use_hals: [bool]
        Whether to refine components with the hals method
  
    Returns
    --------    
    Ain: np.ndarray        
        (d1*d2) x K , spatial filter of each neuron.
    Cin: np.ndarray
        T x K , calcium activity of each neuron.
    center: np.ndarray
        K x 2 , inferred center of each neuron.
    bin: np.ndarray
        (d1*d2) X nb, initialization of spatial background.
    fin: np.ndarray
        nb X T matrix, initalization of temporal background.    
        
    """
    
    if gSiz is None:
        gSiz=(2*gSig[0] + 1,2*gSig[1] + 1)
     
    d1,d2,T=np.shape(Y) 
    # rescale according to downsampling factor
    gSig = np.round([gSig[0]/ssub,gSig[1]/ssub]).astype(np.int)
    gSiz = np.round([gSiz[0]/ssub,gSiz[1]/ssub]).astype(np.int)    
    
    # spatial downsampling
    if ssub!=1 or tsub!=1:
        print "Spatial Downsampling ..."
        Y_ds = downscale_local_mean(Y,(ssub,ssub,tsub))
    else:
        Y_ds = Y
   
    print 'Roi Extraction...'    
    Ain, Cin, _, b_in, f_in = greedyROI2d(Y_ds, nr = K, gSig = gSig, gSiz = gSiz, nIter=nIter, kernel = kernel)
    if use_hals:    
        print 'Refining Components...'    
        Ain, Cin, b_in, f_in = hals_2D(Y_ds, Ain, Cin, b_in, f_in,maxIter=maxIter);
    
    #center = ssub*com(Ain,d1s,d2s) 
    d1s,d2s,Ts=np.shape(Y_ds)
    Ain = np.reshape(Ain, (d1s, d2s,K),order='F')
    Ain = resize(Ain, [d1, d2, K],order=1)
    
    Ain = np.reshape(Ain, (d1*d2, K),order='F') 
    
    b_in=np.reshape(b_in,(d1s, d2s),order='F')
    
    b_in = resize(b_in, [d1, d2]);
    
    b_in = np.reshape(b_in, (d1*d2, 1),order='F')
    
    Cin = resize(Cin, [K, T])
    f_in = resize(np.atleast_2d(f_in), [1, T])    
    center = com(Ain,d1,d2)
    
    return Ain, Cin, b_in, f_in, center
    

#%%
def greedyROI2d(Y, nr=30, gSig = [5,5], gSiz = [11,11], nIter = 5, kernel = None, Cn = None):
    """
    Greedy initialization of spatial and temporal components using spatial Gaussian filtering
    Inputs:
    Y: np.array
        3d array of fluorescence data with time appearing in the last axis.
        Support for 3d imaging will be added in the future
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
        2d array of size nr x 2 with the components centroids
        
    Author: Eftychios A. Pnevmatikakis based on a matlab implementation by Yuanjun Gao
            Simons Foundation, 2015
    """
    d = np.shape(Y)
    med = np.median(Y,axis = -1)
    Y = Y - med[...,np.newaxis]
    gHalf = np.array(gSiz)/2
    gSiz = 2*gHalf + 1
    
    A = np.zeros((np.prod(d[0:-1]),nr))    
    C = np.zeros((nr,d[-1]))    
    center = np.zeros((nr,2))
    
    [M, N, T] = np.shape(Y)

    rho = imblur(Y, sig = gSig, siz = gSiz, nDimBlur = Y.ndim-1, kernel = kernel)

    if Cn is not None:
        rho=rho*Cn[...,np.newaxis]
        
    v = np.sum(rho**2,axis=-1)
    
    for k in range(nr):
        ind = np.argmax(v)
        ij = np.unravel_index(ind,d[0:-1])
        center[k,0] = ij[0]
        center[k,1] = ij[1]
        iSig = [np.maximum(ij[0]-gHalf[0],0),np.minimum(ij[0]+gHalf[0]+1,d[0])]
        jSig = [np.maximum(ij[1]-gHalf[1],0),np.minimum(ij[1]+gHalf[1]+1,d[1])]
        dataTemp = Y[iSig[0]:iSig[1],jSig[0]:jSig[1],:].copy()
        traceTemp = np.squeeze(rho[ij[0],ij[1],:])
        coef, score = finetune2d(dataTemp, traceTemp, nIter = nIter)
        C[k,:] = np.squeeze(score) 
        dataSig = coef[...,np.newaxis]*score[np.newaxis,np.newaxis,...]
        [xSig,ySig] = np.meshgrid(np.arange(iSig[0],iSig[1]),np.arange(jSig[0],jSig[1]),indexing = 'xy')
        arr = np.array([np.reshape(xSig,(1,np.size(xSig)),order='F').squeeze(),np.reshape(ySig,(1,np.size(ySig)),order='F').squeeze()])
        indeces = np.ravel_multi_index(arr,d[0:-1],order='F')  
        A[indeces,k] = np.reshape(coef,(1,np.size(coef)),order='C').squeeze()
        Y[iSig[0]:iSig[1],jSig[0]:jSig[1],:] -= dataSig.copy()
        if k < nr-1:
            iMod = [np.maximum(ij[0]-2*gHalf[0],0),np.minimum(ij[0]+2*gHalf[0]+1,d[0])]
            iModLen = iMod[1]-iMod[0]
            jMod = [np.maximum(ij[1]-2*gHalf[1],0),np.minimum(ij[1]+2*gHalf[1]+1,d[1])]
            jModLen = jMod[1]-jMod[0]
            iLag = iSig - iMod[0] + 0
            jLag = jSig - jMod[0] + 0
            dataTemp = np.zeros((iModLen,jModLen))
            dataTemp[iLag[0]:iLag[1],jLag[0]:jLag[1]] = coef
            dataTemp = imblur(dataTemp[...,np.newaxis],sig=gSig,siz=gSiz,kernel = kernel)            
            rhoTEMP = dataTemp*score[np.newaxis,np.newaxis,...]
            rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:] -= rhoTEMP.copy()
            v[iMod[0]:iMod[1],jMod[0]:jMod[1]] = np.sum(rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:]**2,axis = -1)            

    res = np.reshape(Y,(M*N,T), order='F') + med.flatten(order='F')[:,None]

    model = NMF(n_components=1, init='random', random_state=0)
    
    b_in = model.fit_transform(np.maximum(res,0));
    f_in = model.components_.squeeze();    
   
    return A, C, center, b_in, f_in

#%%
def finetune2d(Y, cin, nIter = 5):
    """Fine tuning of components within greedyROI using rank-1 NMF
    """
    for iter in range(nIter):
        a = np.maximum(np.dot(Y,cin),0)
        a = a/np.sqrt(np.sum(a**2))
        c = np.sum(Y*a[...,np.newaxis],tuple(np.arange(Y.ndim-1)))
    
    return a, c

#%%    
def imblur(Y, sig = 5, siz = 11, nDimBlur = None, kernel = None):
    """Spatial filtering with a Gaussian or user defined kernel
    The parameters are specified in GreedyROI2d
    """
    from scipy.ndimage.filters import correlate     
    
    X = np.zeros(np.shape(Y))
    
    if kernel is None:
        if nDimBlur is None:
            nDimBlur = Y.ndim - 1
        else:
            nDimBlur = np.min((Y.ndim,nDimBlur))
            
        if np.isscalar(sig):
            sig = sig*np.ones(nDimBlur)
            
        if np.isscalar(siz):
            siz = siz*np.ones(nDimBlur)
        
        xx = np.arange(-np.floor(siz[0]/2),np.floor(siz[0]/2)+1)
        yy = np.arange(-np.floor(siz[1]/2),np.floor(siz[1]/2)+1)
    
        hx = np.exp(-xx**2/(2*sig[0]**2))
        hx /= np.sqrt(np.sum(hx**2))
        
        hy = np.exp(-yy**2/(2*sig[1]**2))
        hy /= np.sqrt(np.sum(hy**2))    
    
        temp = correlate(Y,hx[:,np.newaxis,np.newaxis],mode='constant')
        X = correlate(temp,hy[np.newaxis,:,np.newaxis],mode='constant')  

        ## the for loop helps with memory
        #for t in range(np.shape(Y)[-1]):
            #temp = correlate(Y[:,:,t],hx[:,np.newaxis])#,mode='constant', cval=0.0)
            #X[:,:,t] = correlate(temp,hy[np.newaxis,:])#,mode='constant', cval=0.0)

    else:
        X = correlate(Y,kernel[...,np.newaxis],mode='constant')
        #for t in range(np.shape(Y)[-1]):
        #    X[:,:,t] = correlate(Y[:,:,t],kernel,mode='constant', cval=0.0)
            
    return X

#%%
def hals_2D(Y,A,C,b,f,bSiz=3,maxIter=5):
    """ Hierarchical alternating least square method for solving NMF problem  
    Y = A*C + b*f 
    
    input: 
       Y:      d1 X d2 X T, raw data. It will be reshaped to (d1*d2) X T in this
       function 
       A:      (d1*d2) X K, initial value of spatial components 
       C:      K X T, initial value of temporal components 
       b:      (d1*d2) X 1, initial value of background spatial component 
       f:      1 X T, initial value of background temporal component
    
       bSiz:   blur size. A box kernel (bSiz X bSiz) will be convolved
       with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0. 
       maxIter: maximum iteration of iterating HALS. 
    
    output:
    the updated A, C, b, f
    
    Author: Andrea Giovannucci, Columbia University, based on a matlab implementation from Pengcheng Zhou
       """
    
    #%% smooth the components
    d1, d2, T = np.shape(Y)
    ind_A = nd.filters.uniform_filter(np.reshape(A,(d1,d2, A.shape[1]),order='F'),size=(bSiz,bSiz,0))
    ind_A = np.reshape(ind_A>1e-10,(d1*d2, A.shape[1]),order='F')
    
    #%%
    ind_A = spr.csc_matrix(ind_A);  #indicator of nonnero pixels 
    K = np.shape(A)[1] #number of neurons 
    #%% update spatial and temporal components neuron by neurons
    Yres = np.reshape(Y, (d1*d2, T),order='F') - A.dot(C) - b.dot(f[None,:]);
    print 'First residual is ' + str(scipy.linalg.norm(Yres, 'fro')) + '\n';
     
    for miter in range(maxIter):
        for mcell in range(K):
            ind_pixels = np.squeeze(np.asarray(ind_A[:, mcell].todense()))
            tmp_Yres = Yres[ind_pixels, :]
           # print 'First residual is ' + str(scipy.linalg.norm(tmp_Yres, 'fro')) + '\n';
            # update temporal component of the neuron
            c0 = C[mcell, :].copy();
            a0 = A[ind_pixels, mcell].copy()
            norm_a2 = scipy.linalg.norm(a0, ord=2)**2;
            C[mcell, :] = np.maximum(0, c0 + a0.T.dot(tmp_Yres/norm_a2))
            tmp_Yres = tmp_Yres + np.dot(a0[:,None],(c0-C[mcell,:])[None,:])
           # print 'First residual is ' + str(scipy.linalg.norm(tmp_Yres, 'fro')) + '\n';
            
            # update spatial component of the neuron
            norm_c2 = scipy.linalg.norm(C[mcell,:],ord=2)**2
            tmp_a = np.maximum(0, a0 + np.dot(tmp_Yres,C[mcell, :].T/norm_c2))
            A[ind_pixels, mcell] = tmp_a
            Yres[ind_pixels,:] = tmp_Yres + np.dot(a0[:,None]-tmp_a[:,None],C[None,mcell,:])
            #print 'First residual is ' + str(scipy.linalg.norm(tmp_Yres, 'fro')) + '\n';
        
        # update temporal component of the background
        f0 = f.copy();
        b0 = b.copy();
        norm_b2 = scipy.linalg.norm(b0,ord=2)**2
        f = np.maximum(0, f0 + np.dot(b0.T,Yres/norm_b2))
        Yres = Yres + np.dot(b0,f0-f)
        #print 'First residual is ' + str(scipy.linalg.norm(Yres, 'fro')) + '\n';        
        
        # update spatial component of the background
        norm_f2 = scipy.linalg.norm(f, ord=2)**2
        b = np.maximum(0, b0 + np.dot(Yres,f.T/norm_f2))
        Yres = Yres + np.dot(b0-b,f)
        
        print 'Iteration:' + str(miter) + ', the norm of residual is ' + str(scipy.linalg.norm(Yres, 'fro'))
    
    return A, C, b, f