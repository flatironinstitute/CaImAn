# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:22:26 2015

@author: agiovann
"""



#%%
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label,center_of_mass
import scipy 
import numpy as np
import time
from scipy.optimize import linear_sum_assignment   
import json
from skimage.filters import sobel
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.draw import polygon
import pylab as pl
#from cell_magic_wand import cell_magic_wand,cell_magic_wand_3d_IOU
#%%
def com(A, d1, d2):
    """Calculation of the center of mass for spatial components
     Inputs:
     A:   np.ndarray
          matrix of spatial components (d x K)
     d1:  int
          number of pixels in x-direction
     d2:  int
          number of pixels in y-direction

     Output:
     cm:  np.ndarray
          center of mass for spatial components (K x 2)
    """
    nr = np.shape(A)[-1]
    Coor = dict()
    Coor['x'] = np.kron(np.ones((d2, 1)), np.expand_dims(range(d1), axis=1))
    Coor['y'] = np.kron(np.expand_dims(range(d2), axis=1), np.ones((d1, 1)))
    cm = np.zeros((nr, 2))        # vector for center of mass
    cm[:, 0] = np.dot(Coor['x'].T, A) / A.sum(axis=0)
    cm[:, 1] = np.dot(Coor['y'].T, A) / A.sum(axis=0)

    return cm
#%% 
def mask_to_2d(mask):
    if mask.ndim > 2:
        ncomps,d1,d2 = np.shape(mask)
        dims = d1,d2
        return scipy.sparse.coo_matrix(np.reshape(mask[:].transpose([1,2,0]),(np.prod(dims),-1,),order='F'))
    else:    
        dims  = np.shape(mask)    
        return scipy.sparse.coo_matrix(np.reshape(mask,(np.prod(dims),-1,),order='F'))
#%%
def nf_match_neurons_in_binary_masks(masks_gt,masks_comp,thresh_cost=.7, min_dist = 10, print_assignment= False, plot_results = False, Cn=None):
    '''
    Match neurons expressed as binary masks. Uses Hungarian matching algorithm
    
    Parameters:
    -----------
    
    masks_gt: ndarray  ncomponents x d1 x d2
        ground truth masks
    
    masks_comp: ndarray  ncomponents x d1 x d2
        mask to compare to
    
    thresh_cost: double 
        max cost accepted 
 
    min_dist: min distance between cm
    
    print_assignment:
        for hungarian algorithm
    
    plot_results: bool    

    Cn: 
        correlation image or median

    Returns:
    --------
    idx_tp_1:
        indeces true pos ground truth mask 
    
    idx_tp_2:
        indeces ground truth comp

    idx_fn_1:
        indeces false neg

    idx_fp_2:
        indeces false pos

    performance:  
    
    '''
    #%%
    ncomps,d1,d2 = np.shape(masks_gt)
    dims = d1,d2
    A_ben = scipy.sparse.csc_matrix(np.reshape(masks_gt[:].transpose([1,2,0]),(np.prod(dims),-1,),order='F'))
    A_cnmf = scipy.sparse.csc_matrix(np.reshape(masks_comp[:].transpose([1,2,0]),(np.prod(dims),-1,),order='F'))
    
    cm_ben = [ scipy.ndimage.center_of_mass(mm) for mm in masks_gt]
    cm_cnmf = [ scipy.ndimage.center_of_mass(mm) for mm in masks_comp]
    #%% find distances and matches
    D=distance_masks([A_ben,A_cnmf],[cm_ben,cm_cnmf], min_dist )    
    matches,costs=find_matches(D,print_assignment=print_assignment)
    matches=matches[0]
    costs=costs[0]

    #%% compute precision and recall
    TP = np.sum(np.array(costs)<thresh_cost)*1.
    FN = np.shape(masks_gt)[0] - TP
    FP = np.shape(masks_comp)[0] - TP
    TN = 0
    
    performance = dict() 
    performance['recall'] = TP/(TP+FN)
    performance['precision'] = TP/(TP+FP) 
    performance['accuracy'] = (TP+TN)/(TP+FP+FN+TN)
    performance['f1_score'] = 2*TP/(2*TP+FP+FN)
    print performance
    #%%
    idx_tp = np.where(np.array(costs)<thresh_cost)[0]
    idx_tp_ben = matches[0][idx_tp] 
    idx_tp_cnmf = matches[1][idx_tp]
     
    idx_fn = np.setdiff1d(range(np.shape(masks_gt)[0]),idx_tp)
    
    idx_fp =  np.setdiff1d(range(np.shape(masks_comp)[0]),matches[1][idx_tp])
    
    idx_fp_cnmf = idx_fp
    
    idx_tp_gt,idx_tp_comp, idx_fn_gt, idx_fp_comp = idx_tp_ben,idx_tp_cnmf, idx_fn, idx_fp_cnmf
    
    if plot_results:
        pl.rcParams['pdf.fonttype'] = 42
        font = {'family' : 'Myriad Pro',
                'weight' : 'regular',
                'size'   : 20}
        pl.rc('font', **font)
        lp,hp = np.nanpercentile(Cn,[5,95])
        pl.subplot(1,2,1)
        pl.imshow(Cn,vmin=lp,vmax=hp)
        [pl.contour(mm,levels=[0],colors='w',linewidths=2) for mm in masks_comp[idx_tp_comp]] 
        [pl.contour(mm,levels=[0],colors='r',linewidths=2) for mm in masks_gt[idx_tp_gt]] 
        pl.title('TRUE POSITIVE')
        #pl.legend([comp_str,'GT'])
        pl.axis('off')
        pl.subplot(1,2,2)
        pl.imshow(Cn,vmin=lp,vmax=hp)
        [pl.contour(mm,levels=[0],colors='w',linewidths=2) for mm in masks_comp[idx_fp_comp]] 
        [pl.contour(mm,levels=[0],colors='r',linewidths=2) for mm in masks_gt[idx_fn]] 
        pl.title('FALSE POSITIVE (w), FALSE NEGATIVE (r)')
        #pl.legend([comp_str,'GT'])
        pl.axis('off')
    return  idx_tp_gt,idx_tp_comp, idx_fn_gt, idx_fp_comp, performance 
    
    
#%% compute mask distances
def distance_masks(M_s,cm_s,max_dist):
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order, with matrix i compared with matrix i+1
    
    Parameters
    ----------
    M_s: list of ndarrays
        The thresholded A matrices (masks) to compare, output of threshold_components
    
    cm_s: list of list of 2-ples
        the centroids of the components in each M_s
    
    max_dist: float
        maximum distance among centroids allowed between components. This corresponds to a distance at which two components are surely disjoined
    
    
    
    Returns:
    --------
    D_s: list of matrix distances
    
    """
    D_s=[]

    for M1,M2,cm1,cm2 in zip(M_s[:-1],M_s[1:],cm_s[:-1],cm_s[1:]):
        print 'New Pair **'
        M1= M1.copy()[:,:]
        M2= M2.copy()[:,:]
        d_1=np.shape(M1)[-1]
        d_2=np.shape(M2)[-1]
        D = np.ones((d_1,d_2));
        
        cm1=np.array(cm1)
        cm2=np.array(cm2)
        for i in range(d_1):
            if i%100==0:
                print i
            k=M1[:,np.repeat(i,d_2)]+M2
    #        h=M1[:,np.repeat(i,d_2)].copy()
    #        h.multiply(M2)
            for j  in range(d_2): 
    
                dist = np.linalg.norm(cm1[i]-cm2[j])
                if dist<max_dist:
                    union = k[:,j].sum()
    #                intersection = h[:,j].nnz
                    intersection= np.array(M1[:,i].T.dot(M2[:,j]).todense()).squeeze()
        ##            intersect= np.sum(np.logical_xor(M1[:,i],M2[:,j]))
        ##            union=np.sum(np.logical_or(M1[:,i],M2[:,j]))
                    if union  > 0:
                        D[i,j] = 1-1.*intersection/(union-intersection)
                    else:
#                        print 'empty component: setting distance to max'
                        D[i,j] = 1.
                        
                    if np.isnan(D[i,j]):
                        raise Exception('Nan value produced. Error in inputs')
                else:
                    D[i,j] = 1
        
        D_s.append(D)            
    return D_s   

#%% find matches
def find_matches(D_s, print_assignment=False):
    
    matches=[]
    costs=[]
    t_start=time.time()
    for ii,D in enumerate(D_s):
        DD=D.copy()    
        if np.sum(np.where(np.isnan(DD)))>0:
            raise Exception('Distance Matrix contains NaN, not allowed!')
        
       
    #    indexes = m.compute(DD)
#        indexes = linear_assignment(DD)
        indexes = linear_sum_assignment(DD)
        indexes2=[(ind1,ind2) for ind1,ind2 in zip(indexes[0],indexes[1])]
        matches.append(indexes)
        DD=D.copy()   
        total = []
        for row, column in indexes2:
            value = DD[row,column]
            if print_assignment:
                print '(%d, %d) -> %f' % (row, column, value)
            total.append(value)      
        print  'FOV: %d, shape: %d,%d total cost: %f' % (ii, DD.shape[0],DD.shape[1], np.sum(total))
        print time.time()-t_start
        costs.append(total)      
        
    return matches,costs
      
#%%
def link_neurons(matches,costs,max_cost=0.6,min_FOV_present=None):
    """
    Link neurons from different FOVs given matches and costs obtained from the hungarian algorithm
    
    Parameters
    ----------
    matches: lists of list of tuple
        output of the find_matches function
    
    costs: list of lists of scalars
        cost associated to each match in matches
        
    max_cost: float
        maximum allowed value of the 1- intersection over union metric    
    
    min_FOV_present: int
        number of FOVs that must consequently contain the neuron starting from 0. If none 
        the neuro must be present in each FOV
    Returns:
    --------
    neurons: list of arrays representing the indices of neurons in each FOV
    
    """
    if min_FOV_present is None:
        min_FOV_present=len(matches)
    
    neurons=[]
    num_neurons=0
#    Yr_tot=[]
    num_chunks=len(matches)+1
    for idx in range(len(matches[0][0])):
        neuron=[]
        neuron.append(idx)
#        Yr=YrA_s[0][idx]+C_s[0][idx]
        for match,cost,chk in zip(matches,costs,range(1,num_chunks)):
            rows,cols=match        
            m_neur=np.where(rows==neuron[-1])[0].squeeze()
            if m_neur.size > 0:                           
                if cost[m_neur]<=max_cost:
                    neuron.append(cols[m_neur])
#                    Yr=np.hstack([Yr,YrA_s[chk][idx]+C_s[chk][idx]])
                else:                
                    break
            else:
                break
        if len(neuron)>min_FOV_present:           
            num_neurons+=1        
            neurons.append(neuron)
#            Yr_tot.append(Yr)
            
    
    neurons=np.array(neurons).T
    print 'num_neurons:' + str(num_neurons)
#    Yr_tot=np.array(Yr_tot)
    return neurons
    
   
#%%
def nf_load_masks(file_name,dims):
     # load the regions (training data only)
    with open(file_name) as f:
        regions = json.load(f)
    
    def tomask(coords):
        mask = np.zeros(dims)
        mask[zip(*coords)] = 1
        return mask
    
    masks = np.array([tomask(s['coordinates']) for s in regions])
    return masks
#%%
def nf_masks_to_json(binary_masks,json_filename):
    """
    Take as input a tensor of binary mask and produces json format for neurofinder 
    
    Parameters:
    -----------
    binary_masks: 3d ndarray (components x dimension 1  x dimension 2)
    
    json_filename: str
    
    Returns:
    --------
    regions: list of dict
        regions in neurofinder format
        
    """
    regions=[]
    for m in binary_masks:
        coords = [[x,y] for x,y in zip(*np.where(m))]
        regions.append({"coordinates":coords})
    
    
    with open(json_filename, 'w') as f:
        f.write(json.dumps(regions))
        
    return regions            
    
#%%
# Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
# License: MIT

def nf_read_roi(fileobj):
    '''
    points = read_roi(fileobj)

    Read ImageJ's ROI format
    '''
# This is based on:
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html


    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256


    pos = [4]
    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != 'Iout':
        raise IOError('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used

    roi_type = get8()
    # Discard second Byte:
    get8()

    if not (0 <= roi_type < 11):
        print('roireader: ROI type %s not supported' % roi_type)

    if roi_type != 7:

        print('roireader: ROI type %s not supported (!= 7)' % roi_type)

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()

    x1 = getfloat() 
    y1 = getfloat() 
    x2 = getfloat() 
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise ValueError('roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)
    points[:,1] = [getc() for i in xrange(n_coordinates)]
    points[:,0] = [getc() for i in xrange(n_coordinates)]
    points[:,1] += left
    points[:,0] += top
    points -= 1
    
    
    return points
#%%
def nf_read_roi_zip(fname,dims):
    
    import zipfile
    with zipfile.ZipFile(fname) as zf:
        coords = [nf_read_roi(zf.open(n))
                    for n in zf.namelist()]
    
    def tomask(coords):

        mask = np.zeros(dims)
        coords=np.array(coords)
#        rr, cc = polygon(coords[:,0]+1, coords[:,1]+1)        
        rr, cc = polygon(coords[:,0]+1, coords[:,1]+1)        
        mask[rr,cc]=1
#        mask[zip(*coords)] = 1
        
        return mask
    
    masks = np.array([tomask(s-1) for s in coords])
    return masks

#%%    
def extract_binary_masks_blob(A,  neuron_radius,dims,num_std_threshold=1, minCircularity= 0.5, minInertiaRatio = 0.2,minConvexity = .8):
    """
    Function to extract masks from data. It will also perform a preliminary selectino of good masks based on criteria like shape and size
    
    Parameters:
    ----------
    A: scipy.sparse matris
        contains the components as outputed from the CNMF algorithm
        
    neuron_radius: float 
        neuronal radius employed in the CNMF settings (gSiz)
    
    num_std_threshold: int
        number of times above iqr/1.349 (std estimator) the median to be considered as threshold for the component
    
     
    
    minCircularity: float
        parameter from cv2.SimpleBlobDetector
    
    minInertiaRatio: float
        parameter from cv2.SimpleBlobDetector
    
    minConvexity: float
        parameter from cv2.SimpleBlobDetector
    
    Returns:
    --------
    masks: np.array
    pos_examples:
    neg_examples:
    
    """    
    import cv2

    
    params = cv2.SimpleBlobDetector_Params()
    params.minCircularity = minCircularity
    params.minInertiaRatio = minInertiaRatio 
    params.minConvexity = minConvexity    
    
    # Change thresholds
    params.blobColor=255
    
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep= 3
    
#    min_elevation_map=
#    max_elevation_map=
    
    params.minArea = np.pi*((neuron_radius*.75)**2)
    #params.maxArea = 4*np.pi*((gSig[0]-1)**2)
    
    
    
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = True
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    
    masks_ws=[]
    pos_examples=[] 
    neg_examples=[]


    for count,comp in enumerate(A.tocsc()[:].T):

        print count
        comp_d=np.array(comp.todense())
#        comp_d=comp_d*(comp_d>(np.max(comp_d)*max_fraction))
#        comp_d=comp_d*(comp_d>(np.max(comp_d)*0))        
        comp_orig=np.reshape(comp.todense(),dims,order='F')
        comp_orig=(comp_orig-np.min(comp_orig))/(np.max(comp_orig)-np.min(comp_orig))*255
        gray_image=np.reshape(comp_d,dims,order='F')
        gray_image=(gray_image-np.min(gray_image))/(np.max(gray_image)-np.min(gray_image))*255
        gray_image=gray_image.astype(np.uint8)    

        
        # segment using watershed
        markers = np.zeros_like(gray_image)
        elevation_map = sobel(gray_image)
        thr_1=np.percentile(gray_image[gray_image>0],50)
        iqr=np.diff(np.percentile(gray_image[gray_image>0],(25,75)))
        thr_2=thr_1 + num_std_threshold*iqr/1.35
        markers[gray_image < thr_1] = 1
        markers[gray_image > thr_2] = 2  
#        local_maxi = peak_local_max(elevation_map, num_peaks=1,indices=False)
#        markers = ndi.label(local_maxi)[0]
        edges = watershed(elevation_map, markers)-1
        # only keep largest object 
        label_objects, nb_labels = ndi.label(edges)
        sizes = np.bincount(label_objects.ravel())
#        pl.subplot(2,2,1)
#        pl.imshow(gray_image)
#        pl.subplot(2,2,2)
#        pl.imshow(elevation_map)        
#        pl.subplot(2,2,3)
#        pl.imshow(markers)
#        pl.subplot(2,2,4)
#        pl.imshow(edges)
#        pl.pause(4.)
        if len(sizes)>1:           
            idx_largest = np.argmax(sizes[1:])    
            edges=(label_objects==(1+idx_largest))
            edges=ndi.binary_fill_holes(edges)
        else:
            print 'empty component'
            edges=np.zeros_like(edges)
        
#        edges=skimage.morphology.convex_hull_image(edges)
        if 1:
            masks_ws.append(edges)
            keypoints = detector.detect((edges*200.).astype(np.uint8))
        else:            
            masks_ws.append(gray_image)
            keypoints = detector.detect(gray_image)
            
        if len(keypoints)>0:
    #        im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            pos_examples.append(count)

        else:
            
            neg_examples.append(count)

    return np.array(masks_ws),np.array(pos_examples),np.array(neg_examples)
    
    
#%%
def extract_binary_masks_blob_parallel(A,  neuron_radius,dims,num_std_threshold=1, minCircularity= 0.5, minInertiaRatio = 0.2,minConvexity = .8,dview=None):
    
    pars=[]    
    for a in range(A.shape[-1]):
        pars.append([A[:,a],  neuron_radius,dims,num_std_threshold, minCircularity, minInertiaRatio,minConvexity])
    if dview is not None:
        res=dview.map_sync(extract_binary_masks_blob_parallel_place_holder,pars)
    else:
        res=map(extract_binary_masks_blob_parallel_place_holder,pars)
    
    masks=[]
    is_pos=[]
    is_neg=[]
    for r in res:
        masks.append(np.squeeze(r[0]))
        is_pos.append(r[1])
        is_neg.append(r[2])

    masks=np.dstack(masks).transpose([2,0,1])
    return  masks,is_pos,is_neg
#%%   
def extract_binary_masks_blob_parallel_place_holder(pars):
    A,  neuron_radius, dims, num_std_threshold , minCircularity , minInertiaRatio ,minConvexity =pars
    masks_ws,pos_examples,neg_examples = extract_binary_masks_blob(A,  neuron_radius,dims,num_std_threshold=num_std_threshold, minCircularity= 0.5, minInertiaRatio = minInertiaRatio,minConvexity = minConvexity)
    return masks_ws,len(pos_examples),len(neg_examples)
#%%
def extractROIsFromPCAICA(spcomps, numSTD=4, gaussiansigmax=2 , gaussiansigmay=2,thresh=None):
    """
    Given the spatial components output of the IPCA_stICA function extract possible regions of interest
    The algorithm estimates the significance of a components by thresholding the components after gaussian smoothing
    Parameters
    -----------
    spcompomps, 3d array containing the spatial components
    numSTD: number of standard deviation above the mean of the spatial component to be considered signiificant
    """        
    
    numcomps, width, height=spcomps.shape
    rowcols=int(np.ceil(np.sqrt(numcomps)));  
    
    #%
    allMasks=[];
    maskgrouped=[];
    for k in xrange(0,numcomps):
        comp=spcomps[k]
#            plt.subplot(rowcols,rowcols,k+1)
        comp=gaussian_filter(comp,[gaussiansigmay,gaussiansigmax])
        
        maxc=np.percentile(comp,99);
        minc=np.percentile(comp,1);
#            comp=np.sign(maxc-np.abs(minc))*comp;
        q75, q25 = np.percentile(comp, [75 ,25])
        iqr = q75 - q25
        minCompValuePos=np.median(comp)+numSTD*iqr/1.35;  
        minCompValueNeg=np.median(comp)-numSTD*iqr/1.35;            

        # got both positive and negative large magnitude pixels
        if thresh is None:
            compabspos=comp*(comp>minCompValuePos)-comp*(comp<minCompValueNeg);
        else:
            compabspos=comp*(comp>thresh)-comp*(comp<-thresh);

        #height, width = compabs.shape
        labeledpos, n = label(compabspos>0, np.ones((3,3)))
        maskgrouped.append(labeledpos)
        for jj in range(1,n+1):
            tmp_mask=np.asarray(labeledpos==jj)
            allMasks.append(tmp_mask)
#            labeledneg, n = label(compabsneg>0, np.ones((3,3)))
#            maskgrouped.append(labeledneg)
#            for jj in range(n):
#                tmp_mask=np.asarray(labeledneg==jj)
#                allMasks.append(tmp_mask)
#            plt.imshow(labeled)                             
#            plt.axis('off')         
    return allMasks,maskgrouped 
    
#%% threshold and remove spurious components    
def threshold_components(A_s,shape,min_size=5,max_size=np.inf,max_perc=.3):        
    """
    Threshold components output of a CNMF algorithm (A matrices)
    
    Parameters:
    ----------
    
    A_s: list 
        list of A matrice output from CNMF
    
    min_size: int
        min size of the component in pixels

    max_size: int
        max size of the component in pixels
        
    max_perc: float        
        fraction of the maximum of each component used to threshold 
        
        
    Returns:
    -------        
    
    B_s: list of the thresholded components
    
    lab_imgs: image representing the components in ndimage format

    cm_s: center of masses of each components
    """
    
    B_s=[]
    lab_imgs=[]
    
    cm_s=[]
    for A_ in A_s:
        print '*'
        max_comps=A_.max(0).todense().T
        tmp=[]
        cm=[]
        lim=np.zeros(shape)
        for idx,a in enumerate(A_.T):        
            #create mask by thresholding to 50% of the max
            print idx
            mask=np.reshape(a.todense()>(max_comps[idx]*max_perc),shape)        
            label_im, nb_labels = ndi.label(mask)
            sizes = ndi.sum(mask, label_im, range(nb_labels + 1))            
            l_largest=(label_im==np.argmax(sizes))
            cm.append(scipy.ndimage.measurements.center_of_mass(l_largest,l_largest))
            lim[l_largest] = (idx+1)
    #       #remove connected components that are too small
            mask_size=np.logical_or(sizes<min_size,sizes>max_size)
            if np.sum(mask_size[1:])>1:
                print 'removing ' + str( np.sum(mask_size[1:])-1) + ' components'
            remove_pixel=mask_size[label_im]
            label_im[remove_pixel] = 0           

            label_im=(label_im>0)*1    
            
            tmp.append(label_im.flatten())
        
        
        cm_s.append(cm)    
        lab_imgs.append(lim)        
        B_s.append(csc.csc_matrix(np.array(tmp)).T)
    
    return B_s, lab_imgs, cm_s         
    
#%%
#def get_roi_from_spatial_component(sp_comp, min_radius, max_radius, roughness=2, zoom_factor=1, center_range=2,z_step=1,thresh_iou=.4):
#     
#     
#     if sp_comp.ndim > 2:
#         center=center_of_mass(sp_comp[0])
#         roi=cell_magic_wand_3d_IOU(sp_comp,center,min_radius, max_radius, roughness=roughness, zoom_factor=zoom_factor, center_range=center_range, z_step=z_step,thresh_iou=thresh_iou)
#     else:            
#         center=center_of_mass(sp_comp)
#         roi=cell_magic_wand(sp_comp,center,min_radius, max_radius, roughness=roughness, zoom_factor=zoom_factor, center_range=center_range)
#     print sp_comp.shape,roi.shape
#     pl.subplot(1,2,1)
#     pl.cla()
#     if sp_comp.ndim > 2:
#         pl.imshow(sp_comp[0])
#     else:
#         pl.imshow(sp_comp)
#     pl.subplot(1,2,2)
#     pl.cla()    
#     pl.imshow(roi)
##     pl.plot(center[1],center[0],'k.')     
#     pl.pause(.3)
#     print 'Roi Size:' + str(np.sum(roi)) 
#     return roi
##%%     
#def get_roi_from_spatial_component_place_holder(pars):
#    sp_comp, min_radius, max_radius, roughness, zoom_factor, center_range,z_step,thresh_iou = pars
#    roi=get_roi_from_spatial_component(sp_comp,min_radius, max_radius, roughness=roughness, zoom_factor=zoom_factor, center_range=center_range,z_step=z_step,thresh_iou=thresh_iou)
#    return roi
##%%
#def get_roi_from_spatial_component_parallel(sp_comps,min_radius, max_radius, other_imgs=None,dview=None, roughness=2, zoom_factor=1, center_range=2,z_step=1,thresh_iou=.4):   
#
#    pars=[]
#    for sp_comp in sp_comps:
#        if other_imgs is None:
#            pars.append([sp_comp, min_radius, max_radius, roughness, zoom_factor, center_range,z_step,thresh_iou])
#        else:
#           
#            pars.append([np.dstack([sp_comp]+other_imgs).transpose(2,0,1), min_radius, max_radius, roughness, zoom_factor, center_range,z_step,thresh_iou])
#            
#        
#    if dview is not None:
#        rois=dview.map_sync(get_roi_from_spatial_component_place_holder,pars)
#        
#    else:
#        rois=map(get_roi_from_spatial_component_place_holder,pars)
#        
#    return rois        