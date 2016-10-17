# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 14:23:22 2016

@author: agiovann
"""
import calblitz as cb
import pylab as pl
import cv2

m=cb.load('M_FLUO_1.tif',fr=15)
m,_,_,_=m.motion_correct(10,10)

im1=np.median(m,axis=0)
pl.imshow(im1,cmap=pl.cm.gray,vmax=np.percentile(m,90))

#%%
# Read the images to be aligned
newm=[];

 # Find size of image1
sz = im1.shape


for idx,im2 in enumerate(m):

    print(idx)
     
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
     
    # Specify the number of iterations.
    number_of_iterations = 5000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1,im2,warp_matrix, warp_mode, criteria)
     
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    newm.append(im2_aligned)
    # Show final results
#    cv2.imshow("Image 1", im1)
#    cv2.imshow("Image 2", im2)
#    cv2.imshow("Aligned Image 2", im2_aligned)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#%%
mm=np.concatenate([m,newm],axis=2)
newmn=cb.movie(np.array(mm),fr=m.fr)
pl.imshow(np.median(newmn,axis=0),cmap=pl.cm.gray)