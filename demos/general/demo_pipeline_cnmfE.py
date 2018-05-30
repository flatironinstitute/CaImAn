#!/usr/bin/env python

"""
Complete demo pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.

Demo is also available as a jupyter notebook (see demo_pipeline_cnmfE.ipynb)
"""

from __future__ import division
from __future__ import print_function

try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
except:
    print('Not launched under iPython')

import numpy as np
import matplotlib.pyplot as plt
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.motion_correction import motion_correct_oneP_rigid, motion_correct_oneP_nonrigid

#%% First setup some parameters

# dataset dependent parameters
display_images = False           # Set to true to show movies and images
fnames = ['data_endoscope.tif']  # filename to be processed
frate = 10                       # movie frame rate
decay_time = 0.4                 # length of a typical transient in seconds

# motion correction parameters
do_motion_correction_nonrigid = True
do_motion_correction_rigid = False  # in this case it will also save a rigid motion corrected movie
gSig_filt = (3, 3)   # size of filter, in general gSig (see below),
#                      change this one if algorithm does not work
max_shifts = (5, 5)  # maximum allowed rigid shift
splits_rig = 10      # for parallelization split the movies in  num_splits chuncks across time
strides = (48, 48)   # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
# for parallelization split the movies in  num_splits chuncks across time
# (remember that it should hold that length_movie/num_splits_to_process_rig>100)
splits_els = 10
upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3

# parameters for source extraction and deconvolution
p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None
gSig = 3            # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 13           # average diameter of a neuron, in general 4*gSig+1
merge_thresh = .7   # merging threshold, max correlation allowed
rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 2            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
Ain = None          # if you want to initialize with some preselected components
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                             True performs global low-rank approximation 
gnb = -1            # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb=-2: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb= 0: Don't return background
nb_patch = -1       # number of background components (rank) per patch,
#                     use 0 or -1 for exact background of ring model (cf. gnb)
min_corr = .8       # min peak value from correlation image
min_pnr = 10        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

# parameters for component evaluation
min_SNR = 3            # adaptive way to set threshold on the transient size
r_values_min = 0.85    # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)

#%% start the cluster
try:
    dview.terminate()  # stop it if it was running
except:
    pass

c, dview, n_processes = cm.cluster.setup_cluster(backend='local',  # use this one
                                                 n_processes=24,  # number of process to use, if you go out of memory try to reduce this one
                                                 single_thread=False)

#%% download demo file
fnames = [download_demo(fnames[0])]
filename_reorder = fnames


#%% MOTION CORRECTION
if do_motion_correction_nonrigid or do_motion_correction_rigid:
    # do motion correction rigid
    mc = motion_correct_oneP_rigid(fnames,
                                   gSig_filt=gSig_filt,
                                   max_shifts=max_shifts,
                                   dview=dview,
                                   splits_rig=splits_rig,
                                   save_movie=not(do_motion_correction_nonrigid)
                                   )

    new_templ = mc.total_template_rig

    plt.subplot(1, 2, 1)
    plt.imshow(new_templ)  # % plot template
    plt.subplot(1, 2, 2)
    plt.plot(mc.shifts_rig)  # % plot rigid shifts
    plt.legend(['x shifts', 'y shifts'])
    plt.xlabel('frames')
    plt.ylabel('pixels')

    # borders to eliminate from movie because of motion correction
    bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
    filename_reorder = mc.fname_tot_rig

    # do motion correction nonrigid
    if do_motion_correction_nonrigid:
        mc = motion_correct_oneP_nonrigid(
            fnames,
            gSig_filt=gSig_filt,
            max_shifts=max_shifts,
            strides=strides,
            overlaps=overlaps,
            splits_els=splits_els,
            upsample_factor_grid=upsample_factor_grid,
            max_deviation_rigid=max_deviation_rigid,
            dview=dview,
            splits_rig=None,
            save_movie=True,  # whether to save movie in memory mapped format
            new_templ=new_templ  # template to initialize motion correction
        )

        filename_reorder = mc.fname_tot_els
        bord_px = np.ceil(
            np.maximum(np.max(np.abs(mc.x_shifts_els)),
                       np.max(np.abs(mc.y_shifts_els)))).astype(np.int)

# create memory mappable file in the right order on the hard drive (C order)
fname_new = cm.save_memmap_each(
    filename_reorder,
    base_name='memmap_',
    order='C',
    border_to_0=bord_px,
    dview=dview)
fname_new = cm.save_memmap_join(fname_new, base_name='memmap_', dview=dview)


# load memory mappable file
Yr, dims, T = cm.load_memmap(fname_new)
Y = Yr.T.reshape((T,) + dims, order='F')
#%% compute some summary images (correlation and peak to noise)
# change swap dim if output looks weird, it is a problem with tiffile
cn_filter, pnr = cm.summary_images.correlation_pnr(Y, gSig=gSig, swap_dim=False)
# inspect the summary images and set the parameters
inspect_correlation_pnr(cn_filter, pnr)
# print parameters set above, modify them if necessary based on summary images
print(min_corr) # min correlation of peak (from correlation image)
print(min_pnr)  # min peak to noise ratio


#%% RUN CNMF ON PATCHES
cnm = cnmf.CNMF(
    n_processes=n_processes,
    method_init='corr_pnr',             # use this for 1 photon
    k=K,
    gSig=(gSig, gSig),
    gSiz=(gSiz, gSiz),
    merge_thresh=merge_thresh,
    p=p,
    dview=dview,
    tsub=tsub,
    ssub=ssub,
    Ain=Ain,
    rf=rf,
    stride=stride_cnmf,
    only_init_patch=True,               # just leave it as is
    gnb=gnb,
    nb_patch=nb_patch,
    method_deconvolution='oasis',       # could use 'cvxpy' alternatively
    low_rank_background=low_rank_background,
    update_background_components=True,  # sometimes setting to False improve the results
    min_corr=min_corr,
    min_pnr=min_pnr,
    normalize_init=False,               # just leave as is
    center_psf=True,                    # leave as is for 1 photon
    ssub_B=ssub_B,
    ring_size_factor=ring_size_factor,
    del_duplicates=True,                # whether to remove duplicates from initialization
    border_pix=bord_px)                 # number of pixels to not consider in the borders
cnm.fit(Y)


#%% DISCARD LOW QUALITY COMPONENTS
idx_components, idx_components_bad, comp_SNR, r_values, pred_CNN = \
    estimate_components_quality_auto(
        Y, cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, frate,
        decay_time, gSig, dims, dview=dview,
        min_SNR=min_SNR, r_values_min=r_values_min, use_cnn=False)

print(' ***** ')
print((len(cnm.C)))
print((len(idx_components)))

cm.stop_server(dview=dview)


#%% PLOT COMPONENTS
if display_images:
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    crd_good = cm.utils.visualization.plot_contours(
        cnm.A[:, idx_components], cn_filter, thr=.8, vmax=0.95)
    plt.title('Contour plots of accepted components')
    plt.subplot(122)
    crd_bad = cm.utils.visualization.plot_contours(
        cnm.A[:, idx_components_bad], cn_filter, thr=.8, vmax=0.95)
    plt.title('Contour plots of rejected components')

#%% VISUALIZE IN DETAILS COMPONENTS
    cm.utils.visualization.view_patches_bar(
        Yr, cnm.A[:, idx_components], cnm.C[idx_components], cnm.b, cnm.f,
        dims[0], dims[1], YrA=cnm.YrA[idx_components], img=cn_filter)


#%% MOVIES
if display_images:
    B = cnm.b.dot(cnm.f)
    if 'sparse' in str(type(B)):
        B = B.toarray()
# denoised movie
    cm.movie(np.reshape(cnm.A.tocsc()[:, idx_components].dot(cnm.C[idx_components]) + B,
                        dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=3, gain=1.)
# only neurons
    cm.movie(np.reshape(cnm.A.tocsc()[:, idx_components].dot(
        cnm.C[idx_components]), dims + (-1,), order='F').transpose(2, 0, 1)
    ).play(magnification=3, gain=10.)
# only the background
    cm.movie(np.reshape(B, dims + (-1,), order='F').transpose(2, 0, 1)
             ).play(magnification=3, gain=1.)
# residuals
    cm.movie(np.array(Y) - np.reshape(cnm.A.tocsc()[:, :].dot(cnm.C[:]) + B,
                                      dims + (-1,), order='F').transpose(2, 0, 1)
             ).play(magnification=3, gain=10., fr=10)
# eventually, you can rerun the algorithm on the residuals
    plt.imshow(cm.movie(np.array(Y) - np.reshape(cnm.A.tocsc()[:, :].dot(cnm.C[:]) + B,
                                                 dims + (-1,), order='F').transpose(2, 0, 1)
                        ).local_correlations(swap_dim=False))
