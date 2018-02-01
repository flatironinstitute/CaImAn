#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div

import os
import sys

here = os.path.dirname(os.path.realpath(__file__))
caiman_path = os.path.join(here, "..", "..")
print("Caiman path detected as " + caiman_path)
sys.path.append(caiman_path)

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
from caiman.motion_correction import motion_correct_oneP_rigid,motion_correct_oneP_nonrigid
import os
#%% Set parameters
display_images = False # Set to true to show movies and images
fnames = ['data_endoscope.tif']
frate = 10  # movie frame rate
gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 10  # average diameter of a neuron
do_motion_correction_nonrigid = True
do_motion_correction_rigid = False # in this case it will also save a rigid motion corrected movie
#%% start the cluster
try:
    dview.terminate()  # stop it if it was running
except:
    pass

c, dview, n_processes = cm.cluster.setup_cluster(backend='local',  # use this one
                                                 n_processes=24,  # number of process to use, if you go out of memory try to reduce this one
                                                 single_thread=False)

#%% download demo file
fnames = [download_demo(fnames[0], caiman_base=caiman_path)]
filename_reorder = fnames

#%% motion correction
if do_motion_correction_nonrigid or do_motion_correction_rigid:
    # do motion correction rigid
    mc = motion_correct_oneP_rigid(fnames,                        # name of file to motion correct
                                   # size of filter, xhange this one if
                                   # algorithm does not work
                                   gSig_filt=[gSig] * 2,
                                   # maximum shifts allowed in each direction
                                   max_shifts=[5, 5],
                                   dview=dview,
                                   # number of chunks for parallelizing motion correction (remember that it should hold that length_movie/num_splits_to_process_rig>100)
                                   splits_rig=10,
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

    bord_px = np.ceil(np.max(mc.shifts_rig)).astype(np.int)     #borders to eliminate from movie because of motion correction
    filename_reorder = mc.fname_tot_rig

    # do motion correction nonrigid
    if do_motion_correction_nonrigid:
        mc = motion_correct_oneP_nonrigid(
                          fnames,# name of file to motion correct
                          # size of filter, change this one if
                          # algorithm does not work
                          gSig_filt=[gSig] * 2,
                          max_shifts=[5, 5], # maximum shifts allowed in each direction
                          # start a new patch for pw-rigid
                          # motion correction every x pixels
                          strides=(48, 48),
                          # overlap between pathes (size of
                          # patch strides+overlaps)
                          overlaps=(24, 24),
                          splits_els=10,
                          # number of chunks for parallelizing
                          # motion correction (remember that it
                          # should hold that
                          # length_movie/num_splits_to_process_rig>100)
                          upsample_factor_grid=4,
                          # upsample factor to avoid smearing
                          # when merging patches
                          max_deviation_rigid=3,
                          # maximum deviation allowed for patch
                          # with respect to rigid shifts
                          dview=dview,
                          splits_rig=None,
                          save_movie=True,
                          # whether to save movie in memory
                          # mapped format
                          new_templ=new_templ
                          # template to initialize motion correction
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
cn_filter, pnr = cm.summary_images.correlation_pnr(
    Y, gSig=gSig, swap_dim=False)
#%% inspect the summary images and set the parameters
inspect_correlation_pnr(cn_filter, pnr)
#%%
min_corr = .8  # min correlation of peak (from correlation image)
min_pnr = 10  # min peak to noise ratio
min_SNR = 3  # adaptive way to set threshold on the transient size
# threshold on space consistency (if you lower more components will be accepted, potentially with worst quality)
r_values_min = 0.85
decay_time = 0.4  # decay time of transients/indocator
#%%
cnm = cnmf.CNMF(n_processes=n_processes,
                method_init='corr_pnr',                 # use this for 1 photon
                k=70,                                   # neurons per patch
                gSig=(3, 3),                            # half size of neuron
                gSiz=(10, 10),                          # in general 3*gSig+1
                merge_thresh=.8,                        # threshold for merging
                p=1,                                    # order of autoregressive process to fit
                dview=dview,                            # if None it will run on a single thread
                # downsampling factor in time for initialization, increase if you have memory problems
                tsub=2,
                # downsampling factor in space for initialization, increase if you have memory problems
                ssub=2,
                # if you want to initialize with some preselcted components you can pass them here as boolean vectors
                Ain=None,
                # half size of the patch (final patch will be 100x100)
                rf=(40, 40),
                # overlap among patches (keep it at least large as 4 times the neuron size)
                stride=(20, 20),
                only_init_patch=True,                   # just leave it as is
                gnb=16,                                 # number of background components
                nb_patch=16,                            # number of background components per patch
                method_deconvolution='oasis',  # could use 'cvxpy' alternatively
                low_rank_background=True,  # leave as is
                # sometimes setting to False improve the results
                update_background_components=True,
                min_corr=min_corr,                      # min peak value from correlation image
                min_pnr=min_pnr,                        # min peak to noise ration from PNR image
                normalize_init=False,                   # just leave as is
                center_psf=True,                        # leave as is for 1 photon
                del_duplicates=True,                    # whether to remove duplicates from initialization
                border_pix = bord_px)                   # number of pixels to not consider in the borders
cnm.fit(Y)

# %% DISCARD LOW QUALITY COMPONENTS
idx_components, idx_components_bad, comp_SNR, r_values, pred_CNN = estimate_components_quality_auto(
                            Y, cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, frate,
                            decay_time, gSig, dims, dview = dview,
                            min_SNR=min_SNR, r_values_min = r_values_min, use_cnn = False)

print(' ***** ')
print((len(cnm.C)))
print((len(idx_components)))
#%% PLOT ALL COMPONENTS
crd = cm.utils.visualization.plot_contours(cnm.A, cn_filter, thr=.8, vmax=0.99)
#%% PLOT ONLY GOOD QUALITY COMPONENTS
crd = cm.utils.visualization.plot_contours(
    cnm.A.tocsc()[:, idx_components], cn_filter, thr=.8, vmax=0.95)
#%% PLOT ONLY BAD QUALITY COMPONENTS
crd = cm.utils.visualization.plot_contours(
    cnm.A.tocsc()[:, idx_components_bad], cn_filter, thr=.8, vmax=0.95)

#%% VISUALIZE IN DETAILS COMPONENTS
cm.utils.visualization.view_patches_bar(Yr, cnm.A[:, idx_components], cnm.C[idx_components], cnm.b, cnm.f,
                                        dims[0], dims[1], YrA=cnm.YrA[idx_components], img=cn_filter)

#%%
cm.stop_server(dview=dview)
#%% denoised movie
if display_images:
    cm.movie(np.reshape(cnm.A.tocsc()[:, idx_components].dot(cnm.C[idx_components]) + cnm.b.dot(
        cnm.f), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=3, gain=1.)
#%% only neurons
if display_images:
    cm.movie(np.reshape(cnm.A.tocsc()[:, idx_components].dot(
        cnm.C[idx_components]), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=3, gain=10.)
#%% only the background
if display_images:
    cm.movie(np.reshape(cnm.b.dot(cnm.f), dims + (-1,),
        order='F').transpose(2, 0, 1)).play(magnification=3, gain=1.)
#%% residuals
if display_images:
    cm.movie(np.array(Y) - np.reshape(cnm.A.tocsc()[:, :].dot(cnm.C[:]) + cnm.b.dot(
        cnm.f), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=3, gain=10., fr=10)
#%% eventually, you can rerun the algorithm on the residuals
if display_images:
    plt.imshow(cm.movie(np.array(Y) - np.reshape(cnm.A.tocsc()[:, :].dot(cnm.C[:]) + cnm.b.dot(
        cnm.f), dims + (-1,), order='F').transpose(2, 0, 1)).local_correlations(swap_dim=False))
