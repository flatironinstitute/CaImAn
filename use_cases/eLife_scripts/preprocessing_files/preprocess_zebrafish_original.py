#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using OnACID.
@author: Andrea Giovannucci @agiovann and Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing their data used in this demo.

KERAS_BACKEND=tensorflow; CUDA_VISIBLE_DEVICES=-1; spyder
"""

import os
import sys

#try:
#    here = os.path.dirname(os.path.realpath(__file__))
#    caiman_path = os.path.join(here, "..", "..")
#    print("Caiman path detected as " + caiman_path)
#    sys.path.append(caiman_path)
#except:
#    pass
import numpy as np
try:
    if __IPYTHON__:
        print('Detected iPython')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


from time import time
import caiman as cm
from caiman.utils.visualization import view_patches_bar
from caiman.utils.utils import download_demo, load_object, save_object
import pylab as pl
import scipy
from caiman.motion_correction import motion_correct_iteration_fast
import cv2
from caiman.utils.visualization import plot_contours
import glob
from caiman.source_extraction.cnmf.online_cnmf import bare_initialization
from copy import deepcopy
'''
WILL NOT WORK WITH CURRENT VERSION OF CAIMAN, SEE PREPROCESS ZEBRAFISH
'''
#%%
try:
    import sys
    if 'pydevconsole' in sys.argv[0]:
        raise Exception('Running in PYCHARM')
    ID = sys.argv[1]
    ID = str(np.int(ID)+1)
    print('Processing ID:'+ str(ID))
    ploton = False
    save_results = True
    save_init = True     # flag for saving initialization object. Useful if you want to check OnACID with different parameters but same initialization
except:
    print('ID NOT PASSED')
    ID = 11
    ploton = False
    save_results = False
    save_init = False # flag for saving initialization object. Useful if you want to check OnACID with different parameters but same initialization

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'
#%%
decay_time = 1.5
gSig = (6,6)
rval_thr = 1
epochs = 1
fls = [os.path.join(base_folder,'Zebrafish/Plane' + str(ID) + '.stack.hdf5')];
K = 100
min_num_trial = 50

mmm = cm.load(fls,subindices = 0)
dims = mmm.shape
K = np.maximum(K,np.round(600/1602720*np.prod(mmm.shape)).astype(np.int))
min_num_trial = np.maximum(min_num_trial,np.round(200/1602720*np.prod(mmm.shape)).astype(np.int))
# your list of files should look something like this
print(fls)

print([K,min_num_trial])

# number of passes over the data
#%%   Set up some parameters

# frame rate (Hz)
fr = 2
#fr = 15

# approximate length of transient event in seconds

#decay_time = 0.5

# expected half size of neurons
#gSig = (2.5, 2.5)

# order of AR indicator dynamics
p = 1
# minimum SNR for accepting new components
min_SNR = 2.5
# correlation threshold for new component inclusion

#rval_thr = 0.85

# spatial downsampling factor (increases speed but may lose some fine structure)
ds_factor = 2
# number of background components
gnb = 3
# recompute gSig if downsampling is involved
gSig = tuple((np.array(gSig) / ds_factor))#.astype('int'))
# flag for online motion correction
mot_corr = True
# maximum allowed shift during motion correction
max_shift = np.ceil(10. / ds_factor).astype('int')

# set up some additional supporting parameters needed for the algorithm (these are default values but change according to dataset characteristics)

# number of shapes to be updated each time (put this to a finite small value to increase speed)
max_comp_update_shape = np.inf
# number of files used for initialization
init_files = 1
# number of files used for online
online_files = len(fls) - 1
# number of frames for initialization (presumably from the first file)
initbatch = 200
# maximum number of expected components used for memory pre-allocation (exaggerate here)
expected_comps = 600
# initial number of components
# number of timesteps to consider when testing new neuron candidates
N_samples = np.ceil(fr * decay_time)
# exceptionality threshold
thresh_fitness_raw = scipy.special.log_ndtr(-min_SNR) * N_samples

# upper bound for number of frames in each file (used right below)
len_file = 1885#1885 1815
# total length of all files (if not known use a large number, then truncate at the end)
T1 = len(fls) * len_file * epochs
#%%
compute_corr = False
if compute_corr:
    m = cm.load(fls)
    mc = m.motion_correct(10,10)[0]
    mp = (mc.computeDFF(3))
    Cn = cv2.resize(mp[0].local_correlations(eight_neighbours=True, swap_dim=False),dims[::-1][:-1])
    np.save('/mnt/ceph/neuro/zebra/05292014Fish1-4/results_analysis_online_Plane_CN_' + str(ID) + '.npy', Cn)
#%%    Initialize movie
# load only the first initbatch frames and possibly downsample them
if ds_factor > 1:
    Y = cm.load(fls[0], subindices=slice(0, initbatch, None)).astype(
        np.float32).resize(1. / ds_factor, 1. / ds_factor)
else:
    Y = cm.load(fls[0], subindices=slice(
        0, initbatch, None)).astype(np.float32)

if mot_corr:                                        # perform motion correction on the first initbatch frames
    mc = Y.motion_correct(max_shift, max_shift)
    Y = mc[0].astype(np.float32)
    borders = np.max(mc[1])
else:
    Y = Y.astype(np.float32)

# minimum value of movie. Subtract it to make the data non-negative
img_min = Y.min()
Y -= img_min
img_norm = np.std(Y, axis=0)
# normalizing factor to equalize the FOV
img_norm += np.median(img_norm)
Y = Y / img_norm[None, :, :]                        # normalize data

_, d1, d2 = Y.shape
dims = (d1, d2)                                     # dimensions of FOV
Yr = Y.to_2D().T                                    # convert data into 2D array

Cn_init = Y.local_correlations(swap_dim=False)    # compute correlation image
if ploton:
    pl.imshow(Cn_init)
    pl.title('Correlation Image on initial batch')
    pl.colorbar()

#%% initialize OnACID with bare initialization
t1 = time()
cnm_init = bare_initialization(Y[:initbatch].transpose(1, 2, 0), init_batch=initbatch, k=K, gnb=gnb,
                               gSig=gSig, p=0, minibatch_shape=100, minibatch_suff_stat=5,
                               update_num_comps=True, rval_thr=rval_thr,
                               thresh_fitness_raw=thresh_fitness_raw,
                               batch_update_suff_stat=True, max_comp_update_shape=max_comp_update_shape,
                               deconv_flag=False, use_dense=False,
                               simultaneously=False, n_refit=0)

time_init = time() - t1
#%% Plot initialization results
if ploton:
    crd = plot_contours(cnm_init.A.tocsc(), Cn_init, thr=0.9)
    A, C, b, f, YrA, sn = cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.YrA, cnm_init.sn
    view_patches_bar(Yr, scipy.sparse.coo_matrix(
        A.tocsc()[:, :]), C[:, :], b, f, dims[0], dims[1], YrA=YrA[:, :], img=Cn_init)


#%% create a function for plotting results in real time if needed
def create_frame(cnm2, img_norm, captions):
    A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
    C, f = cnm2.C_on[cnm2.gnb:cnm2.M, :], cnm2.C_on[:cnm2.gnb, :]
    # inferred activity due to components (no background)
    comps_frame = A.dot(C[:, t - 1]).reshape(cnm2.dims,
                                             order='F') * img_norm / np.max(img_norm)

    comps_frame_ = A.dot(C[:, t - 1]).reshape(cnm2.dims,
                                             order='F') * img_norm / np.max(img_norm)/3
#    comps_frame_ = cnm2.sv.reshape(cnm2.dims, order='C')/200


    bgkrnd_frame = b.dot(f[:, t - 1]).reshape(cnm2.dims, order='F') * \
        img_norm / np.max(img_norm)  # denoised frame (components + background)



    if show_residuals:
        all_comps = np.reshape(cnm2.Yres_buf.mean(
            0), cnm2.dims, order='F') * img_norm / np.max(img_norm) / 2
        all_comps = np.minimum(np.maximum(all_comps * 10, 0), 255)
    else:
        all_comps = (np.array(A.sum(-1)).reshape(cnm2.dims, order='F')
                     )                         # spatial shapes


    frame_comp_1 = cv2.resize(np.concatenate([frame_ / np.max(img_norm), all_comps * 3.], axis=-1),
                              (2 * np.int(cnm2.dims[1] * resize_fact), np.int(cnm2.dims[0] * resize_fact)))

    frame_comp_2 = cv2.resize(np.concatenate([comps_frame_ * 10., comps_frame + bgkrnd_frame],
                                             axis=-1), (2 * np.int(cnm2.dims[1] * resize_fact), np.int(cnm2.dims[0] * resize_fact)))

    frame_pn = np.concatenate([frame_comp_1, frame_comp_2], axis=0).T
    vid_frame = np.repeat(frame_pn[:, :, None], 3, axis=-1)
    vid_frame = np.minimum((vid_frame * 255.), 255).astype('u1')

    if show_residuals and cnm2.ind_new_all:
        add_v = np.int(cnm2.dims[1]*resize_fact)
        for ind_new in cnm2.ind_new_all:
            cv2.rectangle(vid_frame,(int(ind_new[0][1]*resize_fact),int(ind_new[1][1]*resize_fact)+add_v),
                                         (int(ind_new[0][0]*resize_fact),int(ind_new[1][0]*resize_fact)+add_v),(255,255,255),1)
    if show_residuals and cnm2.ind_new:
        add_v = np.int(cnm2.dims[1]*resize_fact)
        for ind_new in cnm2.ind_new:
            cv2.rectangle(vid_frame,(int(ind_new[0][1]*resize_fact),int(ind_new[1][1]*resize_fact)+add_v),
                                         (int(ind_new[0][0]*resize_fact),int(ind_new[1][0]*resize_fact)+add_v),(255,0,255),1)


    cv2.putText(vid_frame, captions[0], (5, 20), fontFace=5, fontScale=1.2, color=(
        0, 255, 0), thickness=1)
    cv2.putText(vid_frame, captions[1], (np.int(
        cnm2.dims[0] * resize_fact) + 5, 20), fontFace=5, fontScale=1.2, color=(0, 255, 0), thickness=1)
    cv2.putText(vid_frame, captions[2], (5, np.int(
        cnm2.dims[1] * resize_fact) + 20), fontFace=5, fontScale=1.2, color=(0, 255, 0), thickness=1)
    cv2.putText(vid_frame, captions[3], (np.int(cnm2.dims[0] * resize_fact) + 5, np.int(
        cnm2.dims[1] * resize_fact) + 20), fontFace=5, fontScale=1.2, color=(0, 255, 0), thickness=1)
    cv2.putText(vid_frame, 'Frame = ' + str(t), (vid_frame.shape[1] // 2 - vid_frame.shape[1] //
                                                 10, vid_frame.shape[0] - 20), fontFace=5, fontScale=1.2, color=(0, 255, 255), thickness=1)
    return vid_frame

#%% Prepare object for OnACID
cnm2 = deepcopy(cnm_init)
path_to_model os.path.join(caiman_datadir(), 'model', 'cnn_model_online.h5')


if save_init:
    cnm_init.dview = None
    save_object(cnm_init, fls[0][:-4] + '_DS_' + str(ds_factor) + '.pkl')
    cnm_init = load_object(fls[0][:-4] + '_DS_' + str(ds_factor) + '.pkl')

t1 = time()
cnm2._prepare_object(np.asarray(Yr), T1, expected_comps, idx_components=None,
                         min_num_trial=min_num_trial, max_num_added = min_num_trial, N_samples_exceptionality=int(N_samples),
                         path_to_model = path_to_model,
                         sniper_mode = True, use_peak_max = True)
cnm2.thresh_CNN_noisy = 0.75
time_prepare = time() - t1
#%% Run OnACID and optionally plot results in real time
cnm2.Ab_epoch = []                       # save the shapes at the end of each epoch
t = cnm2.initbatch                       # current timestep
tottime = []
Cn = Cn_init.copy()
cnn_pos = []
# flag for plotting contours of detected components at the end of each file
plot_contours_flag = False
# flag for showing video with results online (turn off flags for improving speed)
play_reconstr = True
# flag for saving movie (file could be quite large..)
save_movie = False
folder_name = '.'
if save_movie:
    movie_name = os.path.join(folder_name, 'output.avi')  # name of movie to be saved

resize_fact = 1.2                        # image resizing factor

if online_files == 0:                    # check whether there are any additional files
    process_files = fls[:init_files]     # end processing at this file
    init_batc_iter = [initbatch]         # place where to start
    end_batch = T1
else:
    process_files = fls[:init_files + online_files]     # additional files
    # where to start reading at each file
    init_batc_iter = [initbatch] + [0] * online_files

t1 = time()
num_comps = []
shifts = []
show_residuals = True
if show_residuals:
    caption = 'Mean Residual Bufer'
else:
    caption = 'Identified Components'
captions = ['Raw Data', 'Inferred Activity', caption, 'Denoised Data']
if save_movie and play_reconstr:
    #fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(movie_name, fourcc, 10.0, tuple(
        [int(2 * x * resize_fact) for x in cnm2.dims]))

for iter in range(epochs):
    if iter > 0:
        # if not on first epoch process all files from scratch
        process_files = fls[:init_files + online_files]
        init_batc_iter = [0] * (online_files + init_files)

    # np.array(fls)[np.array([1,2,3,4,5,-5,-4,-3,-2,-1])]:
    for file_count, ffll in enumerate(process_files):
        print('Now processing file ' + ffll)
        Y_ = cm.load(ffll, subindices=slice(
            init_batc_iter[file_count], T1, None))

        # update max-correlation (and perform offline motion correction) just for illustration purposes
        if plot_contours_flag:
            if ds_factor > 1:
                Y_1 = Y_.resize(1. / ds_factor, 1. / ds_factor, 1)
            else:
                Y_1 = Y_.copy()
            if mot_corr:
                templ = (cnm2.Ab.data[:cnm2.Ab.indptr[1]] * cnm2.C_on[0,
                                                                      t - 1]).reshape(cnm2.dims, order='F') * img_norm
                newcn = (Y_1 - img_min).motion_correct(max_shift, max_shift,
                                                       template=templ)[0].local_correlations(swap_dim=False)
                Cn = np.maximum(Cn, newcn)
            else:
                Cn = np.maximum(Cn, Y_1.local_correlations(swap_dim=False))

        old_comps = cnm2.N                              # number of existing components
        for frame_count, frame in enumerate(Y_):        # now process each file
            if np.isnan(np.sum(frame)):
                raise Exception('Frame ' + str(frame_count) + ' contains nan')
            if t % 100 == 0:
                print('Epoch: ' + str(iter + 1) + '. ' + str(t) + ' frames have beeen processed in total. ' + str(cnm2.N -
                                                                                                                  old_comps) + ' new components were added. Total number of components is ' + str(cnm2.Ab.shape[-1] - gnb))
                old_comps = cnm2.N

            t1 = time()                                 # count time only for the processing part
            frame_ = frame.copy().astype(np.float32)    #
            if ds_factor > 1:
                frame_ = cv2.resize(
                    frame_, img_norm.shape[::-1])   # downsampling

            frame_ -= img_min                                       # make data non-negative

            if mot_corr:                                            # motion correct
                templ = cnm2.Ab.dot(
                    cnm2.C_on[:cnm2.M, t - 1]).reshape(cnm2.dims, order='F') * img_norm
                frame_cor, shift = motion_correct_iteration_fast(
                    frame_, templ, max_shift, max_shift)
                shifts.append(shift)
            else:
                templ = None
                frame_cor = frame_

            frame_cor = frame_cor / img_norm                        # normalize data-frame
            cnm2.fit_next(t, frame_cor.reshape(-1, order='F')
                          )      # run OnACID on this frame
            # store time
            cnn_pos.append(cnm2.cnn_pos)
            tottime.append(time() - t1)
            num_comps.append(cnm2.N)
            t += 1

            if t % 1000 == 0 and plot_contours_flag:
                pl.cla()
                A = cnm2.Ab[:, cnm2.gnb:]
                # update the contour plot every 1000 frames
                crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
                pl.pause(1)

            if play_reconstr:                                               # generate movie with the results
                vid_frame = create_frame(cnm2, img_norm, captions)
                if save_movie:
                    out.write(vid_frame)
                cv2.imshow('frame', vid_frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break

        print('Cumulative processing speed is ' + str((t - initbatch) /
                                                      np.sum(tottime))[:5] + ' frames per second.')
    # save the shapes at the end of each epoch
    cnm2.Ab_epoch.append(cnm2.Ab.copy())

if save_movie:
    out.release()
cv2.destroyAllWindows()
#%%  save results (optional)

if save_results:
    np.savez('/mnt/ceph/neuro/zebra/05292014Fish1-4/results_analysis_online_1EPOCH_gSig6_equalized_Plane_' + str(ID) + '.npz',
             Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
             dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts,
             num_comps = num_comps,
             time_prepare=time_prepare, time_init=time_init)
#%%
reload = False
if reload:
    with np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/results_analysis_online_1EPOCH_gSig6_equalized_Plane_' + str(ID) + '.npz') as ld:
        locals().update(ld)
        print(ld.keys())
        Ab = Ab[()]
        A, b = Ab[:, gnb:], Ab[:, :gnb].toarray()
        C, f = Cf[gnb:Ab.shape[-1],:T1], Cf[:gnb, :T1]
        noisyC = noisyC[:, :T1]

#        m = cm.movie((A.dot(C)+b.dot(f))[:].reshape(list(dims)+[-1],order='F')).transpose([2,0,1])*img_norm[None,:,:]
        m = cm.movie((A.dot(C))[:].reshape(list(dims)+[-1],order='F')).transpose([2,0,1])*img_norm[None,:,:]

#%%
if ploton:
    m = cm.load(fls)
    m = m.motion_correct(10,10)[0]
    mp = (m.computeDFF(3))
    Cn_ = mp[0].local_correlations(eight_neighbours=True, swap_dim=False)
    Cn_ = cv2.resize(Cn_,dims[::-1])
#%% extract results from the objects and do some plotting
if ploton:
    A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
    C, f = cnm2.C_on[cnm2.gnb:cnm2.M, t - t //
                     epochs:t], cnm2.C_on[:cnm2.gnb, t - t // epochs:t]
    noisyC = cnm2.noisyC[:, t - t // epochs:t]
    b_trace = [osi.b for osi in cnm2.OASISinstances] if hasattr(
        cnm2, 'OASISinstances') else [0] * C.shape[0]

    pl.figure()
    crd = cm.utils.visualization.plot_contours(A, Cn_, thr=0.9, vmax = 0.75)
    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                     dims[0], dims[1], YrA=noisyC[cnm2.gnb:cnm2.M] - C, img=Cn_)

    #%%
    #df_f = detrend_df_f_auto(A,b,C,f,YrA)
    A_img = scale(A.toarray(),axis=0).mean(axis=-1).reshape(dims,order='F')
    crd = cm.utils.visualization.plot_contours(A, A_img, thr=0.9, vmax = 0.1,vmin=-0.01,cmap='gray')
    #%%
    from sklearn.preprocessing import scale
    pl.imshow(scale(A.toarray(),axis=0).mean(axis=-1).reshape(dims,order='F'),vmin=-.01,vmax=.1,cmap='gray')
#%% weighted suff stats
    #%% WEIGHTED SUFF STAT
if ploton:
    from caiman.components_evaluation import compute_event_exceptionality
    from scipy.stats import norm

    min_SNR = 2.5
    N_samples = np.ceil(fr*decay_time).astype(np.int)   # number of timesteps to consider when testing new neuron candidates
    fitness, erf, noi, what = compute_event_exceptionality(C+cnm2.noisyC[cnm2.gnb:cnm2.M, t - t // epochs:t],N=N_samples)
    COMP_SNR = -norm.ppf(np.exp(erf/ N_samples))
    COMP_SNR  = np.vstack([np.ones_like(f),COMP_SNR])
    COMP_SNR = np.clip(COMP_SNR, a_min = 0, a_max = 100)

    Cf = cnm2.C_on[:cnm2.M, t-t//epochs:t]

    Cf_ = Cf*COMP_SNR
#    Cf__ = Cf*np.sqrt(COMP_SNR)
    CC_ = Cf_.dot(Cf.T)
#    CC_ = Cf__.dot(Cf__.T)
    CY_ = Cf_.dot([(cv2.resize(yy,dims[::-1]).reshape(-1,order='F')-img_min)/img_norm.reshape(-1, order='F') for yy in Y_])

    Ab_, ind_A_, Ab_dense_ = cm.source_extraction.cnmf.online_cnmf.update_shapes(CY_, CC_, cnm2.Ab.copy(), cnm2.ind_A, indicator_components=None, Ab_dense=None, update_bkgrd=True, iters=55)
    #%%
    A_, b_ = Ab_[:, cnm2.gnb:], Ab_[:, :cnm2.gnb].toarray()

    #
    #view_patches_bar(Yr, scipy.sparse.coo_matrix(A_.tocsc()[:, :]), C[:, :], b_, f,
    #                 dims[0], dims[1], YrA=noisyC - C, img=Cn)
    counter = 0
    for comp1, comp2 in zip(A.T,A_.T):
        counter += 1
        if counter > 380:
            print(scipy.sparse.linalg.norm(comp1-comp2),scipy.sparse.linalg.norm(comp1),scipy.sparse.linalg.norm(comp2))
            pl.subplot(1,3,1)
            pl.imshow((comp1-comp2).toarray().reshape(dims, order='F'))
            pl.subplot(1,3,2)
            pl.imshow((comp1).toarray().reshape(dims, order='F'),vmax = 0.1)
            pl.subplot(1,3,3)
            pl.imshow((comp2).toarray().reshape(dims, order='F'),vmax = 0.1)
            pl.pause(1)
            if counter > 390:
                break

#%% All Planes
if ploton:
    from sklearn.preprocessing import normalize
    num_neur = []
    #tott = np.zeros_like(tottime)
    update_comps_time = []
    tott = []
    totneursum = 0
    time_per_neuron = []
    pl.figure()
    for ID in range(1,46):
#        try:
            with np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/results_analysis_online_1EPOCH_gSig6_equalized_Plane_' + str(ID) + '.npz') as ld:
#                locals().update(ld)
                print(np.sum(ld['tottime'])+ld['time_init'])
                tottime = ld['tottime']
                print(ld.keys())
                totneursum += ld['Cf'].shape[0]-3
                pl.subplot(5,9,ID)
#                img = normalize(Ab[()][:,3:],'l1',axis=0).mean(-1).reshape(dims,order = 'F').T
                Cn_ = np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/results_analysis_online_Plane_CN_'+str(ID)+ '.npy')
                pl.imshow(ld['Cf'][3:],aspect = 'auto', vmax = 10)
                pl.ylim([0,1950])
                pl.axis('off')
                pl.pause(0.1)


#                pl.figure();crd = cm.utils.visualization.plot_contours(
#                        Ab[()][:,3:].toarray().reshape(tuple(dims)+(-1,), order = 'F').transpose([1,0,2]).\
#                        reshape((dims[1]*dims[0],-1),order = 'F'), cv2.resize(Cn_,tuple(dims[::-1])).T, thr=0.9, vmax = 0.75,
#                        display_numbers=False)
#                A_thr = cm.source_extraction.cnmf.spatial.threshold_components(Ab[()].tocsc()[:,gnb:].toarray(), dims, medw=None, thr_method='nrg',
#                                                                  maxthr=0.3, nrgthr=0.95, extract_cc=True,
#                                 se=None, ss=None, dview=dview)
#                np.save('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components' + str(ID) + '.npy',A_thr)
                A_thr = np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components' + str(ID) + '.npy')
#                img = normalize(Ab[()][:,gnb:].multiply(A_thr),'l1',axis=0).mean(-1).reshape(dims,order = 'F').T
#                img = Ab[()][:,gnb:].multiply(A_thr).mean(-1).reshape(dims,order = 'F').T
                Ab_thr = Ab[()][:,gnb:].multiply(A_thr)
                img = (Ab_thr.dot(scipy.sparse.spdiags(np.minimum(1.0/np.max(Ab_thr,0).toarray(),100),0,Ab_thr.shape[-1],Ab_thr.shape[-1]))).mean(-1).reshape(dims,order = 'F').T
                xx,yy = np.subtract((560,860),img.shape)//2+1

#                pl.imshow(cv2.copyMakeBorder(img,xx,xx,yy,yy, cv2.BORDER_CONSTANT,0),vmin=np.percentile(img,5),vmax=np.percentile(img,99.99),cmap = 'gray')

#                A_thr = A_thr > 0

#                pl.imshow(((A_thr*np.random.randint(1,10,A_thr.shape[-1])[None,:]).sum(-1).reshape(dims,order='F')).T, cmap = 'hot', vmin = 0.9, vmax=20)
                pl.axis('off')
                pl.pause(0.05)

                num_neur.append(num_comps[1885-201])
                tottime = tottime[:1885-201]
                num_comps = num_comps[:1885-201]
                update_comps_time.append((np.array(num_comps)[99::100],tottime[99::100].copy()))
                tottime[99::100] = np.nan
                tottime[0] = np.nan
                [(np.where(np.diff([0]+list(num_comps))==cc)[0], tottime[np.where(np.diff([0]+list(num_comps))==cc)[0]]) for cc in range(6)]
                tott.append(tottime)

#        except:
            print(ID)
    pl.tight_layout()
    #%% create movies
    if ploton:
        from sklearn.preprocessing import normalize
        num_neur = []
        #tott = np.zeros_like(tottime)
        update_comps_time = []
        tott = []
        time_per_neuron = []
        pl.figure()
        for ID in range(1,46):

    #        try:
                with np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/results_analysis_online_1EPOCH_gSig6_equalized_Plane_' + str(ID) + '.npz') as ld:
                    fls = ['/mnt/ceph/neuro/zebra/05292014Fish1-4/Plane' + str(ID) + '.stack.hdf5']
                    locals().update(ld)
                    print(ld.keys())
                    Ab = Ab[()]
#                    if ds_factor > 1:
#                        Y = cm.load(fls[0], subindices=slice(0, initbatch, None)).astype(
#                            np.float32).resize(1. / ds_factor, 1. / ds_factor)
#                    else:
#                        Y = cm.load(fls[0], subindices=slice(
#                            0, initbatch, None)).astype(np.float32)
#
#                    if mot_corr:                                        # perform motion correction on the first initbatch frames
#                        mc = Y.motion_correct(max_shift, max_shift)
#                        Y = mc[0].astype(np.float32)
#                        borders = np.max(mc[1])
#                    else:
#                        Y = Y.astype(np.float32)
#
#                    # minimum value of movie. Subtract it to make the data non-negative
#                    img_min = Y.min()
#                    Y -= img_min
#                    img_norm = np.std(Y, axis=0)
#                    # normalizing factor to equalize the FOV
#                    img_norm += np.median(img_norm)



                    A_thr = np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components' + str(ID) + '.npy')
    #                img = normalize(Ab[()][:,gnb:].multiply(A_thr),'l1',axis=0).mean(-1).reshape(dims,order = 'F').T
    #                img = Ab[()][:,gnb:].multiply(A_thr).mean(-1).reshape(dims,order = 'F').T

                    A, b = Ab[:, gnb:].multiply(A_thr), Ab[:, :gnb].toarray()
#                    A, b = Ab[:, gnb:], Ab[:, :gnb].toarray()

                    C, f = Cf[gnb:Ab.shape[-1],:T1], Cf[:gnb, :T1]
#                    C = C/np.max(C,1)[:,None]
                    noisyC = noisyC[:, :T1]

                    m = (cm.movie((A.dot(C[:,600:900])).reshape(list(dims)+[-1],order='F')).transpose([2,0,1]))
                    xx,yy = np.subtract((560,860),dims[::-1])//2+1
                    m1 = cm.movie(np.concatenate([cv2.copyMakeBorder(img.T,xx,xx,yy,yy, cv2.BORDER_CONSTANT,0).T[None,:860,:560] for img in m],0))
                    m1.save('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components_movie_' + str(ID) + '.hdf5')

#%% load each movie and create a super movie
        movies = []
        for ID in range(1,41):
            print(ID)
            movies.append(cm.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components_movie_' + str(ID) + '.hdf5').astype(np.float32))
        movies = np.array(movies)
        movies = movies.transpose([1,0,3,2])
#%%
        montmov = []
        from skimage.util.montage import montage2d
        for idx,fr in enumerate(movies):
            print(idx)
            montmov.append(montage2d(fr,grid_shape=(5,8)).astype(np.float32))
        movies = []
        montmov = np.array(montmov)
        montmov = cm.movie(montmov)
        #%%
        montmov = cm.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components_movie_all.hdf5')
#%% save frames in avi

        import tifffile as tiff
        for idx,vid_frame in enumerate(montmov):
            print(idx)
            tiff.imsave('/mnt/ceph/neuro/zebra/05292014Fish1-4/frames/'+str(idx)+'_.tif',vid_frame)

#%% Plane 11
if ploton:
    from sklearn.preprocessing import normalize
    num_neur = []
    #tott = np.zeros_like(tottime)
    update_comps_time = []
    tott = []
    time_per_neuron = []
    pl.figure()
    for ID in range(11,12):
#        try:
            with np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/results_analysis_online_1EPOCH_gSig6_equalized_Plane_' + str(ID) + '.npz') as ld:
                locals().update(ld)
                print(ld.keys())
                pl.subplot(5,9,ID)
#                img = normalize(Ab[()][:,3:],'l1',axis=0).mean(-1).reshape(dims,order = 'F').T
                Cn_ = np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/results_analysis_online_Plane_CN_'+str(ID)+ '.npy')
#
#                pl.imshow(Cf[3:],aspect = 'auto', vmax = 10)
                pl.figure();crd = cm.utils.visualization.plot_contours(
                        Ab[()][:,3:].toarray().reshape(tuple(dims)+(-1,), order = 'F').transpose([1,0,2]).\
                        reshape((dims[1]*dims[0],-1),order = 'F'), cv2.resize(Cn_,tuple(dims[::-1])).T, thr=0.9, vmax = 0.75,
                        display_numbers=False)
#                A_thr = cm.source_extraction.cnmf.spatial.threshold_components(Ab[()].tocsc()[:,gnb:].toarray(), dims, medw=None, thr_method='nrg',
#                                                                  maxthr=0.3, nrgthr=0.95, extract_cc=True,
#                                 se=None, ss=None, dview=dview)
#                np.save('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components' + str(ID) + '.npy',A_thr)
                A_thr = np.load('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components' + str(ID) + '.npy')
#                img = normalize(Ab[()][:,gnb:].multiply(A_thr),'l1',axis=0).mean(-1).reshape(dims,order = 'F').T
#                img = Ab[()][:,gnb:].multiply(A_thr).mean(-1).reshape(dims,order = 'F').T
                Ab_thr = Ab[()][:,gnb:].multiply(A_thr)
                img = (Ab_thr.dot(scipy.sparse.spdiags(np.minimum(1.0/np.max(Ab_thr,0).toarray(),100),0,Ab_thr.shape[-1],Ab_thr.shape[-1]))).mean(-1).reshape(dims,order = 'F').T
                pl.imshow(img,vmin=np.percentile(img,5),vmax=np.percentile(img,99.99),cmap = 'hot')

#                A_thr = A_thr > 0

#                pl.imshow(((A_thr*np.random.randint(1,10,A_thr.shape[-1])[None,:]).sum(-1).reshape(dims,order='F')).T, cmap = 'hot', vmin = 0.9, vmax=20)
                pl.axis('off')
                pl.pause(0.05)

                num_neur.append(num_comps[1885-201])
                tottime = tottime[:1885-201]
                num_comps = num_comps[:1885-201]
                update_comps_time.append((np.array(num_comps)[99::100],tottime[99::100].copy()))
                tottime[99::100] = np.nan
                tottime[0] = np.nan
                [(np.where(np.diff([0]+list(num_comps))==cc)[0], tottime[np.where(np.diff([0]+list(num_comps))==cc)[0]]) for cc in range(6)]
                tott.append(tottime)

#        except:
            print(ID)
    pl.tight_layout()
    #%% predictions for Plan 11
    from skimage.util.montage import  montage2d
    predictions, final_crops = cm.components_evaluation.evaluate_components_CNN(Ab[()][:,gnb:], dims, np.array(gSig).astype(np.int), model_name='model/cnn_model', patch_size=50, loaded_model=None, isGPU=False)
    #%%
    idx = np.argsort(predictions[:,0])[:10]#[[0,1,2,3,5,9]]
    Ab_part = Ab[()][:,gnb:][:,idx]
    pl.imshow(montage2d(final_crops[idx]))
    pl.figure();crd = cm.utils.visualization.plot_contours(
                        Ab_part.toarray().reshape(tuple(dims)+(-1,), order = 'F').transpose([1,0,2]).\
                        reshape((dims[1]*dims[0],-1),order = 'F'), cv2.resize(Cn_,tuple(dims[::-1])).T, thr=0.9, vmax = 0.95,
                        display_numbers=True)
    pl.figure();crd = cm.utils.visualization.plot_contours(
                        Ab_part.toarray().reshape(tuple(dims)+(-1,), order = 'F').transpose([1,0,2]).\
                        reshape((dims[1]*dims[0],-1),order = 'F'), img, thr=0.9, display_numbers=True, vmax = .001)
    #%%
    count = 0
    for cf,sp_c in zip(Cf[idx+gnb], final_crops[idx]):
        pl.subplot(10,2,2*count+1)
        pl.imshow(sp_c[10:-10,10:-10])
        pl.axis('off')
        pl.subplot(10,2,2*count+2)
        pl.plot(cf)
        count+=1
        pl.axis('off')

#%%
    pl.rcParams['pdf.fonttype'] = 42
    font = {'family' : 'Arial',
    'weight' : 'regular',
    'size'   : 20}

    pl.rc('font', **font)
    pl.close()
    pl.subplot(1,2,1)
    for ttt in update_comps_time:
        pl.plot(ttt[0],ttt[1],'o')
    pl.xlabel('number of components')
    pl.ylabel('time(s)')
    pl.title('updating shapes')
    pl.subplot(1,2,2)
    #for ttt in tott:
    pl.plot(np.arange(1885-201)*1,np.max(tott,0))
    pl.plot([0,(1885-201)*1],[1,1],'k--')
    pl.ylabel('time (s)')
    pl.xlabel('time (s)')
    pl.title('neural activity tracking')
    #%%
    try:
#        cm.stop_server()
        dview.terminate()
    except:
        print('No clusters to stop')

    c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=24)

    A_thr = cm.source_extraction.cnmf.spatial.threshold_components(Ab[()].tocsc()[:,gnb:].toarray(), dims, medw=None, thr_method='nrg',
                                                                  maxthr=0.3, nrgthr=0.95, extract_cc=True,
                                 se=None, ss=None, dview=dview)

    A_thr = A_thr > 0
    size_neurons = A_thr.sum(0)
#        idx_size_neuro = np.where((size_neurons>min_size_neuro) & (size_neurons<max_size_neuro) )[0]
#    A_thr = A_thr[:,idx_size_neuro]
    print(A_thr.shape)
    #%
    pl.imshow((A_thr*np.random.randint(1,20,A_thr.shape[-1])[None,:]).sum(-1).reshape(dims,order='F'), cmap = 'tab20c')