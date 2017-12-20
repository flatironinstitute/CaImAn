#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:35:22 2017

@author: agiovann
"""
import os
import numpy as np
import caiman as cm
import scipy
import pylab as pl
import cv2
import glob
#%%
import itertools


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


#%% file name
inputs = [{'fname': '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
            'gSig': [8, 8]},
          {'fname': '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
              'gSig': [5, 5]},
          {'fname': '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
              'gSig': [5, 5]},
          {'fname': '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
              'gSig': [5, 5]},
          {'fname': '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
              'gSig': [6, 6]},
          {'fname': '/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
              'gSig': [6, 6]},
          {'fname': '/mnt/ceph/neuro/labeling/k53_20160530/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
              'gSig': [6, 6]},  # SEE BELOW
          {'fname': '/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
              'gSig': [7, 7]},  # SEE BELOW
          {'fname': '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
             'gSig': [12, 12]}]  # SEE BELOW


#fname_new = '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap'
#fname_new = '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap'
#fname_new = '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap'
#fname_new = '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap'
#fname_new = '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap'
#fname_new = '/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap'
fract_allowed_overlap = .5
for inps in inputs[:5]:
    fname_new = inps['fname']
    gSig = inps['gSig']
    gt_file = os.path.join(os.path.split(fname_new)[0], os.path.split(
        fname_new)[1][:-4] + 'match_masks.npz')

    base_name = fname_new.split('/')[5]

    # %% you guys sort out how to deal with these big beasts
    #fname_new = glob.glob('/mnt/ceph/neuro/labeling/k53_20160530/images/mmap/*.mmap')
    # fname_new.sort()
    #gt_file = '/mnt/ceph/neuro/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.match_masks.npz'
    #
    #fname_new = glob.glob('/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/mmap/*.mmap')
    # fname_new.sort()
    #gt_file = ' /mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.match_masks.npz'
    #
    #fname_new = glob.glob('/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/mmap/*.mmap')
    # fname_new.sort()
    #gt_file = '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.match_masks.npz'
    #%%
    maxT = 8100
    with np.load(gt_file, encoding='latin1') as ld:
        print(ld.keys())
        locals().update(ld)
        A_gt = scipy.sparse.coo_matrix(A_gt[()])
        dims = (d1, d2)
        C_gt = C_gt[:, :maxT]
        YrA_gt = YrA_gt[:, :maxT]
        f_gt = f_gt[:, :maxT]
        try:
            fname_new = fname_new[()].decode('unicode_escape')
        except:
            fname_new = fname_new[()]

        A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt.tocsc()[:, :].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                                                                          se=None, ss=None, dview=None)
    #%%
#    crd = cm.utils.visualization.plot_contours(A_gt, Cn, thr=.99)
    #%%
    Yr, dims, T = cm.load_memmap(fname_new)
    T = np.minimum(T, maxT)
    Yr = Yr[:, :T]
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # TODO: needinfo
    Y = np.reshape(Yr, dims + (T,), order='F')
    m_orig = np.array(images)
    #%%
    from caiman.components_evaluation import evaluate_components_CNN
    predictions, final_crops = evaluate_components_CNN(
        A_gt, dims, gSig, model_name='use_cases/CaImAnpaper/cnn_model', isGPU=False)
    #%%
    thresh = .95
    idx_components_cnn = np.where(predictions[:, 1] >= thresh)[0]
    #%%
    if T > 8000:
        num_neurons_per_group = 100
    else:
        num_neurons_per_group = 50

    patch_size = 50
    half_crop = np.minimum(gSig[0] * 3 + 1, patch_size) // 2
    for idx_included in grouper(num_neurons_per_group, idx_components_cnn, fillvalue=None):
        # idx_included = idx_components_cnn[np.arange(50)] # neurons that are displayed
        all_pos_crops = []
        all_neg_crops = []
        all_dubious_crops = []
        base_name = fname_new.split('/')[5]
        dims = (d1, d2)
        idx_included = np.array([x for x in idx_included if x is not None])
        base_name = base_name + '_' + \
            str(idx_included[0]) + '_' + str(idx_included[-1])
        print(base_name)

        idx_excluded = np.setdiff1d(np.arange(A_gt.shape[-1]), idx_included)
        Npeaks = 5
        YrA_gt[np.abs(YrA_gt) > np.std(YrA_gt) * 10] = 0
        traces_gt = YrA_gt + C_gt
        downsampfact = 500
        elm_missing = int(np.ceil(T * 1.0 / downsampfact) * downsampfact - T)
        padbefore = int(np.floor(elm_missing / 2))
        padafter = int(np.ceil(elm_missing / 2))
        tr_tmp = np.pad(
            traces_gt.T, ((padbefore, padafter), (0, 0)), mode='reflect')
        numFramesNew, num_traces = np.shape(tr_tmp)
        #% compute baseline quickly
        print("binning data ...")
        tr_BL = np.reshape(tr_tmp, (downsampfact, int(
            numFramesNew / downsampfact), num_traces), order='F')
        tr_BL = np.percentile(tr_BL, 8, axis=0)
        print("interpolating data ...")
        print(tr_BL.shape)
        tr_BL = scipy.ndimage.zoom(np.array(tr_BL, dtype=np.float32), [
                                   downsampfact, 1], order=3, mode='constant', cval=0.0, prefilter=True)
        if padafter == 0:
            traces_gt -= tr_BL.T
        else:
            traces_gt -= tr_BL[padbefore:-padafter].T

        #traces_gt = scipy.signal.savgol_filter(traces_gt,5,2)
        #%
        fitness, exceptionality, sd_r, md = cm.components_evaluation.compute_event_exceptionality(
            traces_gt[idx_included], robust_std=False, N=5, use_mode_fast=False)
        fitness_d, exceptionality_d, sd_r_d, md_d = cm.components_evaluation.compute_event_exceptionality(
            np.diff(traces_gt[idx_included], axis=1), robust_std=False, N=3, use_mode_fast=False)

        #%
        m_res = m_orig - cm.movie(np.reshape(A_gt.tocsc()[:, idx_excluded].dot(
            C_gt[idx_excluded]) + b_gt.dot(f_gt), dims + (-1,), order='F').transpose([2, 0, 1]))
        #%
#        mean_proj = np.mean(m_res,0)
        #%
#        max_mov = mean_proj.max()
        #%%
#        min_mov = mean_proj.min()
        #%%
#        m_res = (m_res-min_mov)/(max_mov-min_mov)
        #%%
#        m_res.play()
        #%%
#        cnn = False
#        if cnn:
#            import json as simplejson
#            from keras.models import model_from_json
#            model_name = 'use_cases/CaImAnpaper/cnn_model'
#            json_file = open(model_name +'.json', 'r')
#            loaded_model_json = json_file.read()
#            json_file.close()
#            loaded_model = model_from_json(loaded_model_json)
#            loaded_model.load_weights(model_name +'.h5')
#            print("Loaded model from disk")
#

        #%%
#        def my_rank1_nmf(Ypx,dims):
#            ain = np.maximum(np.mean(Ypx, 1), 0)
#            na = ain.dot(ain)
#            if not na:
#                lsls
#            ain /= np.sqrt(na)
#            ain, cin, cin_res = cm.source_extraction.cnmf.online_cnmf.rank1nmf(Ypx, ain)  # expects and returns normalized ain
#            return ain.reshape(dims,order = 'F'), cin, cin_res
        #%%
        count_start = 30
        dims = np.array(dims)

        #bin_ = 10
        cms_total = [np.array(scipy.ndimage.center_of_mass(np.reshape(a.toarray(
        ), dims, order='F'))).astype(np.int) for a in A_gt.tocsc()[:, idx_included].T]
        for count in range(count_start, T):
            if count % 100 == 0:
                print(count)
#            print(count)
            img_avg = m_res[count - count_start:count].mean(0)
            image_no_neurons = (img_avg).astype(np.float32)
        #    img_temp = m_res[count].copy().astype(np.float32)/max_mov
            possibly_active = np.where((exceptionality[:, count - count_start:count].min(-1) < -10) & (
                exceptionality[:, count - count_start:count].min(-1) >= -25))[0]
            super_active = np.where(
                (exceptionality[:, count - count_start:count].min(-1) < -25))[0]

#            cms_1 = [np.array(scipy.ndimage.center_of_mass(np.reshape(a.toarray(),dims,order = 'F'))).astype(np.int) for a in  A_gt.tocsc()[:,idx_included[possibly_active]].T]
#            cms_2 = [np.array(scipy.ndimage.center_of_mass(np.reshape(a.toarray(),dims,order = 'F'))).astype(np.int) for a in  A_gt.tocsc()[:,idx_included[super_active]].T]
            cms_1 = np.array(cms_total)[possibly_active]
            cms_2 = np.array(cms_total)[super_active]
            cms_1 = np.maximum(cms_1, half_crop)
            cms_1 = np.array([np.minimum(cms, dims - half_crop)
                              for cms in cms_1]).astype(np.int)
            cms_2 = np.maximum(cms_2, half_crop)
            cms_2 = np.array([np.minimum(cms, dims - half_crop)
                              for cms in cms_2]).astype(np.int)

#            if cnn:
#                As = []
#                final_crops = []
#                for cm__ in cms_1:
#                    cm_=cm__[::]
#
#                #        Ypx = m_res[count-count_start+5:count+5,cm_[0]-gSig[0]*2:cm_[0]+gSig[0]*2,cm_[1]-gSig[0]*2: cm_[1]+gSig[0]*2 ].T.reshape([-1,count_start],order = 'F')
#                #        ain,cin, cin_res = my_rank1_nmf(Ypx,[gSig[0]*4,gSig[0]*4])
#                    Ypx = m_res[count-count_start+5:count+5,cm_[0]-half_crop:cm_[0]+half_crop, cm_[1]-half_crop:cm_[1]+half_crop].T.reshape([-1,count_start],order = 'F')
#                    ain,cin, cin_res = my_rank1_nmf(Ypx,[half_crop*2,half_crop*2])
#                    As.append(ain)
#                    final_crops.append(cv2.resize(ain/np.linalg.norm(ain),(patch_size ,patch_size)))
#
#                predictions = loaded_model.predict(np.array(final_crops)[:,:,:,np.newaxis], batch_size=32, verbose=1)
#                for pred,cm_ in zip(predictions,cms_1):
#                    if pred[1]>1e-8:
#                        img_temp = cv2.rectangle(img_temp,(cm_[1]-half_crop, cm_[0]-half_crop),(cm_[1]+half_crop, cm_[0]+half_crop),.3/2)
#
#            image_no_neurons = img_temp
            image_orig = np.array(image_no_neurons.copy())
            image_orig = (image_orig -np.median(image_orig))/np.std(image_orig)
            img_temp = np.array(image_no_neurons.copy())
            for cm__ in cms_total:
                cm_ = cm__[::]
#                img_temp = cv2.rectangle(img_temp,(cm_[1]-half_crop, cm_[0]-half_crop),(cm_[1]+half_crop, cm_[0]+half_crop),0)
                image_no_neurons[cm_[0] - half_crop:cm_[0] + half_crop,
                                 cm_[1] - half_crop:cm_[1] + half_crop] = np.nan

            for cm__ in cms_1:
                cm_ = cm__[::]
#                img_temp = cv2.rectangle(img_temp,(cm_[1]-half_crop, cm_[0]-half_crop),(cm_[1]+half_crop, cm_[0]+half_crop),.5)
                all_dubious_crops.append(image_orig[cm_[
                                         0] - half_crop:cm_[0] + half_crop, cm_[1] - half_crop:cm_[1] + half_crop])

        #
            for cm__ in cms_2:
                cm_ = cm__[::]
#                img_temp = cv2.rectangle(img_temp,(cm_[1]-half_crop, cm_[0]-half_crop),(cm_[1]+half_crop, cm_[0]+half_crop),1)
                all_pos_crops.append(image_orig[cm_[
                                     0] - half_crop:cm_[0] + half_crop, cm_[1] - half_crop:cm_[1] + half_crop])

            total_patches = 0
            min_overlap = 0 # the first patches are chosen so to not overlap to neurons
            # extract classes without
            for (px, py) in zip(np.random.randint(low=half_crop, high=dims[0] - half_crop, size=10000), np.random.randint(low=half_crop, high=dims[1] - half_crop, size=10000)):
                overlap_ = np.sum(np.isnan(np.array(image_no_neurons)[px - half_crop:px + half_crop, py - half_crop:py + half_crop]))
                if min_overlap <= overlap_ < np.prod(gSig)*4*fract_allowed_overlap:
                    total_patches += 1
#                    img_temp = cv2.rectangle(img_temp,(py-half_crop, px-half_crop),(py+half_crop, px+half_crop),1000)
                    all_neg_crops.append(
                        image_orig[px - half_crop:px + half_crop, py - half_crop:py + half_crop])
                    if total_patches >= len(cms_2):
                        break
                    if total_patches > 1:
                        # the first patches are chosen so to not to overlap to neurons
                        min_overlap = 1


        #
        #    cms_1 = cms_2q
        #    cms_1 = np.maximum(cms_1,half_crop)
        #    cms_1 = np.array([np.minimum(cms,dims-half_crop) for cms in cms_1]).astype(np.int)
        #    crop_imgs = [img_avg[com[0]-half_crop:com[0]+half_crop, com[1]-half_crop:com[1]+half_crop] for com in cms_1]
        #    final_crops = np.array([cv2.resize(im/np.linalg.norm(im),(patch_size ,patch_size)) for im in crop_imgs])

        #

        #    cm.movie(np.array(final_crops[predictions[:,1]>=.01])).play(magnification = 10, gain = 3,fr = 5)
        #    cv2.imshow('frame', cv2.resize(image_no_neurons*img_temp*3,(dims[1]*3,dims[0]*3)))
#            cv2.imshow('frame', cv2.resize(img_temp*1,(dims[1]*3,dims[0]*3))*.001)
#            cv2.waitKey(200)
        np.savez(base_name+str(count_start)+'_tight_overlap_norm.npz',all_dubious_crops = all_dubious_crops, all_pos_crops = all_pos_crops, all_neg_crops = all_neg_crops, gSig = gSig)
        print([len(all_neg_crops), len(all_pos_crops), len(all_dubious_crops)])

#%%
total_crops = []
total_labels = []
patch_size = 50
for fl in glob.glob('*.npz'):
    print(fl)
    try:
        with np.load(fl) as ld:
            all_neg_crops = ld['all_neg_crops']
            all_pos_crops = ld['all_pos_crops']
            all_dubious_crops = ld['all_dubious_crops']
            gSig = ld['gSig']
            # , all_dubious_crops]):
            for class_id, pos in enumerate([all_pos_crops, all_neg_crops]):
                pos = pos - np.median(pos, axis=(1, 2))[:, None, None]
                pos = pos / np.std(pos, axis=(1, 2))[:, None, None]
                total_crops += [cv2.resize(ain / np.linalg.norm(ain),
                                           (patch_size, patch_size)) for ain in pos]
                total_labels += [class_id] * len(pos)

            print([len(all_neg_crops), len(all_pos_crops), len(all_dubious_crops)])
    except:
        pass
rand_perm = np.random.permutation(len(total_crops))
total_crops = np.array(total_crops)[rand_perm]
total_labels = np.array(total_labels)[rand_perm]
np.savez('use_cases/edge-cutter/residual_crops_all_classes_tight.npz',
         all_masks_gt=total_crops, labels_gt=total_labels)
#%%

pos = np.array(all_neg_crops)[np.random.permutation(len(all_neg_crops))]
pos = pos - np.median(pos, axis=(1, 2))[:, None, None]
pos = pos / np.std(pos, axis=(1, 2))[:, None, None]
cm.movie(pos).play(magnification=7, gain=2, fr=3)
#%%
crd = cm.utils.visualization.plot_contours(
    A_gt.tocsc()[:, idx_included], Cn, thr=.99)
