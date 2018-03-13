#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:12:46 2017

@author: agiovann
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:22:48 2017

@author: agiovann
"""
#%%
import numpy as np
try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

from time import time
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.visualization import view_patches_bar
import pylab as pl
import scipy
from caiman.motion_correction import motion_correct_iteration_fast
from caiman.source_extraction.cnmf.online_cnmf import RingBuffer
from caiman.components_evaluation import evaluate_components
import cv2
from caiman.utils.visualization import plot_contours
import cPickle as pickle
import glob
import os
#%%


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'r') as input_obj:
        obj = pickle.load(input_obj)
    return obj

#%%


def sliding_window_new(image, overlaps, strides):
    ''' efficiently and lazily slides a window across the image
     Parameters
     ----------
     img:ndarray 2D
         image that needs to be slices
     windowSize: tuple 
         dimension of the patch
     strides: tuple
         stride in wach dimension    

     Returns:
     -------
     iterator containing five items
     dim_1, dim_2 coordinates in the patch grid 
     x, y: bottom border of the patch in the original matrix  
     patch: the patch
     '''
    windowSize = np.add(overlaps, strides)
    range_1 = list(range(
        0, image.shape[0] - windowSize[0], strides[0])) + [image.shape[0] - windowSize[0]]
    range_2 = list(range(
        0, image.shape[1] - windowSize[1], strides[1])) + [image.shape[1] - windowSize[1]]
    for dim_1, x in enumerate(range_1):
        for dim_2, y in enumerate(range_2):
            x0 = x
            x1 = x + windowSize[0]
            y0 = y
            y1 = y + windowSize[1]
            # yield the current window
            yield (x0, x1, y0, y1, image[x0:x1, y0:y1])
#%%


def remount_window(generator_images, shape_img):
    image = np.zeros(shape_img)
    for (x, y, x0, x1, y0, y1, fragm) in generator_images:
        #        image[ x:x + windowSize[0],y:y + windowSize[1]]
        print(dim_1, dim_2, x, y)
        image[x:x1, y0:y1] = fragm
    return image


#%%
def initialize_movie(Y, K, gSig, rf, stride, base_name,
                     p=1, merge_thresh=0.95, rval_thr_online=0.9, thresh_fitness_delta_online=-30, thresh_fitness_raw_online=-50,
                     rval_thr_init=.5, thresh_fitness_delta_init=-20, thresh_fitness_raw_init=-20,
                     rval_thr_refine=0.95, thresh_fitness_delta_refine=-100, thresh_fitness_raw_refine=-100,
                     final_frate=10, Npeaks=10, single_thread=True, dview=None):

    _, d1, d2 = Y.shape
    dims = (d1, d2)
    Yr = Y.to_2D().T
    # merging threshold, max correlation allowed
    # order of the autoregressive system
    #T = Y.shape[0]
    base_name = base_name + '.mmap'
    fname_new = Y.save(base_name, order='C')
    #%
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.reshape(Yr, dims + (T,), order='F')
    Cn2 = cm.local_correlations(Y)
#    pl.imshow(Cn2)
    #%
    #% RUN ALGORITHM ON PATCHES
#    pl.close('all')
    cnm_init = cnmf.CNMF(n_processes, method_init='greedy_roi', k=K, gSig=gSig, merge_thresh=merge_thresh,
                         p=0, dview=dview, Ain=None, rf=rf, stride=stride, method_deconvolution='oasis', skip_refinement=False,
                         normalize_init=False, options_local_NMF=None,
                         minibatch_shape=100, minibatch_suff_stat=5,
                         update_num_comps=True, rval_thr=rval_thr_online, thresh_fitness_delta=thresh_fitness_delta_online, thresh_fitness_raw=thresh_fitness_raw_online,
                         batch_update_suff_stat=True, max_comp_update_shape=5)

    cnm_init = cnm_init.fit(images)
    A_tot = cnm_init.A
    C_tot = cnm_init.C
    YrA_tot = cnm_init.YrA
    b_tot = cnm_init.b
    f_tot = cnm_init.f

    print(('Number of components:' + str(A_tot.shape[-1])))

    #%

    traces = C_tot + YrA_tot
    #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
    #        traces_b=np.diff(traces,axis=1)
    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = evaluate_components(
        Y, traces, A_tot, C_tot, b_tot, f_tot, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

    idx_components_r = np.where(r_values >= rval_thr_init)[0]
    idx_components_raw = np.where(fitness_raw < thresh_fitness_raw_init)[0]
    idx_components_delta = np.where(
        fitness_delta < thresh_fitness_delta_init)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(('Keeping ' + str(len(idx_components)) +
           ' and discarding  ' + str(len(idx_components_bad))))

    A_tot = A_tot.tocsc()[:, idx_components]
    C_tot = C_tot[idx_components]
    #%
    cnm_refine = cnmf.CNMF(n_processes, method_init='greedy_roi', k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, rf=None, stride=None,
                           p=p, dview=dview, Ain=A_tot, Cin=C_tot, f_in=f_tot, method_deconvolution='oasis', skip_refinement=True,
                           normalize_init=False, options_local_NMF=None,
                           minibatch_shape=100, minibatch_suff_stat=5,
                           update_num_comps=True, rval_thr=rval_thr_refine, thresh_fitness_delta=thresh_fitness_delta_refine, thresh_fitness_raw=thresh_fitness_raw_refine,
                           batch_update_suff_stat=True, max_comp_update_shape=5)

    cnm_refine = cnm_refine.fit(images)
    #%
    A, C, b, f, YrA, sn = cnm_refine.A, cnm_refine.C, cnm_refine.b, cnm_refine.f, cnm_refine.YrA, cnm_refine.sn
    #%
    final_frate = 10
    Npeaks = 10
    traces = C + YrA

    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
        evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                            N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

    idx_components_r = np.where(r_values >= rval_thr_refine)[0]
    idx_components_raw = np.where(fitness_raw < thresh_fitness_raw_refine)[0]
    idx_components_delta = np.where(
        fitness_delta < thresh_fitness_delta_refine)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(' ***** ')
    print((len(traces)))
    print((len(idx_components)))
    #%
    cnm_refine.idx_components = idx_components
    cnm_refine.idx_components_bad = idx_components_bad
    cnm_refine.r_values = r_values
    cnm_refine.fitness_raw = fitness_raw
    cnm_refine.fitness_delta = fitness_delta
    cnm_refine.Cn2 = Cn2

    #%

#    cnm_init.dview = None
#    save_object(cnm_init,fls[0][:-4]+ '_DS_' + str(ds)+ '_init.pkl')

    return cnm_refine, Cn2, fname_new


#%%
#m = cm.load('/mnt/xfs1/home/agiovann/SOFTWARE/CaImAn/example_movies/demoMovie.tif')
m = cm.load('/opt/local/Data/Johannes/Ahrens/TM_layer20_crop_3000_5150_orig.hdf5')
#%%
initbatch = 100
K = 8
gSig = [2, 2]
rf = 15
stride = 6
#expected_comps =120
base_name_total = 'demo_'
max_shifts = (8, 8)
expected_comps = 400
mot_corr = True
img_min = m[:initbatch].min()
overlaps = (32, 32)
strides = (192, 192)
T1 = m.shape[0] + initbatch

#m_init, shifts_init, template_init, corrs_init = m[:initbatch].copy().motion_correct()
# %%
#Cn_init = m_init.local_correlations(eight_neighbours = True,swap_dim  = False)
# pl.imshow(Cn_init)
#%%
#ccnnmm = initialize_movie(m,4,[7,7],14,6,50,'./test_andrea_')
#crd = plot_contours(ccnnmm.A.tocsc()[:, idx_components], Cn2, thr=0.9)

movies = []
cnms = []
imgs_norm = []
Cns = []
base_names = []
pl.figure(figsize=(15, 10))
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
for (x0, x1, y0, y1, _) in sliding_window_new(m[0], overlaps, strides):
    print([x0, y0])
    base_name = base_name_total + str(x0) + '_' + str(y0)

    mc_init, shifts_init, template_init, corrs_init = m[:initbatch, x0:x1, y0:y1].copy(
    ).motion_correct(max_shifts[0], max_shifts[1])
    mc_init -= img_min
#     img_norm = np.std(mc_init, axis=0)[None,:,:]
#     img_norm  += np.median(img_norm)
    img_norm = 1
    mc_init = mc_init / img_norm
    imgs_norm.append(img_norm)


#     pl.figure()
    cnm, Cn2, fname_new = initialize_movie(mc_init, K, gSig, rf, stride, base_name,  merge_thresh=1.1,
                                           rval_thr_online=0.9, thresh_fitness_delta_online=-30, thresh_fitness_raw_online=-50,
                                           rval_thr_init=.5, thresh_fitness_delta_init=-20, thresh_fitness_raw_init=-20,
                                           rval_thr_refine=0.995, thresh_fitness_delta_refine=-200, thresh_fitness_raw_refine=-200,
                                           final_frate=2, Npeaks=10, single_thread=False, dview=dview)
    cnms.append(cnm)
    Cns.append(Cn2)
    base_names.append(fname_new)
    A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
    crd = plot_contours(scipy.sparse.coo_matrix(A), Cns[-1], thr=0.9)
    pl.pause(1)


cm.stop_server()
log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
#%% vompute all the Cns
Cns_all = []
templates_all = []
shifts_all = []
corrs_all = []
for (x0, x1, y0, y1, _) in sliding_window_new(m[0], overlaps, strides):
    print([x0, y0])
    mc, shifts, template, corrs = m[:, x0:x1, y0:y1].copy(
    ).motion_correct(max_shifts[0], max_shifts[1])
    Cns_all.append(mc.local_correlations(
        eight_neighbours=True, swap_dim=False))
    templates_all.append(shifts)
    shifts_all.append(template)
    corrs_all.append(corrs)
#%%
for fname, cnm in zip(base_names, cnms):
    cnm.dview = None
    save_object(cnm, fname[:-5] + '.pkl')
#%%
np.savez('demo_zebra_variables', Cns_all=Cns_all, templates_all=templates_all,
         shifts_all=shifts_all, corrs_all=corrs_all, imgs_norm=imgs_norm, Cns=Cns, base_names=base_names)
#%%
with np.load('demo_zebra_variables.npz') as ld:
    locals().update(ld)
#%%
pl.close('all')
for count, fname in enumerate(base_names):
    pl.subplot(3, 5, count + 1)
    pl.imshow(Cns[count])
#%%
pl.close('all')
for count, fname in enumerate(base_names):
    pl.subplot(3, 5, count + 1)
    cnm = load_object(fname[:-5] + '.pkl')
    idx_components = np.where((cnm.r_values > 0.95) | (
        cnm.fitness_raw < -50) | (cnm.fitness_delta < -50))[0]
    A, C, b, f, YrA, sn = cnm.A.tocsc()[
        :, idx_components], cnm.C[idx_components], cnm.b, cnm.f, cnm.YrA[idx_components], cnm.sn
    crd = plot_contours(scipy.sparse.coo_matrix(A), Cns[count], thr=0.9)
#%%
# for  Cn,cnm,img_norm, base_name in zip(Cns,cnms,imgs_norm,base_names):
#    mc_init = cm.load(base_name)
#    pl.figure()
#    idx_components = np.where((cnm.r_values>0.95) | (cnm.fitness_raw<-50) | (cnm.fitness_delta<-50))[0]
#    A, C, b, f, YrA, sn = cnm.A.tocsc()[:,idx_components], cnm.C[idx_components], cnm.b, cnm.f, cnm.YrA[idx_components], cnm.sn
#    crd = plot_contours(scipy.sparse.coo_matrix(A), Cn, thr=0.9)
#    (mc_init*img_norm).resize(1,1,1).play(gain = 2., magnification=3, fr =100)
#%%
cnms_2 = []
for count, fname in enumerate(base_names):
    cnm = load_object(fname[:-5] + '.pkl')
    mc_init = cm.load(fname)
    _, d1, d2 = mc_init.shape
    dims = (d1, d2)
    Yr = mc_init .to_2D().T
    idx_components = np.where((cnm.r_values > 0.95) | (
        cnm.fitness_raw < -50) | (cnm.fitness_delta < -50))[0]
    cnm._prepare_object(np.asarray(Yr), T1, expected_comps,
                        idx_components=cnm.idx_components)
    cnm.rval_thr = .75
    cnm.thresh_fitness_delta = -20
    cnm.thresh_fitness_raw = -20
    cnm.gSig = [4, 4]
    cnm.gSiz[:] = 9
    cnm.max_comp_update_shape = np.inf
    cnm.update_num_comps = True
    cnms_2.append(cnm)

#%%
#

# print(np.max(cnm2.CC))
# print(np.max(cnm2.CY))
# print(np.max(cnm2.C_on[:,:t-1]))
# print(np.sum(cnm2.Ab.power(2),0))

t = initbatch
tottime = []
shifts = []
cnm_res = cnms_2

for whole_frame in m[:]:
    if np.isnan(np.sum(whole_frame)):
        raise Exception('Frame ' + str(count) + ' contains nan')
    if t % 10 == 0:
        print(t)
    count = 0
    for (x0, x1, y0, y1, frame) in sliding_window_new(whole_frame, overlaps, strides):
        if count != 0:
            continue
        cnm2 = cnms_2[count]
#        print(count)
        img_norm = imgs_norm[count]
        t1 = time()
        frame_ = frame.copy().astype(np.float32)
        frame_ -= img_min
        templ = cnm2.Ab.dot(
            cnm2.C_on[:, t - 1]).reshape(cnm2.dims, order='F') * img_norm
        frame_cor, shift = motion_correct_iteration_fast(
            frame_, templ, max_shifts[0], max_shifts[1])
        shifts.append(shift)

        frame_cor = frame_cor / img_norm
        cnm2.fit_next(t, frame_cor.reshape(-1, order='F'))
        tottime.append(time() - t1)
        cnm_res[count] = cnm2
        count += 1
#        break
    t += 1

#    A,b = cnm2.Ab[:,:-1],cnm2.Ab[:,-1].toarray()
#    C, f = cnm2.C_on[:-1,:], cnm2.C_on[-1:,:]
#    noisyC=cnm2.noisyC
#    pl.cla()
#    crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
#    pl.pause(.1)
#    if mot_corr:
#        np.savez('results_analysis_online_SUE_' + '_DS_' + str(ds)+ '_' +  str(t) + '.npz', Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f, dims=cnm2.dims, tottime = tottime,noisyC = cnm2.noisyC, shifts = shifts, templ = templ)
#    else:
#        np.savez('results_analysis_online_JEFF_LAST_' + '_DS_' + str(ds)+ '_' +  str(t) + '.npz', Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f, dims=cnm2.dims, tottime = tottime,noisyC = cnm2.noisyC)

print((t - initbatch) / np.sum(tottime) / len(cnms_2))
cv2.destroyAllWindows()

#%%
for count, cnm in enumerate(cnm_res):
    #    pl.subplot(3,5,count+1)
    A, C, b, f, YrA, sn = cnm.Ab.tocsc()[:, :-1], cnm.C_on[:-1], cnm.Ab.tocsc()[
        :, -1:], cnm.C_on[-1:], cnm.noisyC[:-1], cnm.sn
    crd = plot_contours(scipy.sparse.coo_matrix(A), Cns_all[count], thr=0.9)
    break
#    break
#%%
for (x0, x1, y0, y1, _) in sliding_window_new(m[0], overlaps, strides):
    print([x0, y0])
    mc, shifts, template, corrs = m[:, x0:x1, y0:y1].copy(
    ).motion_correct(max_shifts[0], max_shifts[1])
    break
