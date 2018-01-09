#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sun Jul 24 17:06:17 2016

@author: agiovann
"""

from __future__ import division
from __future__ import print_function
#%%
from builtins import str
from builtins import range
from past.utils import old_div
try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    print((1))
except:
    print('NOT IPYTHON')

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
# plt.ion()

import sys
import numpy as np


# sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import glob
import os
import scipy
from ipyparallel import Client

import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.source_extraction import cnmf as cnmf
#%%
# backend='SLURM'
backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    # roughly number of cores on your machine minus 1
    n_processes = np.maximum(np.int(psutil.cpu_count()), 1)
print(('using ' + str(n_processes) + ' processes'))
#%% start cluster for efficient computation
single_thread = False

if single_thread:
    dview = None
else:
    try:
        c.close()
    except:
        print('C was not existing, creating one')
    print("Stopping  cluster to avoid unnencessary use of memory....")
    sys.stdout.flush()
    if backend == 'SLURM':
        try:
            cm.stop_server(is_slurm=True)
        except:
            print('Nothing to stop')
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cm.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:
        cm.stop_server()
        cm.start_server()
        c = Client()

    print(('Using ' + str(len(c)) + ' processes'))
    dview = c[:len(c)]
#%%
os.chdir('/mnt/ceph/users/agiovann/ImagingData/eyeblink/b38/20160706154257')
fls = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith("1.npz"):
            print((os.path.join(root, file)))
            fls.append(os.path.join(root, file))

fls.sort()
for fl in fls:
    print(fl)
    with np.load(fl) as ld:
        print((list(ld.keys())))

        tmpls = ld['template']
        lq, hq = np.percentile(tmpls, [5, 95])
        pl.imshow(tmpls, cmap='gray', vmin=lq, vmax=hq)
        pl.pause(.001)
        pl.cla()
#%%
all_movs = []
for f in fls:
    with np.load(f) as fl:
        print(f)
#        pl.subplot(1,2,1)
#        pl.imshow(fl['template'],cmap=pl.cm.gray)
#        pl.subplot(1,2,2)

        all_movs.append(fl['template'][np.newaxis, :, :])
#        pl.plot(fl['shifts'])
#        pl.pause(.001)
all_movs = cb.movie(np.concatenate(all_movs, axis=0), fr=30)
all_movs, shifts, _, _ = all_movs.motion_correct(20, 20, template=None)
all_movs[30:80].play(backend='opencv', gain=5., fr=10)
all_movs = all_movs[30:80]
fls = fls[30:80]
final_template = np.median(all_movs, 0)
#%%
new_fls = []
for fl in fls:
    new_fls.append(fl[:-3] + 'tif')
#%%
file_res = cb.motion_correct_parallel(new_fls, fr=6, template=final_template, margins_out=0,
                                      max_shift_w=25, max_shift_h=25, dview=c[:], apply_smooth=True, save_hdf5=False, remove_blanks=False)
#%%
xy_shifts = []
for fl in new_fls:
    if os.path.exists(fl[:-3] + 'npz'):
        print((fl[:-3] + 'npz'))
        with np.load(fl[:-3] + 'npz') as ld:
            xy_shifts.append(ld['shifts'])
    else:
        raise Exception('*********************** ERROR, FILE NOT EXISTING!!!')
#%%
resize_facts = (1, 1, .2)
name_new = cm.save_memmap_each(
    new_fls, dview=c[:], base_name=None, resize_fact=resize_facts, remove_init=0, xy_shifts=xy_shifts)
#%%
fname_new = cm.save_memmap_join(
    name_new, base_name='TOTAL_', n_chunks=6, dview=c[:])
#%%
m = cm.load('TOTAL__d1_512_d2_512_d3_1_order_C_frames_2300_.mmap', fr=6)
#%%
tmp = np.median(m, 0)
#%%
Cn = m.local_correlations(eight_neighbours=True, swap_dim=False)
pl.imshow(Cn, cmap='gray')
#%%
lq, hq = np.percentile(tmp, [10, 98])
pl.imshow(tmp, cmap='gray', vmin=lq, vmax=hq)
#%%
pl.imshow(tmp[10:160, 120:450], cmap='gray', vmin=lq, vmax=hq)
#%%
m1 = m[:, 10:160, 120:450]
m1.save('MOV_EXAMPLE_20160706154257.tif')
#%%
name_new = cm.save_memmap_each(
    ['MOV_EXAMPLE_20160706154257.tif'], dview=c[:], base_name=None)
#%%
n_chunks = 6  # increase this number if you have memory issues at this point
fname_new = cm.save_memmap_join(
    name_new, base_name='MOV_EXAMPLE_20160706154257__', n_chunks=6, dview=dview)
#%%
Yr, dims, T = cm.load_memmap(
    'MOV_EXAMPLE_20160706154257___d1_150_d2_330_d3_1_order_C_frames_2300_.mmap')
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
#%%
Cn = cm.local_correlations(Y[:, :, :3000], swap_dim=True)
pl.imshow(Cn, cmap='gray')
#%%
rf = 10  # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 4  # amounpl.it of overlap between the patches in pixels
K = 4  # number of neurons expected per patch
gSig = [5, 5]  # expected half size of neurons
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 2  # order of the autoregressive system
memory_fact = 1  # unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
save_results = False
#%% RUN ALGORITHM ON PATCHES
cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=dview, Ain=None,
                rf=rf, stride=stride, memory_fact=memory_fact,
                method_init='greedy_roi', alpha_snmf=10e2)
cnm = cnm.fit(images)

A_tot = cnm.A
C_tot = cnm.C
YrA_tot = cnm.YrA
b_tot = cnm.b
f_tot = cnm.f
sn_tot = cnm.sn

print(('Number of components:' + str(A_tot.shape[-1])))
#%%

final_frate = 2  # approx final rate  (after eventual downsampling )
tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
Npeaks = 10
traces = C_tot + YrA_tot
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = evaluate_components(
    Y, traces, A_tot, C_tot, b_tot, f_tot, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

idx_components_r = np.where(r_values >= .4)[0]
idx_components_raw = np.where(fitness_raw < -20)[0]
idx_components_delta = np.where(fitness_delta < -10)[0]

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))
#%%
pl.figure()
crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)
#%%
A_tot = A_tot.tocsc()[:, idx_components]
C_tot = C_tot[idx_components]
#%%
cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
                f_in=f_tot, rf=None, stride=None)
cnm = cnm.fit(images)
#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%%
final_frate = 1
tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
Npeaks = 10
traces = C + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f, remove_baseline=True,
                        N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

idx_components_r = np.where(r_values >= .5)[0]
idx_components_raw = np.where(fitness_raw < -50)[0]
idx_components_delta = np.where(fitness_delta < -30)[0]


min_radius = gSig[0] - 2
# masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
#    A.tocsc(), min_radius, dims, num_std_threshold=1,
#    minCircularity=0.5, minInertiaRatio=0.2, minConvexity=.7)

#% LOOK FOR BLOB LIKE STRUCTURES!
masks_ws, is_blob, is_non_blob = cm.base.rois.extract_binary_masks_blob_parallel(A.tocsc(), min_radius, dims, num_std_threshold=1,
                                                                                 minCircularity=0.1, minInertiaRatio=0.1, minConvexity=.1, dview=dview)

idx_blobs = np.where(is_blob)[0]
idx_non_blobs = np.where(is_non_blob)[0]

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

print(' ***** ')
print((len(traces)))
print((len(idx_components)))
print((len(idx_blobs)))
#%%
save_results = False
if save_results:
    np.savez('results_analysis.npz', Cn=Cn, A=A.todense(), C=C, b=b, f=f, YrA=YrA, sn=sn,
             d1=d1, d2=d2, idx_components=idx_components, idx_components_bad=idx_components_bad)
    scipy.io.savemat('results_analysis.mat', {'C': Cn, 'A': A.toarray(), 'C': C, 'b': b, 'f': f, 'YrA': YrA,
                                              'sn': sn, 'd1': d1, 'd2': d2, 'idx_components': idx_components, 'idx_components_blobs': idx_blobs})
    np.savez('results_blobs.npz', spatial_comps=A.tocsc().toarray().reshape(dims + (-1,), order='F').transpose(
        [2, 0, 1]), masks=masks_ws, idx_components=idx_components, idx_blobs=idx_blobs, idx_components_bad=idx_components_bad)


#%% visualize components
# pl.figure();
pl.subplot(1, 3, 1)
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
pl.subplot(1, 3, 2)
crd = plot_contours(A.tocsc()[:, idx_blobs], Cn, thr=0.9)
pl.subplot(1, 3, 3)
crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
#%%
#idx_very_nice=[2, 19, 23, 27,32,43,45,49,51,94,100]
# idx_very_nice=np.array(idx_very_nice)[np.array([3,4,8,10])]
# idx_very_nice=idx_blobs[idx_very_nice]
idx_very_nice = idx_blobs
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_very_nice]), C[
    idx_very_nice, :], b, f, dims[0], dims[1], YrA=YrA[idx_very_nice, :], img=Cn)
#%%
new_m = cm.movie(np.reshape(A.tocsc()[
                 :, idx_blobs] * C[idx_blobs] + b.dot(f), dims + (-1,), order='F').transpose([2, 0, 1]))
new_m.play(fr=30, backend='opencv', gain=7., magnification=3.)
#%%
new_m = cm.movie(np.reshape(A.tocsc()[:, idx_blobs] * C[idx_blobs] +
                            b * np.median(f), dims + (-1,), order='F').transpose([2, 0, 1]))
new_m.play(fr=30, backend='opencv', gain=7., magnification=3.)
#%%
new_m = cm.movie(np.reshape(A.tocsc()[
                 :, idx_blobs] * C[idx_blobs], dims + (-1,), order='F').transpose([2, 0, 1]))
new_m.play(fr=30, backend='opencv', gain=30., magnification=3.)
#%%
# idx_to_show=[0,1,5,8,14,17,18,23,24,25,26,28,29,31,32,33,34,36,43,45,47,51,53,54,57,60,61,62,63,64,65,66,67,71,72,74,75,78,79,80,81,91,95,96,97,99,102]
#cm.view_patches_bar(Yr,scipy.sparse.coo_matrix(A.tocsc()[:,sure_in_idx[idx_to_show]]),C[sure_in_idx[idx_to_show],:],b,f, dims[0],dims[1], YrA=YrA[sure_in_idx[idx_to_show],:],img=np.mean(Y,-1))
#%%
# idx_to_show=[0,1,5,8,14,17,18,23,24,25,26,28,29,31,32,33,34,36,43,45,47,51,53,54,57,60,61,62,63,64,65,66,67,71,72,74,75,78,79,80,81,91,95,96,97,99,102]
# idx_to_show=np.array(idx_to_show)[[2,19,23,26,34]]
#%%
import numpy as np
import caiman as cm
import scipy
with np.load('results_analysis.npz') as ld:
    locals().update(ld)

A = scipy.sparse.coo_matrix(A)

with np.load('results_blobs.npz') as ld:
    locals().update(ld)

m = cm.load(
    'MOV_EXAMPLE_20160706154257___d1_150_d2_330_d3_1_order_C_frames_2300_.mmap')
Yr, dims, T = cm.load_memmap(
    'MOV_EXAMPLE_20160706154257___d1_150_d2_330_d3_1_order_C_frames_2300_.mmap')
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
#%%

ylimit = 100
pl.figure()
pl.subplot(3, 1, 1)
pl.imshow(np.mean(Y, -1), cmap='gray', vmin=10, vmax=60)
pl.ylim([0, ylimit])
pl.axis('off')
pl.subplot(3, 1, 2)
pl.imshow(np.mean(Y, -1), cmap='gray', vmin=10, vmax=60)
msk = np.reshape(A.tocsc()[:, sure_in_idx[7]].sum(-1), dims, order='F')
msk[msk < 0.01] = np.nan
pl.imshow(msk, cmap='Greens', alpha=.3)
msk = np.reshape(
    A.tocsc()[:, sure_in_idx[idx_to_show]].sum(-1), dims, order='F')
msk[msk < 0.01] = np.nan
pl.ylim([0, ylimit])
pl.imshow(msk, cmap='hot', alpha=.3)
pl.axis('off')
pl.subplot(3, 1, 3)
pl.imshow(np.reshape(
    A.tocsc()[:, sure_in_idx[idx_to_show]].mean(-1), dims, order='F'), cmap='hot')
pl.ylim([0, ylimit])
pl.axis('off')


font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 30}

pl.rc('font', **font)
#%

pl.figure()
counter = 0
for iid in sure_in_idx[np.hstack([idx_to_show, 7])]:
    counter += 1
    pl.subplot(7, 7, counter)
    mmsk = np.reshape(A.tocsc()[:, iid].todense(), dims, order='F')
    cx, cy = scipy.ndimage.measurements.center_of_mass(np.array(mmsk))
    cx = np.int(cx)
    cy = np.int(cy)
    print((cx, cy))
    pl.imshow(mmsk[np.maximum(cx - 15, 0):cx + 15,
                   np.maximum(cy - 15, 0):cy + 15], cmap='gray')
    pl.ylim([0, 30])

    pl.axis('off')
    pl.title(np.hstack([idx_to_show, 7])[counter - 1])


font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 30}

pl.rc('font', **font)
#%
pl.figure()

m = np.array(Yr)
bckg_1 = b.dot(f)
nA = (A.power(2)).sum(0)

m = m - bckg_1


Y_r_sig = A.T.dot(m)
Y_r_sig = scipy.sparse.linalg.spsolve(
    scipy.sparse.spdiags(np.sqrt(nA), 0, nA.size, nA.size), Y_r_sig)
Y_r_bl = A.T.dot(bckg_1)
Y_r_bl = scipy.sparse.linalg.spsolve(
    scipy.sparse.spdiags(np.sqrt(nA), 0, nA.size, nA.size), Y_r_bl)
Y_r_bl = cm.mode_robust(Y_r_bl, 1)
trs = old_div(Y_r_sig, Y_r_bl[:, np.newaxis])

cb.trace(trs[np.hstack([sure_in_idx[idx_to_show], 7])].T, fr=6).plot()
# pl.figure()
# cb.trace(trs[sure_in_idx[7]].T,fr=6).plot()


# %%
# done '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627105123/'
# errors:  '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160623161504/',
# base_folders=[
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627154015/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160624105838/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160625132042/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160626175708/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627110747/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628100247/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160705103903/',
#
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628162522/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160629123648/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160630120544/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160701113525/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160702152950/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160703173620/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160704130454/',
# ]
# error:               '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711104450/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712105933/',
# base_folders=[
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160710134627/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160710193544/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711164154/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711212316/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712101950/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712173043/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160713100916/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160713171246/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714094320/',
# '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714143248/'
# ]
# for base_folder in base_folders:
#    img_descr=cb.utils.get_image_description_SI(glob(base_folder+'2016*.tif')[0])[0]
#    f_rate=img_descr['scanimage.SI.hRoiManager.scanFrameRate']
#    print f_rate
#    #%%
#    fls=glob(os.path.join(base_folder,'2016*.tif'))
#    fls.sort()
#    print fls
#    # verufy they are ordered
#    #%%
#    triggers_img,trigger_names_img=gc.extract_triggers(fls,read_dictionaries=False)
#    np.savez(base_folder+'all_triggers.npz',triggers=triggers_img,trigger_names=trigger_names_img)
#    #%% get information from eyelid traces
#    t_start=time()
#    camera_file=glob(os.path.join(base_folder,'*_cam2.h5'))
#    assert len(camera_file)==1, 'there are none or two camera files'
#    res_bt=gc.get_behavior_traces(camera_file[0],t0=0,t1=8.0,freq=60,ISI=.25,draw_rois=False,plot_traces=False,mov_filt_1d=True,window_lp=5)
#    t_end=time()-t_start
#    print t_end
#    #%%
#    np.savez(base_folder+'behavioral_traces.npz',**res_bt)
#    #%%
#    with np.load(base_folder+'behavioral_traces.npz') as ld:
#        res_bt=dict(**ld)
#    #%%
#    pl.close()
#    tm=res_bt['time']
#    f_rate_bh=1/np.median(np.diff(tm))
#    ISI=res_bt['trial_info'][0][3]-res_bt['trial_info'][0][2]
#    eye_traces=np.array(res_bt['eyelid'])
#    idx_CS_US=res_bt['idx_CS_US']
#    idx_US=res_bt['idx_US']
#    idx_CS=res_bt['idx_CS']
#
#    idx_ALL=np.sort(np.hstack([idx_CS_US,idx_US,idx_CS]))
#    eye_traces,amplitudes_at_US, trig_CRs=gc.process_eyelid_traces(eye_traces,tm,idx_CS_US,idx_US,idx_CS,thresh_CR=.15,time_CR_on=-.1,time_US_on=.05)
#
#    idxCSUSCR = trig_CRs['idxCSUSCR']
#    idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
#    idxCSCR = trig_CRs['idxCSCR']
#    idxCSNOCR = trig_CRs['idxCSNOCR']
#    idxNOCR = trig_CRs['idxNOCR']
#    idxCR = trig_CRs['idxCR']
#    idxUS = trig_CRs['idxUS']
#    idxCSCSUS=np.concatenate([idx_CS,idx_CS_US])
#
#
#    pl.plot(tm,np.mean(eye_traces[idxCSUSCR],0))
#    pl.plot(tm,np.mean(eye_traces[idxCSUSNOCR],0))
#    pl.plot(tm,np.mean(eye_traces[idxCSCR],0))
#    pl.plot(tm,np.mean(eye_traces[idxCSNOCR],0))
#    pl.plot(tm,np.mean(eye_traces[idx_US],0))
#    pl.legend(['idxCSUSCR','idxCSUSNOCR','idxCSCR','idxCSNOCR','idxUS'])
#    pl.xlabel('time to US (s)')
#    pl.ylabel('eyelid closure')
#    plt.axvspan(-ISI,ISI, color='g', alpha=0.2, lw=0)
#    plt.axvspan(0,0.03, color='r', alpha=0.2, lw=0)
#
#    pl.xlim([-.5,1])
#    pl.savefig(base_folder+'behavioral_traces.pdf')
#    #%%
#    #pl.close()
#    #bins=np.arange(0,1,.01)
#    #pl.hist(amplitudes_at_US[idxCR],bins=bins)
#    #pl.hist(amplitudes_at_US[idxNOCR],bins=bins)
#    #pl.savefig(base_folder+'hist_behav.pdf')
#
#
#    #%%
#    pl.close()
#    f_results= glob(base_folder+'*results_analysis.npz')
#    f_results.sort()
#    for rs in f_results:
#        print rs
#    #%% load results and put them in lists
#    A_s,C_s,YrA_s, Cn_s, b_s, f_s, shape =  gc.load_results(f_results)
#    B_s, lab_imgs, cm_s  = gc.threshold_components(A_s,shape, min_size=5,max_size=50,max_perc=.5)
#    #%%
#    if not batch_mode:
#        for i,A_ in enumerate(B_s):
#             sizes=np.array(A_.sum(0)).squeeze()
#             pl.subplot(2,3,i+1)
#             pl.imshow(np.reshape(A_.sum(1),shape,order='F'),cmap='gray',vmax=.5)
#    #%% compute mask distances
#    if len(B_s)>1:
#        max_dist=30
#        D_s=gc.distance_masks(B_s,cm_s,max_dist)
#        np.savez(base_folder+'distance_masks.npz',D_s=D_s)
#        #%%
#        if not batch_mode:
#            for ii,D in enumerate(D_s):
#                pl.subplot(3,3,ii+1)
#                pl.imshow(D,interpolation='None')
#
#        #%% find matches
#        matches,costs =  gc.find_matches(D_s, print_assignment=False)
#        #%%
#        neurons=gc.link_neurons(matches,costs,max_cost=0.6,min_FOV_present=None)
#    else:
#        neurons=[np.arange(B_s[0].shape[-1])]
#    #%%
#    np.savez(base_folder+'neurons_matching.npz',matches=matches,costs=costs,neurons=neurons,D_s=D_s)
#    #%%
#    re_load = False
#    if re_load:
#        import calblitz as cb
#        from calblitz.granule_cells import utils_granule as gc
#        from glob import glob
#        import numpy as np
#        import os
#        import scipy
#        import pylab as pl
#        import ca_source_extraction as cse
#
#        if is_blob:
#            with np.load(base_folder+'distance_masks.npz') as ld:
#                D_s=ld['D_s']
#            with np.load(base_folder+'neurons_matching.npz') as ld:
#                locals().update(ld)
#
#
#
#        with np.load(base_folder+'all_triggers.npz') as at:
#            triggers_img=at['triggers']
#            trigger_names_img=at['trigger_names']
#
#        with np.load(base_folder+'behavioral_traces.npz') as ld:
#            res_bt = dict(**ld)
#            tm=res_bt['time']
#            f_rate_bh=1/np.median(np.diff(tm))
#            ISI=res_bt['trial_info'][0][3]-res_bt['trial_info'][0][2]
#            eye_traces=np.array(res_bt['eyelid'])
#            idx_CS_US=res_bt['idx_CS_US']
#            idx_US=res_bt['idx_US']
#            idx_CS=res_bt['idx_CS']
#
#            idx_ALL=np.sort(np.hstack([idx_CS_US,idx_US,idx_CS]))
#            eye_traces,amplitudes_at_US, trig_CRs=gc.process_eyelid_traces(eye_traces,tm,idx_CS_US,idx_US,idx_CS,thresh_CR=.15,time_CR_on=-.1,time_US_on=.05)
#
#            idxCSUSCR = trig_CRs['idxCSUSCR']
#            idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
#            idxCSCR = trig_CRs['idxCSCR']
#            idxCSNOCR = trig_CRs['idxCSNOCR']
#            idxNOCR = trig_CRs['idxNOCR']
#            idxCR = trig_CRs['idxCR']
#            idxUS = trig_CRs['idxUS']
#            idxCSCSUS=np.concatenate([idx_CS,idx_CS_US])
#
#
#        f_results= glob(base_folder+'*results_analysis.npz')
#        f_results.sort()
#        for rs in f_results:
#            print rs
#        print '*****'
#        A_s,C_s,YrA_s, Cn_s, b_s, f_s, shape =  gc.load_results(f_results)
#        if is_blob:
#            remove_unconnected_components=True
#        else:
#            remove_unconnected_components=False
#
#            neurons=[]
#            for xx in A_s:
#                neurons.append(np.arange(A_s[0].shape[-1]))
#
#        B_s, lab_imgs, cm_s  = gc. threshold_components(A_s,shape, min_size=5,max_size=50,max_perc=.5,remove_unconnected_components=remove_unconnected_components)
#    #%%
#
#    row_cols=np.ceil(np.sqrt(len(A_s)))
#    for idx,B in enumerate(A_s):
#         pl.subplot(row_cols,row_cols,idx+1)
#         pl.imshow(np.reshape(B[:,neurons[idx]].sum(1),shape,order='F'))
#    pl.savefig(base_folder+'neuron_matches.pdf')
#
#    #%%
#    if not batch_mode:
#        num_neurons=neurons[0].size
#        for neuro in range(num_neurons):
#            for idx,B in enumerate(A_s):
#                 pl.subplot(row_cols,row_cols,idx+1)
#                 pl.imshow(np.reshape(B[:,neurons[idx][neuro]].sum(1),shape,order='F'))
#            pl.pause(.01)
#            for idx,B in enumerate(A_s):
#                pl.subplot(row_cols,row_cols,idx+1)
#                pl.cla()
#
#    #%%
#    if 0:
#        idx=0
#        for  row, column in zip(matches[idx][0],matches[idx][1]):
#            value = D_s[idx][row,column]
#            if value < .5:
#                pl.cla()
#                pl.imshow(np.reshape(B_s[idx][:,row].todense(),(512,512),order='F'),cmap='gray',interpolation='None')
#                pl.imshow(np.reshape(B_s[idx+1][:,column].todense(),(512,512),order='F'),alpha=.5,cmap='hot',interpolation='None')
#                if B_s[idx][:,row].T.dot(B_s[idx+1][:,column]).todense() == 0:
#                    print 'Flaw'
#                pl.pause(.3)
#
#    #%%
#    tmpl_name=glob(base_folder+'*template_total.npz')[0]
#    print tmpl_name
#    with np.load(tmpl_name) as ld:
#        mov_names_each=ld['movie_names']
#
#
#    traces=[]
#    traces_BL=[]
#    traces_DFF=[]
#    all_chunk_sizes=[]
#
#    for idx, mov_names in enumerate(mov_names_each):
#        idx=0
#        A=A_s[idx][:,neurons[idx]]
#    #    C=C_s[idx][neurons[idx]]
#    #    YrA=YrA_s[idx][neurons[idx]]
#        b=b_s[idx]
#        f=f_s[idx]
#        chunk_sizes=[]
#        for mv in mov_names:
#                base_name=os.path.splitext(os.path.split(mv)[-1])[0]
#                with np.load(base_folder+base_name+'.npz') as ld:
#                    TT=len(ld['shifts'])
#                chunk_sizes.append(TT)
#
#
#        all_chunk_sizes.append(chunk_sizes)
#
#        traces_,traces_DFF_,traces_BL_ = gc.generate_linked_traces(mov_names,chunk_sizes,A,b,f)
#        traces=traces+traces_
#        traces_DFF=traces_DFF+traces_DFF_
#        traces_BL=traces_BL+traces_BL_
#
#    #%%
#    import pickle
#    with open(base_folder+'traces.pk','w') as f:
#        pickle.dump(dict(traces=traces,traces_BL=traces_BL,traces_DFF=traces_DFF),f)
#
#    #%%
#    if not batch_mode:
#        with open(base_folder+'traces.pk','r') as f:
#            locals().update(pickle.load(f)   )
#    #%%
#    chunk_sizes=[]
#    for idx,mvs in enumerate(mov_names_each):
#        print idx
#        for mv in mvs:
#            base_name=os.path.splitext(os.path.split(mv)[-1])[0]
#            with np.load(os.path.join(base_folder,base_name+'.npz')) as ld:
#                TT=len(ld['shifts'])
#            chunk_sizes.append(TT)
#
#
#    min_chunk=np.min(chunk_sizes)
#    max_chunk=np.max(chunk_sizes)
#    num_chunks=np.sum(chunk_sizes)
#    #%%
#    import copy
#    Ftraces=copy.deepcopy(traces_DFF[:])
#
#    #%%
#
#    #%%
#    interpolate=False
#    CS_ALONE=0
#    US_ALONE=   1
#    CS_US=2
#
#    samples_before=np.int(2.8*f_rate)
#    samples_after=np.int(7.3*f_rate)-samples_before
#
#
#    if interpolate:
#        Ftraces_mat=np.zeros([len(chunk_sizes),len(traces[0]),max_chunk])
#        abs_frames=np.arange(max_chunk)
#    else:
#        Ftraces_mat=np.zeros([len(chunk_sizes),len(traces[0]),samples_after+samples_before])
#
#    crs=idxCR
#    nocrs=idxNOCR
#    uss=idxUS
#
#    triggers_img=np.array(triggers_img)
#
#    idx_trig_CS=triggers_img[:][:,0]
#    idx_trig_US=triggers_img[:][:,1]
#    trial_type=triggers_img[:][:,2]
#    length=triggers_img[:][:,-1]
#    ISI=np.int(np.nanmedian(idx_trig_US)-np.nanmedian(idx_trig_CS))
#
#    for idx,fr in enumerate(chunk_sizes):
#
#        print idx
#
#        if interpolate:
#
#            if fr!=max_chunk:
#
#                f1=scipy.interpolate.interp1d(np.arange(fr) , Ftraces[idx] ,axis=1, bounds_error=False, kind='linear')
#                Ftraces_mat[idx]=np.array(f1(abs_frames))
#
#            else:
#
#                Ftraces_mat[idx]=Ftraces[idx][:,trigs_US-samples_before]
#
#
#        else:
#
#            if trial_type[idx] == CS_ALONE:
#                    Ftraces_mat[idx]=Ftraces[idx][:,np.int(idx_trig_CS[idx]+ISI-samples_before):np.int(idx_trig_CS[idx]+ISI+samples_after)]
#            else:
#                    Ftraces_mat[idx]=Ftraces[idx][:,np.int(idx_trig_US[idx]-samples_before):np.int(idx_trig_US[idx]+samples_after)]
#
#    #%%
#    wheel_traces, movement_at_CS, trigs_mov = gc.process_wheel_traces(np.array(res_bt['wheel']),tm,thresh_MOV_iqr=1000,time_CS_on=-.25,time_US_on=0)
#    print trigs_mov
#    mn_idx_CS_US=np.intersect1d(idx_CS_US,trigs_mov['idxNO_MOV'])
#    nm_idx_US=np.intersect1d(idx_US,trigs_mov['idxNO_MOV'])
#    nm_idx_CS=np.intersect1d(idx_CS,trigs_mov['idxNO_MOV'])
#    nm_idxCSUSCR = np.intersect1d(idxCSUSCR,trigs_mov['idxNO_MOV'])
#    nm_idxCSUSNOCR = np.intersect1d(idxCSUSNOCR,trigs_mov['idxNO_MOV'])
#    nm_idxCSCR = np.intersect1d(idxCSCR,trigs_mov['idxNO_MOV'])
#    nm_idxCSNOCR = np.intersect1d(idxCSNOCR,trigs_mov['idxNO_MOV'])
#    nm_idxNOCR = np.intersect1d(idxNOCR,trigs_mov['idxNO_MOV'])
#    nm_idxCR = np.intersect1d(idxCR,trigs_mov['idxNO_MOV'])
#    nm_idxUS = np.intersect1d(idxUS,trigs_mov['idxNO_MOV'])
#    nm_idxCSCSUS=np.intersect1d(idxCSCSUS,trigs_mov['idxNO_MOV'])
#    #%%
#    threshold_responsiveness=0.1
#    ftraces=Ftraces_mat.copy()
#    ftraces=ftraces-np.median(ftraces[:,:,:samples_before-ISI],axis=(2))[:,:,np.newaxis]
#    amplitudes_responses=np.mean(ftraces[:,:,samples_before+ISI-1:samples_before+ISI+1],-1)
#    cell_responsiveness=np.median(amplitudes_responses[nm_idxCSCSUS],axis=0)
#    fraction_responsive=len(np.where(cell_responsiveness>threshold_responsiveness)[0])*1./np.shape(ftraces)[1]
#    print fraction_responsive
#    ftraces=ftraces[:,cell_responsiveness>threshold_responsiveness,:]
#    amplitudes_responses=np.mean(ftraces[:,:,samples_before+ISI-1:samples_before+ISI+1],-1)
#    #%%
#    np.savez('ftraces.npz',ftraces=ftraces,samples_before=samples_before,samples_after=samples_after,ISI=ISI)
#
#
#    #%%pl.close()
#    pl.close()
#    t=np.arange(-samples_before,samples_after)/f_rate
#    pl.plot(t,np.median(ftraces[nm_idxCR],axis=(0,1)),'-*')
#    pl.plot(t,np.median(ftraces[nm_idxNOCR],axis=(0,1)),'-d')
#    pl.plot(t,np.median(ftraces[nm_idxUS],axis=(0,1)),'-o')
#    plt.axvspan((-ISI)/f_rate, 0, color='g', alpha=0.2, lw=0)
#    plt.axvspan(0, 0.03, color='r', alpha=0.5, lw=0)
#    pl.xlabel('Time to US (s)')
#    pl.ylabel('DF/F')
#    pl.xlim([-.5, 1])
#    pl.legend(['CR+','CR-','US'])
#    pl.savefig(base_folder+'eyelid_resp_by_trial.pdf')
#
#    #%%
#    if not batch_mode:
#        pl.close()
#        for cell in range(ftraces.shape[1]):
#        #    pl.cla()
#            pl.subplot(11,10,cell+1)
#            print cell
#            tr_cr=np.median(ftraces[crs,cell,:],axis=(0))
#            tr_nocr=np.median(ftraces[nocrs,cell,:],axis=(0))
#            tr_us=np.median(ftraces[uss,cell,:],axis=(0))
#            pl.imshow(ftraces[np.concatenate([uss,nocrs,crs]),cell,:],aspect='auto',vmin=0,vmax=1)
#            pl.xlim([samples_before-10,samples_before+10])
#            pl.axis('off')
#        #    pl.plot(tr_cr,'b')
#        #    pl.plot(tr_nocr,'g')
#        #    pl.plot(tr_us,'r')
#        #    pl.legend(['CR+','CR-','US'])
#        #    pl.pause(1)
#    #%%
#    import pandas
#
#    bins=np.arange(-.1,.3,.05)
#    n_bins=6
#    dfs=[];
#    dfs_random=[];
#    x_name='ampl_eye'
#    y_name='ampl_fl'
#    for resps in amplitudes_responses.T:
#        idx_order=np.arange(len(idxCSCSUS))
#        dfs.append(pandas.DataFrame(
#            {y_name: resps[idxCSCSUS[idx_order]],
#             x_name: amplitudes_at_US[idxCSCSUS]}))
#
#        idx_order=np.random.permutation(idx_order)
#        dfs_random.append(pandas.DataFrame(
#            {y_name: resps[idxCSCSUS[idx_order]],
#             x_name: amplitudes_at_US[idxCSCSUS]}))
#
#
#    r_s=[]
#    r_ss=[]
#
#    for df,dfr in zip(dfs,dfs_random): # random scramble
#
#        if bins is None:
#            [_,bins]=np.histogram(dfr.ampl_eye,n_bins)
#        groups = dfr.groupby(np.digitize(dfr.ampl_eye, bins))
#        grouped_mean = groups.mean()
#        grouped_sem = groups.sem()
#        (r,p_val)=scipy.stats.pearsonr(grouped_mean.ampl_eye,grouped_mean.ampl_fl)
#    #    r=np.corrcoef(grouped_mean.ampl_eye,grouped_mean.ampl_fl)[0,1]
#
#        r_ss.append(r)
#
#        if bins is None:
#            [_,bins]=np.histogram(df.ampl_eye,n_bins)
#
#        groups = df.groupby(np.digitize(df.ampl_eye, bins))
#        grouped_mean = groups.mean()
#        grouped_sem= groups.sem()
#        (r,p_val)=scipy.stats.pearsonr(grouped_mean.ampl_eye,grouped_mean.ampl_fl)
#    #    r=np.corrcoef(grouped_mean.ampl_eye,grouped_mean.ampl_fl)[0,1]
#        r_s.append(r)
#        if r_s[-1]>.86:
#            pl.subplot(1,2,1)
#            print 'found'
#            pl.errorbar(grouped_mean.ampl_eye,grouped_mean.ampl_fl,grouped_sem.ampl_fl.as_matrix(),grouped_sem.ampl_eye.as_matrix(),fmt='.')
#            pl.scatter(grouped_mean.ampl_eye,grouped_mean.ampl_fl,s=groups.apply(len).values*3)#
#            pl.xlabel(x_name)
#            pl.ylabel(y_name)
#
#    mu_scr=np.mean(r_ss)
#
#    std_scr=np.std(r_ss)
#    [a,b]=np.histogram(r_s,20)
#
#    pl.subplot(1,2,2)
#    pl.plot(b[1:],scipy.signal.savgol_filter(a,3,1))
#    plt.axvspan(mu_scr-std_scr, mu_scr+std_scr, color='r', alpha=0.2, lw=0)
#    pl.xlabel('correlation coefficients')
#    pl.ylabel('bin counts')
#    pl.savefig(base_folder+'correlations.pdf')
#
#
#
#    #%%
#    if not batch_mode:
#        r_s=[]
#        for resps in amplitudes_responses.T:
#            r=np.corrcoef(amplitudes_at_US[idxCSCSUS],resps[idxCSCSUS])[0,1]
#        #    if r>.25:
#        #        pl.scatter(amplitudes_at_US[idxCSCSUS],resps[idxCSCSUS])
#        #    bins=np.arange(-.3,1.5,.2)
#        #    a,b=np.histogram(resps,bins)
#        #    new_dat=[]
#        #    for bb in a:
#        #
#            r_s.append(r)
#            pl.xlabel('Amplitudes CR')
#            pl.ylabel('Amplitudes GC responses')
#
#        pl.hist(r_s)
#
# %%
###
# base_name='20160518133747_'
# cam1=base_name+'cam1.h5'
# cam2=base_name+'cam2.h5'
# meta_inf=base_name+'data.h5'
###
# mtot=[]
# eye_traces=[]
# tims=[]
# trial_info=[]
###
# with h5py.File(cam2) as f:
###
# with h5py.File(meta_inf) as dt:
###
# rois=np.asarray(dt['roi'],np.float32)
###
###        trials = f.keys()
# trials.sort(key=lambda(x): np.int(x.replace('trial_','')))
###        trials_idx=[np.int(x.replace('trial_',''))-1 for x in trials]
###
###
###
###
# for tr,idx_tr in zip(trials,trials_idx):
###
# print tr
###
# trial=f[tr]
###
# mov=np.asarray(trial['mov'])
###
# if 0:
###
# pl.imshow(np.mean(mov,0))
# pts=pl.ginput(-1)
###                pts = np.asarray(pts, dtype=np.int32)
###                data = np.zeros(np.shape(mov)[1:], dtype=np.int32)
# if CV_VERSION == 2:
# lt = cv2.CV_AA
# elif CV_VERSION == 3:
###                lt = cv2.LINE_AA
###                cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)
# rois[0]=data
# eye_trace=np.mean(mov*rois[0],axis=(1,2))
# mov_trace=np.mean((np.diff(np.asarray(mov,dtype=np.float32),axis=0)**2)*rois[1],axis=(1,2))
# mov=np.transpose(mov,[0,2,1])
###
# mov=mov[:,:,::-1]
###
# if  mov.shape[0]>0:
# ts=np.array(trial['ts'])
# if np.size(ts)>0:
# print (ts[-1,0]-ts[0,0])
# new_ts=np.linspace(0,ts[-1,0]-ts[0,0],np.shape(mov)[0])
###
# print 1/np.mean(np.diff(new_ts))
# tims.append(new_ts)
###
# mov=cb.movie(mov*rois[0][::-1].T,fr=1/np.mean(np.diff(new_ts)))
# x_max,y_max=np.max(np.nonzero(np.max(mov,0)),1)
# x_min,y_min=np.min(np.nonzero(np.max(mov,0)),1)
# mov=mov[:,x_min:x_max,y_min:y_max]
###                mov=np.mean(mov, axis=(1,2))
###
# if mov.ndim == 3:
# window_hp=(177,1,1)
# window_lp=(7,1,1)
# bl=signal.medfilt(mov,window_hp)
# mov=signal.medfilt(mov-bl,window_lp)
###
# else:
# window_hp=201
# window_lp=3
# bl=signal.medfilt(mov,window_hp)
# bl=cm.mode_robust(mov)
# mov=signal.medfilt(mov-bl,window_lp)
###
###
# if mov.ndim == 3:
###                    eye_traces.append(np.mean(mov, axis=(1,2)))
# else:
# eye_traces.append(mov)
###
# mtot.append(mov)
# trial_info.append(dt['trials'][idx_tr,:])
# cb.movie(mov,fr=1/np.mean(np.diff(new_ts)))
##
# %%
# %%
# sub_trig_img=downsample_triggers(triggers_img.copy(),fraction_downsample=.3)
# %%
# if num_frames_movie != triggers[-1,-1]:
##        raise Exception('Triggers values do not match!')
##
# %%
# fnames=[]
# sub_trig_names=trigger_names[39:95].copy()
# sub_trig=triggers[39:95].copy().T
# for a,b in zip(sub_trig_names,sub_trig):
# fnames.append(a+'.hdf5')
###
# fraction_downsample=.333333333333333333333; # useful to downsample the movie across time. fraction_downsample=.1 measn downsampling by a factor of 10
# sub_trig[:2]=np.round(sub_trig[:2]*fraction_downsample)
# sub_trig[-1]=np.floor(sub_trig[-1]*fraction_downsample)
# sub_trig[-1]=np.cumsum(sub_trig[-1])
# fname_new=cm.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample),remove_init=0,idx_xy=(slice(90,-10,None),slice(30,-120,None)))
# %%
# m=cb.load(fname_new,fr=30*fraction_downsample)
# T,d1,d2=np.shape(m)
# %%
# if T != sub_trig[-1,-1]:
###    raise Exception('Triggers values do not match!')
# %% how to take triggered aligned movie
# wvf=mmm.take(trg)
# %%
# newm=m.take(trg,axis=0)
# newm=newm.mean(axis=1)
# %%
# (newm-np.mean(newm,0)).play(backend='opencv',fr=3,gain=2.,magnification=1,do_loop=True)
# %%v
# Yr,d1,d2,T=cm.load_memmap(fname_new)
# d,T=np.shape(Yr)
# Y=np.reshape(Yr,(d1,d2,T),order='F') # 3D version of the movie
##
# %%
##
# pl.plot(np.nanmedian(np.array(eye_traces).T,1))
##
# %%
##mov = np.concatenate(mtot,axis=0)
# m1=cb.movie(mov,fr=1/np.mean(np.diff(new_ts)))
# x_max,y_max=np.max(np.nonzero(np.max(m,0)),1)
# x_min,y_min=np.min(np.nonzero(np.max(m,0)),1)
# m1=m[:,x_min:x_max,y_min:y_max]
# %% filters
##b, a = signal.butter(8, [.05, .5] ,'bandpass')
# pl.plot(np.mean(m1,(1,2))-80)
# pl.plot(signal.lfilter(b,a,np.mean(m1,(1,2))),linewidth=2)
# %%
# m1.play(backend='opencv',gain=1.,fr=f_rate,magnification=3)
# %% NMF
##comps, tim,_=cb.behavior.extract_components(np.maximum(0,m1-np.min(m1,0)),n_components=4,init='nndsvd',l1_ratio=1,alpha=0,max_iter=200,verbose=True)
# pl.plot(np.squeeze(np.array(tim)).T)
# %% ICA
##from sklearn.decomposition import FastICA
# fica=FastICA(n_components=3,whiten=True,max_iter=200,tol=1e-6)
# X=fica.fit_transform(np.reshape(m1,(m1.shape[0],m1.shape[1]*m1.shape[2]),order='F').T,)
# pl.plot(X)
# %%
# for count,c in enumerate(comps):
# pl.subplot(2,3,count+1)
# pl.imshow(c)
##
# %%
# md=cm.mode_robust(m1,0)
# mm1=m1*(m1<md)
# rob_std=np.sum(mm1**2,0)/np.sum(mm1>0,0)
# rob_std[np.isnan(rob_std)]=0
# mm2=m1*(m1>(md+rob_std))
# %%
##
##dt = h5py.File('20160423165229_data.h5')
# sync for software
# np.array(dt['sync'])
# dt['sync'].attrs['keys']
# dt['trials']
# dt['trials'].attrs
# dt['trials'].attrs['keys']
# you needs to apply here the sync on dt['sync'], like,
##us_time_cam1=np.asarray(dt['trials'])[:,3] - np.array(dt['sync'])[1]
# main is used as the true time stamp, and you can adjust the value with respect to main sync value
# np.array(dt['sync']) # these are the values read on a unique clock from the three threads
# %%
##from skimage.external import tifffile
##
# tf=tifffile.TiffFile('20160423165229_00001_00001.tif')
# imd=tf.pages[0].tags['image_description'].value
# for pag in tf.pages:
# imd=pag.tags['image_description'].value
# i2cd=si_parse(imd)['I2CData']
##    print (i2cd)
# %%
# with h5py.File('20160705103903_cam2.h5') as f1:
# for k in f1.keys()[:1]:
###        m = np.array(f1[k]['mov'])
###
###
# pl.imshow(np.mean(m,0),cmap='gray')
# %%
# with h5py.File('20160705103903_data.h5') as f1:
# print f1.keys()
###    rois= np.array(f1['roi'])
# %%
# with h5py.File('20160705103903_cam2.h5') as f1:
# for k in f1.keys()[:1]:
###        m = np.array(f1[k]['mov'])
###
###
# pl.imshow(np.mean(m,0),cmap='gray')
# pl.imshow(rois[0],alpha=.3)
# pl.imshow(rois[1],alpha=.3)
###
