# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
#%%
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')


try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import caiman as cm
import numpy as np
import time
import pylab as pl
import psutil
import sys
import os
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
# , motion_correction_piecewise
from caiman.motion_correction import tile_and_correct
#%%
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob

#%% in parallel


def tile_and_correct_wrapper(params):

    from skimage.external.tifffile import imread
    import numpy as np
    import cv2
    try:
        cv2.setNumThreads(1)
    except:
        1  # 'Open CV is naturally single threaded'

    from caiman.motion_correction import tile_and_correct

    img_name,  out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides = params

    imgs = imread(img_name, key=idxs)
    mc = np.zeros(imgs.shape, dtype=np.float32)
    shift_info = []
    for count, img in enumerate(imgs):
        if count % 10 == 0:
            print(count)
        mc[count], total_shift, start_step, xy_grid = tile_and_correct(img, template, strides, overlaps, max_shifts, add_to_movie=add_to_movie, newoverlaps=newoverlaps, newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid, upsample_factor_fft=10, show_movie=False, max_deviation_rigid=max_deviation_rigid)
        shift_info.append([total_shift, start_step, xy_grid])
    if out_fname is not None:
        outv = np.memmap(out_fname, mode='r+', dtype=np.float32,
                         shape=shape_mov, order='F')
        outv[:, idxs] = np.reshape(
            mc.astype(np.float32), (len(imgs), -1), order='F').T

    return shift_info, idxs, np.nanmean(mc, 0)


#%%
def motion_correction_piecewise(fname, splits, strides, overlaps, add_to_movie=0, template=None, max_shifts=(12, 12), max_deviation_rigid=3, newoverlaps=None, newstrides=None,
                                upsample_factor_grid=4, order='F', dview=None, save_movie=True, base_name='none'):
    '''

    '''

    with TiffFile(fname) as tf:
        d1, d2 = tf[0].shape
        T = len(tf)

    if type(splits) is int:
        idxs = np.array_split(list(range(T)), splits)
    else:
        idxs = splits
        save_movie = False

    if template is None:
        raise Exception('Not implemented')

    shape_mov = (d1 * d2, T)

    dims = d1, d2

    if save_movie:
        if base_name is None:
            base_name = fname[:-4]

        fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(
            1 if len(dims) == 2 else dims[2]) + '_order_' + str(order) + '_frames_' + str(T) + '_.mmap'
        fname_tot = os.path.join(os.path.split(fname)[0], fname_tot)

        np.memmap(fname_tot, mode='w+', dtype=np.float32,
                  shape=shape_mov, order=order)
    else:
        fname_tot = None

    pars = []

    for idx in idxs:
        pars.append([fname, fname_tot, idx, shape_mov, template, strides, overlaps, max_shifts, np.array(
            add_to_movie, dtype=np.float32), max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides])

    t1 = time.time()
    if dview is not None:
        res = dview.map_sync(tile_and_correct_wrapper, pars)
    else:
        res = list(map(tile_and_correct_wrapper, pars))

    print((time.time() - t1))

    return fname_tot, res


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
#%% set parameters and create template by rigid motion correction
t1 = time.time()
fname = '20_12__002_cr.tif'
max_shifts = (12, 12)
splits = 28  # for parallelization split the movies in  num_sqplits chuncks across time
m = cm.load(fname, subindices=slice(None, None, None))
#%%
m.play(gain=1, magnification=1, fr=60)
#%% initiali template
template = cm.motion_correction.bin_median(m[100:500].copy().motion_correct(
    max_shifts[0], max_shifts[1], template=None)[0])
pl.imshow(template)
#%%
new_templ = template
add_to_movie = -np.min(template)
save_movie = False
num_iter = 2
for iter_ in range(num_iter):
    print(iter_)
    old_templ = new_templ.copy()
    if iter_ == num_iter - 1:
        save_movie = True
        print('saving!')
#        templ_to_save = old_templ

    fname_tot, res = motion_correction_piecewise(fname, splits, None, None,
                                                 add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts, max_deviation_rigid=0,
                                                 newoverlaps=None, newstrides=None,
                                                 upsample_factor_grid=4, order='F', dview=dview, save_movie=save_movie, base_name=fname[:-4] + '_rig_')

    new_templ = np.nanmedian(np.dstack([r[-1] for r in res]), -1)
    print((old_div(np.linalg.norm(new_templ - old_templ), np.linalg.norm(old_templ))))

t2 = time.time() - t1
print(t2)
pl.imshow(new_templ, cmap='gray', vmax=np.percentile(new_templ, 95))
#%%
import scipy
np.save(fname[:-4] + '_templ_rigid.npy', new_templ)
#scipy.io.savemat('/mnt/xfs1/home/agiovann/dropbox/Python_progress/' + str(np.shape(m)[-1])+'_templ_rigid.mat',{'template':new_templ})
#%%
template = new_templ
#%%
mr = cm.load(fname_tot)
#%%
fnames = [fname_tot]
#%%
idx_x = slice(12, 500, None)
idx_y = slice(12, 500, None)
idx_xy = (idx_x, idx_y)
add_to_movie = -np.nanmin(mr)  # the movie must be positive!!!
downsample_factor = .2  # use .2 or .1 if file is large and you want a quick answer
# idx_xy=slice*None
base_name = 'Yr'
name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy, add_to_movie=add_to_movie)
name_new.sort()
print(name_new)
#%%
fname_new = cm.save_memmap_join(
    name_new, base_name='Yr', n_chunks=12, dview=dview)
#%%
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
#%%
if np.min(images) < 0:
    raise Exception('Movie too negative, add_to_movie should be larger')
#%%
Cn = cm.local_correlations(Y[:, :, :3000])
pl.imshow(Cn, cmap='gray')

#%%
# THIS METHOd CAN GIVE POSSIBLY INCONSISTENT RESULTS ON SOMAS WHEN NOT USED WITH PATCHES
init_method = 'sparse_nmf'
alpha_snmf = 10e3  # this controls sparsity!

#%%
rf = 30  # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 10  # amounpl.it of overlap between the patches in pixels
K = 10  # number of neurons expected per patch
gSig = [6, 6]  # expected half size of neurons
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 1  # order of the autoregressive system
memory_fact = 1  # unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
save_results = False
#%% RUN ALGORITHM ON PATCHES

cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=memory_fact,
                method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1, method_deconvolution='oasis')
cnm = cnm.fit(images)

A_tot = cnm.A
C_tot = cnm.C
YrA_tot = cnm.YrA
b_tot = cnm.b
f_tot = cnm.f
sn_tot = cnm.sn

print(('Number of components:' + str(A_tot.shape[-1])))
#%%
pl.figure()
crd = plot_contours(A_tot, Cn, thr=0.9)
#%%
final_frate = 16  # approx final rate  (after eventual downsampling )
Npeaks = 10
traces = C_tot + YrA_tot
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = evaluate_components(
    Y, traces, A_tot, C_tot, b_tot, f_tot, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

idx_components_r = np.where(r_values >= .5)[0]
idx_components_raw = np.where(fitness_raw < -40)[0]
idx_components_delta = np.where(fitness_delta < -20)[0]

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
save_results = True
if save_results:
    np.savez('results_analysis_patch.npz', A_tot=A_tot, C_tot=C_tot,
             YrA_tot=YrA_tot, sn_tot=sn_tot, d1=d1, d2=d2, b_tot=b_tot, f=f_tot)

#%%
cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
                f_in=f_tot, rf=None, stride=None)
cnm = cnm.fit(images)

#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%%
final_frate = 6

Npeaks = 10
traces = C + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                        N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

idx_components_r = np.where(r_values >= .95)[0]
idx_components_raw = np.where(fitness_raw < -100)[0]
idx_components_delta = np.where(fitness_delta < -100)[0]


#min_radius = gSig[0] - 2
# masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
#    A.tocsc(), min_radius, dims, num_std_threshold=1,
#    minCircularity=0.7, minInertiaRatio=0.2, minConvexity=.5)

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
#idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

print(' ***** ')
print((len(traces)))
print((len(idx_components)))
# print((len(idx_blobs)))
#%%
save_results = True
if save_results:
    np.savez(os.path.join(os.path.split(fname_new)[0], 'results_analysis.npz'), Cn=Cn, A=A.todense(
    ), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components, idx_components_bad=idx_components_bad)

#%% visualize components
# pl.figure();
pl.subplot(1, 2, 1)
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
#pl.subplot(1, 3, 2)
#crd = plot_contours(A.tocsc()[:, idx_blobs], Cn, thr=0.9)
pl.subplot(1, 2, 2)
crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
#%%

view_patches_bar(Yr, A.tocsc()[:, idx_components], C[
    idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[
    idx_components_bad, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
#%% STOP CLUSTER and clean up log files


cm.stop_server()

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
#
