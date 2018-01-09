#!/usr/bin/env python
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
    print('Not launched under iPython')

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
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, shifts_opencv = params

    imgs = imread(img_name, key=idxs)
    mc = np.zeros(imgs.shape, dtype=np.float32)
    shift_info = []
    for count, img in enumerate(imgs):
        if count % 10 == 0:
            print(count)
        mc[count], total_shift, start_step, xy_grid = tile_and_correct(img, template, strides, overlaps, max_shifts, add_to_movie=add_to_movie, newoverlaps=newoverlaps, newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid, upsample_factor_fft=10, show_movie=False, max_deviation_rigid=max_deviation_rigid, shifts_opencv=shifts_opencv)
        shift_info.append([total_shift, start_step, xy_grid])
    if out_fname is not None:
        outv = np.memmap(out_fname, mode='r+', dtype=np.float32,
                         shape=shape_mov, order='F')
        outv[:, idxs] = np.reshape(
            mc.astype(np.float32), (len(imgs), -1), order='F').T

    return shift_info, idxs, np.nanmean(mc, 0)


#%%
def motion_correction_piecewise(fname, splits, strides, overlaps, add_to_movie=0, template=None, max_shifts=(12, 12), max_deviation_rigid=3, newoverlaps=None, newstrides=None,
                                upsample_factor_grid=4, order='F', dview=None, save_movie=True, base_name='none', num_splits=None, shifts_opencv=False):
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
    if num_splits is not None:
        idxs = np.array(idxs)[np.random.randint(0, len(idxs), num_splits)]
        save_movie = False
        print('**** MOVIE NOT SAVED BECAUSE num_splits is not None ****')

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
            add_to_movie, dtype=np.float32), max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, shifts_opencv])

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
#fname = 'k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif'
#fname = 'Sue_1000.tif'
fname = 'Sue_2000.tif'
max_shifts = (12, 12)
# splits = 56 # for parallelization split the movies in  num_splits chuncks across time
#num_splits_to_process = 28
#fname = 'M_FLUO_t_1000.tif'
#max_shifts = (10,10)
splits = 56  # for parallelization split the movies in  num_splits chuncks across time
num_splits_to_process = 28
#fname = 'M_FLUO_4.tif'
m = cm.load(fname, subindices=slice(0, 500, None))
template = cm.motion_correction.bin_median(m[100:400].copy().motion_correct(
    max_shifts[0], max_shifts[1], template=None)[0])
print(time.time() - t1)
#%
# pl.imshow(template)
#%
shifts_opencv = False
new_templ = template
add_to_movie = -np.min(template)
save_movie = False
num_iter = 1
for iter_ in range(num_iter):
    print(iter_)
    old_templ = new_templ.copy()
    if iter_ == num_iter - 1:
        save_movie = True
        print('saving!')
        num_splits_to_process = None
#        templ_to_save = old_templ

    fname_tot, res = motion_correction_piecewise(fname, splits, None, None,
                                                 add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts, max_deviation_rigid=0,
                                                 newoverlaps=None, newstrides=None,
                                                 upsample_factor_grid=4, order='F', dview=dview, save_movie=save_movie, base_name=fname[:-4] + '_rig_', num_splits=num_splits_to_process, shifts_opencv=shifts_opencv)

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
#%% online does not seem to work!
#overlaps = (16,16)
# if template.shape == (512,512):
#    strides = (128,128)# 512 512
#    #strides = (48,48)# 128 64
# elif template.shape == (64,128):
#    strides = (48,48)# 512 512
# else:
#    raise Exception('Unknown size, set manually')
#upsample_factor_grid = 4
#
#T = m.shape[0]
#idxs_outer = np.array_split(range(T),T/1000)
# for iddx in idxs_outer:
#    num_fr = len(iddx)
#    splits = np.array_split(iddx,num_fr/n_processes)
#    print (splits[0][0]),(splits[-1][-1])
#    fname_tot, res = motion_correction_piecewise(fname,splits, strides, overlaps,\
#                            add_to_movie=add_to_movie, template = template, max_shifts = (12,12),max_deviation_rigid = 3,\
#                            upsample_factor_grid = upsample_factor_grid,dview = dview)
#%%
# for 512 512 this seems good
t1 = time.time()

if template.shape == (512, 512):
    strides = (128, 128)  # 512 512
    overlaps = (32, 32)

#    strides = (16,16)# 512 512
    newoverlaps = None
    newstrides = None
    # strides = (48,48)# 128 64
elif template.shape == (64, 128):
    strides = (32, 32)
    overlaps = (16, 16)

    newoverlaps = None
    newstrides = None
else:
    raise Exception('Unknown size, set manually')
splits = 56
num_splits_to_process = 28
upsample_factor_grid = 4
max_deviation_rigid = 3
new_templ = template
add_to_movie = -np.min(m)
num_iter = 2
save_movie = False

for iter_ in range(num_iter):
    print(iter_)
    old_templ = new_templ.copy()

    if iter_ == num_iter - 1:
        save_movie = True
        num_splits_to_process = None
        print('saving!')

    fname_tot, res = motion_correction_piecewise(fname, splits, strides, overlaps,
                                                 add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts, max_deviation_rigid=max_deviation_rigid,
                                                 newoverlaps=newoverlaps, newstrides=newstrides,
                                                 upsample_factor_grid=upsample_factor_grid, order='F', dview=dview, save_movie=save_movie, base_name=fname[:-4] + '_els_opencv_', num_splits=num_splits_to_process, shifts_opencv=shifts_opencv)

    new_templ = np.nanmedian(np.dstack([r[-1] for r in res]), -1)
#    print((old_div(np.linalg.norm(new_templ-old_templ),np.linalg.norm(old_templ))))
#    pl.imshow(new_templ,cmap = 'gray',vmax = np.percentile(new_templ,99))
#    pl.pause(.1)
t2 = time.time() - t1
print(t2)

mc = cm.load(fname_tot)
#%%
pl.imshow(new_templ, cmap='gray', vmax=np.percentile(new_templ, 95))
#%%
np.save(fname[:-4] + '_templ_pw_rigid.npy', new_templ)
#scipy.io.savemat('/mnt/xfs1/home/agiovann/dropbox/Python_progress/' + str(np.shape(m)[-1])+'_templ_pw_rigid.mat',{'template':templ_to_save})
#%%
#%%


def compute_metrics_motion_correction(fname, final_size_x, final_size_y, swap_dim, pyr_scale=.5, levels=3, winsize=100, iterations=15, poly_n=5, poly_sigma=1.2 / 5, flags=0,
                                      play_flow=False, resize_fact_flow=.2, template=None):

    # cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    import scipy
    vmin, vmax = -1, 1
    m = cm.load(fname)

    max_shft_x = np.int(np.ceil((np.shape(m)[1] - final_size_x) / 2))
    max_shft_y = np.int(np.ceil((np.shape(m)[2] - final_size_y) / 2))
    max_shft_x_1 = - ((np.shape(m)[1] - max_shft_x) - (final_size_x))
    max_shft_y_1 = - ((np.shape(m)[2] - max_shft_y) - (final_size_y))
    if max_shft_x_1 == 0:
        max_shft_x_1 = None

    if max_shft_y_1 == 0:
        max_shft_y_1 = None
#    print ([max_shft_x,max_shft_x_1,max_shft_y,max_shft_y_1])
    m = m[:, max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]
    print('Local correlations..')
    img_corr = m.local_correlations(eight_neighbours=True, swap_dim=swap_dim)
    print(m.shape)
    if template is None:
        tmpl = cm.motion_correction.bin_median(m)
    else:
        tmpl = template
#    tmpl = tmpl[max_shft_x:-max_shft_x,max_shft_y:-max_shft_y]

    print('Compute Smoothness.. ')
    smoothness = np.sqrt(
        np.sum(np.sum(np.array(np.gradient(np.mean(m, 0)))**2, 0)))
    smoothness_corr = np.sqrt(
        np.sum(np.sum(np.array(np.gradient(img_corr))**2, 0)))

    print('Compute correlations.. ')
    correlations = []
    count = 0
    for fr in m:
        if count % 100 == 0:
            print(count)

        count += 1
        correlations.append(scipy.stats.pearsonr(
            fr.flatten(), tmpl.flatten())[0])

    print('Compute optical flow .. ')

    m = m.resize(1, 1, resize_fact_flow)
    norms = []
    flows = []
    count = 0
    for fr in m:

        if count % 100 == 0:
            print(count)

        count += 1
        flow = cv2.calcOpticalFlowFarneback(
            tmpl, fr, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

        if play_flow:
            pl.subplot(1, 3, 1)
            pl.cla()
            pl.imshow(fr, vmin=0, vmax=300, cmap='gray')
            pl.title('movie')

            pl.subplot(1, 3, 3)
            pl.cla()
            pl.imshow(flow[:, :, 1], vmin=vmin, vmax=vmax)
            pl.title('y_flow')

            pl.subplot(1, 3, 2)
            pl.cla()
            pl.imshow(flow[:, :, 0], vmin=vmin, vmax=vmax)
            pl.title('x_flow')
            pl.pause(.05)

        n = np.linalg.norm(flow)
        flows.append(flow)
        norms.append(n)

    np.savez(fname[:-4] + '_metrics', flows=flows, norms=norms, correlations=correlations,
             smoothness=smoothness, tmpl=tmpl, smoothness_corr=smoothness_corr, img_corr=img_corr)
    return tmpl, correlations, flows, norms, smoothness


#%% run comparisons MLK
m_res = glob.glob('MKL*hdf5')
final_size = (512 - 24, 512 - 24)
winsize = 100
swap_dim = False
resize_fact_flow = .2
for mv in m_res:
    tmpl, correlations, flows_orig, norms, smoothness = compute_metrics_motion_correction(
        mv, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)


#%% run comparisons NORMCORRE
m_fluos = glob.glob('M_FLUO*.mmap') + glob.glob('M_FLUO*.tif')
final_size = (64 - 20, 128 - 20)
winsize = 32
resize_fact_flow = 1
for mv in m_fluos:
    tmpl, correlations, flows_orig, norms, smoothness = compute_metrics_motion_correction(
        mv, final_size[0], final_size[1], winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
#% run comparisons resonant
m_res = glob.glob('Sue*mmap') + glob.glob('Sue*.tif')
final_size = (512 - 24, 512 - 24)
winsize = 100
swap_dim = False
resize_fact_flow = .2
for mv in m_res:
    tmpl, correlations, flows_orig, norms, smoothness = compute_metrics_motion_correction(
        mv, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

#%% run comparisons SIMA
m_fluos = glob.glob('plane*.tif') + glob.glob('row*.tif')
final_size = (64 - 20, 128 - 20)
winsize = 32
resize_fact_flow = 1
for mv in m_fluos:
    tmpl, correlations, flows_orig, norms, smoothness = compute_metrics_motion_correction(
        mv, final_size[0], final_size[1], winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
#% run comparisons resonant
m_res = glob.glob('Sue*.tif')
final_size = (512 - 24, 512 - 24)
winsize = 100
resize_fact_flow = .2
for mv in m_res:
    tmpl, correlations, flows_orig, norms, smoothness = compute_metrics_motion_correction(
        mv, final_size[0], final_size[1], winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
#%% run comparisons SUITE2P
for mvs in glob.glob('Sue*2000*16*.mat'):
    print(mvs)
    cm.movie(scipy.io.loadmat(mvs)['data'].transpose(
        [2, 0, 1])).save(mvs[:-3] + '.hdf5')
#%%
m_fluos = glob.glob('M_FLUO*.hdf5')
final_size = (64 - 20, 128 - 20)
winsize = 32
resize_fact_flow = 1
for mv in m_fluos:
    tmpl, correlations, flows_orig, norms, smoothness = compute_metrics_motion_correction(
        mv, final_size[0], final_size[1], winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
#% run comparisons resonant
m_res = glob.glob('Sue_2000*16*.hdf5')
final_size = (512 - 24, 512 - 24)
winsize = 100
resize_fact_flow = .2
for mv in m_res:
    tmpl, correlations, flows_orig, norms, smoothness = compute_metrics_motion_correction(
        mv, final_size[0], final_size[1], winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
#%% plot the results
files_img = [u'/mnt/xfs1/home/agiovann/DataForPublications/Piecewise-Rigid-Analysis-paper/NORM_CORRE_OPENCV/Sue_2000_els_opencv__d1_512_d2_512_d3_1_order_F_frames_2000_._metrics.npz',
             u'/mnt/xfs1/home/agiovann/DataForPublications/Piecewise-Rigid-Analysis-paper/NORMCORRE_EFF/Sue_2000_els__d1_512_d2_512_d3_1_order_F_frames_2000_._metrics.npz',
             #             u'/mnt/xfs1/home/agiovann/DataForPublications/Piecewise-Rigid-Analysis-paper/MLK/Sue_2000_MLK_metrics.npz',
             #             u'/mnt/xfs1/home/agiovann/DataForPublications/Piecewise-Rigid-Analysis-paper/SIMA_RESULTS/Sue_1000_T.tifrow1_example_sima_Trow1_example_sima_metrics.npz',
             #             u'/mnt/xfs1/home/agiovann/DataForPublications/Piecewise-Rigid-Analysis-paper/SUITE_2P_RES/Sue_2000_t_NB_16.._metrics.npz',
             u'/mnt/xfs1/home/agiovann/DataForPublications/Piecewise-Rigid-Analysis-paper/MLK/MKL16T._metrics.npz']
# for fl in glob.glob('*.npz'):
for fl in files_img:
    with np.load(fl) as ld:
        print(ld.keys())
        pl.figure()
        print(fl + ':' + str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) + ' ; ' + str(np.mean(ld['correlations'])
                                                                                                    ) + '+/-' + str(np.std(ld['correlations'])) + ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))
        pl.subplot(1, 2, 1)
        try:
            mean_img = np.mean(cm.load(fl[:-12] + 'mmap'), 0)[12:-12, 12:-12]
        except:
            try:
                mean_img = np.mean(
                    cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
            except:
                mean_img = np.mean(
                    cm.load(fl[:-12] + 'hdf5'), 0)[12:-12, 12:-12]
#        lq,hq = np.nanpercentile(mean_img,[.1,99.9])
        lq, hq = 13.3, 318.01
        pl.imshow(mean_img, vmin=lq, vmax=hq)
        pl.colorbar()
#        pl.plot(ld['correlations'])

        pl.subplot(1, 2, 2)
        pl.imshow(ld['img_corr'], vmin=0, vmax=.5)
        pl.colorbar()
#%%
for fl in glob.glob('Mf*.npz'):
    with np.load(fl) as ld:
        print(ld.keys())
        pl.figure()
        print(fl + ':' + str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) + ' ; ' + str(np.mean(ld['correlations'])
                                                                                                    ) + '+/-' + str(np.std(ld['correlations'])) + ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))


#%%
#%
#total_shifts = []
#start_steps = []
#xy_grids = []
#mc = np.zeros(m.shape)
# for count,img in enumerate(np.array(m)):
#    if count % 10  == 0:
#        print(count)
#    mc[count],total_shift,start_step,xy_grid = tile_and_correct(img, template, strides, overlaps,(12,12), newoverlaps = None, \
#                newstrides = newstrides, upsample_factor_grid=upsample_factor_grid,\
#                upsample_factor_fft=10,show_movie=False,max_deviation_rigid=2,add_to_movie=add_to_movie)
#
#    total_shifts.append(total_shift)
#    start_steps.append(start_step)
#    xy_grids.append(xy_grid)


#mc = cm.load('M_FLUO_4_d1_64_d2_128_d3_1_order_F_frames_4620_.mmap')
#mc = cm.load('M_FLUO_t_d1_64_d2_128_d3_1_order_F_frames_6764_.mmap')

#%%
mc.resize(1, 1, .1).play(gain=10., fr=30, offset=100, magnification=1.)
#%%
m.resize(1, 1, .2).play(gain=10, fr=30, offset=0, magnification=1.)
#%%
cm.concatenate([mr.resize(1, 1, .5), mc.resize(1, 1, .5)], axis=1).play(
    gain=10, fr=100, offset=300, magnification=1.)

#%%
import h5py
with h5py.File('sueann_pw_rigid_movie.mat') as f:
    mef = np.array(f['M2'])

mef = cm.movie(mef.transpose([0, 2, 1]))

#%%
cm.concatenate([mef.resize(1, 1, .15), mc.resize(1, 1, .15)], axis=1).play(
    gain=30, fr=40, offset=300, magnification=1.)
#%%
(mef - mc).resize(1, 1, .1).play(gain=50, fr=20, offset=0, magnification=1.)
#%%
(mc - mef).resize(1, 1, .1).play(gain=50, fr=20, offset=0, magnification=1.)
#%%
T, d1, d2 = np.shape(m)
shape_mov = (d1 * d2, m.shape[0])

Y = np.memmap('M_FLUO_4_d1_64_d2_128_d3_1_order_F_frames_4620_.mmap',
              mode='r', dtype=np.float32, shape=shape_mov, order='F')
mc = cm.movie(np.reshape(Y, (d2, d1, T), order='F').transpose([2, 1, 0]))
mc.resize(1, 1, .25).play(gain=10., fr=50)
#%%
total_shifts = [r[0][0][0] for r in res]


pl.plot(np.reshape(np.array(total_shifts), (len(total_shifts), -1)))
#%%
#m_raw = cm.motion_correction.bin_median(m,exclude_nans=True)
#m_rig = cm.motion_correction.bin_median(mr,exclude_nans=True)
#m_el = cm.motion_correction.bin_median(mc,exclude_nans=True)

m_raw = np.nanmean(m, 0)
m_rig = np.nanmean(mr, 0)
m_el = np.nanmean(mc, 0)
m_ef = np.nanmean(mef, 0)
#%%
import scipy
r_raw = []
r_rig = []
r_el = []
r_ef = []
max_shft_x, max_shft_y = max_shifts
for fr_id in range(m.shape[0]):
    fr = m[fr_id].copy()[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y]
    templ_ = m_raw.copy()[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y]
    r_raw.append(scipy.stats.pearsonr(fr.flatten(), templ_.flatten())[0])

    fr = mr[fr_id].copy()[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y]
    templ_ = m_rig.copy()[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y]
    r_rig.append(scipy.stats.pearsonr(fr.flatten(), templ_.flatten())[0])

    fr = mc[fr_id].copy()[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y]
    templ_ = m_el.copy()[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y]
    r_el.append(scipy.stats.pearsonr(fr.flatten(), templ_.flatten())[0])
    if 1:
        fr = mef[fr_id].copy()[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y]
        templ_ = m_ef.copy()[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y]
        r_ef.append(scipy.stats.pearsonr(fr.flatten(), templ_.flatten())[0])

r_raw = np.array(r_raw)
r_rig = np.array(r_rig)
r_el = np.array(r_el)
r_ef = np.array(r_ef)
#%%
#r_ef = scipy.io.loadmat('sueann.mat')['cM2'].squeeze()
#r_efr = scipy.io.loadmat('sueann.mat')['cY'].squeeze()
# pl.close()
#%%
pl.plot(r_raw)
pl.plot(r_rig)
pl.plot(r_el)
# pl.plot(r_ef)
#%%
pl.scatter(r_el, r_ef)
pl.plot([0, 1], [0, 1], 'r--')

#%%
pl.plot(old_div((r_ef - r_el), np.abs(r_el)))
#%%
import pylab as pl
vmax = -100
max_shft = 3

#%


pl.subplot(3, 3, 1)
pl.imshow(np.nanmean(m, 0)[max_shft:-max_shft, max_shft:-
                           max_shft], cmap='gray', vmax=vmax, interpolation='none')
pl.title('raw')
pl.axis('off')
pl.xlim([0, 100])
pl.ylim([220, 320])
pl.axis('off')


pl.subplot(3, 3, 2)
pl.title('rigid mean')
pl.imshow(np.nanmean(mr, 0)[max_shft:-max_shft, max_shft:-
                            max_shft], cmap='gray', vmax=vmax, interpolation='none')
pl.xlim([0, 100])
pl.ylim([220, 320])
pl.axis('off')

pl.subplot(3, 3, 3)
pl.imshow(np.nanmean(mc, 0)[max_shft:-max_shft, max_shft:-
                            max_shft], cmap='gray', vmax=vmax, interpolation='none')
pl.title('pw-rigid mean')
pl.axis('off')
pl.xlim([0, 100])
pl.ylim([220, 320])
pl.axis('off')

pl.subplot(3, 3, 5)
pl.scatter(r_raw, r_rig)
pl.plot([0, 1], [0, 1], 'r--')
pl.xlabel('raw')
pl.ylabel('rigid')
pl.xlim([0, 1])
pl.ylim([0, 1])
pl.subplot(3, 3, 6)
pl.scatter(r_rig, r_el)
pl.plot([0, 1], [0, 1], 'r--')
pl.ylabel('pw-rigid')
pl.xlabel('rigid')
pl.xlim([0, 1])
pl.ylim([0, 1])

if 0:
    pl.subplot(2, 3, 3)
    pl.scatter(r_el, r_ef)
    pl.plot([0, 1], [0, 1], 'r--')
    pl.ylabel('pw-rigid')
    pl.xlabel('pw-rigid eft')
    pl.xlim([0, 1])
    pl.ylim([0, 1])

    pl.subplot(2, 3, 6)
    pl.imshow(np.nanmean(mef, 0)[max_shft:-max_shft, max_shft:-
                                 max_shft], cmap='gray', vmax=vmax, interpolation='none')
    pl.title('pw-rigid eft mean')
    pl.axis('off')

#%%
pl.plot(r_ef)
#%%
mc = cm.movie(mc)
mc[np.isnan(mc)] = 0
#%% play movie
(mc + add_to_movie).resize(1, 1, .25).play(gain=10., fr=50)
#%% compute correlation images
ccimage = m.local_correlations(eight_neighbours=True, swap_dim=False)
ccimage_rig = mr.local_correlations(eight_neighbours=True, swap_dim=False)
ccimage_els = mc.local_correlations(eight_neighbours=True, swap_dim=False)
ccimage_ef = mef.local_correlations(eight_neighbours=True, swap_dim=False)
#%% check correlation images
pl.subplot(2, 2, 1)
pl.imshow(ccimage, vmin=0, vmax=0.4, interpolation='none')
pl.subplot(2, 2, 2)
pl.imshow(ccimage_rig, vmin=0, vmax=0.4, interpolation='none')
pl.subplot(2, 2, 3)
pl.imshow(ccimage_els, vmin=0, vmax=0.4, interpolation='none')
pl.subplot(2, 2, 4)
pl.imshow(ccimage_ef, vmin=0, vmax=0.4, interpolation='none')
#%%
all_mags = []
all_mags_eig = []
for chunk in res:
    for frame in chunk[0]:
        shifts, pos, init = frame
        x_sh = np.zeros(np.add(init[-1], 1))
        y_sh = np.zeros(np.add(init[-1], 1))

        for nt, sh in zip(init, shifts):
            x_sh[nt] = sh[0]
            y_sh[nt] = sh[1]

        jac_xx = x_sh[1:, :] - x_sh[:-1, :]
        jac_yx = y_sh[1:, :] - y_sh[:-1, :]
        jac_xy = x_sh[:, 1:] - x_sh[:, :-1]
        jac_yy = y_sh[:, 1:] - y_sh[:, :-1]

        mag_norm = np.sqrt(jac_xx[:, :-1]**2 + jac_yx[:, :-1]
                           ** 2 + jac_xy[:-1, :]**2 + jac_yy[:-1, :]**2)
        all_mags.append(mag_norm)


#        pl.cla()
#        pl.imshow(mag_norm,vmin=0,vmax =1,interpolation = 'none')
#        pl.pause(.1)
#%%
mam = cm.movie(np.dstack(all_mags)).transpose([2, 0, 1])
#mam.play(magnification=10,gain = 5.)
#%%
pl.imshow(np.max(mam, 0), interpolation='none')
#%%
m = cm.load('rig_sue__d1_512_d2_512_d3_1_order_F_frames_3000_.mmap')
m1 = cm.load('els_sue__d1_512_d2_512_d3_1_order_F_frames_3000_.mmap')
m0 = cm.load('k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif')
tmpl = cm.motion_correction.bin_median(m)
tmpl1 = cm.motion_correction.bin_median(m1)
tmpl0 = cm.motion_correction.bin_median(m0)

#%%
vmin, vmax = -1, 1
count = 0
pyr_scale = .5
levels = 3
winsize = 100
iterations = 15
poly_n = 5
poly_sigma = old_div(1.2, 5)
flags = 0  # cv2.OPTFLOW_FARNEBACK_GAUSSIAN
norms = []
flows = []
for fr, fr1, fr0 in zip(m.resize(1, 1, .2), m1.resize(1, 1, .2), m0.resize(1, 1, .2)):
    count += 1
    print(count)

    flow1 = cv2.calcOpticalFlowFarneback(tmpl1[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y], fr1[max_shft_x:-
                                                                                                    max_shft_x, max_shft_y:-max_shft_y], None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    flow = cv2.calcOpticalFlowFarneback(tmpl[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y], fr[max_shft_x:-
                                                                                                 max_shft_x, max_shft_y:-max_shft_y], None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    flow0 = cv2.calcOpticalFlowFarneback(tmpl0[max_shft_x:-max_shft_x, max_shft_y:-max_shft_y], fr0[max_shft_x:-
                                                                                                    max_shft_x, max_shft_y:-max_shft_y], None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
#
#    pl.subplot(2,3,1)
#    pl.cla()
#    pl.imshow(flow1[:,:,1],vmin=vmin,vmax=vmax)
#    pl.subplot(2,3,2)
#    pl.cla()
#    pl.imshow(flow[:,:,1],vmin=vmin,vmax=vmax)
#    pl.subplot(2,3,3)
#    pl.cla()
#    pl.imshow(flow0[:,:,1],vmin=vmin,vmax=vmax)
#
#    pl.subplot(2,3,4)
#    pl.cla()
#    pl.imshow(flow1[:,:,0],vmin=vmin,vmax=vmax)
#    pl.subplot(2,3,5)
#    pl.cla()
#    pl.imshow(flow[:,:,0],vmin=vmin,vmax=vmax)
#    pl.subplot(2,3,6)
#    pl.cla()
#    pl.imshow(flow0[:,:,0],vmin=vmin,vmax=vmax)
#    pl.pause(.1)
    n1, n, n0 = np.linalg.norm(flow1), np.linalg.norm(
        flow), np.linalg.norm(flow0)
    flows.append([flow1, flow, flow0])
    norms.append([n1, n, n0])
#%%
flm1_x = cm.movie(np.dstack([fl[0][:, :, 0]
                             for fl in flows])).transpose([2, 0, 1])
flm_x = cm.movie(np.dstack([fl[1][:, :, 0]
                            for fl in flows])).transpose([2, 0, 1])
flm0_x = cm.movie(np.dstack([fl[2][:, :, 0]
                             for fl in flows])).transpose([2, 0, 1])

flm1_y = cm.movie(np.dstack([fl[0][:, :, 1]
                             for fl in flows])).transpose([2, 0, 1])
flm_y = cm.movie(np.dstack([fl[1][:, :, 1]
                            for fl in flows])).transpose([2, 0, 1])
flm0_y = cm.movie(np.dstack([fl[2][:, :, 1]
                             for fl in flows])).transpose([2, 0, 1])

#%%
pl.figure()
pl.subplot(2, 1, 1)
pl.plot(norms)
pl.subplot(2, 1, 2)
pl.plot(np.arange(0, 3000 * .2, 0.2), r_el)
pl.plot(np.arange(0, 3000 * .2, 0.2), r_rig)
pl.plot(np.arange(0, 3000 * .2, 0.2), r_raw)
#%%
#%% compare to optical flow
pl.figure()
vmin = -.5
vmax = .5
cmap = 'hot'
pl.subplot(2, 3, 1)
pl.imshow(np.mean(np.abs(flm1_x), 0), vmin=vmin, vmax=vmax, cmap=cmap)
pl.title('PW-RIGID')
pl.ylabel('optical flow x')
pl.colorbar()
pl.subplot(2, 3, 2)
pl.title('RIGID')
pl.imshow(np.mean(np.abs(flm_x), 0), vmin=vmin, vmax=vmax, cmap=cmap)
pl.colorbar()
pl.subplot(2, 3, 3)
pl.imshow(np.mean(np.abs(flm0_x), 0), vmin=vmin * 4, vmax=vmax * 4, cmap=cmap)
pl.title('RAW')
pl.colorbar()
pl.subplot(2, 3, 4)
pl.imshow(np.mean(np.abs(flm1_y), 0), vmin=vmin, vmax=vmax, cmap=cmap)
pl.ylabel('optical flow y')
pl.colorbar()
pl.subplot(2, 3, 5)
pl.imshow(np.mean(np.abs(flm_y), 0), vmin=vmin, vmax=vmax, cmap=cmap)
pl.colorbar()
pl.subplot(2, 3, 6)
pl.imshow(np.mean(np.abs(flm0_y), 0), vmin=vmin * 4, vmax=vmax * 4, cmap=cmap)
pl.colorbar()
#%%
fl_rig = [n[1] / 1000 for n in norms]
fl_raw = [n[2] / 1000 for n in norms]
fl_el = [n[0] / 1000 for n in norms]
#%%

font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 15}

pl.rc('font', **font)

vmax = -100
max_shft = 3

pl.subplot(4, 3, 1)
pl.imshow(np.nanmean(m, 0)[max_shft:-max_shft, max_shft:-
                           max_shft], cmap='gray', vmax=vmax, interpolation='none')
pl.title('raw')
pl.axis('off')
pl.xlim([0, 100])
pl.ylim([220, 320])
pl.axis('off')


pl.subplot(4, 3, 2)
pl.title('rigid mean')
pl.imshow(np.nanmean(mr, 0)[max_shft:-max_shft, max_shft:-
                            max_shft], cmap='gray', vmax=vmax, interpolation='none')
pl.xlim([0, 100])
pl.ylim([220, 320])
pl.axis('off')

pl.subplot(4, 3, 3)
pl.imshow(np.nanmean(mc, 0)[max_shft:-max_shft, max_shft:-
                            max_shft], cmap='gray', vmax=vmax, interpolation='none')
pl.title('pw-rigid mean')
pl.axis('off')
pl.xlim([0, 100])
pl.ylim([220, 320])
pl.axis('off')

pl.subplot(4, 3, 5)
pl.scatter(r_raw, r_rig, s=50, c='red')
pl.axis('tight')
pl.plot([0, 1], [0, 1], 'k--')
pl.xlabel('raw')
pl.ylabel('rigid')
pl.xlim([0.2, .45])
pl.ylim([.2, .45])
pl.locator_params(nbins=4)

pl.subplot(4, 3, 6)
pl.scatter(r_rig, r_el, s=50, c='red')
pl.plot([0, 1], [0, 1], 'k--')
pl.ylabel('pw-rigid')
pl.xlabel('rigid')
pl.xlim([0.3, .45])
pl.ylim([.3, .45])
pl.locator_params(nbins=4)


pl.subplot(4, 3, 4)
pl.plot(np.arange(0, 3000 * .2, 0.2), r_el)
pl.plot(np.arange(0, 3000 * .2, 0.2), r_rig)
pl.plot(np.arange(0, 3000 * .2, 0.2), r_raw)
pl.xlim([220, 320])
pl.ylabel('correlation')
pl.locator_params(nbins=4)

pl.subplot(4, 3, 7)
pl.plot(norms)
pl.xlim([220, 320])
pl.ylabel('norm of optical flow')
pl.xlabel('frames')
pl.locator_params(nbins=4)

pl.subplot(4, 3, 8)
pl.scatter(fl_raw, fl_rig, s=50, c='red')
pl.axis('tight')
pl.plot([0, 3000], [0, 3000], 'k--')
pl.xlabel('raw')
pl.ylabel('rigid')
pl.xlim([0, 3])
pl.ylim([0, 3])
pl.locator_params(nbins=4)

pl.subplot(4, 3, 9)
pl.scatter(fl_rig, fl_el, s=50, c='red')
pl.plot([0, 1000], [0, 1000], 'k--')
pl.ylabel('pw-rigid')
pl.xlabel('rigid')
pl.xlim([0, 1])
pl.ylim([0, 1])
pl.locator_params(nbins=4)


ofl_mod_rig = np.mean(np.sqrt(flm_x**2 + flm_y**2), 0)
ofl_mod_el = np.mean(np.sqrt(flm1_x**2 + flm1_y**2), 0)


pl.subplot(4, 3, 10)
pl.imshow(ofl_mod_el, cmap='hot', vmin=0, vmax=1, interpolation='none')
pl.axis('off')
pl.colorbar()

pl.subplot(4, 3, 11)
pl.imshow(ofl_mod_rig, cmap='hot', vmin=0, vmax=1, interpolation='none')
pl.axis('off')
# pl.xlim([0,100])
# pl.ylim([220,320])
pl.axis('off')


pl.subplot(4, 3, 12)
pl.imshow(ofl_mod_el, cmap='hot', vmin=0, vmax=1, interpolation='none')
pl.axis('off')
# pl.xlim([0,100])
# pl.ylim([220,320])
pl.axis('off')

# font = {'family' : 'Myriad Pro',
#        'weight' : 'regular',
#        'size'   : 15}
#
#pl.rc('font', **font)
pl.rcParams['pdf.fonttype'] = 42
#%% test against SIMA
import sima
import sima.motion
from sima.motion import HiddenMarkov2D
#fname_gr = 'M_FLUO_t.tif'
#fname_gr = 'Sue_1000.tif'
#fname_gr = 'Sue_2000.tif'
fname_gr = 'Sue_1000_T.tif'
fname_gr = 'Sue_1000_T.tifrow1_example_sima_T.tif'
sequences = [sima.Sequence.create('TIFF', fname_gr)]
dataset = sima.ImagingDataset(sequences, fname_gr)
#%%
import time
t1 = time.time()
granularity = 'row'
gran_n = 1
mc_approach = sima.motion.HiddenMarkov2D(granularity=(
    granularity, gran_n), max_displacement=max_shifts, verbose=True, n_processes=14)
new_dataset = mc_approach.correct(dataset, None)
t2 = time.time() - t1
print(t2)
#%
new_dataset.export_frames(
    [[[fname_gr[:-4] + granularity + str(gran_n) + '_example_sima.tif']]], fmt='TIFF16')
#%%
m_s = cm.load(granularity + str(gran_n) + '_example_sima.tif')
m_s_row = cm.load('example_sima.tif')

#%%


def compute_jacobians(res):
    all_mags = []
    all_mags_eig = []
    for chunk in res:
        for frame in chunk[0]:
            shifts, pos, init = frame
            x_sh = np.zeros(np.add(init[-1], 1))
            y_sh = np.zeros(np.add(init[-1], 1))

            for nt, sh in zip(init, shifts):
                x_sh[nt] = sh[0]
                y_sh[nt] = sh[1]

            jac_xx = x_sh[1:, :] - x_sh[:-1, :]
            jac_yx = y_sh[1:, :] - y_sh[:-1, :]
            jac_xy = x_sh[:, 1:] - x_sh[:, :-1]
            jac_yy = y_sh[:, 1:] - y_sh[:, :-1]

            mag_norm = np.sqrt(
                jac_xx[:, :-1]**2 + jac_yx[:, :-1]**2 + jac_xy[:-1, :]**2 + jac_yy[:-1, :]**2)
            for a, b, c, d in zip(jac_xx, jac_xy, jac_yy, jac_yy):
                jc = np.array([[a, b], [c, d]])
                w, vl, vr = scipy.linalg.eig(jc)
                lsl

            all_mags_eig.append(mag_eig)
            all_mags.append(mag_norm)


# %%
#m = cm.load('M_FLUO_t_1000.tif')
#tmpl, correlations, flows_rig, norms = compute_metrics_motion_correction('M_FLUO_t_1000_rig__d1_64_d2_128_d3_1_order_F_frames_1000_.mmap',10,10,winsize=32, play_flow=False, resize_fact_flow=1)
#tmpl, correlations, flows_els, norms = compute_metrics_motion_correction('M_FLUO_t_1000_els__d1_64_d2_128_d3_1_order_F_frames_1000_.mmap',10,10,winsize=32, play_flow=False, resize_fact_flow=1)
#tmpl, correlations, flows_orig, norms = compute_metrics_motion_correction('M_FLUO_t_1000.tif',10,10,winsize=32, play_flow=False, resize_fact_flow=1)
#mfl_orig = cm.movie(np.concatenate([np.sqrt(np.sum(ff**2,-1))[np.newaxis,:,:] for ff in flows_orig],axis=0))
#mfl_rig = cm.movie(np.concatenate([np.sqrt(np.sum(ff**2,-1))[np.newaxis,:,:] for ff in flows_rig],axis=0))
#mfl_els = cm.movie(np.concatenate([np.sqrt(np.sum(ff**2,-1))[np.newaxis,:,:] for ff in flows_els],axis=0))
# %%
#cm.concatenate([mfl_orig/5.,mfl_rig,mfl_els],axis = 1).zproject(vmax = .5)
# %%
#cm.concatenate([m[:,10:-10,10:-10]/500,mfl_orig,mfl_rig,mfl_els],axis = 1).play(magnification = 5,gain = 5)
#%% TEST OPT FLOW
nmf = 'M_FLUO_t_shifted_flow.tif'
m = cm.load('M_FLUO_t_1000_els__d1_64_d2_128_d3_1_order_F_frames_1000_.mmap')
#shfts = [(a,b) for a,b in zip(np.random.randint(-2,3,m.shape[0]),np.random.randint(-2,3,m.shape[0]))]
shfts = [(a, b) for a, b in zip(np.random.randn(
    m.shape[0]), np.random.randn(m.shape[0]))]
msh = m.copy().apply_shifts(shfts)
msh[:, 10:-10, 10:-10].save(nmf)
template = np.nanmean(m[:, 10:-10, 10:-10], 0)
tmpl, correlations, flows_orig, norms, smoothness = compute_metrics_motion_correction(
    'M_FLUO_t_shifted_flow.tif', template.shape[0], template.shape[1], winsize=32, play_flow=False, resize_fact_flow=1, template=template)

with np.load('M_FLUO_t_shifted_flow_metrics.npz') as ld:
    flows = ld['flows']
    ff_1 = [np.nanmean(f[:, :, 1]) for f in flows]
    ff_0 = [np.nanmean(f[:, :, 0]) for f in flows]


pl.subplot(2, 1, 1)
pl.plot(np.array(shfts)[:, 1])
pl.plot(np.array(ff_0))
pl.legend(['shifts', 'optical flow'])
pl.xlim([400, 600])
pl.ylabel('x shifts')
pl.subplot(2, 1, 2)
pl.plot(np.array(shfts)[:, 0])
pl.plot(np.array(ff_1))
pl.xlim([400, 600])
pl.xlabel('frames (15 Hz)')
pl.ylabel('y shifts')
