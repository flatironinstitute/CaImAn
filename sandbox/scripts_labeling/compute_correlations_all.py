#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  2 16:27:17 2017

@author: agiovann
"""
import cv2
try:
    cv2.setNumThreads(1)
except:
    print('OpenCV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import caiman as cm
import numpy as np
import os
import glob
import pylab as pl

from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
#%%

fls_1 = ['neurofinder.00.00',
         'neurofinder.02.00',
         'neurofinder.03.00.test',
         'neurofinder.04.00',
         'neurofinder.04.00.test',
         'neurofinder.01.01',
         'k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16',
         'yuste.Single_150u',
         'packer.001',
         'J123_2015-11-20_L01_0',
         'Yi.data.001',
         'FN.151102_001',
         'Jan-AMG1_exp2_new_001',
         'Jan-AMG_exp3_001',
         'k31_20160107_MMP_150um_65mW_zoom2p2_00001_1-15',
         'k31_20160106_MMA_400um_118mW_zoom2p2_00001_1-19',
         'k36_20151229_MMA_200um_65mW_zoom2p2_00001_1-17',
         ]

base = '/mnt/ceph/neuro/labeling/'

fls = []
for fl in fls_1:
    fls.append(os.path.join(base, fl))
    print(fls[-1])

# fls = [
# 'Jan25_2015_07_13',
# 'k53_20160530',
# 'neurofinder.00.01',
# 'k26_v1_176um_target_pursuit_002_013',
# 'neurofinder.02.01',
# 'J115_2015-12-09_L01',
# 'neurofinder.00.04',
# 'k31_20160104_MMA_150um_65mW_zoom2p2',
# 'k36_20160127_RL_150um_65mW_zoom2p2_00002_22-41',
# 'neurofinder.00.02',
# 'neurofinder.01.00',
# 'neurofinder.03.00',
# 'neuro_finder_tests',
# 'neurofinder.00.03',
# 'k31_20151223_AM_150um_65mW_zoom2p2',
# 'k36_20160115_RSA_400um_118mW_zoom2p2_00001_20-38',
# 'neurofinder.04.01',
# 'Yi.data.002',
# 'Jan42_exp4_001',
# 'Jan40_exp2_001',
#]
#%%
import os
import glob
import caiman as cm

#%% RUN ANALYSIS
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=20, single_thread=False)
#%%
max_shifts = [20, 20]
niter_rig = 1
splits_rig = 10
num_splits_to_process_rig = None

count = 1
for fl in fls:
    nms = glob.glob(os.path.join(fl, 'images/*.tif'))
    nms.sort()
    for nm_ in nms:
        if '_BL' in nm_:
            nms.remove(nm_)
    if len(nms) > 0:
        pl.subplot(5, 4, count)
        count += 1
        templ = cm.load(os.path.join(fl, 'projections/median_projection.tif'))
        mov_tmp = cm.load(nms[0], subindices=range(400))
        if mov_tmp.shape[1:] != templ.shape:
            diffx, diffy = np.subtract(mov_tmp.shape[1:], templ.shape) // 2 + 1

        vmin, vmax = np.percentile(templ, 5), np.percentile(templ, 95)
        pl.imshow(templ, vmin=vmin, vmax=vmax)

        min_mov = np.nanmin(mov_tmp)
        mc_list = []
        mc_templs_part = []
        mc_templs = []
        mc_fnames = []
        for each_file in nms:
            # TODO: needinfo how the classes works
            mc = MotionCorrect(each_file, min_mov,
                               dview=dview, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
                               num_splits_to_process_rig=num_splits_to_process_rig,
                               shifts_opencv=True, nonneg_movie=True)
            mc.motion_correct_rigid(template=templ, save_movie=True)
            new_templ = mc.total_template_rig

            #TODO : needinfo
            pl.imshow(new_templ, cmap='gray')
            pl.pause(.1)
            mc_list += mc.shifts_rig
            mc_templs_part += mc.templates_rig
            mc_templs += [mc.total_template_rig]
            mc_fnames += mc.fname_tot_rig

        np.savez(os.path.join(fl, 'images/mot_corr_res.npz'), mc_list=mc_list,
                 mc_templs_part=mc_templs_part, mc_fnames=mc_fnames, mc_templs=mc_templs)

    print([os.path.split(nm)[-1] for nm in nms])
    print([int(os.path.getsize(nm) / 1e+9 * 100) / 100. for nm in nms])

#%% fix files
