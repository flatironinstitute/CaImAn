#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Jun 28 15:27:24 2017

@author: epnevmatikakis
"""

import numpy as np
import caiman as cm
import os
#%%

foldernames = ['/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS',
               '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0',
               '/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001',
               '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16',
               '/mnt/ceph/neuro/labeling/k53_20160530',
               '/mnt/ceph/neuro/labeling/neurofinder.00.00',
               '/mnt/ceph/neuro/labeling/neurofinder.02.00',
               '/mnt/ceph/neuro/labeling/neurofinder.04.00',
               '/mnt/ceph/neuro/labeling/neurofinder.03.00.test',
               '/mnt/ceph/neuro/labeling/neurofinder.04.00.test']

foldernames = ['/mnt/ceph/neuro/labeling/FINAL_NO_USED_FOR_CONSENSUS/neurofinder.01.01',
               '/mnt/ceph/neuro/labeling/FINAL_NO_USED_FOR_CONSENSUS/packer.001',
               '/mnt/ceph/neuro/labeling/FINAL_NO_USED_FOR_CONSENSUS/Yi.data.001',
               '/mnt/ceph/neuro/labeling/FINAL_NO_USED_FOR_CONSENSUS/yuste.Single_150u',
               '/mnt/ceph/neuro/labeling/FINAL_NO_USED_FOR_CONSENSUS/Jan-AMG1_exp2_new_001']
#%%

import glob
from caiman.base.rois import detect_duplicates, nf_merge_roi_zip, nf_read_roi_zip
from shutil import copyfile
#%%
for fldname in foldernames:
    current_folder = os.path.join(
        '/mnt/ceph/neuro/labeling', fldname, 'regions')
    img_shape = cm.load(os.path.join('/mnt/ceph/neuro/labeling',
                                     fldname, 'projections/correlation_image.tif')).shape
    filenames = glob.glob(os.path.join(current_folder, '*active*regions.zip'))
    for flname in filenames:
        ind_dup, ind_keep = detect_duplicates(flname, 0.25, FOV=img_shape)
        rois = nf_read_roi_zip(flname, img_shape)
        new_fname = flname[:-4] + '_nd.zip'
        print(flname)
        if not ind_dup:
            copyfile(flname, new_fname)
        else:
            nf_merge_roi_zip([flname], [ind_dup], flname[:-4] + '_copy')
            nf_merge_roi_zip([flname], [ind_keep], new_fname[:-4])
            print('FOUND!!')
