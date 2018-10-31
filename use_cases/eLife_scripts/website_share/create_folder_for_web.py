# %%
import glob
import os
from PIL import Image
import caiman as cm
import numpy as np
from caiman.base.rois import nf_read_roi_zip, nf_masks_to_json
import shutil
# %%
dest_folders = ['J115', 'J123', 'K53', 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']
dest_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE'

base_folder = '/mnt/ceph/neuro/labeling'
base_folders = ['J115_2015-12-09_L01_ELS', 'J123_2015-11-20_L01_0', 'k53_20160530', 'neurofinder.00.00', 'neurofinder.01.01', 'neurofinder.02.00', 'neurofinder.03.00.test', 'neurofinder.04.00.test', 'yuste.Single_150u']
import zipfile
for bfold,dfold in zip(base_folders[:], dest_folders[:]):
    datafold = os.path.join(dest_folder, dfold)
    print(datafold)
    images_fold = os.path.join(datafold, 'images')
    os.makedirs(images_fold, exist_ok=True)
    print(os.path.join(base_folder, bfold) +'/images/mmap_tifs/*.tif')
    movlist = glob.glob(os.path.join(base_folder, bfold) +'/images/mmap_tifs/*.tif')
    movlist.sort()
    regions_fold = os.path.join(os.path.join(base_folder, bfold), 'regions')
    regions_dist = os.path.join(os.path.join(dest_folder, dfold), 'regions')
    os.makedirs(regions_dist, exist_ok=True)
    counter = 0
    z = zipfile.ZipFile(os.path.join(images_fold, 'images.zip'), "w")
    for movname in movlist:
        mov = cm.load(movname)
        print(movname)
        for idx, fr in enumerate(mov):
            if counter%1000 == 0:
                print(counter)
            frame_name = os.path.join(images_fold, 'image' + str(counter).zfill(5) + '.tif')
            im = Image.fromarray(fr)
            im.save(frame_name)
            z.write(frame_name, os.path.basename(frame_name))
            os.remove(frame_name)
            counter += 1
    z.close()
    dims = np.shape(mov)[1:]


    regionslist = glob.glob(os.path.join(regions_fold, '*'))

    for region in regionslist:
        if 'ben_active_regions_nd.zip' in region:
            name = os.path.join(regions_dist , 'L4_regions.json')
            masks = nf_read_roi_zip(region, dims)
        elif 'lindsey_active_regions_nd.zip' in region:
            name = os.path.join(regions_dist , 'L3_regions.json')
            masks = nf_read_roi_zip(region, dims)
        elif 'sonia_active_regions_nd.zip' in region:
            name = os.path.join(regions_dist , 'L2_regions.json')
            masks = nf_read_roi_zip(region, dims)
        elif 'natalia_active_regions_nd.zip' in region:
            name = os.path.join(regions_dist , 'L1_regions.json')
            masks = nf_read_roi_zip(region, dims)
        elif 'joined_consensus_active_regions.npy' in region:
            name = os.path.join(regions_dist , 'consensus_regions.json')
            masks = np.load(region)
        else:
            print('discarded ' + region)
            continue

        nf_masks_to_json(masks, name)

    projection_dest = os.path.join(os.path.join(dest_folder, dfold), 'projections')
    projection_base = os.path.join(os.path.join(base_folder, bfold), 'projections')
    os.makedirs(projection_dest, exist_ok=True)
    shutil.copyfile(os.path.join(projection_base,'correlation_image.tif'),os.path.join(projection_dest,'correlation_image.tif'))
    shutil.copyfile(os.path.join(projection_base,'median_projection.tif'),os.path.join(projection_dest,'median_image.tif'))




#%% JUST MOVIES

dest_folders = ['J115', 'J123', 'K53', 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']
dest_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE'

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/'
base_folders = dest_folders

import zipfile
from glob import glob
for bfold, dfold in zip(base_folders[-1:], dest_folders[-1:]):
    datafold = os.path.join(dest_folder, dfold)
    print(datafold)
    images_fold = os.path.join(datafold, 'images')
    os.makedirs(images_fold, exist_ok=True)
    print(os.path.join(base_folder, bfold) + '/mmap_files/*.mmap')
    movlist = glob(os.path.join(base_folder, bfold) + '/mmap_files/*.mmap')
    movlist.sort()
    print(movlist)
    counter = 0
    z = zipfile.ZipFile(os.path.join(images_fold, 'images.zip'), "w")
    for movname in movlist:
        mov = np.array(cm.load(movname), dtype=np.float32)
        print(movname)
        for idx, fr in enumerate(mov):
            if counter % 1000 == 0:
                print(counter)
            frame_name = os.path.join(images_fold, 'image' + str(counter).zfill(5) + '.tif')
            im = Image.fromarray(fr)
            im.save(frame_name)
            z.write(frame_name, os.path.basename(frame_name))
            os.remove(frame_name)
            counter += 1
    z.close()






