#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Mon Jul 11 10:09:09 2016

@author: agiovann
"""

from glob import glob
import scipy.stats as st
import calblitz as cb
import numpy as np
#%%
for fl in glob('k36*compress_.tif'):
    print(fl)
    m = cb.load(fl, fr=3)

    img = m.local_correlations(eight_neighbours=True)
    im = cb.movie(img, fr=1)
    im.save(fl[:-4] + 'correlation_image.tif')

    m = np.array(m)

    img = st.skew(m, 0)
    im = cb.movie(img, fr=1)
    im.save(fl[:-4] + 'skew.tif')

    img = st.kurtosis(m, 0)
    im = cb.movie(img, fr=1)
    im.save(fl[:-4] + 'kurtosis.tif')

    img = np.std(m, 0)
    im = cb.movie(img, fr=1)
    im.save(fl[:-4] + 'std.tif')

    img = np.median(m, 0)
    im = cb.movie(img, fr=1)
    im.save(fl[:-4] + 'median.tif')

    img = np.max(m, 0)
    im = cb.movie(img, fr=1)
    im.save(fl[:-4] + 'max.tif')

#%%
m = cb.load('All.tif', fr=3)

m = cb.load('All_BL.tif', fr=3)

#%%
m = cb.load('k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_All.tif', fr=3)

img = m.local_correlations(eight_neighbours=True)
im = cb.movie(img, fr=1)
im.save('correlation_image.tif')

m = np.array(m)
img = st.skew(m, 0)
im = cb.movie(img, fr=1)
im.save('skew.tif')

img = st.kurtosis(m, 0)
im = cb.movie(img, fr=1)
im.save('kurtosis.tif')

img = np.std(m, 0)
im = cb.movie(img, fr=1)
im.save('std.tif')

img = np.median(m, 0)
im = cb.movie(img, fr=1)
im.save('median.tif')


img = np.max(m, 0)
im = cb.movie(img, fr=1)
im.save('max.tif')
#%%
m = cb.load('k31_20160104_MMA_150um_65mW_zoom2p2_00001_000_All_BL.tif', fr=3)

img = m.local_correlations(eight_neighbours=True)
im = cb.movie(img, fr=1)
im.save('correlation_image_BL.tif')
m = np.array(m)

img = st.skew(m, 0)
im = cb.movie(img, fr=1)
im.save('skew_BL.tif')

img = st.kurtosis(m, 0)
im = cb.movie(img, fr=1)
im.save('kurtosis_BL.tif')

img = np.std(m, 0)
im = cb.movie(img, fr=1)
im.save('std_BL.tif')

img = np.median(m, 0)
im = cb.movie(img, fr=1)
im.save('median_BL.tif')


img = np.max(m, 0)
im = cb.movie(img, fr=1)
im.save('max_BL.tif')
