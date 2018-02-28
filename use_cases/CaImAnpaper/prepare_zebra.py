#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:11:40 2018
Tool to prepare zebrafish data
@author: agiovann
"""

#%%
import pylab as pl
import numpy as np
import glob
import caiman as cm

fnames = glob.glob("/mnt/ceph/users/jfriedrich/data/05292014Fish1-4/*.stack")
fnames.sort()
print(fnames)
#%%
pl.figure(figsize=(12,10))
fname = fnames[44]
with open(fname, "r") as f:
    a = np.fromfile(f, dtype=np.uint16)
    b = a.reshape((2048,1096,1885), order = 'F')
    m = cm.movie(b.transpose([2,0,1]))

img = m[:300].std(0)
pl.imshow(img, vmax = np.std(img)*2)
coords = pl.ginput(-1)
min_x, min_y = np.min(coords,0).astype(np.int)
max_x, max_y = np.max(coords,0).astype(np.int)

pl.imshow(img[min_y:max_y,min_x:max_x],vmax = np.std(img)*2)
#%%
m[:,min_y:max_y,min_x:max_x].save(fname.split('/')[-1]+'.hdf5')