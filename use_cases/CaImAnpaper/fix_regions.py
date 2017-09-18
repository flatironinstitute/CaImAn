#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:42:20 2017

@author: agiovann
"""
from caiman.base.rois import nf_read_roi_zip
# data transformation because of motion correction mismatch in shape
#%% sue k37
new_templ = cm.load('projections/median_projection.tif')
regions = nf_read_roi_zip('regions/joined_consensus_active_regions.zip',new_templ.shape)

new_templ = np.pad(new_templ.T,((7,8),(4,3)),mode = 'constant', constant_values = new_templ.mean())
regions = np.pad(regions.transpose(0,2,1),((0,0),(7,8),(4,3)),mode = 'constant')


np.save('regions/joined_consensus_active_regions',regions)
np.save('projections/median_projection',new_templ)
#%% if no change 
corr_img =  cm.load('projections/correlation_image.tif')
new_templ = cm.load('projections/median_projection.tif')
regions = nf_read_roi_zip('regions/joined_consensus_active_regions.zip',new_templ.shape)

pl.imshow(new_templ/np.nanmax(new_templ),cmap = 'gray',vmin =0.01,vmax = .3)
#pl.imshow(corr_img/np.nanmax(corr_img),cmap = 'gray',vmax = .25)

pl.imshow(regions.sum(0),alpha =.1,cmap = 'hot',vmax = 3)

#%%
new_templ = cm.load('projections/median_projection.tif')
regions = nf_read_roi_zip('regions/joined_consensus_active_regions.zip',new_templ.shape)
np.save('regions/joined_consensus_active_regions',regions)
np.save('projections/median_projection',new_templ)

