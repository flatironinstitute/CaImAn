#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sun Aug  6 11:43:39 2017

@author: agiovann
"""

import cv2
from IPython import get_ipython
import logging
import numpy as np
import matplotlib.pyplot as plt

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        print("Detected iPython")
        ipython = get_ipython()
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
        ipython.run_line_magic('matplotlib', 'qt')
except NameError:
    pass

import caiman as cm
from caiman.behavior import behavior
from caiman.utils.utils import download_demo

#%%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.WARNING)

#%%
def main():
    pass # For compatibility between running under IDE and the CLI

    #%%
    plt.ion()

    fname = [u'demo_behavior.h5']
    if fname[0] in ['demo_behavior.h5']:
        # TODO: todocument
        fname = [download_demo(fname[0])]
    # TODO: todocument
    m = cm._load_behavior(fname[0])

<<<<<<< HEAD
    #%% load, rotate and eliminate useless pixels
    m = m.transpose([0, 2, 1])
    m = m[:, 150:, :]
=======
    plt.ion()

    # TODO: todocument
    m = cm._load_behavior(fnames[0])
>>>>>>> 7f972213 (demo_behavior: pull some changes from dev)

    #%% visualize movie
    m.play()

    #%% select interesting portion of the FOV (draw a polygon on the figure that pops up, when done press enter)
    # TODO: Put the message below into the image
    print("Please draw a polygon delimiting the ROI on the image that will be displayed after the image; press enter when done")
    mask = np.array(behavior.select_roi(np.median(m[::100], 0), 1)[0], np.float32)

    #%%
    n_components = 4  # number of movement looked for
    resize_fact = 0.5  # for computational efficiency movies are downsampled
    # number of standard deviations above mean for the magnitude that are considered enough to measure the angle in polar coordinates
    num_std_mag_for_angle = .6
    only_magnitude = False  # if onlu interested in factorizing over the magnitude
    method_factorization = 'dict_learn'  # could also use nmf
    # number of iterations for the dictionary learning algorithm (Marial et al, 2010)
    max_iter_DL = -30

    spatial_filter_, time_trace_, of_or = cm.behavior.behavior.extract_motor_components_OF(m, n_components, mask=mask,
                                                                                           resize_fact=resize_fact, only_magnitude=only_magnitude, verbose=True, method_factorization='dict_learn', max_iter_DL=max_iter_DL)

    #%%
    mags, dircts, dircts_thresh, spatial_masks_thrs = cm.behavior.behavior.extract_magnitude_and_angle_from_OF(
        spatial_filter_, time_trace_, of_or, num_std_mag_for_angle=num_std_mag_for_angle, sav_filter_size=3, only_magnitude=only_magnitude)
    
    #%%
    idd = 0
    axlin = plt.subplot(n_components, 2, 2)
    for mag, dirct, spatial_filter in zip(mags, dircts_thresh, spatial_filter_):
        plt.subplot(n_components, 2, 1 + idd * 2)
        min_x, min_y = np.min(np.where(mask), 1)

        spfl = spatial_filter
        spfl = cm.movie(spfl[None, :, :]).resize(
            1 / resize_fact, 1 / resize_fact, 1).squeeze()
        max_x, max_y = np.add((min_x, min_y), np.shape(spfl))

        mask[min_x:max_x, min_y:max_y] = spfl
        mask[mask < np.nanpercentile(spfl, 70)] = np.nan
        plt.imshow(m[0], cmap='gray')
        plt.imshow(mask, alpha=.5)
        plt.axis('off')

        axelin = plt.subplot(n_components, 2, 2 + idd * 2, sharex=axlin)
        plt.plot(mag / 10, 'k')
        dirct[mag < 0.5 * np.std(mag)] = np.nan
        plt.plot(dirct, 'r-', linewidth=2)

        idd += 1

#%%
# This is to mask the differences between running this demo in an IDE
# versus from the CLI
if __name__ == "__main__":
    main()