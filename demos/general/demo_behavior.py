#!/usr/bin/env python

"""
Demonstrate Caiman functions relating to behavioral experiments and optical flow

This demo requires a GUI; it does not make sense to run it noninteractively.
"""

# If this demo crashes right after making it past the initial play of the mouse movie,
# you may need to switch your build of opencv+libopencv+py-opencv to a qt6 build of the
# same (in conda, these have a qt6_ prefix to their build id). 

import argparse
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.behavior import behavior
from caiman.utils.utils import download_demo


def main():
    cfg = handle_args()

    if cfg.logfile:
        logging.basicConfig(format=
            "[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s",
            level=logging.INFO,
            filename=cfg.logfile)
        # You can make the output more or less verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR
    else:
        logging.basicConfig(format=
            "[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s",
            level=logging.INFO)

    if cfg.input is None:
        # If no input is specified, use sample data, downloading if necessary
        fnames = [download_demo('demo_behavior.h5')]
    else:
        fnames = cfg.input
    # If you prefer to hardcode filenames, you could do something like this:
    # fnames = ["/path/to/myfile1.avi", "/path/to/myfile2.avi"]

    plt.ion()

    # TODO: todocument
    m = cm._load_behavior(fnames[0])

    # load, rotate and eliminate useless pixels
    m = m.transpose([0, 2, 1]) # XXX Does this really work outside this dataset?
    m = m[:, 150:, :] # TODO adopt some syntax for clipping, or make this optional and tell the user to clip before running

    # visualize movie
    m.play()

    # select interesting portion of the FOV (draw a polygon on the figure that pops up, when done press enter)
    print("Please draw a polygon delimiting the ROI on the image that will be displayed after the image; press enter when done")
    mask = np.array(behavior.select_roi(np.median(m[::100], 0), 1)[0], np.float32)

    n_components = 4  # number of movement looked for
    resize_fact = 0.5  # for computational efficiency movies are downsampled
    # number of standard deviations above mean for the magnitude that are considered enough to measure the angle in polar coordinates
    num_std_mag_for_angle = .6
    only_magnitude = False  # if onlu interested in factorizing over the magnitude
    method_factorization = 'nmf'
    # number of iterations for the dictionary learning algorithm (Marial et al, 2010)
    max_iter_DL = -30

    spatial_filter_, time_trace_, of_or = cm.behavior.behavior.extract_motor_components_OF(m, n_components, mask=mask,
                                                                                           resize_fact=resize_fact, only_magnitude=only_magnitude, verbose=True, method_factorization='nmf', max_iter_DL=max_iter_DL)


    mags, dircts, dircts_thresh, spatial_masks_thrs = cm.behavior.behavior.extract_magnitude_and_angle_from_OF(
        spatial_filter_, time_trace_, of_or, num_std_mag_for_angle=num_std_mag_for_angle, sav_filter_size=3, only_magnitude=only_magnitude)

    idd = 0
    axlin = pl.subplot(n_components, 2, 2)
    for mag, dirct, spatial_filter in zip(mags, dircts_thresh, spatial_filter_):
        pl.subplot(n_components, 2, 1 + idd * 2)
        min_x, min_y = np.min(np.where(mask), 1)

        spfl = spatial_filter
        spfl = cm.movie(spfl[None, :, :]).resize(
            1 / resize_fact, 1 / resize_fact, 1).squeeze()
        max_x, max_y = np.add((min_x, min_y), np.shape(spfl))

        mask[min_x:max_x, min_y:max_y] = spfl
        mask[mask < np.nanpercentile(spfl, 70)] = np.nan
        pl.imshow(m[0], cmap='gray')
        pl.imshow(mask, alpha=.5)
        pl.axis('off')

        axelin = pl.subplot(n_components, 2, 2 + idd * 2, sharex=axlin)
        pl.plot(mag / 10, 'k')
        dirct[mag < 0.5 * np.std(mag)] = np.nan
        pl.plot(dirct, 'r-', linewidth=2)

        idd += 1

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate behavioural/optic flow functions")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()