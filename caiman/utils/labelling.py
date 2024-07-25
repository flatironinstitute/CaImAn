#!/usr/bin/env python

import logging
from scipy.ndimage import filters as ft

import caiman

def pre_preprocess_movie_labeling(dview, file_names, median_filter_size=(2, 1, 1),
                                  resize_factors=[.2, .1666666666], diameter_bilateral_blur=4):

       #todo: todocument

    def pre_process_handle(args):
        # todo: todocument
        fil, resize_factors, diameter_bilateral_blur, median_filter_size = args

        name_log = fil[:-4] + '_LOG'
        logger = logging.getLogger(name_log)
        hdlr = logging.FileHandler(name_log) # We don't use the caiman logger here b/c we're saving to files

        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)

        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)

        logger.info('START')
        logger.info(fil)

        mov = caiman.load(fil, fr=30)
        logger.info('Read file')

        mov = mov.resize(1, 1, resize_factors[0])
        logger.info('Resize')

        mov = mov.bilateral_blur_2D(diameter=diameter_bilateral_blur)
        logger.info('Bilateral')

        mov1 = caiman.movie(ft.median_filter(mov, median_filter_size), fr=30)
        logger.info('Median filter')

        mov1 = mov1.resize(1, 1, resize_factors[1])
        logger.info('Resize 2')

        mov1 = mov1 - caiman.utils.stats.mode_robust(mov1, 0)
        logger.info('Mode')

        mov = mov.resize(1, 1, resize_factors[1])
        logger.info('Resize')

        mov.save(fil[:-4] + '_compress_.tif')
        logger.info('Save 1')

        mov1.save(fil[:-4] + '_BL_compress_.tif')
        logger.info('Save 2')
        return 1

    args = []
    for name in file_names:
        args.append(
            [name, resize_factors, diameter_bilateral_blur, median_filter_size])

    if dview is not None:
        file_res = dview.map_sync(pre_process_handle, args)
        dview.results.clear()
    else:
        file_res = list(map(pre_process_handle, args))

    return file_res
