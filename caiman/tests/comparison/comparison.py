#!/usr/bin/env python

""" compare how the elements behave

We create a folder ground truth that possess the same thing than the other
in a form of a dictionary containing nparrays and other info.
the other files contains every test and the name is the date of the test

"""

#FIXME this file needs some attention

import copy
import datetime
import logging
import matplotlib.pyplot as pl
import numpy as np
import os
import platform as plt
import scipy

# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.getLogger("caiman").setLevel(logging.DEBUG)

import caiman as cm
from caiman.paths import caiman_datadir


class Comparison(object):
    """
       Comparison(object): It is used for comparison the modification on the CaImAn program


       You need to compare your modification to the groundtruth and then rename the folder created as "proof" for it to be send in you next push request

       Here it has been made for 3 different functions. for it to compare well you need to set your
       ground truth with the same computer with which you are comparing the files
       class you instantiate to compare the different parts of CaImAn




        Attributes
        ----------
        self : object
    LIST OF WHAT YOU CAN FIND IN THE SAVED FILE
      data
        A_full
        A_patch
        rig_shifts
        C_full
        C_patch
        information
                differences
                        params_cnm
                        proc
                        param_movie
                params
                timer
                        rig_shifts
                        cnmf on patch
                        cnmf_full_frame
                time
                processor
                cnmpatch
                platform
                diff
                        cnmfull [same parameters](#732ed9)
                        cnmpatch
                                diff_data
                                        performance
                                                f1_score
                                                accuracy
                                                precision
                                                recall
                                        diffneur
                                        correlations
                                isdifferent
                                diff_timing
                        rig
                                diff_timing
                                isdifferent
                                diff_data
            the user to change it manually

        Methods
        -------
        __init__()
            Initialize the function be instantiating a comparison object

        save(istruth)
            save the comparison object on a file

        save_with_compare(self, istruth, params, dview, avg_min_size_ratio, n_frames_per_bin, dims_test, dims_gt, Cn, cmap)
            the save functions to use !

        see(self,filename=None)
            to look into a particular saved file

        plotall(self)
            when you want to plot the variations of the different tests that have been compared according to the actual groundtruth

        See Also
        --------

        .. image:: caiman/tests/comparison/data.pdf
        """

    def __init__(self):

        self.comparison:dict[str, dict] = {
            'rig_shifts': {},
            'pwrig_shifts': {},
            'cnmf_on_patch': {},
            'cnmf_full_frame': {},
        }

        self.comparison['rig_shifts'] = {
            'ourdata': None,
            'timer': None,
            'sensitivity': 0.001               # the sensitivity USER TO CHOOSE
        }
                                               # apparently pwrig shift are not used any more and the comparison are useless
                                               # self.comparison['pwrig_shifts']={
                                               #                 'ourdata': None,
                                               #                'timer': None,
                                               #               'sensitivity': 0.001
                                               #           }
        self.comparison['cnmf_on_patch'] = {'ourdata': None, 'timer': None, 'sensitivity': 0.01}
        self.comparison['cnmf_full_frame'] = {'ourdata': None, 'timer': None, 'sensitivity': 0.01}

        self.cnmpatch = None
        self.information = None
        self.dims = None

    def save_with_compare(self, istruth=False, params=None, dview=None, Cn=None):
        """save the comparison as well as the images of the precision recall calculations


            depending on if we say this file will be ground truth or not, it will be saved in either the tests or the ground truth folder
            if saved in test, a comparison to groundtruth will be added to the object 
            this comparison will be on 
                data : a normized difference of the normalized value of the arrays
                time : difference
            in order for this function to work, you must
                have previously given it the cnm objects after initializing them ( on patch and full frame)
                give the values of the time and data 
                have a groundtruth


            Args:
                self:  dictionary
                   the object of this class that contains every value

                istruth: Boolean
                    if we want it to be the ground truth

                params:
                    movie parameters

                dview :
                    your dview object

                n_frames_per_bin:
                    you need to know those data before
                    they have been given to the base/rois functions

                dims_test:
                    you need to know those data before
                    they have been given to the base/rois functions

                Cn:
                    your correlation image

                Cmap:
                    a particular colormap for your Cn

            See Also:
                Example of utilisation on Demo Pipeline
\image caiman/tests/comparison/data.pdf

             Raises:
                 ('we now have ground truth\n')

                 ('we were not able to read the file to compare it\n')

                """
        # getting the DATA FOR COMPARISONS
        assert (params != None and self.cnmpatch != None)
        logging.info('we need the parameters in order to save anything\n')
        # actions on the sparse matrix
        cnm = self.cnmpatch.__dict__
        cnmpatch = deletesparse(cnm)

        # initialization
        dims_test = [self.dims[0], self.dims[1]]
        dims_gt = dims_test
        today_dt = datetime.datetime.today()
        today_str = str(today_dt)
        plat = plt.platform()
        plat = str(plat)
        pro = plt.processor()
        pro = str(pro)
        # we store a big file which contains everything (INFORMATION)
        information = {
            'platform': plat,
            'time': today_str,
            'processor': pro,
            'params': params,
            'cnmpatch': cnmpatch,
            'timer': {
                'cnmf_on_patch': self.comparison['cnmf_on_patch']['timer'],
                'cnmf_full_frame': self.comparison['cnmf_full_frame']['timer'],
                'rig_shifts': self.comparison['rig_shifts']['timer']
            }
        }

        file_path = os.path.join(caiman_datadir(), "testdata", "groundtruth.npz")

        # OPENINGS
        # if we want to set this data as truth
        if istruth:
            # we just save it
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                logging.debug("nothing to remove\n")
            np.savez_compressed(file_path,
                                information=information,
                                A_full=self.comparison['cnmf_full_frame']['ourdata'][0],
                                C_full=self.comparison['cnmf_full_frame']['ourdata'][1],
                                A_patch=self.comparison['cnmf_on_patch']['ourdata'][0],
                                C_patch=self.comparison['cnmf_on_patch']['ourdata'][1],
                                rig_shifts=self.comparison['rig_shifts']['ourdata'])
            logging.info('we now have ground truth\n')
            return

        else:                                                                                               # if not we create a comparison first
            try:
                with np.load(file_path, encoding='latin1', allow_pickle=True) as dt:
                    rig_shifts = dt['rig_shifts'][()]
                    A_patch = dt['A_patch'][()]
                    A_full = dt['A_full'][()]
                    C_full = dt['C_full'][()]
                    C_patch = dt['C_patch'][()]
                    data = dt['information'][()]
                                                                                                            # if we cannot manage to open it or it doesn't exist:
            except (IOError, OSError):
                                                                                                            # we save but we explain why there were a problem
                logging.warning('we were not able to read the file ' + str(file_path) + ' to compare it\n')
                file_path = os.path.join(caiman_datadir(), "testdata", "NC" + dt + ".npz")
                np.savez_compressed(file_path,
                                    information=information,
                                    A_full=self.comparison['cnmf_full_frame']['ourdata'][0],
                                    C_full=self.comparison['cnmf_full_frame']['ourdata'][1],
                                    A_patch=self.comparison['cnmf_on_patch']['ourdata'][0],
                                    C_patch=self.comparison['cnmf_on_patch']['ourdata'][1],
                                    rig_shifts=self.comparison['rig_shifts']['ourdata'])
                return
                                                                                                            # creating the FOLDER to store our data
                                                                                                            # XXX Is this still hooked up to anything?
        i = 0
        dr = os.path.join(caiman_datadir(), "testdata")
        for name in os.listdir(dr):
            i += 1
        istr = str(i)
        if not os.path.exists(dr + istr):
            os.makedirs(dr + istr)
        information.update({'diff': {}})
        information.update({'differences': {'proc': False, 'params_movie': False, 'params_cnm': False}})

        if data['processor'] != information['processor']:
            logging.info("You don't have the same processor as was used to generate the ground truth. The processing time can vary.\n" +
                         "For time comparison, Create your own groundtruth standard for future testing.\n" +
                         f"Compare: {data['processor']} to {information['processor']}\n")
            information['differences']['proc'] = True
        if data['params'] != information['params']:
            logging.warning("You are not using the same movie parameters. Results will not be comparable.")
            logging.warning('You must use the same parameters as the groundtruth.\n' +
                            'examine the groundtruth parameters with the see() method\n')
            information['differences']['params_movie'] = True
                                                                                                            # We must cleanup some fields to permit an accurate comparison
        if not normalised_compare_cnmpatches(data['cnmpatch'], cnmpatch):
            if data['cnmpatch'].keys() != cnmpatch.keys():
                logging.error(
                    'DIFFERENCES IN THE FIELDS OF CNMF'
                )                                                                                           # TODO: Now that we have deeply nested data structures, find a module that gives you tight differences.
            diffkeys = [k for k in data['cnmpatch'] if data['cnmpatch'][k] != cnmpatch[k]]
            for k in diffkeys:
                logging.info(f"{k}:{data['cnmpatch'][k]}->{cnmpatch[k]}")

            logging.warning('You are not using the same parameters in your cnmf on patches initialization\n')
            information['differences']['params_cnm'] = True

        # for rigid
        # plotting part

        information['diff'].update({
            'rig':
            plotrig(init=rig_shifts,
                    curr=self.comparison['rig_shifts']['ourdata'],
                    timer=self.comparison['rig_shifts']['timer'] - data['timer']['rig_shifts'],
                    sensitivity=self.comparison['rig_shifts']['sensitivity'])
        })
        information['diff'].update({
            'cnmpatch':
            cnmf(Cn=Cn,
                 A_gt=A_patch,
                 A_test=self.comparison['cnmf_on_patch']['ourdata'][0],
                 C_gt=C_patch,
                 C_test=self.comparison['cnmf_on_patch']['ourdata'][1],
                 dview=dview,
                 sensitivity=self.comparison['cnmf_on_patch']['sensitivity'],
                 dims_test=dims_test,
                 dims_gt=dims_gt,
                 timer=self.comparison['cnmf_on_patch']['timer'] - data['timer']['cnmf_on_patch'])
        })
        # CNMF FULL FRAME
        information['diff'].update({
            'cnmfull':
            cnmf(Cn=Cn,
                 A_gt=A_full,
                 A_test=self.comparison['cnmf_full_frame']['ourdata'][0],
                 C_gt=C_full,
                 C_test=self.comparison['cnmf_full_frame']['ourdata'][1],
                 dview=dview,
                 sensitivity=self.comparison['cnmf_full_frame']['sensitivity'],
                 dims_test=dims_test,
                 dims_gt=dims_gt,
                 timer=self.comparison['cnmf_full_frame']['timer'] - data['timer']['cnmf_full_frame'])
        })
        #try:
        #    pl.gcf().savefig(dr + istr + '/' + 'cnmfull.pdf')
        #    pl.close()
        #except:
        #    pass

        # Saving of everything
        target_dir = os.path.join(caiman_datadir(), "testdata", istr)
        if not os.path.exists(target_dir):
            os.makedirs(os.path.join(caiman_datadir(), "testdata", istr)) # TODO Revise to just use the exist_ok flag to os.makedirs
        file_path = os.path.join(target_dir, istr + ".npz")
        np.savez_compressed(file_path,
                            information=information,
                            A_full=self.comparison['cnmf_full_frame']['ourdata'][0],
                            C_full=self.comparison['cnmf_full_frame']['ourdata'][1],
                            A_patch=self.comparison['cnmf_on_patch']['ourdata'][0],
                            C_patch=self.comparison['cnmf_on_patch']['ourdata'][1],
                            rig_shifts=self.comparison['rig_shifts']['ourdata'])

        self.information = information


def see(filename=None):
    """shows you the important data about a certain test file ( just give the number or name)

        if you give nothing it will give you back the groundtruth infos

        Args:
            self:  dictionary
                the object of this class that tcontains every value
            filename:
                ( just give the number or name)

        See Also:
            @image html caiman/tests/comparison/data.pdf
            """

    if filename == None:
        dr = os.path.join(caiman_datadir(), "testdata", "groundtruth.npz")
    else:
        dr = os.path.join(caiman_datadir(), "testdata", filename, filename + ".npz")
        logging.debug("Loading GT file " + str(dr))
    with np.load(dr) as dt:
        print('Info :\n')
        see_it(dt)


def see_it(data=None):
    for key in data:

        val = data[key]
        if isinstance(val, dict):
            print('\n')
            print(key)
            print('\n')
            see_it(val)
        else:
            if not isinstance(val, scipy.sparse.coo.coo_matrix) \
                    and not isinstance(val, scipy.sparse.csc.csc_matrix) and not isinstance(val, list):

                print(key)
                print(val)


def deletesparse(cnm):
    for keys in cnm:
        val = cnm[keys]
        if isinstance(val, dict):
            val = deletesparse(val)
        if not isinstance(val, scipy.sparse.coo.coo_matrix) and not isinstance(val, np.ndarray) \
                and not isinstance(val, scipy.sparse.csc.csc_matrix) and not keys == 'dview':
            logging.debug(f"type of val is {type(val)}")
            cnm[keys] = val
        else:

            cnm[keys] = None
    return cnm


def cnmf(Cn, A_gt, A_test, C_gt, C_test, dims_gt, dims_test, dview=None, sensitivity=0, timer=0, n_frames_per_bin=10):

    A_test = A_test.toarray()  # coo sparse matrix
    A_gt = A_gt.toarray()

    # proceed to a trhreshold
    A_test_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_test,
                                                                        dims_test,
                                                                        medw=None,
                                                                        thr_method='max',
                                                                        maxthr=0.2,
                                                                        nrgthr=0.99,
                                                                        extract_cc=True,
                                                                        se=None,
                                                                        ss=None,
                                                                        dview=dview)
    A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt,
                                                                      dims_gt,
                                                                      medw=None,
                                                                      thr_method='max',
                                                                      maxthr=0.2,
                                                                      nrgthr=0.99,
                                                                      extract_cc=True,
                                                                      se=None,
                                                                      ss=None,
                                                                      dview=dview)

    # compute C using this A thr
    A_test_thr = A_test_thr.toarray() > 0
    A_gt_thr = A_gt_thr.toarray() > 0
    # we do not compute a threshold on the size of neurons
    C_test_thr = C_test
    C_gt_thr = C_gt
    # we would also like the difference in the number of neurons
    diffneur = A_test_thr.shape[1] - A_gt_thr.shape[1]
    #    print(diffneur+1)
    # computing the values
    C_test_thr = np.array([CC.reshape([-1, n_frames_per_bin]).max(1) for CC in C_test_thr])
    C_gt_thr = np.array([CC.reshape([-1, n_frames_per_bin]).max(1) for CC in C_gt_thr])
    maskgt = A_gt_thr[:, :].reshape([dims_gt[0], dims_gt[1], -1], order='F').transpose([2, 0, 1]) * 1.
    masktest = A_test_thr[:, :].reshape([dims_test[0], dims_test[1], -1], order='F').transpose([2, 0, 1]) * 1.

    idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance_off_on =  \
        cm.base.rois.nf_match_neurons_in_binary_masks(masks_gt=maskgt,
                                                      masks_comp=masktest, Cn=Cn, plot_results=False)

    # the pearson's correlation coefficient of the two Calcium activities thresholded
    # comparing Calcium activities of all the components that are defined by

    corrs = np.array(
        [scipy.stats.pearsonr(C_gt_thr[gt, :], C_test_thr[comp, :])[0] for gt, comp in zip(idx_tp_gt, idx_tp_comp)])
    # todo, change this test when I will have found why I have one additional neuron

    isdiff = True if ((np.linalg.norm(corrs) < sensitivity) or (performance_off_on['f1_score'] < 0.98)) else False
    info = {
        'isdifferent': int(isdiff),
        'diff_data': {
            'performance': performance_off_on,
            'corelations': corrs.tolist(),
                                                       #performance = dict()
                                                       #performance['recall'] = TP/(TP+FN)
                                                       #performance['precision'] = TP/(TP+FP)
                                                       #performance['accuracy'] = (TP+TN)/(TP+FP+FN+TN)
                                                       #performance['f1_score'] = 2*TP/(2*TP+FP+FN)
            'diffneur': diffneur
        },
        'diff_timing': timer
    }
    return info


def plotrig(init, curr, timer, sensitivity):

    diff = np.linalg.norm(np.asarray(init) - np.asarray(curr)) / np.linalg.norm(init)
    isdiff = diff > sensitivity
    info = {'isdifferent': int(isdiff), 'diff_data': diff, 'diff_timing': timer}
    curr = np.asarray(curr).transpose([1, 0])
    init = init.transpose([1, 0])
    xc = np.arange(curr.shape[1])
    xi = np.arange(init.shape[1])
    #try:
    #    pl.figure()
    #    pl.subplot(1, 2, 1)
    #    pl.plot(xc, curr[0], 'r', xi, init[0], 'b')
    #    pl.legend(['x shifts curr', 'x shifts init'])
    #    pl.xlabel('frames')
    #    pl.ylabel('pixels')
    #    pl.subplot(1, 2, 2)
    #    pl.plot(xc, curr[1], 'r', xi, init[1], 'b')
    #    pl.legend(['yshifts curr', 'y shifts init'])
    #    pl.xlabel('frames')
    #    pl.ylabel('pixels')
    #except:
    #    logging.warning("not able to plot")
    return info


def normalised_compare_cnmpatches(a, b):
    # This is designed to copy with fields that make it into these objects that need some normalisation before they
    # are rightly comparable. To deal with that we do a deepcopy and then inline-normalise. Add any new needed keys
    # into this code. Right now this is manual, but if we need to do more of this we should turn this into a nice
    # list of fields to ignore, and another list of fields to perform regular transforms on.
    mutable_a = copy.deepcopy(a)
    mutable_b = copy.deepcopy(b)

    if 'params' in mutable_a and 'params' in mutable_b:
        params_a = mutable_a['params']
        params_b = mutable_b['params']
        if hasattr(params_a, 'online') and hasattr(params_b, 'online'):
            if 'path_to_model' in params_a.online and 'path_to_model' in params_b.online:
                _, params_a.online['path_to_model'] = os.path.split(
                    params_a.online['path_to_model'])                  # Remove all but the last part
                _, params_b.online['path_to_model'] = os.path.split(params_b.online['path_to_model'])
                                                                       # print("Normalised A: " + str(params_a.online['path_to_model']))
                                                                       # print("Normalised B: " + str(params_b.online['path_to_model']))

    return mutable_a == mutable_b
