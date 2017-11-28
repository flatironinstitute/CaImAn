""" compare how the elements behave
 
We create a folder ground truth that possess the same thing than the other
in a form of a dictionnary containing nparrays and other info.
the other files contains every test and the name is the date of the test
 
See Also
------------
 
Link 

\image dev/kalfon/img/datacomparison.pdf
\author: jeremie KALFON
\date Created on Tue Jun 30 21:01:17 2015
\copyright GNU General Public License v2.0
\package CaImAn/comparison
"""
#
#\image html dev/kalfon/img/datacomparison.pdf
#\version   1.0
#
#
#


import platform as plt
import datetime
import numpy as np
import os
######## ONLY IF ON TRAVIS ######

#############################
import matplotlib.pyplot as pl
import caiman as cm
import scipy


class Comparison(object):
    """
       Comparison(object): It is used for comparison the modification on the CaImAn program


       You need to compare your modification to the groundtruth and then rename the folder created as "proof" for it to be send in you next push request

       Here it has been made for 3 different functions. for it to compare well you need to set your
       ground truth with the same computer with which you are comparing the files
       class you instanciate to compare the different parts of CaImAn




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
                diffrences
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
            the user to change it manualy

        Methods
        -------
        __init__()
            Initialize the function be instanciating a comparison object

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

        self.comparison = {'rig_shifts': {},
                           'pwrig_shifts': {},
                           'cnmf_on_patch': {},
                           'cnmf_full_frame': {},
                           }

        self.comparison['rig_shifts'] = {
            'ourdata': None,
            'timer': None,
            'sensitivity': 0.001  # the sensitivity USER TO CHOOSE
        }
        # apparently pwrig shift are not used any more and the comparison are useless
        # self.comparison['pwrig_shifts']={
        #                 'ourdata': None,
        #                'timer': None,
        #               'sensitivity': 0.001
        #           }
        self.comparison['cnmf_on_patch'] = {
            'ourdata': None,
            'timer': None,
            'sensitivity': 0.01
        }
        self.comparison['cnmf_full_frame'] = {
            'ourdata': None,
            'timer': None,
            'sensitivity': 0.01
        }

        self.cnmpatch = None
        self.information = None
        self.dims = None

    def save_with_compare(self, istruth=False, params=None, dview=None, Cn=None):
        """save the comparison as well as the images of the precision recall calculations


            depending on if we say this file will be ground truth or not, it wil be saved in either the tests or the groung truth folder
            if saved in test, a comparison to groundtruth will be add to the object 
            this comparison will be on 
                data : a normized difference of the normalized value of the arrays
                time : difference
            in order for this function to work, you need to
                previously give it the cnm objects after initializing them ( on patch and full frame)
                give the values of the time and data 
                have a groundtruth


            Parameters:
            -----------

            self:  dictionnary
               the object of this class tha tcontains every value

            istruth: Boolean
                if we want it ot be the ground truth

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
                ---------

            Example of utilisation on Demo Pipeline
\image caiman/tests/comparison/data.pdf


             Raise:
             ------

             ('we now have ground truth\n')

             ('we were not able to read the file to compare it\n')

                """
        # getting the DATA FOR COMPARISONS
        assert (params != None and self.cnmpatch != None)
        print('we need the paramters in order to save anything\n')
        # actions on the sparse matrix
        cnm = self.cnmpatch.__dict__
        cnmpatch = deletesparse(cnm)

        # initialization
        dims_test = [self.dims[0], self.dims[1]]
        dims_gt = dims_test
        dt = datetime.datetime.today()
        dt = str(dt)
        plat = plt.platform()
        plat = str(plat)
        pro = plt.processor()
        pro = str(pro)
        # we store a big file which is containing everything ( INFORMATION)
        information = {
            'platform': plat,
            'time': dt,
            'processor': pro,
            'params': params,
            'cnmpatch': cnmpatch,
            'timer': {
                'cnmf_on_patch': self.comparison['cnmf_on_patch']['timer'],
                'cnmf_full_frame': self.comparison['cnmf_full_frame']['timer'],
                'rig_shifts': self.comparison['rig_shifts']['timer']
            }

        }

        rootdir = os.path.abspath(cm.__path__[0])[:-7]
        file_path = rootdir + "/caiman/tests/comparison/groundtruth.npz"

        # OPENNINGS
        # if we want to set this data as truth
        if istruth:
                # we just save it
            if os._exists(file_path):
                os.remove(file_path)
                print("nothing to remove\n")
            np.savez(file_path, information=information, A_full=self.comparison['cnmf_full_frame']['ourdata'][0],
                     C_full=self.comparison['cnmf_full_frame']['ourdata'][
                         1], A_patch=self.comparison['cnmf_on_patch']['ourdata'][0],
                     C_patch=self.comparison['cnmf_on_patch']['ourdata'][1], rig_shifts=self.comparison['rig_shifts']['ourdata'])
            #np.savez('comparison/groundtruth/groundtruth.npz', **information)
            print('we now have ground truth\n')
            return

        else:  # if not we create a comparison first
            try:
                with np.load(file_path, encoding='latin1') as dt:
                    rig_shifts = dt['rig_shifts'][()]
                    A_patch = dt['A_patch'][()]
                    A_full = dt['A_full'][()]
                    C_full = dt['C_full'][()]
                    C_patch = dt['C_patch'][()]
                    data = dt['information'][()]
            except (IOError, OSError):  # if we cannot manage to open it or it doesnt exist:
                # we save but we explain why there were a problem
                print('we were not able to read the file to compare it\n')
                file_path = "comparison/tests/NC"+dt+".npz"
                np.savez(file_path, information=information, A_full=self.comparison['cnmf_full_frame']['ourdata'][0],
                         C_full=self.comparison['cnmf_full_frame']['ourdata'][
                             1], A_patch=self.comparison['cnmf_on_patch']['ourdata'][0],
                         C_patch=self.comparison['cnmf_on_patch']['ourdata'][1], rig_shifts=self.comparison['rig_shifts']['ourdata'])
                return
        # creating the FOLDER to store our data
        i = 0
        dr = rootdir + '/caiman/tests/comparison/tests/'
        for name in os.listdir(dr):
            i += 1
        i = str(i)
        if not os.path.exists(dr+i):
            os.makedirs(dr+i)
        information.update({'diff': {}})
        information.update({'differences': {
            'proc': False,
            'params_movie': False,
            'params_cnm': False}})
        # INFORMATION FOR THE USER
        if data['processor'] != information['processor']:
            print("you don't have the same processor than groundtruth.. the time difference can vary"
                  " because of that\n try recreate your own groundtruth before testing\n")
            information['differences']['proc'] = True
        if data['params'] != information['params']:
            print("you do not use the same movie parameters... Things can go wrong\n\n")
            print('you need to use the same paramters to compare your version of the code with '
                  'the groundtruth one. look for the groundtruth paramters with the see() method\n')
            information['differences']['params_movie'] = True
        if data['cnmpatch'] != cnmpatch:
            if data['cnmpatch'].keys() != cnmpatch.keys():
                print('DIFFERENCES IN THE FIELDS OF CNMF')
                print(set(cnmpatch.keys()) - set(data['cnmpatch'].keys()))
                print(set(data['cnmpatch'].keys()) - set(cnmpatch.keys()))
            diffkeys = [k for k in data['cnmpatch'] if data['cnmpatch'][k] != cnmpatch[k]]
            for k in diffkeys:
                print(k, ':', data['cnmpatch'][k], '->', cnmpatch[k])

            print('you do not use the same paramters in your cnmf on patches initialization\n')
            information['differences']['params_cnm'] = True

        # for rigid
        # plotting part

        information['diff'].update({
            'rig': plotrig(init=rig_shifts, curr=self.comparison['rig_shifts']['ourdata'], timer=self.comparison['rig_shifts']['timer']-data['timer']['rig_shifts'], sensitivity=self.comparison['rig_shifts']['sensitivity'])})
        try:
            pl.gcf().savefig(dr+str(i)+'/'+'rigidcorrection.pdf')
            pl.close()
        except:
            print("\n")

        # for cnmf on patch
        information['diff'].update({
            'cnmpatch': cnmf(Cn=Cn, A_gt=A_patch,
                             A_test=self.comparison['cnmf_on_patch']['ourdata'][0],
                             C_gt=C_patch,
                             C_test=self.comparison['cnmf_on_patch']['ourdata'][1],
                             dview=dview, sensitivity=self.comparison[
                                 'cnmf_on_patch']['sensitivity'],
                             dims_test=dims_test, dims_gt=dims_gt,
                             timer=self.comparison['cnmf_on_patch']['timer']-data['timer']['cnmf_on_patch'])})
        try:
            pl.gcf().savefig(dr+i+'/'+'onpatch.pdf')
            pl.close()
        except:
            print("\n")


# CNMF FULL FRAME
        information['diff'].update({
            'cnmfull': cnmf(Cn=Cn, A_gt=A_full,
                            A_test=self.comparison['cnmf_full_frame']['ourdata'][0],
                            C_gt=C_full,
                            C_test=self.comparison['cnmf_full_frame']['ourdata'][1],
                            dview=dview, sensitivity=self.comparison[
                                'cnmf_full_frame']['sensitivity'],
                            dims_test=dims_test, dims_gt=dims_gt,
                            timer=self.comparison['cnmf_full_frame']['timer']-data['timer']['cnmf_full_frame'])})
        try:
            pl.gcf().savefig(dr+i+'/'+'cnmfull.pdf')
            pl.close()
        except:
            print("\n")

# SAving of everything
        file_path = rootdir + "/caiman/tests/comparison/tests/"+i+"/"+i+".npz"
        np.savez(file_path, information=information, A_full=self.comparison['cnmf_full_frame']['ourdata'][0],
                 C_full=self.comparison['cnmf_full_frame']['ourdata'][
                     1], A_patch=self.comparison['cnmf_on_patch']['ourdata'][0],
                 C_patch=self.comparison['cnmf_on_patch']['ourdata'][1], rig_shifts=self.comparison['rig_shifts']['ourdata'])

        self.information = information


def see(filename=None):
    """shows you the important data about a certain test file ( just give the number or name)

        if you give nothing it will give you back the groundtruth infos

        Parameters:
        -----------
        self:  dictionnary
           the object of this class tha tcontains every value
        filename:
            ( just give the number or name)

        See Also:
        ---------
        @image html caiman/tests/comparison/data.pdf

            """

    if filename == None:
        dr = './caiman/tests/comparison/groundtruth.npz'
    else:
        dr = os.path.abspath(cm.__path__[0]) + '/tests/comparison/tests/'
        dr = dr+filename+'/'+filename+'.npz'

        print(dr)
    with np.load(dr) as dt:
        print('here is the info :\n')
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
            print(type(val))
            cnm[keys] = val
        else:

            cnm[keys] = None
    return cnm


def cnmf(Cn, A_gt, A_test, C_gt, C_test, dims_gt, dims_test, dview=None, sensitivity=0, timer=0, n_frames_per_bin=10):

    A_test = A_test.toarray()  # coo sparse matrix
    A_gt = A_gt.toarray()

   # proceed to a trhreshold
    A_test_thr = cm.source_extraction.cnmf.spatial.threshold_components(
        A_test, dims_test, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
        se=None, ss=None, dview=dview)
    A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(
        A_gt, dims_gt, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
        se=None, ss=None, dview=dview)

    # compute C using this A thr
    A_test_thr = A_test_thr > 0
    A_gt_thr = A_gt_thr > 0
    # we do not compute a threshold on the size of neurons
    C_test_thr = C_test
    C_gt_thr = C_gt
    # we would also like the difference in the number of neurons
    diffneur = A_test_thr.shape[1] - A_gt_thr.shape[1] 
#    print(diffneur+1)
    # computing the values
    C_test_thr = np.array([CC.reshape([-1, n_frames_per_bin]).max(1) for CC in C_test_thr])
    C_gt_thr = np.array([CC.reshape([-1, n_frames_per_bin]).max(1) for CC in C_gt_thr])
    maskgt = A_gt_thr[:, :].reshape([dims_gt[0], dims_gt[1], -1],
                                    order='F').transpose([2, 0, 1])*1.
    masktest = A_test_thr[:, :].reshape(
        [dims_test[0], dims_test[1], -1], order='F').transpose([2, 0, 1])*1.

    idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance_off_on =  \
        cm.base.rois.nf_match_neurons_in_binary_masks(masks_gt=maskgt,
                                                      masks_comp=masktest, Cn=Cn, plot_results=True)

    # the pearson's correlation coefficient of the two Calcium activities thresholded
    # comparing Calcium activities of all the components that are defined by
    
    corrs = np.array([scipy.stats.pearsonr(
        C_gt_thr[gt, :], C_test_thr[comp, :])[0] for gt, comp in zip(idx_tp_gt, idx_tp_comp)])
    # todo, change this test when I will have found why I have one additionnal neuron

    isdiff = True if ((np.linalg.norm(corrs) < sensitivity) or (performance_off_on['f1_score']<0.98)) else False
    info = {'isdifferent': int(isdiff),
            'diff_data': {'performance': performance_off_on,
                          'corelations': corrs.tolist(),
                          #performance = dict()
                          #performance['recall'] = old_div(TP,(TP+FN))
                          #performance['precision'] = old_div(TP,(TP+FP))
                          #performance['accuracy'] = old_div((TP+TN),(TP+FP+FN+TN))
                          #performance['f1_score'] = 2*TP/(2*TP+FP+FN)
                          'diffneur': diffneur},
            'diff_timing': timer}
    return info


def plotrig(init, curr, timer, sensitivity):

    diff = np.linalg.norm(np.asarray(init)-np.asarray(curr))/np.linalg.norm(init)
    isdiff = diff > sensitivity
    info = {'isdifferent': int(isdiff),
            'diff_data': diff,
            'diff_timing': timer}
    curr = np.asarray(curr).transpose([1, 0])
    init = init.transpose([1, 0])
    xc = np.arange(curr.shape[1])
    xi = np.arange(init.shape[1])
    try:
        pl.figure()
        pl.subplot(1, 2, 1)
        pl.plot(xc, curr[0], 'r', xi, init[0], 'b')
        pl.legend(['x shifts curr', 'x shifts init'])
        pl.xlabel('frames')
        pl.ylabel('pixels')
        pl.subplot(1, 2, 2)
        pl.plot(xc, curr[1], 'r', xi, init[1], 'b')
        pl.legend(['yshifts curr', 'y shifts init'])
        pl.xlabel('frames')
        pl.ylabel('pixels')
    except:
        print("not able to plot")
    return info
