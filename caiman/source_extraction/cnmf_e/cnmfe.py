# -*- coding: utf-8 -*-

"""
Constrained Nonnegative Matrix Factorization for microEndoscopic data (CNMF-E)

author: Pengcheng Zhou
email: zhoupc1988@gmail.com
created: 6/15/17
last edited:
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from collections import OrderedDict
import json
import cv2

import caiman as cm
from caiman.gui import caiman_gui as cmg

"""
--------------------------------CLASSES--------------------------------
"""


class Sources2D(object):
    """
    Python class for pre-processing 2D calcium imaging data. It distinguish
    with the plain CNMF in the way of modeling background components

    References:
        Zhou, P., Resendez, S.L., Rodriguez-Romaguera, J., Jimenez, J.C.,
        Neufeld, S.Q., Stuber, G.D., Hen, R., Kheirbek, M.A., Sabatini,
        B.L., Kass, R.E., Paninski, L. (2016). Efficient and accurate
        extraction of in vivo calcium signals from microendoscopic video
        data. arXiv Prepr, arXiv1605.07266.
    """

    def __init__(self, pars4user=None, *args):
        """
        initialize model variables and parameters
        Args:
            pars4user: (dict or str)
                parameters stored in a dict and in a way of user-friendly
                organizing. if it's a str, then it's the path to load a json
                file containing all parameters

                It has to be assigned when initializing a Sources2D
                instance. set it as None if you want to use the default
                options.
            *args: (tuple)
                each element in the tuple has the format of 'x=y', where x
                is the option used in Sources2D and y is the corresponding
                value

                it allows flexible passing of model parameters.
        """
        self.pars4user = default_pars()
        if not pars4user:
            pars4user = self.pars4user

        self.options = type('', (), {})()
        self.options = self.unpack_pars(pars4user)
        # parse args
        for s in args:
            exec('self.options.'+s)
        self.pars4user = self.pack_pars(self.options)

        # results of source extraction
        self.A = None       # neuron shapes
        self.C = None       # neuron activities (denoised)
        self.C_raw = None   # neuron activities (nodenoising)
        self.S = None       # spiking events
        self.b = None       # spatial components of background sources
        self.f = None       # temporal components of temporal sources
        self.W = None       # weight matrix in ring model of background

        # other parameters
        self.options.block_size = 20000
        self.options.check_nan = True
        self.options.skip_refinement = False
        self.options.normalize_init = True
        self.options.options_local_NMF = None
        self.options.remove_very_bad_comps = False

        # intermediate variables during CNMF-E running
        self.P = type('', (), {})()
        self.P.sn = None
        self.P.pars_ca = None

    def set_envs(self):
        """
        Setting up computing environment

        Returns:

        """
        print('\n'+'-'*25+'Setting up computing environment'+'-'*25)

        self.options.client, self.options.dview, n_processes = \
            cm.cluster.setup_cluster(backend=self.options.backend,
                                     n_processes=self.options.n_processes,
                                     single_thread=self.options.single_thread)
        self.pack_pars()

        print('-'*25+'Done'+'-'*25+'\n')
        return self.options.client, self.options.dview

    def load_file(self, file_name=None, frame_rate=10, pixel_size=None):
        """
        load movie data into caiman and update related parameters
        Args:
            file_name: str,
                    full file path
            frame_rate: float, unit-Hz
            pixel_size: list with 1 or two elements
                    if len(pixel_size)==1, then the pixel size is equal in x and y direction;
                    else:  (pixel_x, pixel_y)

        Returns:
            video_data: T*d1*d2 matrix
                    T is the frame number, d1 is the row number, d2 is the
                    column vector
        """
        if not file_name:
            print('You must specify the file name!')
            return

        print('\n'+'-'*25+'Loading data'+'-'*25)

        # file information
        file_name = os.path.realpath(file_name)
        self.options.file_name = file_name
        self.options.dir, temp = os.path.split(file_name)
        self.options.name, self.options.type = os.path.splitext(temp)
        self.options.Fs = frame_rate
        self.options.pixel_size = pixel_size

        # load data and acquire more information
        video_data = cm.load(self.options.file_name)
        total_frame, nrow, ncol = video_data.shape
        self.options.T = total_frame
        self.options.d1 = nrow
        self.options.d2 = ncol
        print("Dimension: %d X %d pixels X %d frames" % (nrow, ncol, total_frame))

        # create a folder to save caiman results
        self.options.dir_result = os.path.join(self.options.dir,
                                               self.options.name + '_caiman')
        if os.path.exists(self.options.dir_result):
            print("The result folder has been created")
        else:
            os.mkdir(self.options.dir_result)
        print("The results will be saved into folder %s\n"
              % self.options.dir_result)

        # update parameters and write down log information
        self.pack_pars()
        file_pars = self.save_parameters()

        # create a log file for keeping the whole analysis
        log_file = os.path.join(self.options.dir_result, 'log_' +
                                datetime.now().strftime('%Y-%m-%d-%H%M%S')+'.log')

        with open(log_file, 'w') as f:
            # write logs
            self.pars4user['export']['log file'] = log_file
            f.write('-'*25+"File has been successfully loaded"+'-'*25+'\n')
            f.write('--File name: %s\n' % self.options.file_name)
            f.write("--Parameters: %s\n" % file_pars)
            f.write('--Log file: %s\n\n' % log_file)

        self.options.export = self.pars4user['export']

        print('-'*25+'Done'+'-'*25+'\n')
        return video_data

    def update_options(self, *args):
        """
        update options

        Args:
            *args: option items to be updated.
                for example, when you want to update an option named as 'a'
                and want to set its value to '10', then you can call
                function as
                    self.update_options('a=10');
                similarly, if you also want to update b to 1, then call
                function as
                    self.update_options('a=10', 'b=1')

        Returns:

        """
        for s in args:
            exec('self.options.'+s)
        self.pars4user = self.pack_pars(self.options)

    def update_packed_pars(self, *args):
        """
        update packed parameters

        Args:
            *args: parameters to be updated.
                example: if you want to update pars4user['neuron', 'neuron
                diameter'] as 15, you can call function as
                    self.update_packed_pars("{'neuron': {'neuron diameter': 15
                    }}")

        Returns:

        """
        for s in args:
            temp = eval(s)
            for key1 in temp.keys():
                if key1 not in self.pars4user.keys():
                    continue

                if not isinstance(temp[key1], dict):
                    self.pars4user[key1] = temp[key1]
                else:
                    for key2 in temp[key1].keys():
                        if key2 not in self.pars4user[key1].keys():
                            continue
                        if not isinstance(temp[key1][key2], dict):
                            self.pars4user[key1][key2] = temp[key1][key2]
                        else:
                            for key3 in temp[key1][key2].keys():
                                if key3 not in self.pars4user[key1][key2].keys():
                                    continue
                                else:
                                    self.pars4user[key1][key2][key3] = temp[
                                        key1][key2][key3]

        self.options = self.unpack_pars(self.pars4user)

    def ui_get_file(self, directory=None):
        """
        select data using GUI
        Args:
            directory: str
                    directory of the data

        Returns:
            video_data: T*d1*d2
                    same as load_data
        """
        if not directory:
            self.options.dir = directory
        if not self.options.dir:
            self.options.dir = os.getcwd()

        file_info = cmg.open_file(directory=self.options.dir)
        if (not file_info['file_name']) or (not os.path.exists(file_info[
                                                                 'file_name'])):
            # no file selected
            print("No valid data file was selected!\n")
            return
        else:
            return self.load_file(file_name=file_info['file_name'],
                                  frame_rate=file_info['Fs'],
                                  pixel_size=file_info['pixel_size'])

        # def run_motion_correction(self):
        # run motion correction

        # def run_initialization(self):
        #     # initializing neurons from the data:
        #
        # def update_spatial(self):
        #     # update sptial components in the data
        #
        # def update_temporal(self):
        #     # update  temporal components in the data
        #
        # def update_background(self):
        #     # update background
        #
        # def view_neurons(self):
        #     # visualize all neurons and do manual interventions
        #
        # def view_neurons_GUI(self):
        #     # gui mode for viewing neurons
        # def gen_video(self):
        # generate videos

    def unpack_pars(self, pars4user=None):
        """
        unpack the parameter set into options that can be accessed internally
        in CaImAn

        Args:
            pars4user: (None, str, dict)
                organize all parameters in an user-friendly format.
                if it's None, then use the default values
                if it's str, then it refers to a json file saving all pars.
                it it's dict, then it's exactly what we want.

        Returns:
            options: (C-like structure variable)
                it stores parameters in a format that can be easily accessed
                in CaImAn.
        """

        # make sure we have the right input of pars4user
        if not pars4user:
            # not defined, use default values
            pars4user = default_pars()
        elif isinstance(pars4user, str):
            # use json file as input, load the file first
            _, file_extension = os.path.splitext(pars4user)
            if os.path.exists(pars4user) and (file_extension.lower() == '.json'):
                with open('pars4user', 'r') as f:
                    pars4user = OrderedDict(json.load(f))
            else:
                pars4user = default_pars()

        options = self.options

        # computing environment
        pars_envs = pars4user['envs']
        options.backend = pars_envs['backend']
        options.client = pars_envs['client']
        options.dview = pars_envs['direct view']
        options.n_processes = pars_envs['processes number']
        options.single_thread = pars_envs['single thread']
        options.memory_fact = pars_envs['memory factor']

        # data information
        pars_data = pars4user['data']
        options.file_name = pars_data['file path']
        options.dir = pars_data['dir name']
        options.name = pars_data['name']
        options.type = pars_data['file type']
        options.dir_result = pars_data['result folder']
        options.fr = pars_data['frame rate']
        options.T = pars_data['frame number']
        options.d1 = pars_data['row number']
        options.d2 = pars_data['column number']
        options.d3 = pars_data['z planes']
        options.pixel_size = pars_data['pixel size']

        # parameters related to neurons
        pars_neuron = pars4user['neuron']
        options.gSig = pars_neuron['gaussian width']
        options.gSiz = pars_neuron['neuron diameter']
        options.merge_thresh = pars_neuron['merge threshold']  # merging
        # threshold,
        options.do_merge = pars_neuron['do merge']
        options.alpha_snmf = pars_neuron['alpha (sparse NMF)']
        options.merge_method = pars_neuron['merge method']
        options.merge_dmin = pars_neuron['minimum distance']

        # motion correction
        options.pars_motion = pars4user['motion']  # save the whole
        # pars4user for loading in motion correction package

        # initialization
        pars_initialization = pars4user['initialization']
        options.use_patch_init = pars_initialization['use patch']
        options.method_init = pars_initialization['method']
        options.seed_method = pars_initialization['seed method']
        options.K = pars_initialization['maximum neuron number']
        options.min_corr = pars_initialization['minimum local correlation']
        options.min_pnr = pars_initialization['minimum peak-to-noise ratio']
        options.thresh_init = pars_initialization['z-score threshold']

        # spatial components
        pars_spatial = pars4user['spatial']
        options.use_patch = pars_spatial['use patch']
        options.ssub = pars_spatial['spatial downsampling factor']
        options.p_ssub = pars_spatial['spatial downsampling factor (patch)']
        options.rf = pars_spatial['patch size']/2.0
        options.stride = pars_spatial['overlap size']
        options.n_pixels_per_process = pars_spatial['n pixels per process']
        options.min_pixel = pars_spatial['minimum number of nonzero pixels']
        options.bd_width = pars_spatial['boundary width']

        # temporal components
        pars_temporal = pars4user['temporal']
        options.tsub = pars_temporal['temporal downsampling factor']
        options.p_tsub = pars_temporal['temporal downsampling factor (patch)']
        options.run_deconvolution = pars_temporal['run deconvolution']
        options.temporal_algorithm = pars_temporal['algorithm']
        options.temporal_max_iter = pars_temporal['iteration number']
        options.pars_deconvolution = pars_temporal['deconvolution options']

        # background components
        pars_bg = pars4user['background']
        options.gnb = pars_bg['background rank']
        options.bg_size_factor = pars_bg['size factor']
        options.bg_ds_factor = pars_bg['downsampling factor']

        # export results
        options.pars_export = pars4user['export']

        return options

    def pack_pars(self, options=None):
        """
        pack the options used in CaImAn into user-friendly parameter sets

        Args:
            options: (C-like structure variable)
                a variable that can be easily accessed in CaImAn

        Returns:
            pars4user: (dict)
                user-friendly format

        """
        if not options:
            options = self.options

        pars4user = self.pars4user

        pars_envs = pars4user['envs']
        pars_envs['backend'] = options.backend
        pars_envs['processes number'] = options.n_processes
        pars_envs['client'] = options.client
        pars_envs['direct view'] = options.dview
        pars_envs['memory factor'] = options.memory_fact
        pars_envs['single thread'] = options.single_thread

        pars_data = pars4user['data']
        pars_data['file path'] = options.file_name
        pars_data['dir name'] = options.dir
        pars_data['name'] = options.name
        pars_data['file type'] = options.type
        pars_data['frame rate'] = options.fr
        pars_data['frame number'] = options.T
        pars_data['row number'] = options.d1
        pars_data['column number'] = options.d2
        pars_data['z planes'] = options.d3
        pars_data['pixel size'] = options.pixel_size
        pars_data['result folder'] = options.dir_result

        pars4user['motion'] = options.pars_motion

        pars_neuron = pars4user['neuron']
        pars_neuron['neuron diameter'] = options.gSiz
        pars_neuron['gaussian width'] = options.gSig
        pars_neuron['do merge'] = options.do_merge
        pars_neuron['merge method'] = options.merge_method
        pars_neuron['merge threshold'] = options.merge_thresh
        pars_neuron['minimum distance'] = options.merge_dmin
        pars_neuron['alpha (sparse NMF)'] = options.alpha_snmf

        pars_initialization = pars4user['initialization']
        pars_initialization['use patch'] = options.use_patch_init
        pars_initialization['method'] = options.method_init
        pars_initialization['seed method'] = options.seed_method
        pars_initialization['maximum neuron number'] = options.K
        pars_initialization['minimum local correlation'] = options.min_corr
        pars_initialization['minimum peak-to-noise ratio'] = options.min_pnr
        pars_initialization['z-score threshold'] = options.thresh_init

        pars_spatial = pars4user['spatial']
        pars_spatial['use patch'] = options.use_patch
        pars_spatial['patch size'] = options.rf * 2.0
        pars_spatial['overlap size'] = options.stride
        pars_spatial['spatial downsampling factor'] = options.ssub
        pars_spatial['spatial downsampling factor (patch)'] = options.p_ssub
        pars_spatial['n pixels per process'] = options.n_pixels_per_process
        pars_spatial['minimum number of nonzero pixels'] = options.min_pixel
        pars_spatial['boundary width'] = options.bd_width

        pars_deconvolution = options.pars_deconvolution

        pars_temporal = pars4user['temporal']
        pars_temporal['run deconvolution'] = options.run_deconvolution
        pars_temporal['deconvolution options'] = pars_deconvolution
        pars_temporal['algorithm'] = options.temporal_algorithm
        pars_temporal['temporal downsampling factor'] = options.tsub
        pars_temporal['temporal downsampling factor (patch)'] = options.p_tsub
        pars_temporal['iteration number'] = options.temporal_max_iter

        pars_background = pars4user['background']
        pars_background['size factor'] = options.bg_size_factor
        pars_background['downsampling factor'] = options.bg_ds_factor
        pars_background['background rank'] = options.gnb

        pars4user['export'] = options.pars_export

        return pars4user

    def print_parameters(self):
        """
        show parameter options

        Returns:

        """
        print_parameters(self.pars4user)

    def save_parameters(self):
        """
        save the current values of parameters into json file

        Returns:
            file_pars: locations of the saved json file for storing parameters

        """
        self.pars4user = self.pack_pars()  # pack parameters
        file_pars = os.path.join(self.options.dir_result, 'pars_' +
                                 datetime.now().strftime('%Y-%m-%d-%H%M%S')
                                 + '.json')
        with open(file_pars, 'w') as f:
            json.dump(self.pars4user, f)

        self.pars4user['export']['saved parameter files'].append(file_pars)
        self.options.export = self.pars4user['export']

        return file_pars

    def gen_filter_kernel(self, width=None, sigma=None, center=True):
        """
        create a gaussian kernel for spatially filtering the raw data

        Args:
            width: (float)
                width of the kernel
            sigma: (float)
                gaussian width of the kernel
            center:
                if True, subtract the mean of gaussian kernel

        Returns:
            psf: (2D numpy array, width x width)
                the desired kernel

        """
        if not width:
            width = self.options.gSiz
            sigma = self.options.gSig
        rmax = (width-1) / 2.0
        y, x = np.ogrid[-rmax:(rmax+1), -rmax:(rmax+1)]
        psf = np.exp(-(x*x + y*y) / (2*sigma*sigma))
        psf = psf / psf.sum()
        if center:
            idx = (psf >= psf[0].max())
            psf[idx] -= psf[idx].mean()
            psf[~idx] = 0

        return psf

    def reshape(self, data, mode=None):
        """
        change the way of representing video data
        Args:
            data: 2d array or 3d array
                if it's 2d, each image is represented as a vector
                if it's 3d, each image is representd as a 2d matrix
            mode: {1, 2}
                if ndim=1, then each image is a vector;
                if ndim=2, then each image is a 2d matrix

        Returns:
            video data with the desired shape
        """
        if not mode:
            mode = 1 if (data.ndim == 2) else 2

        if mode == 2:
            return data.reshape(-1, self.options.d1, self.options.d2).squeeze()
        else:
            return data.reshape(-1, self.options.d1*self.options.d2).squeeze()

    def image(self, img, vmin=None, vmax=None, cmap='jet', axis=True,
              colorbar=True):
        """
        show an image whose size is the same as video data

        Args:
            img: 1-d array (d1*d2, ) or 2-d array (d1, d2)
                the image to be displayed
            vmin: float
                values below vmin will be displayed as 0
            vmax: float
                values ablove vmax will be displayed as 1
            cmap: str
                colormap options. see
                https://matplotlib.org/examples/color/colormaps_reference
                .html for more colormap options.
            axis: bool
                axis on or off
            colorbar: bool
                show colorbar or not

        Returns:
            ax: axes handle

        """
        img = self.reshape(img, 2)
        # plt.figure()
        ax = plt.axes()
        plt.imshow(img, aspect='equal', vmin=vmin, vmax=vmax, cmap=cmap)
        if not axis:
            plt.axis('off')
        if colorbar:
            plt.colorbar()
        plt.show()

        return ax
"""
-------------------------------FUNCTIONS-------------------------------
"""


def print_parameters(pars=None):
    if pars:
        for i in pars:
            if isinstance(pars[i], dict):
                print(i)
                for j in pars[i]:
                    if isinstance(pars[i][j], dict):
                        print('\t', j)
                        for k in pars[i][j]:
                            if isinstance(pars[i][j][k], dict):
                                print('\t\t', k)
                                for l in pars[i][j][k]:
                                    print('\t\t\t', l, pars[i][j][k][l])
                            else:
                                print('\t\t', k, '=', pars[i][j][k])
                    else:
                        print('\t', j, '  = ', pars[i][j])
            else:
                print(i, pars[i])


def default_pars():
    """
    default parameters used in CaImAn

    Returns:
        pars2user: (dict)
            all parameters saved in a dictionary. The keys describe the
            meaning of each parameter.

    """
    pars_envs = OrderedDict([('backend', 'local'),  # {'local', 'SLURM'}
                             ('processes number', None),
                             ('client', None),
                             ('direct view', None),
                             ('memory factor', None),
                             ('single thread', False)
                             ])
    pars_data = OrderedDict([('file path', None),
                             ('dir name', None),
                             ('name', None),
                             ('file type', None),     # {'avi', 'tif', 'hdf5'}
                             ('frame rate', None),
                             ('frame number', None),
                             ('row number', None),
                             ('column number', None),
                             ('z planes', None),
                             ('pixel size', None),
                             ('result folder', None)
                             ])
    pars_motion_parallel = OrderedDict([('splits_rig', 28),
                                        ('num_splits_to_process_rig', None),
                                        ('splits_els', 28),
                                        ('num_splits_to_process_els', [14, None])
                                        ])
    pars_motion = OrderedDict([('run motion correction', True),
                               ('niter_rig', 1),
                               ('max_shifts', (6, 6)),
                               ('num_splits_to_process_rig', None),
                               ('strides', (48, 48)),
                               ('overlaps', (24, 24)),
                               ('upsample_factor_grid', 4),
                               ('max_deviation_rigid', 3),
                               ('shifts_opencv', True),
                               ('min_mov', 0),
                               ('nonneg_movie', False),
                               ('parallel', pars_motion_parallel)
                               ])
    pars_neuron = OrderedDict([('neuron diameter', 16),   # unit: pixel
                               ('gaussian width', 4),     # unit: pixel
                               ('do merge', True),
                               ('merge method', ['correlation', 'distance']),  # can use only one method
                               ('merge threshold', [0.001, 0.85]),  # [minimum spatial correlation, minimum temporal correlation]
                               ('minimum distance', 3),   # unit: pixel,
                               ('alpha (sparse NMF)', None)
                               ])

    pars_initialization = OrderedDict([('use patch', True),
                                       ('method', 'greedyROI'),  # {'greedyROI','greedyROI_corr','greedyROI_endoscope', 'sparseNMF'}
                                       ('seed method', 'auto'),  # {'auto', 'manual'}
                                       ('maximum neuron number', 5),
                                       ('minimum local correlation', 0.85),
                                       ('minimum peak-to-noise ratio', 10),
                                       ('z-score threshold', 1)
                                       ])
    pars_spatial = OrderedDict([('use patch', True),
                                ('patch size', 64),
                                ('overlap size', None),
                                ('spatial downsampling factor', 1),
                                ('spatial downsampling factor (patch)', 1),
                                ('n pixels per process', 4000),
                                ('minimum number of nonzero pixels', 1),
                                ('boundary width', 0)
                                ])
    pars_deconvolution = OrderedDict([('model', 'ar2'),  # {'ar1', 'ar2'}
                                      ('method', 'constrained foopsi'),  # {'foopsi','constrained foopsi','threshold foopsi'}
                                      ('algorithm', 'oasis')  # {'oasis', # 'cvxpy'}
                                      ])
    pars_temporal = OrderedDict([('run deconvolution', True),
                                 ('deconvolution options', pars_deconvolution),
                                 ('algorithm', 'hals'),
                                 ('temporal downsampling factor', 1),
                                 ('temporal downsampling factor (patch)', 1),
                                 ('iteration number', 2)
                                 ])
    pars_background = OrderedDict([('size factor', 1.5),  # {size factor} = # {The radius of the ring}/{neuron diameter}
                                   ('downsampling factor', 2),
                                   ('background rank', 1),    # number of  background components
                                   ])
    pars_export = OrderedDict([('saved parameter files', []),
                               ('log file', None)])
    pars4user = OrderedDict([('envs', pars_envs),
                             ('data', pars_data),
                             ('neuron', pars_neuron),
                             ('motion', pars_motion),
                             ('initialization', pars_initialization),
                             ('spatial', pars_spatial),
                             ('temporal', pars_temporal),
                             ('background', pars_background),
                             ('export', pars_export)
                             ])

    return pars4user


def local_correlation(video_data, sz=None, d1=None, d2=None,
                      normalized=False, chunk_size=3000):
    """
    compute location correlations of the video data
    Args:
        video_data: T*d1*d2 3d array  or T*(d1*d2) matrix
        sz: method for computing location correlation {4, 8, [dmin, dmax]}
            4: use the 4 nearest neighbors
            8: use the 8 nearest neighbors
            [dmin, dmax]: use neighboring pixels with distances in [dmin, dmax]
        d1: row number
        d2: column number
        normalized: boolean
            if True: avoid the step of normalizing data
        chunk_size: integer
            divide long data into small chunks for faster running time

    Returns:
        d1*d2 matrix, the desired correlation image

    """
    total_frames = video_data.shape[0]
    if total_frames > chunk_size:
        # too many frames, compute correlation images in chunk mode
        n_chunk = np.floor(total_frames/chunk_size)

        cn = np.zeros(shape=(n_chunk, d1, d2))
        for idx in np.arange(n_chunk):
            cn[idx, ] = local_correlation(video_data[chunk_size*idx+np.arange(
                                                  chunk_size), ], sz, d1, d2, normalized)
        return np.max(cn, axis=0)

    # reshape data
    data = video_data.copy().astype('float32')

    if data.ndim == 2:
        data = data.reshape(total_frames, d1, d2)
    else:
        _, d1, d2 = data.shape

    # normalize data
    if not normalized:
        data -= np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std[data_std == 0] = np.inf
        data /= data_std

    # construct a matrix indicating the locations of neighbors
    if (not sz) or (sz == 8):
        mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    elif sz == 4:
        mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    elif len(sz) == 2:
        sz = np.array(sz)
        temp = np.arange(-sz.max(), sz.max()+1).reshape(2*sz.max()+1, 0)
        tmp_dist = np.sqrt(temp**2 + temp.transpose()**2)
        mask = (tmp_dist >= sz.min()) & (tmp_dist < sz.max())
    else:
        mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # compute local correlations
    data_filter = data.copy().astype('float32')
    for idx, img in enumerate(data_filter):
        data_filter[idx, ] = cv2.filter2D(img, -1, mask, borderType=0)

    return np.divide(np.mean(data_filter*data, axis=0), cv2.filter2D(
        np.ones(shape=(d1, d2)), -1, mask, borderType=1))


def correlation_pnr_endoscope(data, options=None):
    """
    compute the correlation image and the peak-to-noise ratio (PNR) image for
    microendoscopic data

    Args:
        data: 2d or 3d numpy array.
            video data
        options: C-like struct variable
            it requires at least 5 fields: d1, d2, gSiz, gSig, thresh_init

    Returns:
        cn: d1*d2 matrix
            local correlation image
        pnr: d1*d2 matrix
            PNR image
        psf: gSiz*gSiz matrix
            kernel used for filtering data

    """
    neuron = Sources2D()  # create a Sources2D class instance for using
    # its # methods
    if not options:
        # use the default options
        options = neuron.unpack_pars()
    else:
        neuron.options = options

    # parameters
    d1 = options.d1
    d2 = options.d2
    sig = options.thresh_init

    data_raw = data.reshape(-1, d1, d2).astype('float32')

    # create a spatial filter for removing background
    psf = neuron.gen_filter_kernel(center=True)

    # filter data
    data_filtered = data_raw.copy()
    for idx, img in enumerate(data_filtered):
        data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)

    # compute peak-to-noise ratio
    data_filtered -= np.median(data_filtered, axis=0)
    data_max = np.max(data_filtered, axis=0)
    data_std = get_noise(data_filtered, method='diff2_med')
    pnr = np.divide(data_max, data_std)
    pnr[pnr < 0] = 0

    # remove small values
    tmp_data = data_filtered.copy()/data_std
    tmp_data[tmp_data < sig] = 0

    # compute correlation image
    cn = local_correlation(tmp_data, d1=d1, d2=d2)

    # return
    return cn, pnr, psf


def get_noise(data, method='diff2_med'):
    """
    estimate the noise level of data

    Args:
        data: n dimensional matrix
        method: {'diff2_med', 'fft'}

    Returns:

    """
    if method.lower() == 'diff2_med':
        return np.sqrt(np.mean(np.diff(data, n=2, axis=0)**2,
                               axis=0))/np.sqrt(6)
    elif method.lower() == 'fft':
        print("To do done")
        pass

#
# def greedyROI_endoscope(Y=None, K=None, options=None, debug_on=False,
#                         save_avi=False):
#     neuron = Sources2D()  # create a Sources2D class instance for using its methods
#     if not options:
#         # use the default options
#         options = neuron.unpack_pars()
#     else:
#         neuron.options = options
#
#     # parameters
#     d1 = options.d1
#     d2 = options.d2
#     gSig = options.gSig
#     gSiz = options.gSiz
#     min_corr = options.min_corr
#     min_pnr = options.min_pnr
#     min_v_search = min_corr*min_pnr
#     seed_method = options.seed_method
#     deconv_options_0 = options.pars_deconvolution
#     deconv_flag = options.run_deconvolution
#     min_pixel = options.min_pixel
#     bd = options.bd_width
#     sig = options.thresh_init
#
#     data_raw = Y.reshape(-1, d1, d2).astype('float32')
#     total_frames = Y.shape[0]
#
#     # create a spatial filter for removing background
#     psf = neuron.gen_filter_kernel(center=True)
#
#     # filter data
#     data_filtered = data_raw.copy()
#     for idx, img in enumerate(data_filtered):
#         data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)
#
#     # compute peak-to-noise ratio
#     data_filtered -= np.median(data_filtered, axis=0)
#     data_max = np.max(data_filtered, axis=0)
#     data_std = get_noise(data_filtered, method='diff2_med')
#     pnr = np.divide(data_max, data_std)
#     pnr[pnr < 0] = 0
#     # remove small values
#     tmp_data = data_filtered.copy()/data_std
#     tmp_data[tmp_data < sig] = 0
#
#     # compute correlation image
#     cn = local_correlation(tmp_data, d1=d1, d2=d2)
#     cn0 = cn.copy()
#     cn[np.isnan(cn)] = 0
#
#     # screen seed pixels as neuron centers
#     v_search = cn*pnr
#     v_search[(cn < min_corr) | (pnr < min_pnr)] = 0
#     ind_search = (v_search <= 0)        # indicate whether the pixel has
#     # been searched before. pixels with low correlations or low PNRs are
#     # ignored direction.
#
#     # pixels near the boundaries are ignored because of artifacts
#     bd = neuron.options.bd_width
#     ind_search[:bd, :] = True
#     ind_search[:, :bd] = 0
#     ind_search[-bd:, :] = 0
#     ind_search[:, -bd:] = 0
#
#     if debug_on:
#         # visualize the initialization procedure.
#         h_fig = plt.figure(figsize=(12, 8), facecolor=(0.9, 0.9, 0.9))
#         ax_cn = plt.subplot2grid((2, 3), (0, 0))
#         ax_cn.imshow(cn)
#         ax_cn.set_title('Correlation')
#         ax_cn.set_axis_off()
#
#         ax_pnr_cn = plt.subplot2grid((2, 3), (0, 1))
#         ax_pnr_cn.imshow(cn*pnr)
#         ax_pnr_cn.set_title('Correlation*PNR')
#         ax_pnr_cn.set_axis_off()
#
#         ax_cn_box = plt.subplot2grid((2, 3), (0, 2))
#         ax_cn_box.imshow(cn)
#         ax_cn_box.set_xlim([54, 63])
#         ax_cn_box.set_ylim([54, 63])
#         ax_cn_box.set_title('Correlation')
#         ax_cn_box.set_axis_off()
#
#         ax_traces = plt.subplot2grid((2, 3), (1, 0), colspan=3)
#         ax_traces.set_title('Activity at the seed pixel')
#         plt.ion()
#
#         if save_avi:
#             mpeg_name = os.path.join(options.dir_results, 'initialization_'
#                                      + datetime.now().strftime('%Y-%m-%d-%H%M%S')
#                                      + '.mp4')
#             mpeg_writer = animation.writers('ffmpeg')
#             metadata = {'title': 'initialization procedure',
#                         'artist': 'Matplotlib',
#                         'comment': 'CNMF-E is cool!'}
#             writer = mpeg_writer(fps=options.fr, metadata=metadata)
#             writer.saving(h_fig, mpeg_name)
#             # TO BE DONE
#
#     # creating variables for storing the results
#     if not K:
#         K = np.int32((ind_search.size - ind_search.sum())/10)  # maximum
#         # number of neurons
#
#     Ain = np.zeros(shape=(d1*d2, K))
#     Cin = np.zeros(shape=(K, total_frames))
#     Sin = np.zeros(shape=(K, total_frames))
#     Cin_raw = np.zeros(shape=(K, total_frames))
#     kernel_pars = [] * K    # parameters corresponding to calcium dynamics
#     center = np.zeros(shape=(K, 2))
#     k = 0                   # number of initialized neurons
#     searching_flag = True       # the value is False when we want to stop
#
#     while searching_flag:
#         # apply median filter to remove isolated signals
#         v_search[ind_search] = 0
#
#         # find local maximum as candidate seed pixels
#         tmp_kernel = np.ones(shape=(2*gSig+1, 2*gSig+1))
#         v_max = cv2.dilate(v_search, tmp_kernel)
#
#         if seed_method.lower() == 'manual':
#             pass
#             # manually pick seed pixels
#         else:
#             # automatically select seed pixels
#             v_max[v_search != v_max] = 0
#             [r_max, c_max] = v_max.nonzero()
#             local_max = v_max[r_max, c_max]
#             n_seeds = len(ind_local_max)
#             if n_seeds == 0:
#                 searching_flag = False
#                 break
#             else:
#                 # order seed pixels according to their corr * pnr values
#                 ind_local_max = local_max.argsort()[::-1]
#
#         # try to initialization neurons given all seed pixels
#         for idx in ind_local_max:
#             if local_max[idx] < min_v_search:
#                 continue
#             r = r_max[idx]
#             c = c_max[idx]
#             ind_search[r, c] = True        # this pixel won't be searched
#
#             # roughly check whether this is a good seed pixel
#             y0 = data_filtered[:, r, c]
#             y0_std = get_noise(y0)
#             if np.max(np.diff(y0)) < 3*y0_std:
#                 continue    # we don't use the peak of y0 directly because
#                 # y0  may have large constant baseline
#
#             # crop a small box for estimation of ai and ci
#             r_min = np.max([0, r-gSiz])
#             r_max = np.min([d1, r+gSiz+1])
#             c_min = np.max([0, c-gSiz])
#             c_max = np.min([d2, c+gSiz+1])
#             nr = r_max - r_min
#             nc = c_max - c_min
#             sz = (nr, nc)
#             data_raw_box = data_raw[:, r_min:r_max, c_min:c_max].reshape(-1,
#                                                                          nr*nc)
#             data_filtered_box = data_filtered[:, r_min:r_max,
#                                 c_min:c_max].reshape(-1, nr*nc)
#             ind_ctr = np.ravel_multi_index((r - r_min, c-c_min), dims=(nr, nc))
#
#             # neighbouring pixels to update after initializing one neuron
#             r2_min = np.max([0, r-2*gSiz])
#             r2_max = np.min([d1, r+2*gSiz+1])
#             c2_min = np.max([0, c-2*gSiz])
#             c2_max = np.min([d2, c+2*gSiz+1])
#             nr2 = r2_max - r2_min
#             nc2 = c2_max - c2_min
#
#             # show temporal trace of the seed pixel
#             if debug_on:
#                 pass
#
#
#             # extract ai, ci
#             [ai, ci_raw, ind_success] = extract_ac(data_filtered_box,
#                                                    data_raw_box, ind_ctr)
#
#
#     results = 1
#     center = 1
#     # return
#     return results, center, cn0, pnr, save_avi




def extract_ac(data_filtered, data_raw, ind_ctr, sz):
    # dimensions and parameters
    nr, nc = sz
    min_corr_neuron = 0.7
    max_corr_bg = 0.3

    # compute the temporal correlation between each pixel and the seed pixel
    data_filtered -= data_filtered.mean(axis=0)    # data centering
    data_filtered /= np.sqrt(np.sum(data_filtered**2, axis=0))   # data normalization
    y0 = data_filtered[:, ind_ctr]  # fluorescence trace at the center
    tmp_corr = np.dot(y0.reshape(1, -1), data_filtered)  # corr. coeff. with y0
    ind_neuron = (tmp_corr>min_corr_neuron).squeeze()  # pixels in the central area of neuron
    ind_bg = (tmp_corr<max_corr_bg).squeeze()   # pixels outside of neuron's ROI

    # extract temporal activity
    ci = np.mean(data_filtered[:, ind_neuron], axis=1).reshape(-1, 1) # initialize temporal activity of the neural
    if np.linalg.norm(ci) == 0:   # avoid empty results
        return None, None, False

    # roughly estimate the background fluctuation
    y_bg = np.median(data_raw[:, ind_bg], axis=1).reshape(-1, 1)

    # extract spatial components
    X = np.hstack([ci, y_bg, np.ones(ci.shape)])
    ai = np.linalg.lstsq(np.dot(X.transpose(), X), np.dot(X.transpose(), data_raw))[0][0]

    # post-process neuron shape

    # return results
    return ai, ci, True
"""
----------------------------------RUN----------------------------------
"""
