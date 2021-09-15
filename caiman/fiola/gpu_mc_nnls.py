#!/usr/bin/env python
"""
This file includes class for gpu motion correction, source extraction(NNLS) and
pipeline which process motion correction, source extraction, spike detection frame
by-frame.
@author: @caichangjia, @cynthia, @agiovann
"""
import logging
import math
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from queue import Queue
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from tensorflow.signal import fft3d, ifft3d, ifftshift
from threading import Thread
import timeit
from caiman.fiola.utilities import HALS4activity

class MotionCorrect(keras.layers.Layer):
    def __init__(self, template, batch_size=1, ms_h=5, ms_w=5, min_mov=0, 
                 strides=[1,1,1,1], padding='VALID',use_fft=True, 
                 normalize_cc=True, epsilon=0.00000001, center_dims=None, return_shifts=False, **kwargs):
        """
        Class for GPU motion correction        

        Parameters
        ----------
        template : ndarray
            The template used for motion correction
        batch_size : int
            number of frames used for motion correction each time. The default is 1.
        ms_h : int
            maximum shift horizontal. The default is 5.
        ms_w : int
            maximum shift vertical. The default is 5.
        min_mov: float
            minimum of the movie will be removed. The default is 0.
        strides : list
            stride for convolution. The default is [1,1,1,1].
        padding : str
            padding for convolution. The default is 'VALID'.
        use_fft : bool
            use FFT for convolution or not. Will use tf.nn.conv2D if False. The default is True.
        normalize_cc : bool
            whether to normalize the cross correlations coefficients or not. The default is True.
        epsilon : float
            epsilon to avoid deviding by zero. The default is 0.00000001.
        center_dims : tuple
            size of center crops of template for motion correction. If None, it will not crop. The default is None.
        return_shifts: bool
            if True, return both motion corrected movie and shifts. If False, return only motion corrected movie
        
        Returns
        -------
        motion corrected frames and/or shifts (depending on return_shifts)
        """        
        super().__init__(**kwargs)   
        
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)
                logging.debug(f'{name}, {value}')
        
        self.template_0 = template
        self.shp_0 = self.template_0.shape
        
        self.xmin, self.ymin = self.shp_0[0] - 2 * ms_w, self.shp_0[1] - 2 * ms_h
        if self.xmin < 10 or self.ymin < 10:
            raise ValueError("The frame dimensions you entered are too small. Please provide a larger field of view or resize your movie.") 
        
        if self.center_dims is not None:
            if self.center_dims[0] >= self.shp_0[0] or self.center_dims[1] >= self.shp_0[1]:
                raise ValueError("center_dims can not be larger than movie shape")
            self.shp_c_x, self.shp_c_y = (self.shp_0[0] - center_dims[0])//2, (self.shp_0[1] - center_dims[1])//2
            self.template = self.template_0[self.shp_c_x:-self.shp_c_x, self.shp_c_y:-self.shp_c_y]
        else:
            self.shp_c_x, self.shp_c_y = (0, 0)
        if self.use_fft == False:
            self.template = self.template_0[(ms_w+self.shp_c_x):-(ms_w+self.shp_c_x),(ms_h+self.shp_c_y):-(ms_h+self.shp_c_y)]

        self.template_zm, self.template_var = self.normalize_template(self.template[:,:,None,None], epsilon=self.epsilon)
        self.shp = self.template.shape
        self.shp_m_x, self.shp_m_y = self.shp[0]//2, self.shp[1]//2
        self.target_freq = fft3d(tf.cast(self.template_zm[:,:,:,0], tf.complex128))
        self.target_freq = tf.repeat(self.target_freq[None,:,:,0], repeats=[self.batch_size], axis=0)
                
        self.Nr = tf.cast(ifftshift(tf.range(-tf.math.floor(self.shp_0[1] / 2.), tf.math.ceil(self.shp_0[1] / 2.))), tf.float32)
        self.Nc = tf.cast(ifftshift(tf.range(-tf.math.floor(self.shp_0[0] / 2.), tf.math.ceil(self.shp_0[0] / 2.))), tf.float32)
        self.Nd = tf.cast(ifftshift(tf.range(-tf.math.floor(1 / 2.), tf.math.ceil(1 / 2.))), tf.float32)
        #self.Nr, self.Nc, self.Nd = tf.meshgrid(self.Nr, self.Nc, self.Nd)        
        self.Nr = tf.repeat(self.Nr[None], self.Nc.shape[0], axis=0)[..., None]
        self.Nc = tf.repeat(self.Nc[:, None], self.Nr.shape[0], axis=1)[..., None]
        self.Nd = tf.zeros((self.shp_0[0], self.shp_0[1], 1))
        
    @tf.function
    def call(self, fr):
        if self.min_mov != 0:
            fr = fr - self.min_mov
        if self.center_dims is None:
            fr_center = fr[0]
        else:      
            fr_center = fr[0,:, self.shp_c_x:(self.shp_0[0]-self.shp_c_x), self.shp_c_y:(self.shp_0[1]-self.shp_c_y)]

        imgs_zm, imgs_var = self.normalize_image(fr_center, strides=self.strides,
                                            padding=self.padding, epsilon=self.epsilon)
        denominator = tf.sqrt(self.template_var * imgs_var)

        if self.use_fft:
            fr_freq = fft3d(tf.cast(imgs_zm[:,:,:,0], tf.complex128))
            img_product = fr_freq *  tf.math.conj(self.target_freq)
            cross_correlation = tf.cast(tf.math.abs(ifft3d(img_product)), tf.float32)
            rolled_cc =  tf.roll(cross_correlation,(self.batch_size, self.shp_m_x,self.shp_m_y), axis=(0,1,2))
            nominator = rolled_cc[:,self.shp_m_x-self.ms_w:self.shp_m_x+self.ms_w+1, self.shp_m_y-self.ms_h:self.shp_m_y+self.ms_h+1, None] 
        else:
            nominator = tf.nn.conv2d(imgs_zm, self.template_zm, padding=self.padding, 
                                     strides=self.strides)
           
        if self.normalize_cc:    
            ncc = tf.truediv(nominator, denominator)        
        else:
            ncc = nominator    
        
        ncc = tf.where(tf.math.is_nan(ncc), tf.zeros_like(ncc), ncc)
        sh_x, sh_y = self.extract_fractional_peak(ncc, self.ms_h, self.ms_w)
        self.shifts = [sh_x, sh_y]
        
        #fr_corrected = tfa.image.translate(fr[0], (tf.squeeze(tf.stack([sh_y, sh_x], axis=1))), 
        #                                    interpolation="bilinear")
        fr_corrected = self.apply_shifts_dft_tf(fr[0], [-sh_x, -sh_y])
        
        if self.return_shifts:
            return tf.reshape(tf.transpose(tf.squeeze(fr_corrected, axis=3), perm=[0,2,1]), (self.batch_size, self.shp_0[0]*self.shp_0[1])), self.shifts
        else:
            return tf.reshape(tf.transpose(tf.squeeze(fr_corrected, axis=3), perm=[0,2,1]), (self.batch_size, self.shp_0[0]*self.shp_0[1]))
    
    def normalize_template(self, template, epsilon=0.00000001):
        # remove mean and divide by std
        template_zm = template - tf.reduce_mean(template, axis=[0,1], keepdims=True)
        template_var = tf.reduce_sum(tf.square(template_zm), axis=[0,1], keepdims=True) + epsilon
        return template_zm, template_var
        
    def normalize_image(self, imgs, strides=[1,1,1,1], padding='VALID', epsilon=0.00000001):
        # remove mean and standardize so that normalized cross correlation can be computed
        if self.use_fft:
            shape = [self.template.shape[0]-2*self.ms_w, self.template.shape[1]-2*self.ms_h]            
        else:
            shape = [self.template.shape[0], self.template.shape[1]]
        imgs_zm = imgs - tf.reduce_mean(imgs, axis=[1,2], keepdims=True)
        img_stack = tf.stack([imgs[:,:,:,0], tf.square(imgs)[:,:,:,0]], axis=3)
        localsum_stack = tf.nn.avg_pool2d(img_stack,[1, shape[0], shape[1], 1], 
                                               padding=padding, strides=strides)
        localsum_ustack = tf.unstack(localsum_stack, axis=3)
        localsum_sq = localsum_ustack[1][:,:,:,None]
        localsum = localsum_ustack[0][:,:,:,None]      
        imgs_var = localsum_sq - tf.square(localsum)/np.prod(shape) + epsilon
        # Remove small machine precision errors after subtraction
        imgs_var = tf.where(imgs_var<0, tf.zeros_like(imgs_var), imgs_var)
        return imgs_zm, imgs_var        
        
    def extract_fractional_peak(self, ncc, ms_h, ms_w):
        # use gaussian interpolation to extract a fractional shift
        shifts_int = self.argmax_2d(ncc) 
        shifts_int_cast = tf.cast(shifts_int,tf.int32)
        sh_x, sh_y = shifts_int_cast[:,0],shifts_int_cast[:,1]
        sh_x_n = tf.cast(-(sh_x - ms_h), tf.float32)
        sh_y_n = tf.cast(-(sh_y - ms_w), tf.float32)
        sh_x = tf.squeeze(sh_x, axis=1)
        sh_y = tf.squeeze(sh_y, axis=1)
        ncc_log = tf.math.log(ncc)

        n_batches = np.arange(self.batch_size)
        idx = tf.transpose(tf.stack([n_batches, sh_x-1, sh_y]))
        log_xm1_y = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x+1, sh_y]))
        log_xp1_y = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x, sh_y-1]))
        log_x_ym1 = tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x, sh_y+1]))
        log_x_yp1 =  tf.gather_nd(ncc_log, idx)
        idx = tf.transpose(tf.stack([n_batches, sh_x, sh_y]))
        
        four_log_xy = 4 * tf.gather_nd(ncc_log, idx)
        sh_x_n = sh_x_n - tf.math.truediv((log_xm1_y - log_xp1_y), (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = sh_y_n - tf.math.truediv((log_x_ym1 - log_x_yp1), (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))

        return tf.reshape(sh_x_n, [self.batch_size, 1]), tf.reshape(sh_y_n, [self.batch_size, 1])
    
    def argmax_2d(self, tensor):
        # extract peaks from 2D tensor (takes batches as input too)
        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))
        argmax= tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

        # convert indexes into 2D coordinates
        argmax_x = tf.cast(argmax, tf.int32) // tf.shape(tensor)[2]
        argmax_y = tf.cast(argmax, tf.int32) % tf.shape(tensor)[2]

        # stack and return 2D coordinates
        return tf.cast(tf.stack((argmax_x, argmax_y), axis=1), tf.float32)
    
    def apply_shifts_dft_tf(self, img, shifts, diffphase=[0]):
        img = tf.cast(img, dtype=tf.complex128)
        if len(shifts) == 3:
            shifts =  (shifts[1], shifts[0], shifts[2]) 
        elif len(shifts) == 2:
            shifts = (shifts[1], shifts[0], tf.zeros(shifts[1].shape))
        src_freq = fft3d(img)
        
        sh_0 = tf.tensordot(-shifts[0], self.Nr[None], axes=[[1], [0]]) / self.shp_0[1]
        sh_1 = tf.tensordot(-shifts[1], self.Nc[None], axes=[[1], [0]]) / self.shp_0[0]
        sh_2 = tf.tensordot(-shifts[2], self.Nd[None], axes=[[1], [0]]) / 1
        sh_tot = (sh_0 + sh_1 + sh_2)
        Greg = src_freq * tf.math.exp(-1j * 2 * math.pi * tf.cast(sh_tot, dtype=tf.complex128))
        
        #todo: check difphase and eventually dot product?
        diffphase = tf.cast(diffphase,dtype=tf.complex128)
        Greg = Greg * tf.math.exp(1j * diffphase)
        new_img = tf.math.real(ifft3d(Greg)) 
        new_img = tf.cast(new_img, tf.float32)
        
        return new_img
        
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "template": self.template_0, "batch_size":self.batch_size, "ms_h": self.ms_h,"ms_w": self.ms_w, 
                "min_mov": self.min_mov, "strides": self.strides, "padding": self.padding, "epsilon": self.epsilon, 
                "use_fft":self.use_fft, "normalize_cc":self.normalize_cc, "center_dims":self.center_dims, "return_shifts":self.return_shifts}
    
class NNLS(keras.layers.Layer):
    def __init__(self, theta_1, name="NNLS",**kwargs):
        """
        Tensforflow layer which perform Non Negative Least Squares. Using  https://angms.science/doc/NMF/nnls_pgd.pdf
            arg min f(x) = 1/2 || Ax − b ||_2^2
              {x≥0}
             
        Notice that the input must be a tensorflow tensor with dimension batch 
        x width x height x channel
        Args:
            theta_1: ndarray
                theta_1 = (np.eye(A.shape[-1]) - AtA/n_AtA)
            theta_2: ndarray
                theta_2 = (Atb/n_AtA)[:,None]            
        
        Returns:
            x regressed values
        
        """
        super().__init__(**kwargs)
        self.th1 = theta_1.astype(np.float32)

    def call(self, X):
        """
        NNLS for each iteration. see https://angms.science/doc/NMF/nnls_pgd.pdf
        
        Parameters
        ----------
        X : tuple
            output of previous iteration

        Returns
        -------
        Y_new : ndarray
            auxilary variables
        new_X : ndarray
            new extracted traces
        k : int
            number of iterations
        weight : ndarray
            equals to theta_2

        """        
        (Y,X_old,k,weight) = X
        mm = tf.matmul(self.th1, Y)
        new_X = tf.nn.relu(mm + weight)

        Y_new = new_X + (k - 1)/(k + 2)*(new_X - X_old)  
        k += 1
        return (Y_new, new_X, k, weight)
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "theta_1": self.th1} 

class compute_theta2(keras.layers.Layer):
    def __init__(self, A, n_AtA, **kwargs): 
        super().__init__(**kwargs)
        self.A = A
        self.n_AtA = n_AtA
        
    def call(self, X):
        Y = tf.matmul(X, self.A)
        Y = tf.divide(Y, self.n_AtA)
        Y = tf.transpose(Y)
        return Y    
    
    def get_config(self):
        base_config = super().get_config().copy()
        return {**base_config, "A":self.A, "n_AtA":self.n_AtA}
    
class Empty(keras.layers.Layer):
    def call(self,  fr):
        fr = fr[0, ..., 0]
        return tf.reshape(tf.transpose(fr, perm=[0, 2, 1]), (fr.shape[0], -1))

def get_mc_model(template, batch_size, **kwargs):
    """
    build gpu motion correction model
    """
    # dimensions of the movie
    shp_x, shp_y = template.shape[0], template.shape[1] 
    template = template.astype(np.float32)
    
    # input layer for one frame of the movie
    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([batch_size, shp_x, shp_y, 1]), name="m")  

    # initialization of the motion correction layer, initialized with the template
    mc_layer = MotionCorrect(template, batch_size, **kwargs)   
    mc, shifts = mc_layer(fr_in)    

    # create final model, returns it and the first weight
    model = keras.Model(inputs=[fr_in], outputs=[mc, shifts])   
    return model

def get_nnls_model(dims, Ab, batch_size, num_layers=10):
    """
    build gpu nnls model
    """
    shp_x, shp_y = dims[0], dims[1] 
    Ab = Ab.astype(np.float32)
    num_components = Ab.shape[-1]

    y_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, batch_size]), name="y") # Input Layer for components
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, batch_size]), name="x") # Input layer for components
    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([batch_size, shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 
    k_in = tf.keras.layers.Input(shape=(1,), name="k") #Input layer for the counter within the NNLS layers
 
    # calculations to initialize NNLS
    AtA = Ab.T@Ab
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)

    # empty motion correction layer
    mc_layer = Empty()
    mc = mc_layer(fr_in)
    
    # chains motion correction layer to weight-calculation layer
    c_th2 = compute_theta2(Ab, n_AtA)
    th2 = c_th2(mc)
    
    # connects weights, to the NNLS layer
    nnls = NNLS(theta_1)
    x_kk = nnls([y_in, x_in, k_in, th2])
    
    # stacks NNLS 
    for j in range(1, num_layers):
        x_kk = nnls(x_kk)
   
    #create final model
    model = keras.Model(inputs=[fr_in, y_in, x_in, k_in], outputs=[x_kk])   
    return model  

def get_model(template, Ab, batch_size, num_layers=10, **kwargs):
    """
    build full gpu mc-nnls model
    """
    shp_x, shp_y = template.shape[0], template.shape[1] #dimensions of the movie
    num_components = Ab.shape[-1]
    y_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, batch_size]), name="y") # Input Layer for components
    x_in = tf.keras.layers.Input(shape=tf.TensorShape([num_components, batch_size]), name="x") # Input layer for components
    fr_in = tf.keras.layers.Input(shape=tf.TensorShape([batch_size, shp_x, shp_y, 1]), name="m") #Input layer for one frame of the movie 
    k_in = tf.keras.layers.Input(shape=(1,), name="k")

    # calculations to initialize 
    AtA = Ab.T@Ab
    n_AtA = np.linalg.norm(AtA, ord='fro') #Frob. normalization
    theta_1 = (np.eye(Ab.shape[-1]) - AtA/n_AtA)

    # initialization of the motion correction layer, initialized with the template   
    mc_layer = MotionCorrect(template, batch_size, **kwargs)   
    mc = mc_layer(fr_in)

    # chains motion correction layer to weight-calculation layer
    c_th2 = compute_theta2(Ab, n_AtA)
    th2 = c_th2(mc)
    
    # connects weights, calculated from the motion correction, to the NNLS layer
    nnls = NNLS(theta_1)
    x_kk = nnls([y_in, x_in, k_in, th2])
    # stacks NNLS 
    for j in range(1, num_layers):
        x_kk = nnls(x_kk)
   
    # create final model, returns it and the first weight
    model = keras.Model(inputs=[fr_in, y_in, x_in, k_in], outputs=[x_kk])   
    return model

    
class Pipeline_mc_nnls(object):    
    def __init__(self, mov, template, batch_size, Ab, **kwargs):
        """
        class for processing motion correction, source extraction frame
        by-frame (or batch).

        Parameters
        ----------
        mov : ndarray
            input movie
        template : ndarray
            the template used for motion correction
        batch_size : int
            number of frames processing each time using gpu 
        Ab : ndarray
            spatial footprints for neurons and background 
        """
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)
                logging.debug(f'{name}, {value}')
        
        b = self.mov[0:self.batch_size].T.reshape((-1, self.batch_size), order='F')         
        C_init = np.dot(Ab.T, b)
        x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
        self.x_0, self.y_0 = np.array(x0[None,:]), np.array(x0[None,:]) 
        self.num_components = self.Ab.shape[-1]
        self.dim_x, self.dim_y = self.mov.shape[1], self.mov.shape[2]
        self.mc_0 = mov[0:batch_size, :, :, None][None, :]

        # build model, load estimator        
        self.model = get_model(self.template, self.Ab, self.batch_size, **kwargs)
        self.model.compile(optimizer='rmsprop', loss='mse')   
        self.estimator = tf.keras.estimator.model_to_estimator(self.model)
        
        # set up queues
        self.frame_input_q = Queue()
        self.nnls_input_q = Queue()
        self.nnls_output_q = Queue()
        
        # seed the queues
        self.frame_input_q.put(self.mc_0)
        self.nnls_input_q.put((self.y_0, self.x_0))

        # start extracting frames: extract calls the estimator to predict using the outputs from the dataset, which
            #pull from the generator.
        self.extraction_thread = Thread(target=self.extract, daemon=True)
        self.extraction_thread.start()
        
    def generator(self):
        #generator waits until data has been enqueued, then yields the data to the dataset
        while True:
            fr = self.frame_input_q.get()
            out = self.nnls_input_q.get()
            (y, x) = out
            yield {"m":fr, "y":y, "x":x, "k":[[0.0]]}
    
    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generator, 
                                                 output_types={"m": tf.float32,
                                                               "y": tf.float32,
                                                               "x": tf.float32,
                                                               "k": tf.float32}, 
                                                 output_shapes={"m":(1, self.batch_size, self.dim_x, self.dim_y, 1),
                                                                "y":(1, self.num_components, self.batch_size),
                                                                "x":(1, self.num_components, self.batch_size),
                                                                "k":(1, 1)})
        return dataset
            
    def extract(self):
        for i in self.estimator.predict(input_fn=self.get_dataset, yield_single_examples=False):
            self.nnls_output_q.put(i)    

    def get_traces(self, bound):
        #to be called separately. Input "bound" represents the number of frames. Starts at one because of initial values put on queue.
        length = bound//self.batch_size
        output = [0]*length
        times = [0]*length
        start = timeit.default_timer()
        for idx in range(self.batch_size, bound, self.batch_size):
            out = self.nnls_output_q.get()
            i = (idx-1)//self.batch_size
            output[i] = out["nnls_1"]
            self.frame_input_q.put(self.mov[idx:idx+self.batch_size, :, :, None][None, :])
            self.nnls_input_q.put((out["nnls"], out["nnls_1"]))
            times[i]  = timeit.default_timer()-start
#            output.append(timeit.default_timer()-st)
        output[-1] = self.nnls_output_q.get()["nnls_1"]
        times[-1] = timeit.default_timer() - start
        logging.info(timeit.default_timer()-start)
        return output

class Pipeline(object):    
    def __init__(self, mode, mov, template, batch_size, Ab, saoz, **kwargs):
        """
        class for processing motion correction, source extraction, spike detection frame
        by-frame (or batch). It has four queues: frame_input_q gets the frame from the raw movie;
        nnls_input_q gets necessary input for nnls algorithm; nnls_output_q stores extracted traces; 
        saoz_input_q gets extracted traces as input for further online processing         

        Parameters
        ----------
        mode : str
            'voltage' or 'calcium 'fluorescence indicator
        mov : ndarray
            input movie
        template : ndarray
            the template used for motion correction
        batch_size : int
            number of frames processing each time using gpu 
        Ab : ndarray
            spatial footprints for neurons and background 
        saoz : TYPE
            object encapsulating online spike extraction 
        """
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)
                logging.debug(f'{name}, {value}')
        
        self.n = mov.shape[0]
        b = self.mov[0:self.batch_size].T.reshape((-1, self.batch_size), order='F')         
        C_init = np.dot(Ab.T, b)
        x0 = np.array([HALS4activity(Yr=b[:,i], A=Ab, C=C_init[:, i].copy(), iters=10) for i in range(batch_size)]).T
        self.x_0, self.y_0 = np.array(x0[None,:]), np.array(x0[None,:]) 
        self.num_components = self.Ab.shape[-1]
        self.dim_x, self.dim_y = self.mov.shape[1], self.mov.shape[2]
        self.mc_0 = mov[0:batch_size, :, :, None][None, :]

        # build model, load estimator        
        self.model = get_model(self.template, self.Ab, self.batch_size, **kwargs)
        self.model.compile(optimizer='rmsprop', loss='mse')   
        self.estimator = tf.keras.estimator.model_to_estimator(self.model)
        
        # set up queues
        self.frame_input_q = Queue()
        self.nnls_input_q = Queue()
        self.nnls_output_q = Queue()
        self.saoz_input_q = Queue()
        
        #seed the queues
        self.frame_input_q.put(self.mc_0)
        self.nnls_input_q.put((self.y_0, self.x_0))

        #start extracting frames: extract calls the estimator to predict using the outputs from the dataset, which
            #pull from the generator.
        self.extraction_thread = Thread(target=self.extract, daemon=True)
        self.extraction_thread.start()
        self.detection_thread = Thread(target=self.detect, daemon=True)
        self.detection_thread.start()
        
    def generator(self):
        #generator waits until data has been enqueued, then yields the data to the dataset
        while True:
            fr = self.frame_input_q.get()
            out = self.nnls_input_q.get()
            (y, x) = out
            yield {"m":fr, "y":y, "x":x, "k":[[0.0]]}
    
    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generator, 
                                                 output_types={"m": tf.float32,
                                                               "y": tf.float32,
                                                               "x": tf.float32,
                                                               "k": tf.float32}, 
                                                 output_shapes={"m":(1, self.batch_size, self.dim_x, self.dim_y, 1),
                                                                "y":(1, self.num_components, self.batch_size),
                                                                "x":(1, self.num_components, self.batch_size),
                                                                "k":(1, 1)})
        return dataset
    
    def load_frame(self, mov_online):
        if len(mov_online) % self.batch_size > 0:
            logging.info('batch size is not a factor of number of frames')
            logging.info(f'take the first {len(mov_online)-len(mov_online) % self.batch_size} frames instead')
        for i in range(int(len(mov_online) / self.batch_size)):
            self.frame_input_q.put(mov_online[i * self.batch_size : (i + 1) * self.batch_size, :, :, None][None,:])
            
    def detect(self):
        self.flag=0 # note the first batch is for initialization, it should not be passed to spike extraction

        while True:
            traces_input = self.saoz_input_q.get()
            if traces_input.ndim == 1:
                traces_input = traces_input[None, :]   # make sure dimension is timepoints * # of neurons

            if self.flag > 0:
                if self.mode == 'voltage':
                    for i in range(len(traces_input)):
                        self.saoz.fit_next(traces_input[i:i+1].T, self.n)
                        if (self.n + 1) % 1000 == 0:
                            logging.info(f'{self.n+1} frames processed')
                        self.n += 1

                elif self.mode == 'calcium':
                    for i in range(len(traces_input)):
                        self.saoz.trace[:, self.n:(self.n+1)] = traces_input[i:i+1].T                                
                        if (self.n + 1) % 1000 == 0:
                            logging.info(f'{self.n+1} frames processed')
                        self.n += 1                    

            self.flag = self.flag + 1
        
    def extract(self):
        for i in self.estimator.predict(input_fn=self.get_dataset, yield_single_examples=False):
            self.nnls_output_q.put(i)

    def get_spikes(self):
        #to be called separately. Input "bound" represents the number of frames. Starts at one because of initial values put on queue.
        while self.frame_input_q.qsize() > 0:
            out = self.nnls_output_q.get().copy()    
            self.nnls_input_q.put((out["nnls"], out["nnls_1"]))
            traces_input = np.array(out["nnls_1"]).squeeze().T
            self.saoz_input_q.put(traces_input)

