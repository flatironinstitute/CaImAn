#!/usr/bin/env python

"""
This file contains a set of methods for the online analysis of microendoscopic
one photon data using a "ring-CNN" background model.
"""

import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"
import time

import torch 
import torch.nn.functional as F
import torch.nn as nn
import keras 
import keras.ops as ops  
from keras.constraints import Constraint 
from keras.layers import Input, Dense, Reshape, Layer, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.initializers import Constant, RandomUniform
from keras.utils import Sequence

import caiman.base.movies 
from caiman.paths import caiman_datadir

class CalciumDataset(Sequence):
    def __init__(self, files, random_state=42, batch_size=32, train=True,
                 var_name_hdf5='mov', subindices=None):
        """ 
        Create a Sequence object for Ca datasets. Not used at the moment
        """
        if isinstance(files, str):
            files = [files]
        self.files = files
        dims, T = caiman.base.movies.get_file_size(files, var_name_hdf5=var_name_hdf5)
        if subindices is not None:
            T = len(range(T)[subindices])
        if isinstance(T, int):
            T = [T]
        self.num_examples_per_npy = T[0]
        self.random_state = random_state
        self.n_channels = 1
        self.dim = dims
        self.T = T
        self.batch_size = batch_size
        self.train = train
        self.on_epoch_end()
        self.var_name_hdf5 = var_name_hdf5
    
    def __len__(self):
        return (np.array(self.T)//self.batch_size).sum()

    def __getitem__(self, index):
        batches_per_npy = int(np.floor(self.num_examples_per_npy / self.batch_size))
        file_id = int(index / batches_per_npy)
        batch_id = int(index % batches_per_npy)
        lb, ub = batch_id*self.batch_size, (batch_id + 1)*self.batch_size
        X = caiman.base.movies.load(os.path.join(self.files[file_id]), subindices=slice(lb, ub),
                                    var_name_hdf5=self.var_name_hdf5)
        X = X.astype(np.float32)
        X = np.expand_dims(X, axis=-1)
        return X, X

    def on_epoch_end(self):
        # loop thru files, and shuffle them up. 
        # shuffles order for next epoch. 
        np.random.shuffle(self.files)

class Masked_Conv2D(Layer): #Keras.layer
    """ Creates a trainable ring convolutional kernel with non zero entries between
    user specified radius_min and radius_max. Uses a random uniform non-negative
    initializer unless specified otherwise.

    Args:
        output_dim: int, default: 1
            number of output channels (number of kernels)

        kernel_size: (int, int), default: (5, 5)
            dimension of 2d boundaing box

        strides: (int, int), default: (1, 1)
            stride for convolution (modifying that will downsample)

        radius_min: int, default: 2
            inner radius of kernel

        radius_max: int, default: 3
            outer radius of kernel (typically: 2*radius_max - 1 = kernel_size[0])

        initializer: 'uniform' or Keras initializer, default: 'uniform'
            initializer for ring weights. 'uniform' will choose from a non-negative
            random uniform distribution such that the expected value of the sum
            is 2.

        use_bias: bool, default: True
            add a bias term to each convolution kernel

    Returns:
        Masked_Conv2D: keras.layer
            A trainable layer implementing the convolution with a ring
    """
    def __init__(self, output_dim=1, kernel_size=(5,5), strides=(1,1),
               radius_min=2, radius_max=3, initializer='uniform',
               use_bias=True): #, output_dim):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.strides = strides
        self.use_bias = use_bias
        xx = np.arange(-(kernel_size[0]-1)//2, (kernel_size[0]+1)//2)
        yy = np.arange(-(kernel_size[0]-1)//2, (kernel_size[0]+1)//2)
        [XX, YY] = np.meshgrid(xx, yy)
        R = np.sqrt(XX**2 + YY**2)
        R[R<radius_min] = 0
        R[R>radius_max] = 0
        R[R>0] = 1
        self.mask = R  # the mask defines the non-zero pattern and will multiply the weights
        if initializer == 'uniform':
            self.initializer = RandomUniform(minval=0, maxval=2/R.sum())
        else:
            self.initializer = initializer
        super(Masked_Conv2D, self).__init__()

    def build(self, input_shape):
        n_filters = input_shape[-1] 
        self.h = self.add_weight(name='h',
                                 shape= self.kernel_size + (n_filters, self.output_dim,),
                                 initializer=self.initializer,
                                 constraint=MaskedConstraint(self.mask), 
                                 trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(self.output_dim,),
                                 initializer=Constant(0),
                                 trainable=self.use_bias) 
        super(Masked_Conv2D, self).build(input_shape)

    def call(self, x):
        y = ops.conv(x, self.h, strides=self.strides, padding='same')
        if self.use_bias:
            y = y + torch.unsqueeze(self.b, dim=0)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (self.output_dim,)

def get_mask(gSig=5, r_factor=1.5, width=5):
    """ Computes the mask to create the ring model

    Args:
        gSig: int, default: 5
            radius of average neuron

        r_factor: float, default: 1.5
            expansion factor to determine inner radius

        width: int, default: 5
            width of ring kernel

    Returns:
        R: np.array
            mask that is non-zero around the ring
    """
    radius_min = int(gSig*r_factor)
    radius_max = radius_min + width
    kernel_size = (2*radius_max + 1, 2*radius_max + 1)
    xx = np.arange(-(kernel_size[0]-1)//2, (kernel_size[0]+1)//2)
    yy = np.arange(-(kernel_size[0]-1)//2, (kernel_size[0]+1)//2)
    [XX, YY] = np.meshgrid(xx, yy)
    R = np.sqrt(XX**2 + YY**2)
    R[R<radius_min] = 0
    R[R>radius_max] = 0
    R[R>0] = 1
    return R

class MaskedConstraint(keras.constraints.Constraint):
    def __init__(self, R):
        self.R = R

    def __call__(self, x):
        self.R = torch.tensor(self.R).float() 
        R_exp = torch.unsqueeze(torch.unsqueeze(self.R, dim=-1), dim=-1)
        Rt = torch.tile(R_exp, [1, 1, 1, x.shape[-1]])
        Z = torch.zeros_like(x)
        return torch.where(Rt > 0, x, Z)

class Hadamard(Layer):
    """ Creates a keras multiplicative layer that performs
    pointwise multiplication with a set of learnable weights.

    Args:
        initializer: keras initializer, default: Constant(0.1)
    """
    def __init__(self, initializer=Constant(0.1), **kwargs):
        self.initializer = initializer
        super(Hadamard, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                     shape=input_shape[1:],
                                     initializer=self.initializer,
                                     trainable=True)
        super(Hadamard, self).build(input_shape)

    def call(self, x):
        hm = torch.multiply(x, self.kernel)
        sm = torch.sum(hm, dim=-1, keepdim=True)
        return sm

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class Additive(Layer):
    """ Creates a keras additive layer that performs
    pointwise addition with a set of learnable weights.

    Args:
        initializer: keras initializer, default: Constant(0)
    """
    def __init__(self, data=None, initializer=Constant(0), pct=1, **kwargs):
        self.data = data
        self.pct = pct
        self.initializer = initializer
        super(Additive, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='additive',
                                     shape=input_shape[1:],
                                     initializer=self.initializer,
                                     trainable=True)
        super(Additive, self).build(input_shape)

    def call(self, x):
        hm = torch.add(x, self.kernel)
        return hm

    def compute_output_shape(self, input_shape):
        return input_shape


def cropped_loss(gSig=0):
    """ Returns a cropped loss function to exclude boundaries (not used)

    Args:
        gSig: int, default: 0
            number of pixels to crop from each boundary

    Returns:
        my_loss: cropped loss function
    """
    def my_loss(y_true, y_pred):
        if gSig > 0:
            error = torch.square(y_true[gSig:-gSig, gSig:-gSig] - y_pred[gSig:-gSig, gSig:-gSig])
        else:
            error = torch.square(y_true - y_pred) 
        return error
    return my_loss

def quantile_loss(qnt=.50):
    """ Returns a quantile loss function that can be used for training.

    Args:
        qnt: float, default: 0.5
            desired quantile (0 < qnt < 1)

    Returns:
        my_qnt_loss: quantile loss function
    """
    def my_qnt_loss(y_true, y_pred):
        error = y_true - y_pred
        pos_error = error > 0
        return torch.where(pos_error, error*qnt, error*(qnt-1))
    return my_qnt_loss

def rate_scheduler(factor=0.5, epoch_length=200, samples_length=1e4):
    """ Creates a simple learning rate scheduler that decreases the learning
    rate exponentially. It is recommended that you provide your own scheduler
    if you want to use one.
    """
    def my_scheduler(epoch, lr):
        '''decrease by a factor of factor every nepochs epochs'''
        nepochs = samples_length/epoch_length
        f = factor**(1/nepochs)
        rate = lr*f if epoch > 0 else lr
        return rate
    return my_scheduler

def total_variation(image):
    """
    Implements PyTorch version of the the anisotropic 2-D version of the formula described here:
    https://en.wikipedia.org/wiki/Total_variation_denoising

    Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    name: A name for the operation (optional).

    Raises:
        ValueError: if images.shape is not a 3-D or 4-D vector.

    Returns:
        The total variation of `images`.
    """
    ndim = image.ndim
    if ndim == 3: 
        # The input is a single image with shape [height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
        pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
        sum_axis = None
    elif ndims == 4:
        # The input is a batch of images with shape:
        # [batch, height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

        # Only sum for the last 3 axis.
        # This results in a 1-D tensor with the total variation for each image.
        sum_axis = [1, 2, 3]
    else:
      raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (torch.sum(torch.abs(pixel_dif1), axis=sum_axis) +
        torch.sum(torch.abs(pixel_dif2), axis=sum_axis))

    return tot_var 


def total_variation_loss():
    """ Returns a total variation norm loss function that can be used for training.
    """
    def my_total_variation_loss(y_true, y_pred):
        error = torch.mean(total_variation(y_true - y_pred))
        return error 
    return my_total_variation_loss

def b0_initializer(Y, pct=10):
    """ Returns a percentile based initializer for the additive layer (not used)

    Args:
        Y: np.array
            loaded dataset with time in the first axis
        pct: float, default: 10
            percentile to be extracted as initializer

    Returns:
        b0_init: keras initializer
    """
    def b0_init(shape, dtype=torch.float32): 
        mY = np.percentile(Y, pct, 0)
        mY = torch.from_numpy(mY)
        if mY.ndim == 2:
            mY = torch.unsqueeze(mY, dim=-1)
        mY = mY.float()
        return mY
    return b0_init


def get_run_logdir():
    """ Returns the path to the directory where the model will be saved.
    The directory will be locates inside the caiman_data/my_logs.
    """
    root_logdir = os.path.join(caiman_datadir(), "my_logs")
    if not os.path.exists(root_logdir):
        os.mkdir(root_logdir)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def create_LN_model(Y=None, shape=(None, None, 1), n_channels=2, gSig=5, r_factor=1.5,
                    use_add=True, initializer='uniform', lr=1e-4,
                    pct=10, loss='mse', width=5, use_bias=False):
    """ Creates a convolutional neural network with ring shape convolutions
    and multiplicative layers. User needs to specify the radius of the average
    neuron through gSig and the number of channels. The other parameters can be
    modified or left to the default values. The inner and outer radius of the
    ring kernel will be int(gSig*r_factor) and int(gSig*r_factor) + width,
    respectively.

    Args:
        Y: np.array, default: None
            dataset to be fit, used only if a percentile based initializer is
            used for the additive layer and can be left to None

        shape: tuple, default: (None, None, 1)
            dimensions of the FOV. Can be left to its default value

        n_channels: int, default: 2
            number of convolutional kernels

        gSig: int, default: 5
            radius of average neuron

        r_factor: float, default: 1.5
            expansion factor to determine inner radius

        width: int, default: 5
            width of ring kernel

        use_add: bool, default: True
            flag for using an additive layer

        initializer: 'uniform' or Keras initializer, default: 'uniform'
            initializer for ring weights. 'uniform' will choose from a non-negative
            random uniform distribution such that the expected value of the sum
            is 2.

        lr: float, default: 1e-4
            (initial) learning rate

        pct: float, default: 10
            percentile used for initializing additive layer

        loss: str or keras loss function
            loss function used for training

        use_bias: bool, default: False
            add a bias term to each convolution kernel

    Returns:
        model_LIN: keras model compiled and ready to be trained.
    """
    x_in = Input(shape=shape)
    radius_min = int(gSig*r_factor)
    radius_max = radius_min + width
    ks = 2*radius_max + 1
    conv1 = Masked_Conv2D(output_dim=n_channels, kernel_size=(ks, ks),
                          radius_min=radius_min, radius_max=radius_max,
                          initializer=initializer, use_bias=use_bias)
    x = conv1(x_in)  # apply convolutions 
    x_out = Hadamard()(x)  # apply Hadamard layer
    if use_add:
        x_out = Additive(data=Y)(x_out)
    model_LIN = Model(x_in, x_out)
    adam = Adam(learning_rate=lr)
    model_LIN.compile(optimizer=adam, loss=loss) # compile model
    return model_LIN

def create_NL_model(Y=None, shape=(None, None, 1), n_channels=2, gSig=5, r_factor=1.5,
                    use_add=True, initializer='he_normal', lr=1e-4, pct=10,
                    activation='relu', loss='mse', width=5, use_bias=True):
    """ Creates a two layer nonlinear convolutional neural network with ring 
    shape convolutions (but no multiplicative layers). User needs to specify
    the radius of the average neuron through gSig and the number of channels.
    The other parameters can be modified or left to the default values. The
    inner and outer radius of the ring kernel will be int(gSig*r_factor) and
    int(gSig*r_factor) + width, respectively.

    Args:
        Y: np.array, default: None
            dataset to be fit, used only if a percentile based initializer is
            used for the additive layer and can be left to None

        shape: tuple, default: (None, None, 1)
            dimensions of the FOV. Can be left to its default value

        n_channels: int, default: 2
            number of convolutional kernels

        gSig: int, default: 5
            radius of average neuron

        r_factor: float, default: 1.5
            expansion factor to determine inner radius

        width: int, default: 5
            width of ring kernel

        use_add: bool, default: True
            flag for using an additive layer

        initializer: 'uniform' or Keras initializer, default: 'uniform'
            initializer for ring weights. 'uniform' will choose from a non-negative
            random uniform distribution such that the expected value of the sum
            is 2.

        lr: float, default: 1e-4
            (initial) learning rate

        pct: float, default: 10
            percentile used for initializing additive layer
 
        activation: str or keras initializer, default: 'relu'
            (nonlinear) activation function 

        loss: str or keras loss function
            loss function used for training

        use_bias: bool, default: False
            add a bias term to each convolution kernel

    Returns:
        model_LIN: torch.keras model compiled and ready to be trained.
    """
    x_in = Input(shape=shape)
    radius_min = int(gSig*r_factor)
    radius_max = radius_min + width
    ks = 2*radius_max + 1
    conv1 = Masked_Conv2D(output_dim=n_channels, kernel_size=(ks, ks),
                          radius_min=radius_min, radius_max=radius_max,
                          initializer=initializer, use_bias=use_bias)
    x = conv1(x_in)
    x = Activation(activation)(x)
    x = Reshape((-1, n_channels))(x)
    use_bias = False if use_add else True
    x_out = Dense(1, use_bias=use_bias, activation=None)(x)
    x_out = Reshape(shape)(x_out)
    if use_add:
        x_out = Additive(data=Y, pct=pct)(x_out)
    model_NL = Model(x_in, x_out)
    adam = Adam(learning_rate=lr)
    model_NL.compile(optimizer=adam, loss=loss)
    return model_NL

def fit_NL_model(model_NL, Y, patience=5, val_split=0.2, batch_size=32,
                 epochs=500, schedule=None):
    """
    Fit either the linear or the non-linear model. The model is fit for a
    use specified maximum number of epochs and early stopping is used based on the 
    validation loss. A Tensorboard compatible log is also created.

    Args:
        model_LN: Keras Ring-CNN model
            see create_LN_model and create_NL_model above
        patience: int, default: 5
            patience value for early stopping criterion
        val_split: float, default: 0.2
            fraction of data to keep for validation (value between 0 and 1)
        batch_size: int, default: 32
            batch size during training
        epochs: int, default: 500
            maximum number of epochs
        schedule: keras learning rate scheduler

    Returns:
        model_NL: 
            Keras Ring-CNN model
            trained model loaded with best weights according to validation loss
        history_NL:
            contains data related to the training history
        path_to_model:
            path to where the weights are stored

    """
    if Y.ndim < 4:
        Y = np.expand_dims(Y, axis=-1)
    run_logdir = get_run_logdir()
    os.mkdir(run_logdir)
    path_to_model = os.path.join(run_logdir, 'model.weights.h5')
    chk = ModelCheckpoint(filepath=path_to_model,
                          verbose=0, save_best_only=True, save_weights_only=True)
    es = EarlyStopping(monitor='val_loss', patience=patience,
                       restore_best_weights=True)
    sch = [] if schedule is None else [LearningRateScheduler(schedule, verbose=1)]
    callbacks = [es, chk] + sch

    history_NL = model_NL.fit(Y, Y, epochs=epochs, batch_size=batch_size,
                            shuffle=True, validation_split=val_split,
                            callbacks=callbacks)
    model_NL.load_weights(os.path.join(run_logdir, 'model.weights.h5'))
    return model_NL, history_NL, path_to_model

def get_MCNN_model(Y, gSig=5, n_channels=8, lr=1e-4, pct=10, r_factor=1.5,
                   patience=5, val_split=0.2, batch_size=32, epochs=500,
                   activation='relu', is_linear=True):
    """ Creates and fits a linear Ring-CNN model (not used)"""
    shape = Y.shape[1:] + (1,)
    model_NL = create_NL_model(Y, shape=shape, n_channels=n_channels, gSig=gSig,
                               r_factor=r_factor, lr=lr, pct=pct,
                               activation=activation)
    model_NL = fit_NL_model(model_NL, Y, patience=patience, val_split=val_split,
                            batch_size=batch_size, epochs=epochs)
    return model_NL
