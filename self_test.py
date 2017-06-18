from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d,conv_3d,avg_pool_3d,max_pool_3d,conv_3d_transpose
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
from sklearn.metrics import  mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn import preprocessing
import math



def extract_data(filename):
        """Extract the images into a 4D tensor [image index, y, x, channels].


        """
        print('Extracting', filename)
        # get data from h5py
        file = h5py.File(filename, 'r')
        train_data = file['train_data'].value
        train_label = file['train_label']
        test_data = file['test_data'].value
        test_label = file['test_label']
        train_label = np.int64(train_label)
        test_label = np.int64(test_label)
        train_num = train_data.shape[0]
        test_num = test_data.shape[0]

        max,min=train_data.max(),train_data.min()
        train_data_new=(train_data-min)/(max-min)
        train_data_out=np.zeros([train_data.shape[0],train_data.shape[3],train_data.shape[1],train_data.shape[2],1])
        for i in range(train_data.shape[3]):
            train_data_out[:,i,:,:,:]=train_data_new[:,:,:,i]

        max, min = test_data.max(), test_data.min()
        test_data_new = (test_data - min) / (max - min)
        test_data_out = np.zeros(
            [test_data.shape[0], test_data.shape[3], test_data.shape[1], test_data.shape[2], 1])
        for i in range(test_data.shape[3]):
            test_data_out[:, i, :, :, :] = test_data_new[:, :, :, i]



        train_data_out, train_label = shuffle(train_data_out, train_label)
        train_label = to_categorical(train_label, 20)
        test_label = to_categorical(test_label, 20)

        return train_data_out, train_label, test_data_out, test_label


def extract_data_self(filename):
    """Extract the images into a 4D tensor [image index, y, x, channels].


    """
    print('Extracting', filename)
    # get data from h5py
    file = h5py.File(filename, 'r')
    train_data = file['train_data']

    test_data = file['test_data']
    test_label = file['test_label']

    test_label = np.int64(test_label)
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]


    train_data_2dim = train_data.value.reshape([train_num, 1 * 5 * 5 * 224])
    train_data_to1 = preprocessing.minmax_scale(train_data_2dim, feature_range=(0, 1), axis=1, copy=True)
    train_data_new = train_data_to1.reshape([train_num, 224, 5, 5, 1])

    test_data_2dim = test_data.value.reshape([test_num, 1 * 5 * 5 * 224])
    test_data_to1 = preprocessing.minmax_scale(test_data_2dim, feature_range=(0, 1), axis=1, copy=True)
    test_data_new = test_data_to1.reshape([test_num, 224, 5, 5, 1])

    '''
    train_data_new=train_data.value.reshape([train_num, 224, 5, 5, 1])
    test_data_new=test_data.value.reshape([test_num, 224, 5, 5, 1])
    '''

    train_label=train_data_new.reshape([-1])
    test_label=test_data_new.reshape([-1])

    return train_data_new, train_label, test_data_new, test_label


def residual_block_concat(incoming, nb_blocks, out_channels, downsample=False,
                   downsample_strides=2, activation='relu', batch_norm=True,
                   bias=True, weights_init='variance_scaling',
                   bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="ResidualBlock"):
    """ Residual Block.

    A residual block as described in MSRA's Deep Residual Network paper.
    Full pre-activation architecture is used here.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        out_channels: `int`. The number of convolutional filters of the
            convolution layers.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'ShallowBottleneck'.

    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    with vscope as scope:
        name = scope.name #TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_3d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_3d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_3d(identity, 1,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0],[0,0], [ch, ch]])
                in_channels = out_channels

            resnet =tf.concat(1,[resnet,identity])
    return resnet
def self(x_train, y_train, x_test, y_test):
    int_put = input_data(shape=[None, 224, 5, 5, 1], )

    conv1 = conv_3d(int_put, 24, [24, 3, 3], padding='VALID', strides=[1, 1, 1, 1, 1], activation='prelu',)
    print('conv1', conv1.get_shape().as_list())
    batch_norm = batch_normalization(conv1)

    conv2 = conv_3d(batch_norm, 12, [24, 3, 3], padding='VALID', strides=[1, 1, 1, 1, 1], activation='prelu',)
    print('conv2', conv2.get_shape().as_list())
    batch_norm_con = batch_normalization(conv2)


    decon2=conv_3d_transpose(batch_norm_con,24,[24,3,3],padding='VALID',output_shape=[201,3,3,24])
    batch_norm=batch_normalization(decon2)
    print ('a')
    decon2 = conv_3d_transpose(batch_norm, 1, [24, 3, 3], padding='VALID',output_shape=[224,5,5,1])
    batch_norm = batch_normalization(decon2)


    network = regression(batch_norm,optimizer='Adagrad', loss='mean_square', learning_rate=0.01,metric='R2')

    feature_model = tflearn.DNN(network)
    feature_model.load('my_model_self.tflearn')
    x_feature = feature_model.predict(x_train)
    save_hdf5(x_feature)
    print('asd')

def showpicture(data):
    #plt.imshow(data[0,2,:, :], plt.cm.gray)
    plt.imshow(data, plt.cm.gray)
    plt.show()

def psnr( x_true,x_pre):
        psnr_summ=0
        mse_sum=0
        for j in range( x_true.shape[1]):
            x_true_mse = x_true[:, j, :, :, :].reshape([-1])
            x_pre_mse = x_pre[:, j, :, :, :].reshape([-1])
            mse = mean_squared_error(x_true_mse, x_pre_mse, )
            mse = mse / (np.float64(x_true.shape[0]))
            mse_sum=mse_sum+mse
            maxx = np.max(x_true[:, j, 2, 2, :])
            maxx = math.pow(maxx, 2)
            maxx = maxx / mse
            if maxx==0:
                maxx=0
            else:
                maxx = math.log10(maxx)
                maxx = 10 * maxx
            psnr_summ = psnr_summ + maxx

        psnr=psnr_summ /(np.float64(x_true.shape[1]))

        #print ('mse',mse)
        print('psnr', psnr)
        print('mse',mse)

def save_hdf5(data):
    filename_new = 'feature.h5'
    file = h5py.File(filename_new, 'w')
    file.create_dataset('train_data', data=data)
    file.close()

def picture():
    file = h5py.File('feature.h5', 'r')
    train_data = file['train_data'].value
    showpicture(train_data[0,2,:,:,0])

def truepicture():
    file = h5py.File('indian_pines_tensor_3d_2.h5', 'r')
    train_data = file['train_data'].value
    showpicture(train_data[0,:,:,2,0])


if __name__ == '__main__':
    #file_name = 'indian_pines_tensor_3d_2.h5'
    #x_train, y_train, x_test, y_test = extract_data_self(file_name)
    #print ('self')
    #self(x_train, y_train, x_test, y_test)
    picture()
    truepicture()
