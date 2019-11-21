from keras import backend as K
import numpy as np
import math
from keras.constraints import MinMaxNorm
from keras.initializers import Constant
from keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
    Add,
    Multiply,
    )
from keras.layers.pooling import(
    _Pooling2D,
    _GlobalPooling2D,
    )
from pooling import ow_pool

class _OWPooling2D(_Pooling2D):
    """OW pooling implementation
       Ordered Weighted Average - pooling
       Weights are learned during the training.
       """
    # @interfaces.legacy_pooling2d_support
    def __init__(self, pool_size=(2, 2),
                 strides=None, padding='valid',
                 data_format=None,
                 weights_initializer='ow_avg',
                 weights_regularizer=None,
                 weights_constraint=None,
                 weights_op='None',
                 sort=True, **kwargs):
        super(_OWPooling2D, self).__init__(pool_size, strides, padding,
                                            data_format, **kwargs)
        self.weights_initializer=weights_initializer
        self.weights_regularizer=weights_regularizer
        self.weights_constraint=weights_constraint
        self.weights_op=weights_op
        self.sort=sort

    def ow_weight_initializer(self, weights_shape):
        if self.weights_initializer == 'ow_avg':
            ini = np.ones(weights_shape) / weights_shape[-1]
            w_initializer = Constant(value=ini)
        else:
            w_initializer = self.weights_initializer
        return w_initializer

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        outputs =  ow_pool.ow_pooling(
                inputs,
                weights = self.kernel,
                padding = self.padding,
                strides = self.strides,
                pool_size = self.pool_size,
                norm=self.weights_op,
                sort=self.sort)
        return outputs


class OW1Pooling2D(_OWPooling2D):
    """OW1 pooling implementation
       Ordered Weighted Average - pooling
       Weights are learned during the training.
       """

    def build(self, input_shape):
        weights_shape = [self.pool_size[0] * self.pool_size[1]]
        weights_initializer = self.ow_weight_initializer(weights_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=weights_shape,
                                      initializer=weights_initializer,
                                      regularizer=self.weights_regularizer,
                                      constraint=self.weights_constraint,
                                      trainable=True)
        super(OW1Pooling2D, self).build(input_shape)


class OW2Pooling2D(_OWPooling2D):
    """OW2 pooling implementation
       Ordered Weighted Average - pooling
       Weights are learned during the training.
       """

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        weights_shape = [input_dim, self.pool_size[0] * self.pool_size[1]]
        weights_initializer = self.ow_weight_initializer(weights_shape)
        self.kernel = self.add_weight(shape=weights_shape,
                                      initializer=weights_initializer,
                                      name='kernel',
                                      regularizer=self.weights_regularizer,
                                      constraint=self.weights_constraint,
                                      trainable=True)
        super(OW2Pooling2D, self).build(input_shape)


class OW3Pooling2D(_OWPooling2D):
    """OW3 pooling implementation
       Ordered Weighted Average - pooling
       Weights are learned during the training.
       """

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = math.ceil((rows - self.pool_size[0]) / self.strides[0]) + 1
        cols = math.ceil((cols - self.pool_size[1]) / self.strides[1]) + 1
        weights_shape = [rows, cols, input_dim, self.pool_size[0] *
                         self.pool_size[1]]
        weights_initializer = self.ow_weight_initializer(weights_shape)
        self.kernel = self.add_weight(shape=weights_shape,
                                      initializer=weights_initializer,
                                      name='kernel',
                                      regularizer=self.weights_regularizer,
                                      constraint=self.weights_constraint,
                                      trainable=True)
        super(OW3Pooling2D, self).build(input_shape)
