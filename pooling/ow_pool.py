# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pooling.legacy_sort import sort_p2x2, sort_p3x3

def sort_p(x):

    _, pool_height, pool_width, channels, elems = x.get_shape().as_list()
    x = tf.reshape(x, [-1, elems]) # Reshape tensor
    x = tf.contrib.framework.sort (x, axis=-1,
                            direction='DESCENDING', name=None)
    # Reshape
    x = tf.reshape(x, [-1, pool_height, pool_width, channels, elems]) # Reshape tensor
    return x

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor

def ow_pooling(x, weights, padding='SAME', strides=(2, 2), pool_size = (2, 2), norm='None', sort=True):

    _, height, width, channels = x.get_shape().as_list()
    pad_bottom = pool_size[0] * height%pool_size[0]
    pad_right = pool_size[1] * width%pool_size[1]

    if(padding=='SAME'): # Complete size to pad 'SAME'
        paddings = tf.constant([[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT")

    # Extract pooling regions
    stride = [1, strides[0], strides[1], 1]
    ksize = [1, pool_size[0], pool_size[1], 1]

    x = tf.extract_image_patches(x, ksizes = ksize, strides = stride,
                        rates = [1, 1, 1, 1], padding='SAME')

    _, pool_height, pool_width, elems = x.get_shape().as_list()

    # Extract pooling regions for each channel
    elems =  int(elems / channels)
    x = tf.reshape(x, [-1, pool_height, pool_width, elems, channels]) # Reshape tensor
    x = tf.transpose(x,perm = [0, 1, 2, 4, 3])

    # Sort values for pooling
    if sort:
        if ((pool_size[0] == 2) and (pool_size[1] == 2)):
            x = sort_p2x2(x)
        if ((pool_size[0] == 3) and (pool_size[1] == 3)):
            x = sort_p3x3(x)
        else:
            x = sort_p(x)

    if norm == 'w_norm':
        assign_op = weights.assign(tf.div(weights,tf.reduce_sum(tf.abs(weights))))
        with tf.control_dependencies([assign_op]):
            x = weights * x
    elif norm == 'w_norm_p':
        assign_op = weights.assign(tf.div(tf.maximum(weights, 0.0001),
                                          tf.reduce_sum(tf.maximum(weights, 0.0001))))
        with tf.control_dependencies([assign_op]):
            x = weights * x
    elif norm == 'w2_norm':
        assign_op = weights.assign(
                        tf.div(weights,
                        tf.transpose(tf_repeat([tf.reduce_sum(tf.abs(weights),1)],
                                             [tf.shape(weights)[1],1])))
                        )
        with tf.control_dependencies([assign_op]):
            x = weights * x
    elif norm == 'w2_norm_p':
        assign_op = weights.assign(
                        tf.div(tf.maximum(weights, 0.0001),
                        tf.transpose(tf_repeat([tf.reduce_sum(tf.maximum(weights, 0.0001),1)],
                                             [tf.shape(weights)[1],1])))
                        )
        with tf.control_dependencies([assign_op]):
            x = weights * x
    else:
        x = weights * x

    x = tf.reduce_sum(x,4)  #Reduce de 4th dimension
    return x
