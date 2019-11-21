''' Owa regularizers'''

import tensorflow as tf
from keras import backend as K

def ow1_reg_un(weight_matrix, a1, a2):
    return (a1 * K.abs(1-K.sum(weight_matrix)) +
            a2 * tf.reduce_sum(-tf.minimum(0.0, weight_matrix)))

def ow1_reg_uns(weight_matrix, a1, a2, a3):
    result = weight_matrix - tf.manip.roll(weight_matrix, shift=1, axis=0)
    result = tf.reduce_sum (tf.square(tf.slice(result,[1],[tf.shape(weight_matrix)[-1]-1])))
    return (a3 * result + ow1_reg_un(weight_matrix, a1, a2))

def ow2_reg_un(weight_matrix, a1, a2):
    return (a1 * K.sum(K.abs(1-K.sum(weight_matrix, axis=1))) +
            a2 * K.sum(-tf.minimum(0.0, weight_matrix)))

def ow2_reg_uns(weight_matrix, a1, a2, a3):
    result = weight_matrix - tf.manip.roll(weight_matrix, shift=1, axis=1)
    shape = tf.shape(weight_matrix)
    result = tf.reduce_sum (tf.square(tf.slice(result,[0, 1],[shape[0], shape[-1]-1])))
    return (a3 * result + ow2_reg_un(weight_matrix, a1, a2))
