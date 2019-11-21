
import tensorflow as tf
from keras import backend as K
import numpy as np

def ow2_reg_un(weight_matrix, a1, a2):
    return (a1 * K.sum(K.abs(1-K.sum(weight_matrix, axis=1))) +
            a2 * K.sum(-tf.minimum(0.0, weight_matrix)))



def ow2_reg_uns(weight_matrix, a1, a2, a3):
    result = weight_matrix - tf.manip.roll(weight_matrix, shift=1, axis=1)
    shape = tf.shape(weight_matrix)
    result = tf.reduce_sum (tf.square(tf.slice(result,[0, 1],[shape[0], shape[-1]-1])))
    return (a3 * result + ow2_reg_un(weight_matrix, a1, a2))


not_equal_weights_ow2 = tf.constant(np.array([[0.25, -0.25, 0.25, 0.25],
                                                        [0.25, 0.25, -0.25, 0.25]]),
                                                       dtype=tf.float32)


reg_op = ow2_reg_uns(not_equal_weights_ow2, 0, 0, 1)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    reg_val = sess.run(reg_op)

print (reg_val)