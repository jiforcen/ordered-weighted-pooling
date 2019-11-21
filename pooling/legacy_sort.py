import numpy as np
import tensorflow as tf

def sort_p2x2(x):

    _, pool_height, pool_width, channels, elems = x.get_shape().as_list()
    x = tf.reshape(x, [-1, elems])
    rows, _ = x.get_shape().as_list()

    # 1st stage
    x_1 = tf.slice(x, [0, 0], [-1, 1])
    x_2 = tf.slice(x, [0, 1], [-1, 1])
    x_3 = tf.slice(x, [0, 2], [-1, 1])
    x_4 = tf.slice(x, [0, 3], [-1, 1])

    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x_34_greater = tf.greater(x_3, x_4)
    x_aux = tf.where(x_34_greater, x_3, x_4)
    x_4 = tf.where(tf.logical_not(x_34_greater), x_3, x_4)
    x_3 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4], axis=1))

    # 2nd stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4], axis=1))

    # 3rd stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4], axis=1))
    x = tf.reshape(x, [-1, pool_height, pool_width, channels, elems]) # Reshape tensor

    return x

def sort_p3x3(x):

    _, pool_height, pool_width, channels, elems = x.get_shape().as_list()
    x = tf.reshape(x, [-1, elems])
    rows, _ = x.get_shape().as_list()

    # 1st stage
    x_1 = tf.slice(x, [0, 0], [-1, 1])
    x_2 = tf.slice(x, [0, 1], [-1, 1])
    x_3 = tf.slice(x, [0, 2], [-1, 1])
    x_4 = tf.slice(x, [0, 3], [-1, 1])
    x_5 = tf.slice(x, [0, 4], [-1, 1])
    x_6 = tf.slice(x, [0, 5], [-1, 1])
    x_7 = tf.slice(x, [0, 6], [-1, 1])
    x_8 = tf.slice(x, [0, 7], [-1, 1])
    x_9 = tf.slice(x, [0, 8], [-1, 1])

    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x_34_greater = tf.greater(x_3, x_4)
    x_aux = tf.where(x_34_greater, x_3, x_4)
    x_4 = tf.where(tf.logical_not(x_34_greater), x_3, x_4)
    x_3 = x_aux

    x_45_greater = tf.greater(x_4, x_5)
    x_aux = tf.where(x_45_greater, x_4, x_5)
    x_5 = tf.where(tf.logical_not(x_45_greater), x_4, x_5)
    x_4 = x_aux

    x_56_greater = tf.greater(x_5, x_6)
    x_aux = tf.where(x_56_greater, x_5, x_6)
    x_6 = tf.where(tf.logical_not(x_56_greater), x_5, x_6)
    x_5 = x_aux

    x_67_greater = tf.greater(x_6, x_7)
    x_aux = tf.where(x_67_greater, x_6, x_7)
    x_7 = tf.where(tf.logical_not(x_67_greater), x_6, x_7)
    x_6 = x_aux

    x_78_greater = tf.greater(x_7, x_8)
    x_aux = tf.where(x_78_greater, x_7, x_8)
    x_8 = tf.where(tf.logical_not(x_78_greater), x_7, x_8)
    x_7 = x_aux

    x_89_greater = tf.greater(x_8, x_9)
    x_aux = tf.where(x_89_greater, x_8, x_9)
    x_9 = tf.where(tf.logical_not(x_89_greater), x_8, x_9)
    x_8 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], axis=1))

    # 2nd stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x_34_greater = tf.greater(x_3, x_4)
    x_aux = tf.where(x_34_greater, x_3, x_4)
    x_4 = tf.where(tf.logical_not(x_34_greater), x_3, x_4)
    x_3 = x_aux

    x_45_greater = tf.greater(x_4, x_5)
    x_aux = tf.where(x_45_greater, x_4, x_5)
    x_5 = tf.where(tf.logical_not(x_45_greater), x_4, x_5)
    x_4 = x_aux

    x_56_greater = tf.greater(x_5, x_6)
    x_aux = tf.where(x_56_greater, x_5, x_6)
    x_6 = tf.where(tf.logical_not(x_56_greater), x_5, x_6)
    x_5 = x_aux

    x_67_greater = tf.greater(x_6, x_7)
    x_aux = tf.where(x_67_greater, x_6, x_7)
    x_7 = tf.where(tf.logical_not(x_67_greater), x_6, x_7)
    x_6 = x_aux

    x_78_greater = tf.greater(x_7, x_8)
    x_aux = tf.where(x_78_greater, x_7, x_8)
    x_8 = tf.where(tf.logical_not(x_78_greater), x_7, x_8)
    x_7 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], axis=1))

    # 3rd stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x_34_greater = tf.greater(x_3, x_4)
    x_aux = tf.where(x_34_greater, x_3, x_4)
    x_4 = tf.where(tf.logical_not(x_34_greater), x_3, x_4)
    x_3 = x_aux

    x_45_greater = tf.greater(x_4, x_5)
    x_aux = tf.where(x_45_greater, x_4, x_5)
    x_5 = tf.where(tf.logical_not(x_45_greater), x_4, x_5)
    x_4 = x_aux

    x_56_greater = tf.greater(x_5, x_6)
    x_aux = tf.where(x_56_greater, x_5, x_6)
    x_6 = tf.where(tf.logical_not(x_56_greater), x_5, x_6)
    x_5 = x_aux

    x_67_greater = tf.greater(x_6, x_7)
    x_aux = tf.where(x_67_greater, x_6, x_7)
    x_7 = tf.where(tf.logical_not(x_67_greater), x_6, x_7)
    x_6 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], axis=1))


    # 4th stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x_34_greater = tf.greater(x_3, x_4)
    x_aux = tf.where(x_34_greater, x_3, x_4)
    x_4 = tf.where(tf.logical_not(x_34_greater), x_3, x_4)
    x_3 = x_aux

    x_45_greater = tf.greater(x_4, x_5)
    x_aux = tf.where(x_45_greater, x_4, x_5)
    x_5 = tf.where(tf.logical_not(x_45_greater), x_4, x_5)
    x_4 = x_aux

    x_56_greater = tf.greater(x_5, x_6)
    x_aux = tf.where(x_56_greater, x_5, x_6)
    x_6 = tf.where(tf.logical_not(x_56_greater), x_5, x_6)
    x_5 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], axis=1))

    # 5th stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x_34_greater = tf.greater(x_3, x_4)
    x_aux = tf.where(x_34_greater, x_3, x_4)
    x_4 = tf.where(tf.logical_not(x_34_greater), x_3, x_4)
    x_3 = x_aux

    x_45_greater = tf.greater(x_4, x_5)
    x_aux = tf.where(x_45_greater, x_4, x_5)
    x_5 = tf.where(tf.logical_not(x_45_greater), x_4, x_5)
    x_4 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], axis=1))

    # 6th stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x_34_greater = tf.greater(x_3, x_4)
    x_aux = tf.where(x_34_greater, x_3, x_4)
    x_4 = tf.where(tf.logical_not(x_34_greater), x_3, x_4)
    x_3 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], axis=1))

    # 7th stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x_23_greater = tf.greater(x_2, x_3)
    x_aux = tf.where(x_23_greater, x_2, x_3)
    x_3 = tf.where(tf.logical_not(x_23_greater), x_2, x_3)
    x_2 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], axis=1))

    # 8th stage
    x_12_greater = tf.greater(x_1, x_2)
    x_aux = tf.where(x_12_greater, x_1, x_2)
    x_2 = tf.where(tf.logical_not(x_12_greater),x_1, x_2)
    x_1 = x_aux

    x = tf.squeeze(tf.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], axis=1))

    # Reshape
    x = tf.reshape(x, [-1, pool_height, pool_width, channels, elems]) # Reshape tensor

    return x
