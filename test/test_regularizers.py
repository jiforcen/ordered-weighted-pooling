""" Unit test to check the correct behaviour of constraints"""
import unittest
import numpy as np
import keras
from keras.initializers import Constant
from keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    Input,
    Dense,
    Flatten,
    Activation,
    )
from keras.models import Model
from pooling.pooling_layers import (
    OW1Pooling2D,
    OW2Pooling2D,
    OW3Pooling2D,
)
from pooling.ow_regularizers import (
    ow1_reg_un,
    ow1_reg_uns,
    ow2_reg_un,
    ow2_reg_uns,
)
import tensorflow as tf

class Test_PosUnitConstraint(unittest.TestCase):
    """ Test OW1 Regularizers"""
    def setUp(self):
        self.x_input = np.random.rand(10, 4, 4, 2)
        self.input_tensor = Input(shape=self.x_input.shape[1:])
        self.y = np.array([0, 1]).reshape(1,2)
        self.y = np.ones((10, 2))
        self.pool_size = (4, 4)
        self.optimizer = keras.optimizers.Adam(lr=.01)

    def test_ow1_regularizer(self):
        """ Test ow-pool with mean weights"""
        def regularizer(weight_matrix):
            return ow1_reg_un(weight_matrix, .1, .1)
        neg_ones_ini = -np.ones(self.pool_size[0] * self.pool_size[1])/2
        w_initializer = Constant(value=neg_ones_ini)
        x = OW1Pooling2D(pool_size=self.pool_size, name='ow', padding='same',
                          weights_regularizer=regularizer,
                          weights_initializer=w_initializer)(self.input_tensor)
        x = Flatten()(x)
        x = Dense(10)(x)
        x = Dense(2)(x)
        x = Activation('softmax')(x)
        ow_model = Model(self.input_tensor, x)
        ow_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                         metrics=['accuracy'])
        ow_layer = ow_model.layers[-5]

        ow_weights = ow_layer.get_weights()[0]
        ow_model.fit(self.x_input, self.y, epochs=1000, verbose=0, batch_size=2)
        ow_layer = ow_model.layers[-5]
        ow_weights = ow_layer.get_weights()[0]
        np.testing.assert_array_almost_equal(np.sum(ow_weights), 1, decimal=2)
        self.assertFalse(np.sum(ow_weights<0))

class Test_RegOwa1(unittest.TestCase):
    """ Test OW1 Regularizers"""
    def setUp(self):
        self.equal_weights_ow1 = tf.constant(np.array([0.25, 0.25, 0.25, 0.25]),
                                                       dtype=tf.float32)
        self.not_equal_weights_ow1 = tf.constant(np.array([0.25, -0.25, 0.25, 0.25]),
                                                       dtype=tf.float32)
    def test_ow1_regularizer_un_case_0(self):
        """ Test ow1 regularizer"""
        reg_op = ow1_reg_un(self.equal_weights_ow1, .1, .1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertEqual(reg_val, 0)

    def test_ow1_regularizer_un_case_1(self):
        """ Test ow1 regularizer"""
        reg_op = ow1_reg_un(self.not_equal_weights_ow1, .1, 0)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertNotEqual(reg_val, 0)

    def test_ow1_regularizer_un_case_2(self):
        """ Test ow1 regularizer"""
        reg_op = ow1_reg_un(self.not_equal_weights_ow1, 0, .1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertNotEqual(reg_val, 0)

    def test_ow1_regularizer_uns_case_0(self):
        """ Test ow1 regularizer"""
        reg_op = ow1_reg_uns(self.equal_weights_ow1, .1, .1, .1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertEqual(reg_val, 0)

    def test_ow1_regularizer_uns_case_1(self):
        """ Test ow1 regularizer"""
        reg_op = ow1_reg_uns(self.not_equal_weights_ow1, 0, 0, 0)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertEqual(reg_val, 0)

    def test_ow1_regularizer_uns_case_2(self):
        """ Test ow1 regularizer"""
        reg_op = ow1_reg_uns(self.not_equal_weights_ow1, 0, 0, 1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertNotEqual(reg_val, 0)


class Test_RegOwa2(unittest.TestCase):
    """ Test OW2 Regularizers"""
    def setUp(self):
        self.equal_weights_ow2 = tf.constant(np.array([[0.25, 0.25, 0.25, 0.25],
                                                        [0.25, 0.25, 0.25, 0.25]]),
                                                       dtype=tf.float32)
        self.not_equal_weights_ow2 = tf.constant(np.array([[0.25, -0.25, 0.25, 0.25],
                                                        [0.25, -0.25, 0.25, 0.25]]),
                                                       dtype=tf.float32)
    def test_ow2_regularizer_un_case_0(self):
        """ Test OW2 Regularizers"""
        reg_op = ow2_reg_un(self.equal_weights_ow2, .1, .1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertEqual(reg_val, 0)

    def test_ow2_regularizer_un_case_1(self):
        """ Test OW2 Regularizers"""
        reg_op = ow2_reg_un(self.not_equal_weights_ow2, .1, 0)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertNotEqual(reg_val, 0)

    def test_ow2_regularizer_un_case_2(self):
        """ Test OW2 Regularizers"""
        reg_op = ow2_reg_un(self.not_equal_weights_ow2, 0, .1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertNotEqual(reg_val, 0)

    def test_ow2_regularizer_uns_case_0(self):
        """ Test OW2 Regularizers"""
        reg_op = ow2_reg_uns(self.equal_weights_ow2, .1, .1, .1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertEqual(reg_val, 0)

    def test_ow2_regularizer_uns_case_1(self):
        """ Test OW2 Regularizers"""
        reg_op = ow2_reg_uns(self.not_equal_weights_ow2, 0, 0, 0)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertEqual(reg_val, 0)

    def test_ow2_regularizer_uns_case_2(self):
        """ Test OW2 Regularizers"""
        reg_op = ow2_reg_uns(self.not_equal_weights_ow2, 0, 0, 1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            reg_val = sess.run(reg_op)
        self.assertNotEqual(reg_val, 0)