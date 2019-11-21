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
from pooling.ow_constraints import (
    PosUnitModule,
)

class Test_PosUnitConstraint(unittest.TestCase):
    """ Test OW1 Pooling"""
    def setUp(self):
        self.x_input = np.random.rand(4, 4, 2)
        self.x_input = np.expand_dims(self.x_input, axis=0)
        self.input_tensor = Input(shape=self.x_input.shape[1:])
        self.y = np.array([0, 1]).reshape(1,2)
        self.pool_size = (4, 4)
        self.optimizer = keras.optimizers.Adam()

    def test_ow1_constraint(self):
        """ Test ow-pool with mean weights"""
        constraint = PosUnitModule()
        neg_ones_ini = -np.ones(self.pool_size[0] * self.pool_size[1])
        w_initializer = Constant(value=neg_ones_ini)
        x = OW1Pooling2D(pool_size=self.pool_size, name='ow', padding='same',
                          weights_constraint=constraint,
                          weights_initializer=w_initializer)(self.input_tensor)
        x = Flatten()(x)
        x = Activation('softmax')(x)
        ow_model = Model(self.input_tensor, x)
        ow_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                         metrics=['accuracy'])
        ow_model.fit(self.x_input, self.y, epochs=1, verbose=0)
        ow_layer = ow_model.layers[-3]
        ow_weights = ow_layer.get_weights()[0]
        np.testing.assert_array_almost_equal(np.sum(ow_weights),1)
        self.assertFalse(np.sum(ow_weights<0))

    def test_ow2_constraint(self):
        """ Test ow-pool with mean weights"""
        constraint = PosUnitModule(axis=1)
        neg_ones_ini = -np.ones((self.x_input.shape[3],
                                 self.pool_size[0] * self.pool_size[1]))
        w_initializer = Constant(value=neg_ones_ini)
        x = OW2Pooling2D(pool_size=self.pool_size, name='ow', padding='same',
                          weights_constraint=constraint,
                          weights_initializer=w_initializer)(self.input_tensor)
        x = Flatten()(x)
        x = Activation('softmax')(x)
        ow_model = Model(self.input_tensor, x)
        ow_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                         metrics=['accuracy'])
        ow_model.fit(self.x_input, self.y, epochs=5, verbose=0)
        ow_layer = ow_model.layers[-3]
        ow_weights = ow_layer.get_weights()[0]

        np.testing.assert_array_almost_equal(np.sum(ow_weights, axis=1), [1, 1])
        self.assertFalse(np.sum(ow_weights<0))

    def test_ow3_constraint(self):
        """ Test ow-pool with mean weights"""
        constraint = PosUnitModule(axis=3)
        neg_ones_ini = -np.ones((1, 1, self.x_input.shape[3],
                                 self.pool_size[0] * self.pool_size[1]))
        w_initializer = Constant(value=neg_ones_ini)
        x = OW3Pooling2D(pool_size=self.pool_size, name='ow', padding='same',
                          weights_constraint=constraint,
                          weights_initializer=w_initializer)(self.input_tensor)
        x = Flatten()(x)
        x = Activation('softmax')(x)
        ow_model = Model(self.input_tensor, x)
        ow_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                         metrics=['accuracy'])
        ow_model.fit(self.x_input, self.y, epochs=5, verbose=0)
        ow_layer = ow_model.layers[-3]
        ow_weights = ow_layer.get_weights()[0]
        np.testing.assert_array_almost_equal(np.sum(ow_weights, axis=3),
                                             [[[1, 1]]], decimal=5)
        self.assertFalse(np.sum(ow_weights<0))
