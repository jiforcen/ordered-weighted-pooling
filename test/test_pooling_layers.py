""" Unit test to check the correct behaviour of custom pooling layers"""
import unittest
import numpy as np
from keras.initializers import Constant
from keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    Input,
    )
from keras.models import Model
from pooling.pooling_layers import (
    OW1Pooling2D,
    OW2Pooling2D,
    OW3Pooling2D,
)

class Test_OW1Pooling2D(unittest.TestCase):
    """ Test OW1 Pooling"""
    def setUp(self):
        self.x_input = np.random.rand(16, 16, 2)
        self.x_input = np.expand_dims(self.x_input, axis=0)
        self.input_tensor = Input(shape=self.x_input.shape[1:])
        self.pool_size = (4, 4)

    def test_ow1_mean(self):
        """ Test ow-pool with mean weights"""
        x = AveragePooling2D(pool_size=self.pool_size,
                             padding='same')(self.input_tensor)
        x = AveragePooling2D(pool_size=self.pool_size,
                             padding='same')(x)
        avg_model = Model(self.input_tensor, x)

        x = OW1Pooling2D(pool_size=self.pool_size,
                          padding='same')(self.input_tensor)
        x = OW1Pooling2D(pool_size=self.pool_size,
                          padding='same')(x)
        ow_model = Model(self.input_tensor, x)

        avg_prediction = avg_model.predict(self.x_input)
        ow_prediction = ow_model.predict(self.x_input)
        np.testing.assert_array_almost_equal(avg_prediction, ow_prediction)

    def test_ow1_max(self):
        """ Test ow-pool with max weights"""
        x = MaxPooling2D(pool_size=self.pool_size,
                         padding='same')(self.input_tensor)
        x = MaxPooling2D(pool_size=self.pool_size,
                         padding='same')(x)
        max_model = Model(self.input_tensor, x)

        max_ini = np.zeros(self.pool_size[0] * self.pool_size[1])
        max_ini[0] = 1
        w_initializer = Constant(value=max_ini)
        x = OW1Pooling2D(pool_size=self.pool_size, padding='same',
                          weights_initializer=w_initializer)(self.input_tensor)
        x = OW1Pooling2D(pool_size=self.pool_size, padding='same',
                          weights_initializer=w_initializer)(x)
        ow_model = Model(self.input_tensor, x)

        avg_prediction = max_model.predict(self.x_input)
        ow_prediction = ow_model.predict(self.x_input)
        np.testing.assert_array_almost_equal(avg_prediction, ow_prediction)


class Test_OW2Pooling2D(unittest.TestCase):
    """ Test OW2 Pooling"""
    def setUp(self):
        self.x_input = np.random.rand(16, 16, 2)
        self.x_input = np.expand_dims(self.x_input, axis=0)
        self.input_tensor = Input(shape=self.x_input.shape[1:])
        self.pool_size = (4, 4)

    def test_ow2_mean(self):
        """ Test ow-pool with mean weights"""
        x = AveragePooling2D(pool_size=self.pool_size,
                             padding='same')(self.input_tensor)
        x = AveragePooling2D(pool_size=self.pool_size,
                             padding='same')(x)
        avg_model = Model(self.input_tensor, x)

        x = OW2Pooling2D(pool_size=self.pool_size,
                          padding='same')(self.input_tensor)
        x = OW2Pooling2D(pool_size=self.pool_size,
                          padding='same')(x)
        ow_model = Model(self.input_tensor, x)

        avg_prediction = avg_model.predict(self.x_input)
        ow_prediction = ow_model.predict(self.x_input)
        np.testing.assert_array_almost_equal(avg_prediction, ow_prediction)

    def test_ow2_max(self):
        """ Test ow-pool with max weights"""
        x = MaxPooling2D(pool_size=self.pool_size,
                         padding='same')(self.input_tensor)
        x = MaxPooling2D(pool_size=self.pool_size,
                         padding='same')(x)
        max_model = Model(self.input_tensor, x)

        max_ini = np.zeros((self.x_input.shape[3],
                            self.pool_size[0] * self.pool_size[1]))
        max_ini[:, 0] = 1
        w_initializer = Constant(value=max_ini)
        x = OW2Pooling2D(pool_size=self.pool_size, padding='same',
                          weights_initializer=w_initializer)(self.input_tensor)
        x = OW2Pooling2D(pool_size=self.pool_size, padding='same',
                          weights_initializer=w_initializer)(x)
        ow_model = Model(self.input_tensor, x)

        avg_prediction = max_model.predict(self.x_input)
        ow_prediction = ow_model.predict(self.x_input)
        np.testing.assert_array_almost_equal(avg_prediction, ow_prediction)


class Test_OW3Pooling2D(unittest.TestCase):
    """ Test OW3 Pooling"""
    def setUp(self):
        self.x_input = np.random.rand(16, 16, 2)
        self.x_input = np.expand_dims(self.x_input, axis=0)
        self.input_tensor = Input(shape=self.x_input.shape[1:])
        self.pool_size = (4, 4)

    def test_ow3_mean(self):
        """ Test ow-pool with mean weights"""
        x = AveragePooling2D(pool_size=self.pool_size,
                             padding='same')(self.input_tensor)
        x = AveragePooling2D(pool_size=self.pool_size,
                             padding='same')(x)
        avg_model = Model(self.input_tensor, x)

        x = OW3Pooling2D(pool_size=self.pool_size,
                          padding='same')(self.input_tensor)
        x = OW3Pooling2D(pool_size=self.pool_size,
                          padding='same')(x)
        ow_model = Model(self.input_tensor, x)

        avg_prediction = avg_model.predict(self.x_input)
        ow_prediction = ow_model.predict(self.x_input)
        np.testing.assert_array_almost_equal(avg_prediction, ow_prediction)

    def test_ow3_max(self):
        """ Test ow-pool with max weights"""
        x = MaxPooling2D(pool_size=self.pool_size,
                         padding='same')(self.input_tensor)
        x = MaxPooling2D(pool_size=self.pool_size,
                         padding='same')(x)
        max_model = Model(self.input_tensor, x)

        max_ini = np.zeros((4, 4, self.x_input.shape[3],
                            self.pool_size[0] * self.pool_size[1]))
        max_ini[:, :, :, 0] = 1
        w_initializer = Constant(value=max_ini)
        x = OW3Pooling2D(pool_size=self.pool_size, padding='same',
                          weights_initializer=w_initializer)(self.input_tensor)

        max_ini = np.zeros((1, 1, self.x_input.shape[3],
                            self.pool_size[0] * self.pool_size[1]))
        max_ini[:, :, :, 0] = 1
        w_initializer = Constant(value=max_ini)
        x = OW3Pooling2D(pool_size=self.pool_size, padding='same',
                          weights_initializer=w_initializer)(x)
        ow_model = Model(self.input_tensor, x)

        avg_prediction = max_model.predict(self.x_input)
        ow_prediction = ow_model.predict(self.x_input)
        np.testing.assert_array_almost_equal(avg_prediction, ow_prediction)



if __name__ == '__main__':
    unittest.main()
