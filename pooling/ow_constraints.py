from keras import backend as K
import numpy as np
import math
from keras.constraints import Constraint
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

class PosUnitModule(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.

    # Arguments
        rate: rate for enforcing the constraint: weights will be
            rescaled to yield positive and unit sum, rate=1.0 stands
            for strict enforcement of the constraint, while rate<1.0
            means that weights will be rescaled at each step to slowly
            move towards a value inside the desired interval.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, rate=1, axis=0):
        self.axis = axis
        self.rate = rate

    def __call__(self, w):
        inc = K.minimum(w, K.epsilon()) * self.rate
        pos = w - K.min(inc, axis=self.axis, keepdims=True)
        abs_sum = K.sum(K.abs(pos), axis=self.axis, keepdims=True)
        desired = self.rate + (1 - self.rate) * abs_sum
        return pos * desired/ (K.maximum(K.epsilon(), abs_sum))

    def get_config(self):
        return {'axis': self.axis,
                'rate': self.rate}