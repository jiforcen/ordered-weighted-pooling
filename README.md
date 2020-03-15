# ordered-weighted-pooling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

In this repository some custom pooling layers published in the paper **Learning ordered pooling weights in image classification** are implemented in Keras/TensorFlow. Also some useful constraints and regularizers for the given layers are implemented.
All layers are tested and explained in this repository.

## pooling-layers

### OW1Pooling2D

```
pooling_layers.OW1Pooling2D(pool_size=(2, 2), strides=None, padding='valid',data_format=None, weights_initializer='ow_avg', weights_regularizer=None, weights_constraint=None, weights_op='None')
```
Ordered Weighted Average Pooling operation for spatial data. For each pooling region activations are sorted and weighted according to a trainable weights learned during training.

##### Arguments

* **pool_size:** integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). `(2, 2)` will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.
* **strides:** Integer, tuple of 2 integers, or None. Strides values. If `None`, it will default to pool_size.
* **padding:** One of `"valid"` or `"same"` (case-insensitive).
* **data_format:** A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
* **weights_initializer:** Initializer of weighted average, by default weights are initialized with the mean, weights_regularizer: Regularizer function applied to the kernel weights,
* **weights_constraint:** Constraint function applied to the kernel matrix (see constraints)

##### Input shape

* If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, rows, cols, channels)`
* If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, rows, cols)`

##### Output shape

* If `data_format='channels_last'`: 4D tensor with shape: `(batch_size, pooled_rows, pooled_cols, channels)`
* If `data_format='channels_first'`: 4D tensor with shape: `(batch_size, channels, pooled_rows, pooled_cols)`

##### Weights shape

* weights_shape = [pool_size[0] * pool_size[1]]


### OW2Pooling2D

```
pooling_layers.OW2Pooling2D(pool_size=(2, 2), strides=None, padding='valid',data_format=None, weights_initializer='ow_avg', weights_regularizer=None, weights_constraint=None, weights_op='None')
```
Similar to OW1Pooling2D, but in this case one set of weights is learned for each channel.

##### Weights shape

* weights_shape = [channels, pool_size[0] * pool_size[1]]

### OW3Pooling2D

```
pooling_layers.OW3Pooling2D(pool_size=(2, 2), strides=None, padding='valid',data_format=None, weights_initializer='ow_avg', weights_regularizer=None, weights_constraint=None, weights_op='None')
```
Similar to OW1Pooling2D and OW2Pooling2D, but in OW3Pooling2D one set of weights is learned for each pooled region.

##### Weights shape

* weights_shape = [pooled_rows, pooled_cols, channels, pool_size[0] * pool_size[1]]


## Constraints
---
```
ow_constraints.PosUnitModule(rate=1.0, axis=0)
```

Constrains the weights to sum one and to have positive values. This constraint is requeride if we want a Weighted Average.


##### Arguments

* **rate:** rate for enforcing the constraint: weights will be rescaled to yield positive and unit sum, `rate=1.0` stands for strict enforcement of the constraint, while `rate<1.0` means that weights will be rescaled at each step to slowly move towrds a value inside the desired interval.
* **axis:** integer, axis along which to calculate weight norms. For instance, in order to ensure weighted average requirements you must set in `OW1: axis=0`, in `OW2: axis=1` and finally for `OW3: axis=2`.






## Testing
---

In order to check the correct behavior of the previous implementations some unit tests were implemented. You can check it with the next command:


```python
python -m unittest -v test/test_*
```
