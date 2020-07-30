from keras.models import Sequential
from keras.layers import Dense
import settings
from tensorflow import map_fn
import sigmoid_activation

from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf


def print_tensor(t):
    print("Type of every element:", t.dtype)
    #print("Number of dimensions:", t.ndim)
    print("Shape of tensor:", t.shape)
    print("Elements along axis 0 of tensor:", t.shape[0])
    print("Elements along the last axis of tensor:", t.shape[-1])
    #print("Total number of elements (3*2*4*5): ", tf.size(t).numpy())
    print("value as list:", t.shape.as_list())


def custom_activation2(x):
    #map_fn(lambda y: 1 / (1 + np.exp(-y)), x)
    if isinstance(x, tf.Tensor):
        if x.ndim > 1:
            return map_fn(custom_activation2, x)
        else:
            return map_fn(custom_activation, x)


def custom_activation(x):
    y = x*2
    z = y*y
    #print_tensor(x)
    #print_tensor(y)
    return z


def hello_world_example():
    elems = tf.ones([1, 2, 3], dtype=tf.int64)
    alternates = custom_activation(elems)
    tf.print(alternates)


def intermediate_world_example():
    elems = tf.ones([1, 2, 3], dtype=tf.float32)
    alternates = sigmoid_activation.get_approx_func_chebyshev(settings.max_degree)(elems*2)
    tf.print(alternates)


def real_world_example():
    model = Sequential()
    model.add(Dense(32, input_dim=784))
    sigmoid_activation.assign_custom_activation(model)


get_custom_objects().update({'custom_activation': Activation(custom_activation)})

if __name__ == "__main__":
    hello_world_example()
    intermediate_world_example()
    real_world_example()

