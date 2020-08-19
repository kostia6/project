from chebyshev import Chebyshev
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import settings
from sigmoid import Sigmoid
from sigmoid_activation import get_real_func_sigmoid
import tensorflow as tf
import keras as kr

chebyshev_func_relo = Chebyshev(settings.min_value, settings.max_value, settings.max_degree, lambda a: max(a, 0))
deriative_func_relo = Sigmoid(get_real_func_sigmoid())


def get_deriative_func_relo():
    return deriative_func_relo;


def get_real_func():
    return lambda a: max(a, 0)


def get_approx_func_polyfit():
    return Sigmoid(lambda a: max(a, 0), False, True)


def get_approx_func(max_degree, min_value=settings.min_value, max_value=settings.max_value):
    return Chebyshev(min_value, max_value, max_degree, lambda a: max(a, 0))


def custom_activation_chebyshev_relu(x):
    return chebyshev_func_relo.eval(x)


def assign_custom_activation(model):
    get_custom_objects().update({'custom_activation': Activation(custom_activation_chebyshev_relu)})
    model.add(Activation(custom_activation_chebyshev_relu, name='ChebyshevActivationRelu'))

#print(tf.__version__)
#print(kr.__version__)