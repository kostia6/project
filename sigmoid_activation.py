import numpy as np
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import settings
from chebyshev import Chebyshev
from sigmoid import Sigmoid

chebyshev_func_sigmoid = Chebyshev(settings.min_value, settings.max_value, settings.max_degree,
                                   lambda a: 1 / (1 + np.exp(-a)))
taylor_func_sigmoid = Sigmoid(lambda a: 1 / (1 + np.exp(-a)))


def get_taylor_func():
    return taylor_func_sigmoid


def get_derivative():
    return lambda a: get_real_func_sigmoid(a)*(1-get_real_func_sigmoid(a))


def get_real_func_sigmoid():
    return lambda a: 1 / (1 + np.exp(-a))


def get_approx_func_taylor_integ():
    return Sigmoid(lambda a: 1 / (1 + np.exp(-a)), True)


def get_approx_func_taylor():
    return Sigmoid(lambda a: 1 / (1 + np.exp(-a)))


def get_approx_func_chebyshev(max_degree):
    return Chebyshev(settings.min_value, settings.max_value, max_degree, lambda a: 1 / (1 + np.exp(-a)))


def custom_activation_chebyshev_sigmoid(x):
    return chebyshev_func_sigmoid.eval(x)


def assign_custom_activation(model):
    get_custom_objects().update({'custom_activation': Activation(chebyshev_func_sigmoid)})
    model.add(Activation(custom_activation_chebyshev_sigmoid, name='ChebyshevActivationSig'))
