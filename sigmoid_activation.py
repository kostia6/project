import numpy as np
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import settings
from chebyshev import Chebyshev

chebyshev_func_sigmoid = Chebyshev(settings.min_value, settings.max_value, settings.max_degree,
                                   lambda a: 1 / (1 + np.exp(-a)))


def get_real_func():
    return lambda a: 1 / (1 + np.exp(-a))


def get_approx_func(max_degree):
    return Chebyshev(settings.min_value, settings.max_value, max_degree, lambda a: 1 / (1 + np.exp(-a)))


def custom_activation_chebyshev_sigmoid(x):
    return chebyshev_func_sigmoid.eval(x)


def assign_custom_activation(model):
    get_custom_objects().update({'custom_activation': Activation(chebyshev_func_sigmoid)})
    model.add(Activation(custom_activation_chebyshev_sigmoid, name='ChebyshevActivationSig'))
