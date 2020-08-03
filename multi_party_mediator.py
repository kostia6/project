import relu_activation
import sigmoid_activation
import settings
import operator
import random
import threading
import sigmoid
import chebyshev
import tempfile
import os
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.models import load_model


# Create n lists whose sum is arr
def divide_coefficients(arr, n):
    current_sum = [0]*len(arr)
    res = [0]*n
    for i in range(0, n):
        current = [0] * len(arr)
        if i == n - 1 or n == 1:
            current = list(map(operator.sub, arr, current_sum))
        else:
            current = [random.uniform(0, 1) * arr[x] for x in range(0, len(current))]

        current_sum = list(map(operator.add, current_sum, current))
        res[i] = current

    return res


class MyThread (threading.Thread):
    def __init__(self, threadID, name, coefs, func, x):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.coefs = coefs
        self.func = func
        self.x = x
        self.res = 0

    def run(self):
        self.res = self.func(self.coefs, self.x)

    def get_result(self):
        return self.res


class MultiPartyMediator:

    def __init__(self, activation_type="Relu", polynom_type="polyfit"):
        self.activation_type = activation_type
        self.polynom_type = polynom_type
        if activation_type == "Relu":
            if polynom_type == "polyfit":
                self.activation = relu_activation.get_approx_func_polyfit()
                self.coefficients = self.activation.get_polyfit_coefficients()
                self.func = sigmoid.eval_polyfit_extern
            elif polynom_type == "chebyshev":
                self.activation = relu_activation.get_approx_func(settings.max_degree)
                self.coefficients = self.activation.get_coefficients()
                self.func = chebyshev.eval_extern
        elif activation_type == "Sigmoid":
            if polynom_type == "chebyshev":
                self.activation = sigmoid_activation.get_approx_func_chebyshev(settings.max_degree)
                self.coefficients = self.activation.get_coefficients()
                self.func = chebyshev.eval_extern


        # create n threads and assign them coefficients
        self.coefs = divide_coefficients(self.coefficients, settings.num_threads)
        self.threads = [0] * settings.num_threads

    # Calculate value of x
    def start(self, x):
        res = 0
        for i in range(0, settings.num_threads):
            self.threads[i] = MyThread(i, "Thread", self.coefs[i], self.func, x)
            self.threads[i].start()

        for t in self.threads:
            t.join()

        for t in self.threads:
            res += t.get_result()

        return res


relu_chab = MultiPartyMediator("Relu", "chebyshev")
sigmoid_chab = MultiPartyMediator("Sigmoid", "chebyshev")


def _apply_activation_model(model, active, active_name, custom_objects=None):
    print(model.get_config())
    for layer in model.layers:
        if isinstance(layer, Activation) and hasattr(layer, 'activation'):
            if active_name in layer.output.name:
                layer.name = layer.name + "_custom"
                layer.activation = active

    # might need parameters:https://stackoverflow.com/questions/43030721/cant-change-activations-in-existing-keras-model
    #model.compile(loss="categorical_crossentropy", optimizer='adam')
    print(model.get_config())

    model_path = os.path.join(tempfile.gettempdir(), "temp_name.h5")
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)

    return model


def relu_cheb_mediator(x):
    return relu_chab.start(x)


def sigmoid_cheb_mediator(x):
    return sigmoid_chab.start(x)


def register_activations():
    get_custom_objects().update({'relu_cheb': Activation(relu_cheb_mediator)})
    get_custom_objects().update({'sigmoid_cheb': Activation(sigmoid_cheb_mediator)})


def get_relu_activation():
    return Activation(relu_cheb_mediator, name="custom_activation_relu")


def get_sigmoid_activation():
    return Activation(sigmoid_cheb_mediator, name="custom_activation_sigmoid")


def replace_activation_model(model, activation_name):
    if activation_name == "relu":
        return _apply_activation_model(model, relu_cheb_mediator, "Relu", custom_objects=
        {"relu_cheb_mediator": relu_cheb_mediator})
    elif activation_name == "sigmoid":
        return _apply_activation_model(model, sigmoid_cheb_mediator, "Sigmoid", custom_objects=
        {"sigmoid_cheb_mediator": sigmoid_cheb_mediator})
    else:
        print("Unknown activation option: " + activation_name)
        return model


if __name__ == "__main__":
    register_activations()
    print("test polyfit")
    dummy = relu_activation.get_approx_func_polyfit()
    print(dummy(10))
    mediator = MultiPartyMediator()
    print(mediator.start(10))

    print("test chebyshev Relu")
    dummy = relu_activation.get_approx_func(settings.max_degree)
    print(dummy(10))
    mediator = MultiPartyMediator("Relu", "chebyshev")
    print(mediator.start(10))

    print("test chebyshev Sigmoid")
    dummy = sigmoid_activation.get_approx_func_chebyshev(settings.max_degree)
    print(dummy(10))
    mediator = MultiPartyMediator("Sigmoid", "chebyshev")
    print(mediator.start(10))
