from tensorflow.keras.models import load_model
import keras
import numpy as np
import multi_party_mediator
import sys
import time
import sympy
import tensorflow as tf

def read_weights(model):
    res = {}
    layer_num = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            res[layer_num] = {'weights': weights, 'name': layer.name, 'layer': layer}

        if layer.name == 'batch_normalization':
            res[layer_num]['epsilon'] = layer.epsilon

        layer_num += 1

    return res


def relu(x):
   return np.maximum(0, x)


def softmax_activation(x):
    return np.exp(x) / sum(np.exp(x))

def calc_batch_normalization2(input, mean, variance, offset, scale, variance_epsilon):
    # see: https://github.com/tensorflow/tensorflow/blob/e500cde7086258b516323d3c62b22853aedde207/tensorflow/python/ops/nn_impl.py#L1569
    inv = 1.0/np.sqrt(variance + variance_epsilon)
    inv *= scale

    return input * inv + offset - mean * inv

def calc_batch_normalization(input, weights, epsilon, is_debug):
    if is_debug:
        return None
    flat_input = input.flatten()
    #output = np.zeros((input.shape[0] * input.shape[1]), dtype=np.uint8)

    # calculate (batch - self.moving_mean) / (self.moving_var + epsilon) * gamma + beta
    # where 0 - gamma, 1 - beta, 2 - moving_mean, 3 - moving_var
    # see https://keras.io/api/layers/normalization_layers/batch_normalization/
    #output = (flat_input - weights[2]) / ((weights[3] + epsilon) * weights[0] + weights[1])
    output = calc_batch_normalization2(flat_input, weights[2], weights[3], weights[1], weights[0], epsilon)
    return output


def calc_layer(input_val, layer_weights, activation_function, is_debug, layer_name):
    weights = layer_weights[0]
    bias = layer_weights[1]
    if is_debug:
        print("Printing layer: " + layer_name)
        num_neurons_layer = len(layer_weights[0][0])
        weights = weights.transpose()
        values = []
        # apply weights
        for i in range(0, num_neurons_layer):
            current_weights = weights[i]
            current_value = ""

            for c in range(0, len(current_weights)):
                current_value += str(current_weights[c]) + " * " + "x" + str(c) + " + "
            current_value = current_value[:-2]
            values.append(current_value)

        # apply bias
        for i in range(0, num_neurons_layer):
            values[i] = "(" + values[i] + ")" + "+" + str(bias[i])

        # apply activation ???
        for i in range(0, num_neurons_layer):
            print_str = "c" + str(i) + "=" + values[i]
            print(print_str)
        return None
    input_val = input_val.flatten()
    max_degree = None
    if len(sys.argv) > 1:
        max_degree = int(sys.argv[1])

    activation_function = multi_party_mediator.get_relu_activation_numpy(max_degree) if activation_function == 'relu' else \
        softmax_activation

    # apply weights
    #output = np.dot(input_val.transpose(), weights)
    output = weights.transpose() @ input_val

    # check validity for debug
    first_column = weights[:, 0]
    first_out_value = np.dot(first_column, input_val)
    #assert(first_out_value == output[0])

    # apply bias
    output += bias

    # apply activation
    output = activation_function(output)
    #output = relu(output)

    return output


def test_one(model_weights, input, expected_output, is_debug=False):
    current_layer_input = input
    if is_debug:
        max_degree = None
        if len(sys.argv) > 1:
            max_degree = int(sys.argv[1])
        activation_function = multi_party_mediator.get_relu_activation_numpy(max_degree)
        activation_str = activation_function.print_str()
        print(activation_str)
        #print(len(activation_str))

        #x = sympy.symbols('x')
        #exp = sympy.simplify(activation_str)
        #print(exp.evalf(subs={x: 1.0}, n=100))
        #print(activation_function(1.0))

    for layer_num in model_weights:
        layer_weights = model_weights[layer_num]['weights']
        layer_name = model_weights[layer_num]['name']
        layer = model_weights[layer_num]['layer']
        current_sum = np.sum(current_layer_input)
        if layer_name == 'batch_normalization':
            current_layer_input = calc_batch_normalization(current_layer_input, layer_weights, model_weights[layer_num]['epsilon'], is_debug)
        elif layer_name.startswith('hidden'):
            current_layer_input = calc_layer(current_layer_input, layer_weights, "relu", is_debug, layer_name)
        elif layer_name == 'output':
            current_layer_input = calc_layer(current_layer_input, layer_weights, "softmax", is_debug, layer_name)

    return current_layer_input.argmax() == expected_output if not is_debug else None


def start_test(model_weights):
    f_mnist = keras.datasets.fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = f_mnist.load_data()
    num_test_items = X_test.shape[0]
    correct_results = 0
    for i in range(0, num_test_items):
        input_shape = X_test[i]
        expected_output = Y_test[i]
        if test_one(model_weights, input_shape, expected_output):
            correct_results += 1

    print("Correct results: " + str(correct_results/num_test_items))


if __name__ == "__main__":
    print("Started test")
    start = time.time()
    is_debug = True if '--debug' in sys.argv else False
    model = load_model('trained_model.h5')
    #model = load_model('trained_model_no_batch.h5')
    weights_map = read_weights(model)
    if not is_debug:
        start_test(weights_map)
    else:
        test_one(weights_map, None, None, True)

    end = time.time()
    print("Finished test in {:.0f} seconds".format(end - start))
