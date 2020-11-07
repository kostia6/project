from tensorflow.keras.models import load_model
import keras
import numpy as np
import multi_party_mediator
import sys
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

def calc_batch_normalization(input, weights, epsilon):
    flat_input = input.flatten()
    #output = np.zeros((input.shape[0] * input.shape[1]), dtype=np.uint8)

    # calculate (batch - self.moving_mean) / (self.moving_var + epsilon) * gamma + beta
    # where 0 - gamma, 1 - beta, 2 - moving_mean, 3 - moving_var
    # see https://keras.io/api/layers/normalization_layers/batch_normalization/
    #output = (flat_input - weights[2]) / ((weights[3] + epsilon) * weights[0] + weights[1])
    output = calc_batch_normalization2(flat_input, weights[2], weights[3], weights[1], weights[0], epsilon)
    return output


def calc_layer(input_val, layer_weights, activation_function):
    input_val = input_val.flatten()
    weights = layer_weights[0]
    bias = layer_weights[1]
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


def test_one(model_weights, input, expected_output):
    current_layer_input = input
    for layer_num in model_weights:
        layer_weights = model_weights[layer_num]['weights']
        layer_name = model_weights[layer_num]['name']
        layer = model_weights[layer_num]['layer']
        current_sum = np.sum(current_layer_input)
        if layer_name == 'batch_normalization':
            current_layer_input = calc_batch_normalization(current_layer_input, layer_weights, model_weights[layer_num]['epsilon'])
        elif layer_name.startswith('hidden'):
            current_layer_input = calc_layer(current_layer_input, layer_weights, "relu")
        elif layer_name == 'output':
            current_layer_input = calc_layer(current_layer_input, layer_weights, "softmax")

    return current_layer_input.argmax() == expected_output


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
    model = load_model('trained_model.h5')
    #model = load_model('trained_model_no_batch.h5')
    weights_map = read_weights(model)
    start_test(weights_map)
    print("Finished test")
