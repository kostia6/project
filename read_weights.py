from tensorflow.keras.models import load_model
import keras
import numpy as np
import multi_party_mediator

def read_weights(model):
    res = {}
    layer_num = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            res[layer_num] = {'weights': weights, 'name': layer.name}

        layer_num += 1

    return res


def softmax_activation(x):
    return np.exp(x) / sum(np.exp(x))


def calc_batch_normalization(input, weights):
    # TODO
    output = np.zeros((input.shape[0] * input.shape[1]), dtype=np.uint8)
    return output


def calc_layer(input_val, layer_weights, activation_function):
    weights = layer_weights[0]
    bias = layer_weights[1]
    output = np.zeros((bias.shape[0]), dtype=np.float32)
    activation_function = multi_party_mediator.get_relu_activation_numpy() if activation_function == 'relu' else \
        softmax_activation

    # apply weights TODO

    # apply bias
    output += bias

    # apply activation
    output = activation_function(output)

    return output


def test_one(model_weights, input, expected_output):
    current_layer_input = input
    for layer_num in model_weights:
        layer_weights = model_weights[layer_num]['weights']
        layer_name = model_weights[layer_num]['name']
        if layer_name == 'batch_normalization':
            current_layer_input = calc_batch_normalization(current_layer_input, layer_weights)
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
    weights_map = read_weights(model)
    start_test(weights_map)
    print("Finished test")
