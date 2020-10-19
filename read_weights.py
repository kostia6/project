from tensorflow.keras.models import load_model


def read_weights(model):
    res = {}
    layer_num = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            res[layer_num] = weights

        layer_num += 1

    return res


if __name__ == "__main__":
    print("Started test")
    model = load_model('trained_model_bn.h5')
    weights_map = read_weights(model)
    print("Finished test")
