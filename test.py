import random
import relu_activation
import sigmoid_activation

num_examples = 1000
max_value = 100
epsilon = 0.01


def main():
    print("Starting test")
    test_relu()
    test_sigmoid()
    print("Finished test")


def test_sigmoid():
    test_set = create_test_set(num_examples, max_value)
    run_test(test_set, sigmoid_activation.get_approx_func(), sigmoid_activation.get_real_func(), "Sigmoid")


def test_relu():
    test_set = create_test_set(num_examples, max_value)
    run_test(test_set, relu_activation.get_approx_func(), relu_activation.get_real_func(), "Relu")


def run_test(test_set, approx_func, orig_func, test_name):
    error = [None] * len(test_set)
    for i in range(0, len(test_set)):
        real_result = orig_func(test_set[i])
        approx_result = approx_func(test_set[i])
        if abs(real_result) > epsilon:
            error[i] = abs(real_result-approx_result)/real_result
        else:
            error[i] = abs(real_result - approx_result) / (real_result+1)

    average_error = 100*(sum(error)/len(error))
    print("The average error (%s test) is: %.4f%%" % (test_name, average_error))


def create_test_set(number_examples, border):
    res = [None] * number_examples
    for i in range(0, number_examples):
        res[i] = (random.uniform(0, 1) * border*2)-border

    return res


if __name__ == "__main__":
    main()
