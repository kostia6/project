import random
import relu_activation
import sigmoid_activation
import settings

import matplotlib.pyplot as plt
import seaborn as sbs

sbs.set()

num_examples = 1000
max_value = 100
epsilon = 0.01


def main():
    print("Starting test")
    cur_max_degree = settings.max_degree
    test_relu(cur_max_degree)
    test_sigmoid(cur_max_degree)
    print("Finished test")


def test_sigmoid(max_degree):
    test_set = create_test_set(num_examples, max_value)
    run_test(test_set, sigmoid_activation.get_approx_func(max_degree), sigmoid_activation.get_real_func(), "Sigmoid")


def test_relu(max_degree):
    test_set = create_test_set(num_examples, max_value)
    run_test(test_set, relu_activation.get_approx_func(max_degree), relu_activation.get_real_func(), "Relu")


def run_test(test_set, approx_func, orig_func, test_name):
    error = [None] * len(test_set)
    for i in range(0, len(test_set)):
        real_result = orig_func(test_set[i])
        approx_result = approx_func(test_set[i])
        if abs(real_result) > epsilon:
            error[i] = abs(real_result-approx_result)/real_result
        else:
            error[i] = abs(real_result - approx_result) / (real_result+1)

    average_error = (sum(error)/len(error))
    print("The average error (%s test) is: %.4f" % (test_name, average_error))


    real_results = [orig_func(x) for x in test_set]
    approx_results = [approx_func(x) for x in test_set]
    sbs.set_style("whitegrid", {"axes.grid": True})
    plt.figure()
    plt.grid(b=True)
    plt.plot(test_set, real_results, label="real")
    plt.plot(test_set, approx_results, label="approximation")
    plt.legend(fancybox=True)
    plt.title(test_name)
    plt.savefig(test_name)

def create_test_set(number_examples, border):
    res = [None] * number_examples
    for i in range(0, number_examples):
        res[i] = (random.uniform(0, 1) * border*2)-border

    res.sort()
    return res

if __name__ == "__main__":
    main()
