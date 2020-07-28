import relu_activation
import sigmoid_activation
import settings

import matplotlib.pyplot as plt
import seaborn as sbs

sbs.set()

max_value = 100
max_value_taylor = 100
epsilon = 0.01


def main():
    print("Starting test")
    cur_max_degree = settings.max_degree
    test_relu(cur_max_degree)
    test_sigmoid_chabyshev(cur_max_degree)
    test_sigmoid_taylor()
    test_relu_taylor()
    test_relu_polyfit()
    print("Finished test")


def test_sigmoid_chabyshev(max_degree):
    test_set = settings.create_test_set(settings.num_examples, max_value)
    run_test(test_set, sigmoid_activation.get_approx_func_chebyshev(max_degree), sigmoid_activation.get_real_func_sigmoid(),
             "Sigmoid-Chabyshev")


def test_relu_taylor():
    test_set = settings.create_test_set(settings.num_examples, max_value_taylor)
    run_test(test_set, sigmoid_activation.get_approx_func_taylor_integ(), relu_activation.get_real_func(),
             "Relu-taylor-Sigmoid-Integral")


def test_relu_polyfit():
    # Least squares polynomial fit
    test_set = settings.create_test_set(settings.num_examples, max_value)
    run_test(test_set, relu_activation.get_approx_func_polyfit(), relu_activation.get_real_func(),
             "Relu-polyfit")


def test_sigmoid_taylor():
    test_set = settings.create_test_set(settings.num_examples, max_value_taylor)
    run_test(test_set, sigmoid_activation.get_approx_func_taylor(), sigmoid_activation.get_real_func_sigmoid(),
             "Sigmoid-taylor")


def test_relu(max_degree):
    test_set = settings.create_test_set(settings.num_examples, max_value)
    run_test(test_set, relu_activation.get_approx_func(max_degree), relu_activation.get_real_func(), "Relu-Chabyshev")


def run_test(test_set, approx_func, orig_func, test_name):
    test_set.sort()
    error = [None] * len(test_set)
    total_error = 0.0
    real_results = [orig_func(x) for x in test_set]
    approx_results = [approx_func(x) for x in test_set]
    for i in range(0, len(test_set)):
        real_result = real_results[i]
        approx_result = approx_results[i]
        if abs(real_result) > epsilon:
            error[i] = abs(real_result-approx_result)/real_result
        else:
            error[i] = abs(real_result - approx_result) / (real_result+1)

        total_error += abs(real_result - approx_result)

    average_error = (sum(error)/len(error))
    average_error_size = total_error/len(test_set)
    total_error_divided = total_error/sum(real_results)
    print("The average error (%s test) is: %.4f ,average error size is: %.4f ,"
          "total error divided by sum of results %.4f"
          % (test_name, average_error, average_error_size, total_error_divided))
    sbs.set_style("whitegrid", {"axes.grid": True})
    plt.figure()
    plt.grid(b=True)
    plt.plot(test_set, real_results, label="real")
    plt.plot(test_set, approx_results, label="approximation")
    plt.legend(fancybox=True)
    plt.title(test_name)
    plt.savefig(test_name)


if __name__ == "__main__":
    main()
