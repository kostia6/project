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
    test_results_relu = {'name': 'Relu', 'test_name': 'Approximation of degree ' + str(settings.max_degree)}
    test_results_sigmoid = {'name': 'Sigmoid', 'test_name': 'Approximation of degree ' + str(settings.max_degree)}
    test_set = settings.create_test_set(settings.num_examples, max_value)

    #test_range_relu(test_set)
    test_range_sigmoid(test_set)
    #test_relu_taylor(test_results_relu, test_set)
    #test_relu_polyfit(test_results_relu, test_set)
    #test_relu(settings.max_degree, test_results_relu, test_set)
    #test_chabyshev_num_examples()
    #plot_results(test_results_relu, test_set)

    #test_sigmoid_polyfit(test_results_sigmoid, test_set)
    #test_sigmoid_chebyshev(test_results_sigmoid, test_set)
    # test_sigmoid_taylor(test_results_sigmoid, test_set)
    #plot_results(test_results_sigmoid, test_set)
    print("Finished test")


def plot_results(test_results, test_set):
    sbs.set_style("whitegrid", {"axes.grid": True})
    plt.figure()
    plt.grid(b=True)
    test_name_main = test_results['test_name']
    real_func_name = test_results['name']
    test_results.pop('name')
    test_results.pop('test_name')
    first = True
    for test_name, test_result in test_results.items():
        plt.plot(test_set, test_result['approx'], label=test_name)
        if first:
            first = False
            plt.plot(test_set, test_result['real'], label=real_func_name)

    plt.legend(fancybox=True)
    plt.title(test_name_main)
    plt.savefig('plot_result')


def test_chabyshev_num_examples():
    test_name = "Error-by-degree"
    num_test_sets = 20
    test_set = []
    test_set_sizes = []
    test_set_results = []
    test_set_size = 10
    test_set_increase = 10
    for i in range(0, num_test_sets):
        test_set.append(settings.create_test_set(test_set_size, max_value))
        test_set_sizes.append(test_set_size)
        test_set_results.append(run_test(test_set[i], relu_activation.get_approx_func(test_set_size),
                                         relu_activation.get_real_func(), "Relu-Chabyshev", False))
        test_set_size += test_set_increase

    plt.figure()
    plt.grid(b=True)
    plt.plot(test_set_sizes, test_set_results, label="error")
    plt.legend(fancybox=True)
    plt.title(test_name)
    plt.savefig(test_name)


def test_sigmoid_chebyshev(test_results, test_set):
    test_result = run_test(test_set, sigmoid_activation.get_approx_func_chebyshev(settings.max_degree),
                           sigmoid_activation.get_real_func_sigmoid(), "Sigmoid-Chabyshev")
    test_results["Chebyshev"] = test_result


def test_relu_taylor(test_results, test_set):
    #test_set = settings.create_test_set(settings.num_examples, max_value_taylor)
    test_result = run_test(test_set, sigmoid_activation.get_approx_func_taylor_integ(), relu_activation.get_real_func(),
                           "Relu-taylor-Sigmoid-Integral")
    test_results["Sigmoid Integral"] = test_result


def test_relu_polyfit(test_results, test_set):
    # Least squares polynomial fit
    #test_set = settings.create_test_set(settings.num_examples, max_value)
    test_result = run_test(test_set, relu_activation.get_approx_func_polyfit(),
                           relu_activation.get_real_func(), "Relu-polyfit")
    test_results["Least Squares"] = test_result


def test_sigmoid_polyfit(test_results, test_set):
    # Least squares polynomial fit
    test_result = run_test(test_set, sigmoid_activation.get_approx_func_polyfit(),
                           sigmoid_activation.get_real_func_sigmoid(), "Sigmoid-polyfit")
    test_results["Least Squares"] = test_result


def test_sigmoid_taylor(test_results, test_set):
    test_result = run_test(test_set, sigmoid_activation.get_approx_func_taylor(),
                           sigmoid_activation.get_real_func_sigmoid(), "Sigmoid-taylor")
    test_results["Taylor"] = test_result


def test_relu(max_degree, test_results, test_set):
    #test_set = settings.create_test_set(settings.num_examples, max_value)
    test_result = run_test(test_set, relu_activation.get_approx_func(max_degree), relu_activation.get_real_func(), "Relu-Chabyshev")
    test_results["Chebyshev"] = test_result


def test_range_relu(test_set):
    approx_funcs = []
    for degree in range(settings.min_degree, settings.max_degree, settings.step_degree):
        approx_funcs.append((relu_activation.get_approx_func(degree), "Degree: %s" % (degree)) )
    run_multi_test(test_set, approx_funcs, relu_activation.get_real_func(), "Relu-Chebyshev", "Relu Chebyshev Approximation", "Relu")


def test_range_sigmoid(test_set):
    approx_funcs = []
    for degree in range(settings.min_degree, settings.max_degree, settings.step_degree):
        approx_funcs.append((sigmoid_activation.get_approx_func_chebyshev(degree), "Degree: %s" % (degree)) )
    run_multi_test(test_set, approx_funcs, sigmoid_activation.get_real_func_sigmoid(), "Sigmoid-Chebyshev", "Sigmoid Chebyshev Approximation", "Sigmoid")


def run_test(test_set, approx_func, orig_func, test_name, create_plot=True):
    test_set.sort()

    # error = [None] * len(test_set)
    # total_error = 0.0
    real_results = [orig_func(x) for x in test_set]
    approx_results = [approx_func(x) for x in test_set]

    errors = [abs(real_results[x] - approx_results[x]) / (real_results[x] + 1) for x in range(len(test_set))]
    square_errors = [pow(real_results[x] - approx_results[x], 2) for x in range(len(test_set))]
    total_error = sum([abs(real_results[x] - approx_results[x]) for x in range(len(test_set))])
    average_square_error = sum(square_errors)/len(test_set)
    test_result = {'real': real_results, 'approx': approx_results, 'avg_square_error': average_square_error}
    average_error = (sum(errors)/len(errors))
    average_absolute_error = total_error/len(test_set)

    if create_plot:
        print("The average error (%s test) is: %.5f \taverage absolute error: %.5f\taverage square error: %.5f"
            % (test_name, average_error, average_absolute_error, average_square_error))
        sbs.set_style("whitegrid", {"axes.grid": True})
        plt.figure()
        plt.grid(b=True)
        plt.plot(test_set, real_results, label="real")
        plt.plot(test_set, approx_results, label="approximation")
        plt.legend(fancybox=True)
        plt.title(test_name)
        plt.savefig(test_name)

    return test_result


def run_multi_test(test_set, approx_funcs, orig_func, test_name, title, function_name):
    """ runs multiple approximate functions and draws a single graph of them 
        The approx_funcs are a list of tuples, where each tuple has a func and a name
    """
    test_set.sort()
    error = [None] * len(test_set)
    real_results = [orig_func(x) for x in test_set]

    sbs.set_style("whitegrid", {"axes.grid": True})
    plt.figure()
    plt.grid(b=True)
    plt.plot(test_set, real_results, label=function_name)
    
    for (approx_func, approx_func_name) in approx_funcs:
        approx_results = [approx_func(x) for x in test_set]
        errors = [abs(real_results[x] - approx_results[x]) / (real_results[x] + 1) for x in range(len(test_set))]
        square_errors = [pow(real_results[x] - approx_results[x], 2) for x in range(len(test_set))]
        total_error = sum([abs(real_results[x] - approx_results[x]) for x in range(len(test_set))])
        average_square_error = sum(square_errors) / len(test_set)

        average_error = (sum(errors)/len(errors))
        average_absolute_error = total_error/len(test_set)
        print("(%s) The average error (%s test) is: %.4f \taverage absolute error: %.4f\tsquare error: %.4f"
            % (approx_func_name, test_name, average_error, average_absolute_error, average_square_error))

        plt.plot(test_set, approx_results, label=approx_func_name)
    
    plt.legend(fancybox=True)
    plt.title(title)
    plt.savefig(test_name)


if __name__ == "__main__":
    main()
