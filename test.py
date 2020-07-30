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
    #test_relu(cur_max_degree)
    test_range_relu(settings.min_degree, settings.max_degree, settings.step_degree)
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
    
def test_range_relu(min_degree, max_degree, step):
    test_set = settings.create_test_set(settings.num_examples, max_value)
    approx_funcs = []
    for degree in range(min_degree, max_degree, step):
        approx_funcs.append( (relu_activation.get_approx_func(degree), "Poly Degree: %s" % (degree)) )
    run_multi_test(test_set, approx_funcs, relu_activation.get_real_func(), "Relu-Chebyshev")


def run_test(test_set, approx_func, orig_func, test_name):
    test_set.sort()
    error = [None] * len(test_set)
    total_error = 0.0
    real_results = [orig_func(x) for x in test_set]
    approx_results = [approx_func(x) for x in test_set]
    errors = [abs(real_results[x] - approx_results[x]) / (real_results[x] + 1) for x in range(len(test_set))]
    total_error = sum([abs(real_results[x] - approx_results[x]) for x in range(len(test_set))])

    average_error = (sum(errors)/len(errors))
    average_absolute_error = total_error/len(test_set)
    print("The average error (%s test) is: %.4f \taverage absolute error: %.4f"
          % (test_name, average_error, average_absolute_error))
    sbs.set_style("whitegrid", {"axes.grid": True})
    plt.figure()
    plt.grid(b=True)
    plt.plot(test_set, real_results, label="real")
    plt.plot(test_set, approx_results, label="approximation")
    plt.legend(fancybox=True)
    plt.title(test_name)
    plt.savefig(test_name)

def run_multi_test(test_set, approx_funcs, orig_func, test_name):
    """ runs multiple approximate functions and draws a single graph of them 
        The approx_funcs are a list of tuples, where each tuple has a func and a name
    """
    test_set.sort()
    error = [None] * len(test_set)
    real_results = [orig_func(x) for x in test_set]

    sbs.set_style("whitegrid", {"axes.grid": True})
    plt.figure()
    plt.grid(b=True)
    plt.plot(test_set, real_results, label="real")
    
    for (approx_func, approx_func_name) in approx_funcs:
        approx_results = [approx_func(x) for x in test_set]
        errors = [abs(real_results[x] - approx_results[x]) / (real_results[x] + 1) for x in range(len(test_set))]
        total_error = sum([abs(real_results[x] - approx_results[x]) for x in range(len(test_set))])

        average_error = (sum(errors)/len(errors))
        average_absolute_error = total_error/len(test_set)
        print("(%s) The average error (%s test) is: %.4f \taverage absolute error: %.4f"
            % (approx_func_name, test_name, average_error, average_absolute_error))

        plt.plot(test_set, approx_results, label=approx_func_name)
    
    plt.legend(fancybox=True)
    plt.title(test_name)
    plt.savefig(test_name)

if __name__ == "__main__":
    main()
