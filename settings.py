import random

max_value = 100
min_value = -100
min_degree = 30
step_degree = 10
max_degree = 100
max_degree_polyfit = 100
num_examples = 10000
num_threads = 1
max_num_threads = 10


def create_test_set(number_examples, border):
    res = [None] * number_examples
    for i in range(0, number_examples):
        res[i] = (random.uniform(0, 1) * border*2)-border

    return res
