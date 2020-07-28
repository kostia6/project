import random

max_value = 100
min_value = -100
max_degree = 100
max_degree_polyfit = 30
num_examples = 1000


def create_test_set(number_examples, border):
    res = [None] * number_examples
    for i in range(0, number_examples):
        res[i] = (random.uniform(0, 1) * border*2)-border

    return res
