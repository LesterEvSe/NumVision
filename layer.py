import numpy as np


class Layer:
    s_lower_bound = -5
    s_upper_bound = 5

    def __init__(self, a, next_layer_size=16):
        self.a = a  # array
        self.W = np.random.uniform(self.s_lower_bound, self.s_upper_bound, size=(next_layer_size, len(a)))
        self.bias = np.random.uniform(self.s_lower_bound, self.s_upper_bound, next_layer_size)

    def calculate_next_layer(self, f):
        return f(np.dot(self.W, self.a) + self.bias)
