import numpy as np

class Layer:
    s_lower_bound = -5
    s_upper_bound = 5

    def __init__(self, a_prev_size, layer_size):
        self.W = np.random.uniform(self.s_lower_bound, 
                                   self.s_upper_bound, 
                                   size=(layer_size, a_prev_size))
        self.bias = np.random.uniform(self.s_lower_bound, 
                                      self.s_upper_bound, 
                                      layer_size)
        self.a = []

    def calculate(self, a_prev, func):
        self.a = func(np.dot(self.W, a_prev) + self.bias)
        return self.a
