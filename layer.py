import numpy as np

class Layer:
    s_rand_from = -1
    s_rand_to = 1
    def __init__(self, prev_layer_size: int, layer_size: int, func):
        self.func = func
        self.W = np.random.uniform(Layer.s_rand_from, Layer.s_rand_to,
                                   size=(layer_size, prev_layer_size))
        self.bias = np.zeros(layer_size)

        self.z = None
        self.a = None

    def calculate(self, a_prev: np.array):
        self.z = np.dot(self.W, a_prev) + self.bias
        self.a = self.func(self.z)
