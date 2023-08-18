import numpy as np

class Layer:
    s_rand_from = -1
    s_rand_to = 1

    def __init__(self, a_prev: np.array, layer_size: int, func, der_func):
        self.func = func
        self.deriv = der_func
        self.weigh = np.random.uniform(Layer.s_rand_from, Layer.s_rand_to,
                                       size=(layer_size, a_prev.shape[0]))
        self.bias = np.zeros(layer_size)

        self.z = None
        self.a = None
        self.recalculate_layer(a_prev)

    def recalculate_layer(self, a_prev: np.array):
        self.z = np.dot(self.weigh, a_prev) + self.bias
        # self.a = np.zeros(len(self.z))
        # for val in self.z:

        self.a = self.func(self.z)
