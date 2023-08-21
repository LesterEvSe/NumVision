import numpy as np
import functions as fn

class Layer:
    s_rand_from = -1
    s_rand_to = 1
    def __init__(self, prev_layer_size: int, layer_size: int, func):
        self.func = func
        if func == fn.sig:
            self.der_func = fn.der_sig
        else:
            raise Exception("Unknown activation function")
        self.W = np.random.uniform(Layer.s_rand_from, Layer.s_rand_to,
                                   size=(layer_size, prev_layer_size))
        self.bias = np.random.uniform(Layer.s_rand_from, Layer.s_rand_to,
                                      size=(layer_size, 1))

        self.z = None
        self.a = None

    def calculate(self, a_prev: np.array):
        self.z = np.dot(self.W, a_prev) + self.bias
        self.a = self.func(self.z)
