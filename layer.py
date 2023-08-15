import numpy as np

class Layer:
    s_lower_bound = -5
    s_upper_bound = 5

    def __init__(self, a_prev_size: int, layer_size: int):
        self.W = np.random.uniform(self.s_lower_bound, 
                                   self.s_upper_bound, 
                                   size=(layer_size, a_prev_size))
        self.bias = np.random.uniform(self.s_lower_bound, 
                                      self.s_upper_bound, 
                                      layer_size)
        self.a = []

    def calcNextLayer(self, a_prev, func):
        self.a = func(np.dot(self.W, a_prev) + self.bias)
        return self.a

    def getWeight(self, i, j):
        return self.W[i][j]
    def getBias(self, i):
        return self.bias[i]

    def recalc_weights(self, W, bias):
        self.W += W
        self.bias += bias
