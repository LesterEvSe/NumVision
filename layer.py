import numpy as np

class Layer:
    def __init__(self, a_prev: np.array, layer_size: int, func):
        self.W = np.random.rand(layer_size, a_prev.shape[0])
        self.bias = np.zeros(layer_size)
        self.a = self.calcLayer(a_prev, func)

    def calcLayer(self, a_prev, func):
        return func(np.dot(self.W, a_prev) + self.bias)

    def recalcLayer(self, W, bias):
        self.W += W
        self.bias += bias

    def getNeurons(self):
        return self.a

    def getWeight(self, i, j):
        return self.W[i][j]
    def getBias(self, i):
        return self.bias[i]

