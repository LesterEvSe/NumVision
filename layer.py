import numpy as np

class Layer:
    s_from = -1
    s_to = 1
    def __init__(self, a_prev: np.array, layer_size: int, func):
        self.W = np.random.uniform(self.s_from, self.s_to,
                                   size=(layer_size, a_prev.shape[0]))
        self.bias = np.zeros(layer_size)
        self.func = func

        self.z = None
        self.a = None
        self.recalcLayer(a_prev)

    def recalcLayer(self, a_prev):
        self.z = np.dot(self.W, a_prev) + self.bias
        self.a = self.func(self.z)

    def calcLayer(self, a_prev):
        return np.dot(self.W, a_prev) + self.bias

    def recalcWAndBias(self, weighs: np.array, bias: np.array):
        self.W += weighs
        self.bias += bias

    def getNeurons(self):
        return self.a

    def getWShape(self):
        return self.W.shape
    def getBiasSize(self):
        return self.bias.shape[0]
