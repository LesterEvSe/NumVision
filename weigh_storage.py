import neural_network as nn
import numpy as np

class WeighStorage:
    def __init__(self, net: nn):
        # We have Wnk and b(bias)k
        # where n - current layer size, k - prev layer size
        self.storage_W = [0] * len(net.layers)
        self.storage_bias = [0] * len(net.layers)

    def addAnother(self, weighs):
        for i in range(len(self.storage_W)):
            self.storage_W[i] -= weighs.storage_W[i]
            self.storage_bias[i] -= weighs.storage_bias[i]

    def divide(self, val: int):
        for i in range(len(self.storage_W)):
            self.storage_W[i] /= val
            self.storage_bias[i] /= val

    def push(self, layer: int, weigh: np.array, bias: np.array):
        self.storage_W[layer] = weigh
        self.storage_bias[layer] = bias

    def printAll(self):
        print("Biases")
        for elem in self.storage_bias:
            print(elem)

        # print("Weighs")
        # for matrix in self.storage_W:
        #     print(matrix)

    def getW(self, layer: int):
        return self.storage_W[layer]
    def getBias(self, layer: int):
        return self.storage_bias[layer]
