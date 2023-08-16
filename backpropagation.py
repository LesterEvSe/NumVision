import numpy as np
import functions as fn
import neural_network as nn

class WeighStorage:
    def __init__(self, net: nn):
        # We have Wnk and b(bias)k
        # where n - current layer size, k - prev layer size
        self.storage_W = [0] * len(net.layers)
        self.storage_bias = [0] * len(net.layers)

    def add(self, layer: int, weigh, bias):
        self.storage_W[layer] = weigh
        self.storage_bias[layer] = bias

    def getW(self, layer: int):
        return self.storage_W[layer]
    def getBias(self, layer: int):
        return self.storage_bias[layer]

    def printAll(self):
        print("Biases")
        for elem in self.storage_bias:
            print(elem)

        print("Weighs")
        for matrix in self.storage_W:
            print(matrix)

def calcGradient(net: nn, real_ans):
    weighs = WeighStorage(net)
    derC0_prev = fn.derSquareDeviation(net.layers[-1].a, real_ans)

    for ly in range(len(net.layers)-1, -1, -1):
        W = np.zeros(net.layers[ly].getWShape())
        bias = np.zeros(net.layers[ly].getBiasSize())
        derC0_curr = np.zeros(W.shape[1])

        prev_a = net.getLayerA(ly-1)
        curr_z = net.layers[ly].z
        W_old = net.layers[ly].W

        # Calculate gradient. Hard formulas...
        for j in range(W.shape[0]):
            temp_foo = fn.derSig(curr_z[j])

            for k in range(W.shape[1]):
                W[j][k] = prev_a[k] * temp_foo * derC0_prev[j]
                derC0_curr[k] += W_old[j][k]
            bias[j] = temp_foo * derC0_prev[j]

        weighs.add(ly, W, bias)
        derC0_prev = derC0_curr
    return weighs

