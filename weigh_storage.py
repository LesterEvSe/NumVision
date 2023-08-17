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