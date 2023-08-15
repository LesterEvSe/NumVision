import numpy as np
import functions as fn
import neural_network as nn

class WeighStorage:
    def __init__(self, net: nn):
        self.storage = [np.zeros((2,)) for _ in range(len(net.sizes))]


def calcGradient(net: nn):
    # print(net.sizes)
    # Last Layer recalculation
    weigh = WeighStorage(net)

    for i in range(len(net.sizes)-2, -1, -1):
        pass

