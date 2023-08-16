import layer as ly
import numpy as np

class NeuralNetwork:
    def __init__(self, input_data: np.array):
        self.a0 = input_data
        self.layers = []

    def addLayer(self, neurons: int, func):
        if len(self.layers) == 0:
            prev_neurons = self.a0
        else:
            prev_neurons = self.layers[-1].getNeurons()
        self.layers.append(ly.Layer(prev_neurons, neurons, func))

    # Start from 0
    def getLayerA(self, layer: int):
        return self.a0 if layer == -1 else self.layers[layer].a
