import numpy as np
from layer import *

class NeuralNetwork:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.layers = []

    def add_layer(self, layer_size: int, func):
        if len(self.layers) == 0:
            prev_size = self.input_size
        else:
            # bias has a layer length
            prev_size = self.layers[-1].bias.shape[0]
        self.layers.append(Layer(prev_size, layer_size, func))

    def calculate(self, a0: np.array):
        prev = a0.copy()
        for i in range(len(self.layers)):
            self.layers[i].calculate(prev)
            prev = self.layers[i].a

