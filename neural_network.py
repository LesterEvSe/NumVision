import layer as ly
import numpy as np
import weigh_storage as wh

class NeuralNetwork:
    def __init__(self):
        self.a0 = None
        self.layers = []

    def replaceData(self, input_data: np.array):
        self.a0 = input_data
        prev_neurons = self.a0.copy()
        for _layer in self.layers:
            _layer.recalcLayer(prev_neurons)
            prev_neurons = _layer.getNeurons()

    def answerCorrect(self, num: int):
        return np.argmax(self.layers[-1].a) == num

    def changeWeighsAndBiases(self, weighs: wh):
        for i in range(len(self.layers)):
            self.layers[i].recalcWAndBias(weighs.getW(i), weighs.getBias(i))

    def addLayer(self, neurons: int, func):
        if len(self.layers) == 0:
            prev_neurons = self.a0.copy()
        else:
            prev_neurons = self.layers[-1].getNeurons()
        self.layers.append(ly.Layer(prev_neurons, neurons, func))

    def getLayerA(self, layer: int):
        return self.a0 if layer == -1 else self.layers[layer].a

    def getLastA(self):
        return self.layers[-1].a
