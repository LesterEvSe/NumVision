import layer as ly
import functions as fn
import numpy as np
import gradient as gr

class NeuralNetwork:
    def __init__(self, a0: np.array, estimate_func):
        self.a0 = a0
        self.layers = []
        self.estimate_func = estimate_func

        if estimate_func == fn.square_error:
            self.der_estimate_func = fn.der_square_error
        elif estimate_func == fn.cross_entropy:
            self.der_estimate_func = fn.der_cross_entropy
        else:
            raise Exception("Unknown estimate function")

    def change_a0(self, a0: np.array):
        self.a0 = a0
        prev = a0.copy()
        for layer in self.layers:
            layer.recalculate_layer(prev)
            prev = layer.a

    def add_layer(self, layer_size: int, func):
        if len(self.layers) == 0:
            prev = self.a0.copy()
        else:
            prev = self.layers[-1].a.copy()

        if func == fn.sig:
            deriv = fn.der_sig
        elif func == fn.ReLU:
            deriv = fn.der_ReLU
        else:
            raise Exception("Unknown activation function")
        self.layers.append(ly.Layer(prev, layer_size, func, deriv))

    def correct_weighs_biases(self, grad: gr):
        for i in range(len(self.layers)):
            self.layers[i].weigh += grad.storage_weighs[i]
            self.layers[i].bias  += grad.storage_biases[i]


    def calc_C0(self, real_ans: np.array):
        return self.estimate_func(self.get_answer_a(), real_ans)

    def answer_correct(self, num: int):
        return np.argmax(self.get_answer_a()) == num


    def get_z(self, layer: int):
        return self.layers[layer].z

    def get_deriv(self, layer: int):
        return self.layers[layer].deriv

    # getters
    def get_a(self, layer: int):
        return self.a0 if layer == -1 else self.layers[layer].a

    def get_answer_a(self):
        return self.layers[-1].a

    def get_layers_num(self):
        return len(self.layers)

    def get_weigh_dim(self, layer: int):
        return self.layers[layer].weigh.shape

    def get_bias_size(self, layer: int):
        return self.layers[layer].bias.shape[0]
