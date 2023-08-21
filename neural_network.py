import numpy as np
import layer as ly
import functions as fn

class NeuralNetwork:
    def __init__(self, input_size: int, cost_func):
        self.eval_func = cost_func
        if self.eval_func == fn.square_error:
            self.der_eval_func = fn.der_square_error
        elif self.eval_func == fn.cross_entropy:
            self.der_eval_func = fn.der_cross_entropy
        else:
            raise Exception("Unknown evaluation function")

        self.input_size = input_size
        self.a0 = None
        self.layers = []

    def add_layer(self, layer_size: int, func):
        if len(self.layers) == 0:
            prev_size = self.input_size
        else:
            # bias has a layer length
            prev_size = self.layers[-1].bias.shape[0]
        self.layers.append(ly.Layer(prev_size, layer_size, func))

    def calculate(self, a0: np.array):
        self.a0 = np.transpose(a0)
        prev = self.a0.copy()
        for i in range(len(self.layers)):
            self.layers[i].calculate(prev)
            prev = self.layers[i].a

    def answer_correct(self, num: int):
        return np.argmax(self.get_answer_a()) == num

    def backpropagation(self, y: np.array, learning_speed=0.01):
        derC0_prev = np.array(self.der_eval_func(self.get_answer_a(), y))

        new_weigh = [0] * len(self.layers)
        new_bias = [0] * len(self.layers)

        for i in reversed(range(len(self.layers))):
            # arr = der(aL)/der(zL) * der(C0)/der(aL) = der(C)/der(zL)
            prev_a = self.get_a(i-1)
            arr = self.layers[i].der_func(self.layers[i].z) * derC0_prev
            new_bias[i] = arr.copy()

            net_weigh = self.layers[i].W
            net_weigh_tp = np.transpose(net_weigh)
            weigh = np.zeros(net_weigh_tp.shape)

            for k in range(len(prev_a)):
                weigh[k] = np.transpose(prev_a[k] * arr)[0]

            new_weigh[i] = np.transpose(weigh)
            arr_tp = np.transpose(arr)
            if i == 0:
                continue

            derC0_curr = np.zeros((len(prev_a), 1))
            for k in range(len(prev_a)):
                derC0_curr[k][0] = sum(net_weigh_tp[k] * arr_tp[0])
            derC0_prev = derC0_curr.copy()

        for i in range(len(self.layers)):
            self.layers[i].W -= new_weigh[i] * learning_speed
            self.layers[i].bias -= new_bias[i] * learning_speed

    def calc_C0(self, y: np.array):
        return self.eval_func(np.transpose(self.get_answer_a()).flatten(), y)

    def get_a(self, layer: int):
        return self.a0 if layer == -1 else self.layers[layer].a

    def get_answer_a(self):
        return self.layers[-1].a
