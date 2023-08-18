import numpy as np
import neural_network as nn

class Gradient:
    def __init__(self, layers: int):
        self.layers_num = layers
        self.storage_weighs = [0] * layers
        self.storage_biases = [0] * layers

        self.C0 = None

    def backpropagation(self, net: nn, real_ans: np.array):
        derC0_prev = net.der_estimate_func(net.get_answer_a(), real_ans)

        for i in range(self.layers_num-1, -1, -1):
            weigh = np.zeros(net.get_weigh_dim(i))
            bias  = np.zeros(net.get_bias_size(i))
            derC0_curr = np.zeros(weigh.shape[1])

            layer_z = net.get_z(i)
            layer_deriv = net.get_deriv(i)

            prev_a = net.get_a(i - 1)
            weigh_old = net.layers[i].weigh

            for j in range(weigh.shape[0]):
                der_z = layer_deriv(layer_z[j])
                bias[j] = der_z * derC0_prev[j]

                for k in range(weigh.shape[1]):
                    weigh[j][k] = prev_a[k] * der_z * derC0_prev[j]
                    derC0_curr[k] += weigh_old[j][k] * der_z * derC0_prev[j]

            self.storage_weighs[i] = weigh
            self.storage_biases[i] = bias
            derC0_prev = derC0_curr
        return self

    def sub(self, other):
        for i in range(self.layers_num):
            self.storage_weighs[i] -= other.storage_weighs[i]
            self.storage_biases[i] -= other.storage_biases[i]

    def div(self, val: float):
        for i in range(self.layers_num):
            self.storage_weighs[i] /= val
            self.storage_biases[i] /= val

    def __str__(self):
        print("Biases")
        for bias in self.storage_biases:
            print(bias)

        print("\nWeighs")
        for weigh in self.storage_weighs:
            print(weigh)
        return str()
