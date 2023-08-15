from PIL import Image
import numpy as np

import functions as fn
import neural_network as nn
import backpropagation as bp

import test

image_matrix = np.array(Image.open('000000-num5.png').convert('L'))

a = [pix / 255 for pix in image_matrix.flatten()]

net = nn.NeuralNetwork([28*28, 16, 16, 10])
bp.calcGradient()

print(fn.crossEntropy(net.calculate(a, fn.sig), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
