from PIL import Image
import numpy as np

import functions as fn
import neural_network as nn
import backpropagation as bp

import test

image_matrix = np.array(Image.open('000000-num5.png').convert('L'))
inputImage = np.array([pix / 255 for pix in image_matrix.flatten()])

net = nn.NeuralNetwork(inputImage)
net.addLayer(16, fn.sig)
net.addLayer(16, fn.sig)
net.addLayer(10, fn.sig)
num = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

weighs = bp.calcGradient(net, num)
weighs.printAll()
