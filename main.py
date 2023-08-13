import layer as lr
import functions as fn
import neural_network as nn
import numpy as np
import cost as cs
from PIL import Image

image_matrix = np.array(Image.open('000000-num5.png').convert('L'))

a = [a / 255 for a in image_matrix.flatten()]

n = nn.NeuralNetwork([28*28, 16, 16, 10])

print(cs.cost(n.calculate(a, fn.sig), [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))