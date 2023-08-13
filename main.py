import layer as lr
import functions as fn
import neural_network as nn
import numpy as np
from PIL import Image

image_matrix = np.array(Image.open('000000-num5.png').convert('L'))

a = image_matrix.flatten()

n = nn.NeuralNetwork([28*28, 16, 16, 10])

print(n.calculate(a, fn.sig))