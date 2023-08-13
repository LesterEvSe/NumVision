import layer as lr
import functions as func
import numpy as np
from PIL import Image

image_matrix = np.array(Image.open('000000-num5.png').convert('L'))

LAYERS = 4
HIDDEN_LAYERS_SIZE = 16

a_prev = image_matrix.flatten()
layers = []
for lyr in range(2, 5):
    print(lyr)
    neurons = 10 if lyr == LAYERS else 16
    next_layer = lr.Layer(a_prev, neurons, func.sig)

    layers.append(next_layer)
    a_prev = next_layer.a
