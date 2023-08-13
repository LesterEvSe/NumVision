import layer as lr
import functions as func
import numpy as np
from PIL import Image

image_matrix = np.array(Image.open('000000-num5.png').convert('L'))

a0 = image_matrix.flatten()
layers = [lr.Layer(a0)]  # Layers, without output
for i in range(1, 3):
    layers.append(lr.Layer(layers[i-1].calculate_next_layer(func.sig)))


# print(a0)
