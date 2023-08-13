import layer as lr
import numpy as np
from PIL import Image

image_matrix = np.array(Image.open('000000-num5.png').convert('L'))

a0 = image_matrix.flatten()
layer = lr.Layer(a0)


# print(a0)
