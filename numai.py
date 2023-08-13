import numpy as np
from PIL import Image

im = np.array(Image.open('000000-num5.png').convert('L'))

print(im)

gr_im = Image.fromarray(im).save('wowowo-gr.png')