from PIL import Image
import os
import numpy as np

import functions as fn
import neural_network as nn
import backpropagation as bp

import test

imageFolder = "../train"
imageFiles = [f for f in os.listdir(imageFolder)
              if os.path.isfile(os.path.join(imageFolder, f))]

images = []
pic = 1

for imFile in imageFiles:
    imPath = os.path.join(imageFolder, imFile)
    img = np.array(Image.open(imPath).convert('L'))

    numSet = np.zeros(10)
    numSet[int(imFile[-5])] = 1

    imTuple = (numSet, np.array([pix / 255 for pix in img.flatten()]))
    images.append(imTuple)

    if pic > 1_000:
        break
    pic += 1


# net = nn.NeuralNetwork(inputImage)
# net.addLayer(16, fn.sig)
# net.addLayer(16, fn.sig)
# net.addLayer(10, fn.sig)
#
# weighs = bp.calcGradient(net, num)
# weighs.printAll()
