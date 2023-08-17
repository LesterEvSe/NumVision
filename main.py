from PIL import Image
import os
import numpy as np
from random import shuffle

import functions as fn
import neural_network as nn
import backpropagation as bp
import weigh_storage as wh

# Load pictures from folder
imageFolder = "train"
imageFiles = [f for f in os.listdir(imageFolder)
              if os.path.isfile(os.path.join(imageFolder, f))]

# sets of tuple - where images[i][0] - pixels of picture in range [0; 1]
# images[i][1] is num in np.array representation
# images[i][2] num in usual form
images = []
pic = 1

for imFile in imageFiles:
    imPath = os.path.join(imageFolder, imFile)
    img = np.array(Image.open(imPath).convert('L'))

    numSet = np.zeros(10)
    num = int(imFile[-5])
    numSet[num] = 1

    imTuple = (np.array([pix / 255 for pix in img.flatten()]), numSet, num)
    images.append(imTuple)

    if pic > 10_000:
        break
    pic += 1
shuffle(images)

def print_info(neu_net: nn, ind: int, guessed: int, pictures: int):
    print(str(ind) + '.', images[ind][2], str(guessed/pictures) + '%', neu_net.getLastA())


# Create NeuralNetwork
net = nn.NeuralNetwork()
net.replaceData(images[0][0])

net.addLayer(16, fn.sig)
net.addLayer(16, fn.sig)
net.addLayer(10, fn.sig)

currWeighs = wh.WeighStorage(net)
guess = 0
overall = 1
for i in range(len(images)):
    guess += net.answerCorrect(images[i][2])
    net.replaceData(images[i][0])
    currWeighs.addAnother(bp.calcGradient(net, images[i][1]))

    if i % 100 == 0:
        print_info(net, i, guess, overall)
        guess = overall = 0
        currWeighs.divide(100)

        net.changeWeighsAndBiases(currWeighs)
        currWeighs = wh.WeighStorage(net)
        print()
    overall += 1

