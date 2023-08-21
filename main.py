from PIL import Image
import os
import numpy as np
from random import shuffle

import neural_network as nn
import functions as fn

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

    imTuple = (np.array([[pix / 255 for pix in img.flatten()]]), numSet, num)
    images.append(imTuple)
    if pic % 5_000 == 0:
        print("Pictures load: ", pic)

    if pic > 30_000:
        break
    pic += 1
shuffle(images)


# Prepare function
def print_info(coun: int, neu_net: nn, image, guessed: int, pictures: int):
    print(str(coun) + ')', image[2], '[C0 =', round(neu_net.calc_C0(image[1]), 3), end='] ')
    print(str(round(guessed/pictures * 100, 2)) + '%', end=' ')

    print('[', end='')
    for elem in neu_net.get_answer_a():
        print(elem, end=' ')
    print(']')


# Create Neural Network
# images[0][0].shape[0] here 28*28 = 784 pixels
test = images[0][0]
net = nn.NeuralNetwork(test.shape[1], fn.cross_entropy)
net.add_layer(64, fn.sig)
net.add_layer(64, fn.sig)
net.add_layer(10, fn.sig)

guess = 0
counter = overall = 1

for i in range(1, 10):
    print("Generation", i)
    for im in images:
        net.calculate(im[0])
        guess += net.answer_correct(im[2])
        net.backpropagation(im[1])

        if counter % 100 == 0:
            print_info(counter, net, im, guess, overall)

        counter += 1
        overall += 1
    print("\nEnd generation", i, end='\n\n')

    guess = 0
    overall = counter = 1
    shuffle(images)
