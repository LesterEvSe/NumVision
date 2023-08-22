from PIL import Image
import os
import numpy as np

import neural_network as nn
import functions as fn
import saving as sv

# Prepare function
def print_info(coun: int, neu_net: nn, image, guessed: int, quantity: int, train=True):
    if train:
        print(str(coun) + ')', image[2], '[C0 =', round(neu_net.calc_C0(image[1]), 3), end='] ')
    print(str(round(guessed / quantity * 100, 2)) + '%', end=' ')

    print('[', end='')
    for elem in neu_net.get_answer_a():
        print(elem, end=' ')
    print(']')

# Load pictures from folder
def learn_neural_network(pictures=60_000):
    from random import shuffle

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

        # Name of the picture, for example: 042269-num7.png
        # The symbol -5 here is a 7, which is a written digit
        numSet = np.zeros(10)
        num = int(imFile[-5])
        numSet[num] = 1

        imTuple = (np.array([pix / 255 for pix in img.flatten()]), numSet, num)
        images.append(imTuple)
        if pic % 5_000 == 0:
            print("Pictures load: ", pic)

        if pic > pictures:
            break
        pic += 1
    shuffle(images)

    # Create Neural Network
    # images[0][0].shape[0] here 28*28 = 784 pixels
    test = images[0][0]
    net = nn.NeuralNetwork(test.shape[0], fn.cross_entropy)
    net.add_layer(32, fn.sig)
    net.add_layer(64, fn.sig)
    net.add_layer(10, fn.softmax)
    sv.load_to(net)

    for generation in range(20):
        print("Generation", generation+1)
        guess = 0
        overall = counter = 1

        for im in images:
            net.calculate(im[0])
            guess += net.answer_correct(im[2])
            net.backpropagation(im[1])

            if counter % 100 == 0:
                print_info(counter, net, im, guess, overall)

            counter += 1
            overall += 1

        print()
        sv.save(net)
        shuffle(images)

def check_result(folder='test', out=1_000):
    imageFolder = folder
    imageFiles = [f for f in os.listdir(imageFolder)
                  if os.path.isfile(os.path.join(imageFolder, f))]

    # sets of tuple - where images[i][0] - pixels of picture in range [0; 1]
    # images[i][1] num in usual form
    images = []

    for imFile in imageFiles:
        imPath = os.path.join(imageFolder, imFile)
        img = np.array(Image.open(imPath).convert('L'))

        imTuple = (np.array([pix / 255 for pix in img.flatten()]), int(imFile[-5]))
        images.append(imTuple)


    # Create Neural Network
    # images[0][0].shape[0] here 28*28 = 784 pixels
    test = images[0][0]
    net = nn.NeuralNetwork(test.shape[0], fn.cross_entropy)
    net.add_layer(32, fn.sig)
    net.add_layer(64, fn.sig)
    net.add_layer(10, fn.softmax)
    sv.load_to(net)

    guess = counter = 0
    for im in images:
        net.calculate(im[0])
        guess += net.answer_correct(im[1])

        counter += 1
        if counter % out != 0:
            continue

        print(imageFiles[counter-1], "it is a number with a probability")
        answer = net.get_answer_a()
        answer = sorted([(round(answer[i]*100, 2), i) for i in range(len(answer))], reverse=True)

        for pair in answer:
            if pair[0] == 0:
                break
            print(pair[1], '->', str(pair[0]) + '%')
        print()

def main():
    folder = input('Enter name of the folder with pictures: ')
    check_result(folder, 1)


# learn_neural_network()
# check_result()
main()
