import numpy as np
from PIL import Image

def sig(x):
	return 1 / (1 + np.exp(-x))

def dersig(x):
	return sig(x) * (1 - sig(x))

def ReLU(x):
	return max(0, x)

def derReLU(x):
	return 1 if x > 0 else 0