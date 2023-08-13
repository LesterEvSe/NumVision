import numpy as np
import layer as ly
import functions as fn

class NeuralNetwork: 
	def __init__(self, sizes):
		self.a0 = [0 for i in range(sizes[0])]
		self.layers = [ly.Layer(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))]

	def calculate(self, a0, func):
		if len(a0) != len(self.a0):
			raise Exception("Input array data must contain " + str(len(self.a0)) + " elements")
		prev_a = a0
		for lay in self.layers:
			prev_a = lay.calculate(prev_a, func)
		return prev_a
