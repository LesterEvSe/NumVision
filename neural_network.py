import numpy as np
import layer as ly
import functions as fn

class NeuralNetwork: 
	def __init__(self, sizes):
		self.sizes = sizes
		self.a0 = [0 for i in range(sizes[0])]
		self.layers = [ly.Layer(sizes[i - 1], sizes[i]) 
			for i in range(1, len(sizes))]

	def calculate(self, a0, func):
		if len(a0) != len(self.a0):
			raise Exception("Input array data must contain", len(self.a0), "elements")
		pr_a = self.a0
		for lay in self.layers:
			pr_a = lay.calculate(pr_a, func)
		return pr_a