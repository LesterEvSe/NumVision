import numpy as np

# Activation functions
def sig(x):
	return 1 / (1 + np.exp(-x))

def der_sig(x):
	res = sig(x)
	return res * (1 - res)

def ReLU(x):
	return np.array([max(0, elem) for elem in x])

def der_ReLU(x):
	return np.array([1 if elem > 0 else 0 for elem in x])

def softmax(x):
	divider = sum(np.exp(x))
	return np.array([np.exp(elem) / divider for elem in x])

def der_softmax(x):
	res = softmax(x)
	return res * (1 - res)


# Evaluation functions
def square_error(a_final, real_ans):
	return sum((a_final - real_ans) ** 2)

def der_square_error(a_final, real_ans):
	return 2*(a_final - real_ans)

def cross_entropy(a_final, real_ans):
	def formula(a, y):
		return -(y*np.log(a) + (1-y)*np.log(1 - a))
	return sum(formula(a_final[i], real_ans[i]) for i in range(len(real_ans)))

def der_cross_entropy(a_final, real_ans):
	def formula(a, y):
		return -y/a + (1 - y) / (1 - a)
	return np.array([formula(a_final[i], real_ans[i]) for i in range(len(real_ans))])
