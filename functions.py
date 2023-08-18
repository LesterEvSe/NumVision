import numpy as np

# Activation functions
def sig(x):
	return 1 / (1 + np.exp(-x))

def der_sig(x):
	return sig(x) * (1 - sig(x))

def ReLU(x):
	return max(0, x)

def der_ReLU(x):
	return 1 if x > 0 else 0


# Evaluation functions
def square_error(a_final, real_ans):
	return sum([(a_final[i] - real_ans[i]) ** 2 for i in range(len(real_ans))])

def der_square_error(a_final, real_ans):
	return [2*(a_final[i] - real_ans[i]) for i in range(len(real_ans))]

def cross_entropy(a_final, real_ans):
	def formula(a, y):
		return -(y*np.log(a) + (1-y)*np.log(1 - a))
	return sum(formula(a_final[i], real_ans[i]) for i in range(len(real_ans)))

def der_cross_entropy(a_final, real_ans):
	def formula(a, y):
		return -y/a + (1 - y) / (1 - a)
	return [formula(a_final[i], real_ans[i]) for i in range(len(real_ans))]
