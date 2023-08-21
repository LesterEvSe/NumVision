import numpy as np

# Activation functions
def sig(z):
	return 1 / (1 + np.exp(-z))

def der_sig(x):
	return sig(x) * (1 - sig(x))


# Evaluation functions
def square_error(a_final, real_ans):
	return sum([(a_final[i] - real_ans[i]) ** 2 for i in range(len(real_ans))])

def der_square_error(a_final, real_ans):
	return 2 * (a_final - real_ans)
	# return [2*(a_final[i] - real_ans[i]) for i in range(len(real_ans))]

def cross_entropy(a_final, real_ans):
	def formula(a, y):
		return -(y*np.log(a) + (1-y)*np.log(1 - a))
	return sum(formula(a_final[i], real_ans[i]) for i in range(len(real_ans)))

def der_cross_entropy(a_final, real_ans):
	def formula(a, y):
		return -y/a + (1 - y) / (1 - a)
	return [formula(a_final[i], real_ans[i]) for i in range(len(real_ans))]
