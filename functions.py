import numpy as np

# Artificial Neurons
def sig(x):
	return 1 / (1 + np.exp(-x))

def derSig(x):
	return sig(x) * (1 - sig(x))

def ReLU(x):
	return max(0, x)

def derReLU(x):
	return 1 if x > 0 else 0


# Evaluation functions
def squareDeviation(a_final, real_ans):
	return sum([(a_final[i] - real_ans[i]) ** 2 for i in range(len(real_ans))])

def derSquareDeviation(a_final, real_ans):
	return [2*(a_final[i] - real_ans[i]) for i in range(len(real_ans))]

def crossEntropy(a_final, real_ans):
	def formula(a, y):
		return -(y*np.log(a) + (1-y)*np.log(1 - a))
	return sum(formula(a_final[i], real_ans[i]) for i in range(len(real_ans)))

def derCrossEntropy(a_final, real_ans):
	return [np.log(real_ans[i]) - np.log(1 - real_ans[i]) for i in range(len(real_ans))]
