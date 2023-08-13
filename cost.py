import numpy as np

def cost(a_final, real_ans):
	c = [(a_final[i] - real_ans[i]) ** 2 for i in range(0, len(a_final))]
	return sum(c)