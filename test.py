import functions as fn

b = [0.9997, 0.9, 0.6, 0.1192, 0.1192, 1-0.1192]
y = [1, 1, 1, 1, 0, 1]

def entropyFunc():
    print(fn.crossEntropy(b, y))