import numpy as np
import functions as fn
import neural_network as nn
import weigh_storage as wh

def calcGradient(net: nn, real_ans):
    weighs = wh.WeighStorage(net)
    derC0_prev = fn.derSquareDeviation(net.layers[-1].a, real_ans)

    for ly in range(len(net.layers)-1, -1, -1):
        W = np.zeros(net.layers[ly].getWShape())
        bias = np.zeros(net.layers[ly].getBiasSize())
        derC0_curr = np.zeros(W.shape[1])

        prev_a = net.getLayerA(ly-1)
        curr_z = net.layers[ly].z
        W_old = net.layers[ly].W

        # Calculate gradient. Hard formulas...
        for j in range(W.shape[0]):
            temp_foo = fn.derSig(curr_z[j])

            for k in range(W.shape[1]):
                W[j][k] = prev_a[k] * temp_foo * derC0_prev[j]
                derC0_curr[k] += W_old[j][k]
            bias[j] = temp_foo * derC0_prev[j]

        weighs.push(ly, W, bias)
        derC0_prev = derC0_curr
    return weighs

