import numpy as np
import json
import neural_network as nn

def save(net: nn):
    weigh_list = [layer.W.tolist() for layer in net.layers]
    bias_list  = [layer.bias.tolist() for layer in net.layers]

    # Save data to json
    data = {'weigh': weigh_list, 'bias': bias_list}
    with open('network_data.json', 'w') as json_file:
        json.dump(data, json_file)

def load_to(net: nn):
    with open('network_data.json', 'r') as json_file:
        loaded_data = json.load(json_file)

    # Has equal size
    weigh = [np.array(matrix) for matrix in loaded_data['weigh']]
    bias = [np.array(arr) for arr in loaded_data['bias']]

    for i in range(len(weigh)):
        net.layers[i].W = weigh[i]
        net.layers[i].bias = bias[i]