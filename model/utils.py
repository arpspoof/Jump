import torch.nn as nn

def xavier_init(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1)
    nn.init.constant_(module.bias.data, 0)
    return module

def orthogonal_init(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
    return module
