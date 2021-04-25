import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import orthogonal_init

class Critic(nn.Module):

    def __init__(self, s_dim, val_min, val_max, hidden=[1024, 512]):
        super(Critic, self).__init__()

        self.fc = []
        input_dim = s_dim
        for h_dim in hidden:
            self.fc.append(orthogonal_init(nn.Linear(input_dim, h_dim)))
            input_dim = h_dim

        self.fc = nn.ModuleList(self.fc)
        self.fcv = orthogonal_init(nn.Linear(input_dim, 1))

        # value normalizer
        self.v_min = torch.Tensor([val_min])
        self.v_max = torch.Tensor([val_max])
        self.v_mean = nn.Parameter((self.v_max + self.v_min) / 2)
        self.v_std  = nn.Parameter((self.v_max - self.v_min) / 2)
        self.v_min.requires_grad = False
        self.v_max.requires_grad = False
        self.v_mean.requires_grad = False
        self.v_std.requires_grad = False

        self.activation = F.relu

    def forward(self, x):
        layer = x
        for fc_op in self.fc:
            layer = self.activation(fc_op(layer))

        # unnormalize value
        value = self.fcv(layer)
        value = self.v_std * value + self.v_mean

        return value
