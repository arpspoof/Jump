import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from ..utils import orthogonal_init
from ..abstract_policy import AbstractPolicy

class Actor(nn.Module, AbstractPolicy):
    def __init__(self, s_dim, a_dim, a_min, a_max, a_noise=None, hidden=[1024, 512], init_actor_scale=0.001):
        super(Actor, self).__init__()

        from presets import preset
        actor_settings = preset.model.actor

        self.normalize_action = actor_settings.normalize_action
        self.use_squashing = actor_settings.use_squashing
        self.ortho_init_offset = actor_settings.ortho_init_offset

        self.enable_vae = preset.vae.enable

        if a_noise is None:
            a_noise = [actor_settings.std_init]*a_dim
        
        if actor_settings.enable_explore_scheduling:
            a_noise = [actor_settings.std_begin]*a_dim
            from .explore_scheduler import ExploreScheduler
            from scheduler import register_scheduler
            register_scheduler(self, ExploreScheduler, self)

        self.fc = []
        input_dim = s_dim
        for h_dim in hidden:
            self.fc.append(orthogonal_init(nn.Linear(input_dim, h_dim)))
            input_dim = h_dim

        self.fc = nn.ModuleList(self.fc)

        # initialize final layer weight to be [INIT, INIT]
        if self.enable_vae:
            self.fca = orthogonal_init(nn.Linear(input_dim, a_dim - 28))
        self.fc_offset = nn.Linear(input_dim, 28)

        if self.ortho_init_offset:
            self.fc_offset = orthogonal_init(self.fc_offset)
        else:
            nn.init.uniform_(self.fc_offset.weight, -init_actor_scale, init_actor_scale)
            nn.init.constant_(self.fc_offset.bias, 0)

        # set a_norm not trainable
        self.a_min = nn.Parameter(torch.tensor(a_min).float())
        self.a_max = nn.Parameter(torch.tensor(a_max).float())
        std = (a_max - a_min) / 2
        self.a_std = nn.Parameter(torch.tensor(std).float())
        self.a_noise = nn.Parameter(torch.tensor(a_noise).float())
        self.a_min.requires_grad = False
        self.a_max.requires_grad = False
        self.a_std.requires_grad = False
        self.a_noise.requires_grad = False

        self.activation = F.relu

        import utils.gpu as gpu
        self.gpu = gpu.USE_GPU_MODEL

    def set_actor_noise_level(self, a_noise):
        if self.gpu:
            self.a_noise.data += (-self.a_noise.cpu().data + a_noise).cuda()
        else:
            self.a_noise.data += -self.a_noise.data + a_noise

    def forward(self, x):
        layer = x
        for fc_op in self.fc:
            layer = self.activation(fc_op(layer))

        layer_offset = self.fc_offset(layer)
        if self.enable_vae:
            layer_a = self.fca(layer)
            return torch.cat((layer_a, layer_offset), -1)
        else:
            return layer_offset * self.a_std if self.normalize_action else layer_offset

    def __act_distribution(self, x):
        a_mean = self.forward(x)
        m = D.Normal(a_mean, self.a_noise.view(-1))
        return m
    
    def logp(self, x, ac):
        if self.use_squashing:
            ac = torch.atanh(ac)
            m = self.__act_distribution(x)
            logp_pi = m.log_prob(ac).sum(axis=1)
            logp_pi -= (2*(np.log(2) - ac - F.softplus(-2*ac))).sum(axis=1)
            return logp_pi
        else:
            m = self.__act_distribution(x)
            return m.log_prob(ac).sum(axis=1)
    
    def __squash(self, ac):
        return torch.clamp(torch.tanh(ac), -0.999, 0.999)

    def act_deterministic(self, x):
        if self.use_squashing:
            return self.__squash(self.forward(x)).cpu().numpy()
        else:
            return self.forward(x).cpu().numpy()

    def act_stochastic(self, x, withLogP=False):
        m = self.__act_distribution(x)
        ac = m.sample()

        if self.use_squashing:
            if withLogP:
                logp_pi = m.log_prob(ac).sum(axis=1)
                logp_pi -= (2*(np.log(2) - ac - F.softplus(-2*ac))).sum(axis=1)
                return self.__squash(ac).cpu().numpy(), logp_pi.cpu().numpy()
            else:
                return self.__squash(ac).cpu().numpy()
        else:
            if withLogP:
                logp_pi = m.log_prob(ac).sum(axis=1)
                return ac.cpu().numpy(), logp_pi.cpu().numpy()
            else:
                return ac.cpu().numpy()
