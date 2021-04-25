import torch

from ..normalizer import Normalizer
from .actor import Actor
from .critic import Critic

def load_model(ckpt):
    data = torch.load(ckpt)

    from presets import preset
    enable_vae = preset.vae.enable

    # get info from ckpt, then build network
    s_dim = data["actor"]["fc.0.weight"].shape[1]
    if enable_vae:
        a_dim = data["actor"]["fca.bias"].shape[0] + 28
    else:
        a_dim = 28
    a_min = data["actor"]["a_min"].numpy()
    a_max = data["actor"]["a_max"].numpy()
    a_noise = data["actor"]["a_noise"].numpy()
    a_hidden = list(map(lambda i: data["actor"]["fc.%d.bias" % i].shape[0], [0, 1]))
    c_hidden = list(map(lambda i: data["critic"]["fc.%d.bias" % i].shape[0], [0, 1]))

    # build network framework
    s_norm = Normalizer(s_dim, non_norm=[], sample_lim=-1) # TODO auto config non_norm
    actor = Actor(s_dim, a_dim, a_min, a_max, a_noise, a_hidden)
    critic= Critic(s_dim, 0, 1, c_hidden)

    # load checkpoint
    actor.load_state_dict(data["actor"])
    critic.load_state_dict(data["critic"])
    s_state = s_norm.state_dict()
    s_state.update(data["s_norm"])
    s_norm.load_state_dict(s_state)

    print("load from %s" % ckpt)

    # output
    model = {
            "policy": actor,
            "actor": actor,
            "critic": critic,
            "s_norm": s_norm,
            }

    return model
