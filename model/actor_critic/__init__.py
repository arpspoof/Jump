from .actor import Actor
from .critic import Critic
from .loader import load_model
from ..normalizer import Normalizer

from munch import munchify
actor_critic = munchify({
    "Actor": Actor,
    "Critic": Critic,
    "Normalizer": Normalizer,
    "policy": Actor,
    "loader": load_model
})
