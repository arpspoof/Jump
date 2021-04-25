from .PPO_vec import PPO_vec

from munch import munchify
ppo_bundle = munchify({
    "algorithm": PPO_vec,
    "model_class": "actor_critic"
})
