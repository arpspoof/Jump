from .actor_critic import actor_critic

from munch import munchify
model_dict = munchify({
    "actor_critic": actor_critic
})
