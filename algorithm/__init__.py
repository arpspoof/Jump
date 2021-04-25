from .ppo import ppo_bundle

from munch import munchify
algorithm_bundle_dict = munchify({
    "ppo": ppo_bundle
})
