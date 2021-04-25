from .vec_env import spawn_vec_env
from .base import BaseEnv

from .jumper import *

def get_env(env_name, bundle=False):
    if env_name == "jumper_run":
        return jumper_run_bundle if bundle else JumperRunEnv
    elif env_name == "jumper_run2":
        return jumper_run2_bundle if bundle else JumperRun2Env
    elif env_name == "highjump":
        return highjump_bundle if bundle else HighJumpEnv
    else:
        print('environment %s does not exist!' % env_name)
        raise ModuleNotFoundError
