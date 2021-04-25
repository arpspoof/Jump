from .jumper_run import JumperRunEnv
from .jumper_run2 import JumperRun2Env
from .highjump import HighJumpEnv
from .wall_scheduler import JumperWallScheduler

from munch import munchify

jumper_run_bundle = munchify({
    "env": JumperRunEnv,
    "schedulers": []
})
jumper_run2_bundle = munchify({
    "env": JumperRun2Env,
    "schedulers": []
})
highjump_bundle = munchify({
    "env": HighJumpEnv,
    "schedulers": [JumperWallScheduler]
})
