from .scene_object import SceneObject
from .sim_pybullet import PyBulletEngine
from .character import Character
from .plane import Plane

engine_dict = {
                "pybullet": PyBulletEngine,
                }

def engine_builder(engine, timestep):
    if engine in engine_dict:
        return engine_dict[engine](timestep)
    else:
        print("select from %s" % str(engine_dict.keys()))
        assert(False and "not implemented engine %s" % engine)
