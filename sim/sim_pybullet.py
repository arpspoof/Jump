import pybullet as p1
from .scene_object import SceneObject
from .bullet_client import BulletClient
import math

class PyBulletEngine:
    def __init__(self, sim_timestep):
        self.sim_step = sim_timestep
        self._pybullet_client = BulletClient(p1.DIRECT)

        self._pybullet_client.setGravity(0,-9.8,0)
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
        self._pybullet_client.setTimeStep(self.sim_step)
        self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

        self.scene_objects = []

    def add_object(self, obj):
        if not isinstance(obj, SceneObject):
            print('object must be an instance of SceneObject')
            assert False
        self.scene_objects.append(obj)
        obj.engine = self
        obj.sim_client = self._pybullet_client
        obj.initialize()

    def step_sim(self):
        for obj in self.scene_objects:
            obj.pre_step()
        self._pybullet_client.stepSimulation()
    
    def close(self):
        self._pybullet_client.disconnect()
