from .scene_object import SceneObject

class Plane(SceneObject):
    def initialize(self):
        import math
        z2y = self.sim_client.getQuaternionFromEuler([-math.pi*0.5, 0, 0])
        self.object_id = self.sim_client.loadURDF("./data/urdf/plane_implicit.urdf", [0,0,0], z2y, useMaximalCoordinates=True)
        self.sim_client.changeDynamics(self.object_id, linkIndex=-1, lateralFriction=2.0)

    def pre_step(self):
        pass
