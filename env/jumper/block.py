from sim import SceneObject
from ur import URObject
import numpy as np

class Block(SceneObject, URObject):
    def __init__(self, wallAngle=0, wallDistance=0, height=0.5, heightdiff=0.7, min_height=0.02, max_height=0.75, vis_offset=0, real=False):
        self.__wallAngle = wallAngle
        self.__wallDistance = wallDistance + 2 + 0.02
        self.__height = height
        self.__heightdiff = heightdiff
        self.__min_height = min_height
        self.__max_height = max_height
        self.__vis_offset = vis_offset
        self.real = real

    def initialize(self):
        self.object_id = self.sim_client.loadURDF(
            "data/urdf/jumper/real_block.urdf" if self.real else "data/urdf/jumper/block.urdf", 
            basePosition=self.__basePos, baseOrientation=self.__baseOri,
            useMaximalCoordinates=True, useFixedBase=True)
        self.sim_client.changeDynamics(self.object_id, linkIndex=-1, lateralFriction=0.4)

    def pre_step(self):
        pass

    @property
    def __basePos(self):
        angle = self.__wallAngle / 180 * np.pi
        h = np.clip(self.__height - self.__heightdiff, self.__min_height, self.__max_height)
        if not self.real:
            return [self.wallDistance * np.sin(angle), h - 5, -self.wallDistance * np.cos(angle)]
        else:
            p = np.array([self.wallDistance * np.sin(angle), h - 5, -self.wallDistance * np.cos(angle)])
            p += np.array([np.cos(angle), 0, np.sin(angle)])*self.__vis_offset
            return p.tolist()
    
    @property
    def __baseOri(self):
        return list(self.sim_client.getQuaternionFromEuler([0, -self.__wallAngle / 180 * np.pi, 0]))

    def __update(self):
        self.sim_client.resetBasePositionAndOrientation(self.object_id, self.__basePos, self.__baseOri)

    @property
    def wallAngleDeg(self):
        return self.__wallAngle

    @wallAngleDeg.setter
    def wallAngleDeg(self, value):
        self.__wallAngle = value
        self.__update()

    @property
    def wallDistance(self):
        return self.__wallDistance

    @wallDistance.setter
    def wallDistance(self, value):
        self.__wallDistance = value + 2 + 0.02
        self.__update()

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, value):
        self.__height = value
        self.__update()

    @property
    def link_names(self):
        return ["block(shadow)"]

    @property
    def link_shapes(self):
        return ["box"]

    @property
    def link_sizes(self):
        return [[6.1, 10, 4]]
        
    def get_link_states(self):
        angle = self.__wallAngle / 180 * np.pi
        if self.real:
            pos = np.array(self.__basePos)
        else:
            pos = np.array(self.__basePos) + np.array([np.cos(angle), 0, np.sin(angle)])*self.__vis_offset
        return [pos.tolist() + self.__baseOri]
