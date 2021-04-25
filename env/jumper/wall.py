from sim import SceneObject
from ur import URObject
from utils.quaternion import Quaternion
import numpy as np

class Wall(SceneObject, URObject):
    def __init__(self, wallAngle=0, wallDistance=0, height=0.5, vis_offset=0):
        self.__wallAngle = wallAngle
        self.__wallDistance = wallDistance
        self.__height = height
        self.__vis_offset = vis_offset

    def initialize(self):
        self.object_id = self.sim_client.loadURDF("data/urdf/jumper/wall.urdf", 
            basePosition=self.__basePos, baseOrientation=self.__baseOri,
            useMaximalCoordinates=True, useFixedBase=True)
    
    def pre_step(self):
        pass

    @property
    def __basePos(self):
        angle = self.__wallAngle / 180 * np.pi
        return [self.wallDistance * np.sin(angle), self.__height - 5, -self.wallDistance * np.cos(angle)]
    
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
        self.__wallDistance = value
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
        return ["l-bar(shadow)", "r-bar(shadow)", "t-bar(shadow)"]

    @property
    def link_shapes(self):
        return ["box", "box", "capsule"]

    @property
    def link_sizes(self):
        return [[0.1, 5, 0.1], [0.1, 5, 0.1], [0.02, 6, 0.02]]
        
    def get_link_states(self):
        angle = self.__wallAngle / 180 * np.pi
        offset = np.array([np.cos(angle), 0, np.sin(angle)]) 
        l_bar_pos = np.array(self.__basePos) - offset*(3.0 - self.__vis_offset); l_bar_pos[1] = 0
        r_bar_pos = np.array(self.__basePos) + offset*(3.0 + self.__vis_offset); r_bar_pos[1] = 0
        t_bar_quat = Quaternion.fromXYZW(self.__baseOri).mul(Quaternion.fromAngleAxis(np.pi/2, np.array([0,0,1])))
        t_bar_pos = np.array(self.__basePos) + offset*self.__vis_offset; t_bar_pos[1] = self.height - 0.02
        return [
            l_bar_pos.tolist() + self.__baseOri,
            r_bar_pos.tolist() + self.__baseOri,
            t_bar_pos.tolist() + t_bar_quat.xyzw().tolist()
        ]
    
    def point_to_plane_distance(self, pos):
        angle = self.__wallAngle / 180 * np.pi
        return np.sin(angle)*pos[0] - np.cos(angle)*pos[2] - self.wallDistance




class BarStock(SceneObject, URObject):
    def __init__(self, wallAngle=0, wallDistance=0, height=0.5, vis_offset=0):
        self.__wallAngle = wallAngle
        self.__wallDistance = wallDistance
        self.__height = height
        self.__vis_offset = vis_offset

    def initialize(self):
        angle =self.__wallAngle * np.pi/180
        offset_dist = self.__wallDistance - 0.05
        pos = np.array([offset_dist * np.sin(angle), self.__height - 3.075, -offset_dist * np.cos(angle)])
        pos[0] += self.__vis_offset*np.cos(angle)
        pos[2] += self.__vis_offset*np.sin(angle)
        self.pos = pos.copy()
        pos_stick1 = pos.copy()
        pos_stick1[0] -= 2*np.cos(angle)
        pos_stick1[2] -= 2*np.sin(angle)

        pos_stick2 = pos.copy()
        pos_stick2[0] += 2*np.cos(angle)
        pos_stick2[2] += 2*np.sin(angle)

        pos_bar = pos
        pos_bar[1] = self.__height - 0.025
        pos_bar[0] += 0.05*np.sin(angle)
        pos_bar[2] -= 0.05*np.cos(angle)

        rot_stick = [0,np.sin(np.pi/4 - angle/2), 0, np.cos(np.pi/4 - angle/2)]
        self.stick1 = self.sim_client.loadURDF("data/urdf/jumper/bracket.urdf", basePosition= pos_stick1.tolist(), baseOrientation=rot_stick,useMaximalCoordinates=True, useFixedBase=True)
        self.stick2 = self.sim_client.loadURDF("data/urdf/jumper/bracket.urdf", basePosition= pos_stick2.tolist(), baseOrientation= rot_stick,useMaximalCoordinates=True, useFixedBase=True)
        self.object_id = self.sim_client.loadURDF("data/urdf/jumper/bar.urdf", basePosition= pos_bar, baseOrientation=rot_stick,useMaximalCoordinates=True)

    def reset_bar(self):
        angle =self.__wallAngle * np.pi/180
        offset_dist = self.__wallDistance - 0.05
        pos = np.array([offset_dist * np.sin(angle), self.__height - 3.075, -offset_dist * np.cos(angle)])
        pos[0] += self.__vis_offset*np.cos(angle)
        pos[2] += self.__vis_offset*np.sin(angle)
        self.pos = pos.copy()
        pos_stick1 = pos.copy()
        pos_stick1[0] -= 2*np.cos(angle)
        pos_stick1[2] -= 2*np.sin(angle)

        pos_stick2 = pos.copy()
        pos_stick2[0] += 2*np.cos(angle)
        pos_stick2[2] += 2*np.sin(angle)

        pos_bar = pos
        pos_bar[1] = self.__height - 0.025
        pos_bar[0] += 0.05*np.sin(angle)
        pos_bar[2] -= 0.05*np.cos(angle)

        rot_stick = [0,np.sin(np.pi/4 - angle/2), 0, np.cos(np.pi/4 - angle/2)]
        self.sim_client.resetBasePositionAndOrientation(bodyUniqueId=self.object_id, 
            posObj=pos_bar, ornObj=rot_stick)
    
    def pre_step(self):
        pass

    @property
    def __basePos(self):
        angle = self.__wallAngle / 180 * np.pi
        return self.pos.tolist()
    
    @property
    def __baseOri(self):
        return list(self.sim_client.getQuaternionFromEuler([0, np.pi/2-self.__wallAngle / 180 * np.pi, 0]))

    def __update(self):
        # self.sim_client.resetBasePositionAndOrientation(self.object_id, self.__basePos, self.__baseOri)
        pass
    
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
        self.__wallDistance = value
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
        return ["l-bar(shadow)", "r-bar(shadow)", "t-bar(shadow)"]

    @property
    def link_shapes(self):
        return ["box", "box", "capsule"]

    @property
    def link_sizes(self):
        return [[0.05, 7, 0.05], [0.05, 7, 0.05], [0.02, 4, 0.02]]
        
    def get_link_states(self):
        pos_stick1, rot_stick1 = self.sim_client.getBasePositionAndOrientation(self.stick1)
        pos_stick2, rot_stick2 = self.sim_client.getBasePositionAndOrientation(self.stick2)
        pos_bar, rot_bar = self.sim_client.getBasePositionAndOrientation(self.object_id)
        rot_bar = Quaternion.fromXYZW(rot_bar).mul(Quaternion.fromXYZW([np.sin(np.pi/4),0,0, np.cos(np.pi/4)]))
        return [
            pos_stick1+rot_stick1,
            pos_stick2+rot_stick2,
            np.array(pos_bar).tolist()+rot_bar.xyzw().tolist(),

        ]
    
    def point_to_plane_distance(self, pos):
        angle = self.__wallAngle / 180 * np.pi
        return np.sin(angle)*pos[0] - np.cos(angle)*pos[2] - self.wallDistance