import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from utils.humanoid_kin import HumanoidSkeleton
from sim import engine_builder, Plane, Character

def get_mocap_frames(path):
    frames = []
    with open(path) as f:
        j = json.load(f)
        data = j["Frames"]
        for row in data:
            frames.append(row[1:])
    return frames

'''
id  name        parent      dof         mirror
0   root        -1          -1          0
1   chest       0           3           1
2   neck        1           3           2
3   rhip        0           3           9
4   rknee       3           1           10
5   rankle      4           3           11
6   rshoulder   1           3           12
7   relbow      6           1           13
8   rwrist      7           0           14
9   lhip        0           3           3
10  lknee       9           1           4
11  lankle      10          3           5
12  lshoulder   1           3           6
13  lelbow      12          1           7
14  lwrist      13          0           8
'''

class FeatureConverter:
    def __init__(self):
        char_file = "data/characters/humanoid3d.txt"
        ctrl_file = "data/controllers/humanoid3d_ctrl.txt"
        self.__skeleton = HumanoidSkeleton(char_file, ctrl_file)
        '''
        self.__engine = engine_builder("pybullet", 1/600.0)
        self.__character = Character(self.__skeleton, True)
        self.__engine.add_object(self.__character)
        '''
        self.__parent_table = [-1, 0, 1, 0, 3, 4, 1, 6, 7, 0, 9, 10, 1, 12, 13]
        self.__dof = [0, 3, 3, 3, 1, 3, 3, 1, 0, 3, 1, 3, 3, 1, 0]
        self.__mirror = [0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]
    
    def quat_to_pos(self, frame):
        self.__skeleton.set_pose(np.array(frame))
        return list(self.__skeleton.get_feature())
    
    def pos_to_quat(self, pos):
        return list(self.__skeleton.inv_feature(pos))
    
    def exp_to_quat(self, exp):
        return self.__skeleton.exp_to_action(exp)
    
    def quat_to_exp(self, quat, root=[0,0,0,1,0,0,0]):
        return self.__skeleton.targ_pose_to_action(np.array(root + quat))
    
    def pos_to_exp(self, pos):
        quat = self.pos_to_quat(pos)
        return self.quat_to_exp(quat)
    
    def mirror(self, pos):
        mirrored = list(pos).copy()
        for i in range(15):
            idx = i*12
            mirroridx = self.__mirror[i]*12
            for j in range(12):
                mirrored[mirroridx + j] = pos[idx + j] * (-1 if j % 3 == 2 else 1)
        return mirrored
    
    def mirror_root(self, root):
        rot = R.from_quat([root[4], root[5], root[6], root[3]])
        expmap = rot.as_rotvec()
        expmap[0] = -expmap[0]
        expmap[1] = -expmap[1]
        q = R.from_rotvec(expmap).as_quat()
        return [root[0], root[1], -root[2], q[3], q[0], q[1], q[2]]
