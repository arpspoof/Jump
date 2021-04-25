import numpy as np
from numpy.random import uniform
import json
import math
from utils.quaternion import Quaternion
from utils.quaternion import computeAngVel, computeAngVelRel

from Kinematic.py import KinematicCore

class JointType():
    BASE     = 0
    FIXED    = 1
    REVOLUTE = 2
    SPHERE   = 3

# pose dof for joints
Dof = {
    JointType.BASE:       4,
    JointType.FIXED:      0,
    JointType.REVOLUTE:   1,
    JointType.SPHERE:     4,
}

# exp dof for joints
ExpDof = {
    JointType.BASE:       0,
    JointType.FIXED:      0,
    JointType.REVOLUTE:   1,
    JointType.SPHERE:     3,
}

NAME2TYPE={
    "none":       JointType.BASE,
    "fixed":      JointType.FIXED,
    "revolute":   JointType.REVOLUTE,
    "spherical": JointType.SPHERE,
}

POSE2QUAT={
    JointType.BASE:       lambda x: Quaternion(x[0], x[1], x[2], x[3]),
    JointType.FIXED:      lambda x: Quaternion(1, 0, 0, 0),
    JointType.REVOLUTE:   lambda x: Quaternion(math.cos(x/2), 0.0, 0.0, math.sin(x/2)),
    JointType.SPHERE:     lambda x: Quaternion(x[0], x[1], x[2], x[3]),
}

VEL2OMG={
    JointType.BASE:       lambda x: np.array([x[0], x[1], x[2]]),
    JointType.FIXED:      lambda x: np.array([0, 0, 0]),
    JointType.REVOLUTE:   lambda x: np.array([0, 0, x[0]]),
    JointType.SPHERE:     lambda x: np.array([x[0], x[1], x[2]]),
}

class Joint(object):
    def __init__(self, info):
        """
            Inputs:
                info  json item load from character file
        """
        self._pos       = np.array([info["AttachX"], info["AttachY"], info["AttachZ"]])
        self._name      = info["Name"]
        self._parent_id = info["Parent"]
        self._parent    = None
        self._type_name = info['Type']
        self._type      = NAME2TYPE[info['Type']]
        self._dof       = Dof[self._type]
        self._expdof    = ExpDof[self._type]
        self._w         = info["DiffWeight"]
        self._is_end_effector = info["IsEndEffector"]

        if self._type is JointType.REVOLUTE:
            self._torque_lim = [info["TorqueLim"]]
        elif self._type is JointType.SPHERE:
            if "TorqueLimX" not in info:
                self._torque_lim = [info["TorqueLim"]]*4
            else:
                self._torque_lim = [info["TorqueLimX"], info["TorqueLimY"], info["TorqueLimZ"], 0]
        else:
            self._torque_lim = [0]*4

        if self._type is JointType.REVOLUTE:
            self.limlow = np.array([info["LimLow0"]])
            self.limhigh = np.array([info["LimHigh0"]])
        elif self._type is JointType.SPHERE:
            self.limlow = -3.14 * np.ones(3)
            self.limhigh = 3.14 * np.ones(3)
        else:
            self.limlow = np.array([])
            self.limhigh = np.array([])

        self.pose = np.zeros(self._dof)
        self.pose2quat = POSE2QUAT[self._type]
        self.quat = Quaternion(1, 0, 0, 0)

        self.vel = np.zeros(self._dof)
        self.vel2omg = VEL2OMG[self._type]
        self.omg = np.zeros(3)

    def __unicode__(self):
        info = u"Joint: name %s, pos %s, parent %d" % (self._name, str(self._pos), self._parent_id)
        return info

    __str__ = __unicode__
    __repr__= __unicode__

    def dof(self):
        return self._dof

    def expdof(self):
        return self._expdof

    def local_pos(self):
        return self._pos

    def set_pose(self, pose):
        assert(pose.shape == self.pose.shape)
        self.pose = pose
        # calculate quat
        self.quat = self.pose2quat(pose)

    def set_vel(self, vel):
        assert(vel.shape == self.vel.shape)
        self.omg = self.vel2omg(vel)

    def get_quat(self):
        return self.quat

    def get_omg(self):
        return self.omg

    def get_parent(self):
        return self._parent_id

    def get_a_min(self):
        std = (self.limhigh - self.limlow) / 2
        mean = (self.limhigh + self.limlow) / 2
        a_min = mean - std*2
        return a_min

    def get_a_max(self):
        std = (self.limhigh - self.limlow) / 2
        mean = (self.limhigh + self.limlow) / 2
        a_max = mean + std*2
        return a_max

class Body:
    def __init__(self, info):
        self._info = info
        self._pos = np.array([info["AttachX"], info["AttachY"], info["AttachZ"]])
        self._name = info["Name"]
        self._joint_id = info["ID"]

    def get_joint_id(self):
        return self._joint_id

    def get_local_pos(self):
        return self._pos
    
    def get_body_name(self):
        return self._name
    
    def get_body_shape(self):
        return self._info["Shape"]
    
    def get_body_size(self):
        ar = np.array([self._info["Param0"], self._info["Param1"], self._info["Param2"]])
        shape = self.get_body_shape()
        if shape == "sphere":
            ar /= 2.0
        if shape == "capsule":
            ar[0] /= 2.0
            ar[2] /= 2.0
        return ar

class HumanoidSkeleton(object):
    """ Store humanoid skeleton information, can perform following functions:

            1) compute link positions and velocities when given character pose and vel
            2) construct character state when given root position and orintaion
            3) slerp between two character poses
            4) transfer between action and target poses

            __init__()
                 _calc_kin_info()
                 _calc_ctrl_info()

            set_pose()

            set_vel()

            build_state()

            slerp()

            pose representation:
                - root pos (x, y, z)
                - root rot (w, x, y, z) in world frame
                - joint rots theta or (w, z, y, z) in local frame, VelRel

            vel w/ padding representation:
                - root vel (vx, vy, vz)
                - root angular vel (wx, wy, wz) in world frame
                - joint angular vel w or (wx, wy, wz, 0) in local frame, VelRel

            vel w/o padding representation
                - root vel (vx, vy, vz)
                - root angular vel (wx, wy, wz) in world frame
                - joint angular vel w or (wx, wy, wz) in local frame, VelRel

            state
                - Y coordinate of origin pos
                - Root link's pos (x, y, z) in xyz coordinate
                - Root link's quat (W, X, Y, Z) in XYZ coordinate
                - Other links' pos (x, y, z) and quat (w, x, y, z) in xyz coordinate
                - Root link's vel (Vx, Vy, Vz) in XYZ coordinate
                - Root link's omega (Wx, Wy, Wz) in XYZ coordinate
                - Other links' vel (vx, vy, vz) and omg (wx, wy, wz) in xyz coordinate

                xyz's origin is set by origin_pos, or character's root joint
                xyz's rotation is calculated by rotating Y-axis to make X-axis to heading
                direction calculated by origin_rot or root joint's rotation

            action
                - no root pose or joint
                - angle-axis (\theta, nx, ny, nz) for spherical joints and \theta for
                    revolute joints

            action represented in exponential map
                - no root pose or joint
                - exponential map (\Omega_x, \Omega_y, \Omega_z) for spherical joints and
                    \theta for revolute joints
    """

    def __init__(self, char_file, ctrl_file):
        # NOTICE humanoid.txt and humanoid.urdf have different scales, urdf is 4x larger
        self._kin_core = KinematicCore.cKinematicCore(char_file)

        ### all private members
        self.joints = None
        self.bodys  = None
        self.dof    = None
        self.expdof = None

        # TODO move to function
        self.num_joints  = None
        self.joint_types = None
        self.joint_dof   = None
        self.pos_start   = None     # start position in pose/vel vec
        self.act_start   = None     # start position in action vec
        self.joint_w     = None

        # control configuration
        self.kp = None
        self.kd = None
        self.a_max = None
        self.a_min = None

        ### all private members
        self._calc_kin_info(char_file)
        self._calc_ctrl_info(ctrl_file)

    def _calc_kin_info(self, char_file):
        # handle character skeleton information
        with open(char_file, 'r') as f:
            json_data = json.load(f)
            joints_data = json_data["Skeleton"]["Joints"]
            bodys_data = json_data["BodyDefs"]

        # build for pose/vel transfer
        self.joints = list(map(lambda x: Joint(x),  joints_data))
        self.bodys  = list(map(lambda x: Body(x),   bodys_data))

        self.dof = sum(map(lambda x: x.dof(), self.joints)) + 3
        self.expdof = sum(map(lambda x: x.expdof(), self.joints))

        # build velocity padding - compressing
        self.comp2pad = [0, 1, 2]  # root vel
        p_off = 3
        for j in self.joints:
            if j.dof() == 1:
                self.comp2pad += [p_off]
                p_off += 1
            if j.dof() == 4:
                self.comp2pad += [p_off, p_off+1, p_off+2]
                p_off += 4

        self.comp2pad = np.array(self.comp2pad, dtype=np.int)

        # build joint infomation mat
        self.num_joints     = len(self.joints)
        self.joint_types    = list(map(lambda x: x._type,   self.joints))
        self.joint_w        = list(map(lambda x: x._w,      self.joints))
        self.joint_dof      = list(map(lambda x: x.dof(),   self.joints))
        self.joint_dof      = np.array(self.joint_dof, dtype=np.int)
        self.joint_expdof   = list(map(lambda x: x.expdof(),self.joints))
        self.joint_expdof   = np.array(self.joint_expdof, dtype=np.int)
        self.end_effectors  = list(filter(
                                                        lambda x: self.joints[x]._is_end_effector,
                                                        range(self.num_joints)))

        self.pos_start = np.zeros(self.num_joints, dtype=np.int)  # start offset for each joint in pos vector
        for i in range(1, self.num_joints):
            self.pos_start[i] = self.pos_start[i-1] + self.joint_dof[i-1]

        self.exp_start = np.zeros(self.num_joints, dtype=np.int)  # start offset for each joint in exp vector
        for i in range(1, self.num_joints):
            self.exp_start[i] = self.exp_start[i-1] + self.joint_expdof[i-1]

        self.pos_start += 3                     # offset for root pos
        self.act_start = self.pos_start - 7     # 7 dim for root pos and root joint
                                                                                        # so the act_start[0] is meaningless

        self.joint_mat = np.array([
                                                                self.joint_types, self.joint_dof,
                                                                self.pos_start, self.act_start,
                                                            ], dtype=np.int)
        self.joint_mat = self.joint_mat.transpose()

    def _calc_ctrl_info(self, ctrl_file):
        """ Build Kp, Kd and torque_lim
        """
        # handle controller information
        with open(ctrl_file, 'r') as f:
            data = json.load(f)
            controller = data["PDControllers"]
            assert(len(data["PDControllers"]) == self.num_joints)

        self.kp = [0,0,0] # for root pos
        self.kd = [0,0,0] # for root pos
        self.torque_lim = [0, 0, 0] # for root pos
        for i in range(self.num_joints):
            if self.joint_types[i] is not JointType.FIXED:
                self.kp += [controller[i]["Kp"]] * self.joint_dof[i]
                self.kd += [controller[i]["Kd"]] * self.joint_dof[i]
                self.torque_lim += self.joints[i]._torque_lim

        self.kp = np.array(self.kp)
        self.kd = np.array(self.kd)
        self.torque_lim = np.array(self.torque_lim)

    def build_a_min(self):
        a_mins = [j.get_a_min() for j in self.joints]
        a_min = np.concatenate(a_mins)
        assert(a_min.size == self.expdof)
        return a_min

    def build_a_max(self):
        a_maxs = [j.get_a_max() for j in self.joints]
        a_max = np.concatenate(a_maxs)
        assert(a_max.size == self.expdof)
        return a_max

    def get_link_ids(self):
        return list(range(len(self.bodys)))

    def get_link_names(self):
        return list(map(lambda body: body.get_body_name(), self.bodys))

    def get_link_shapes(self):
        return list(map(lambda body: body.get_body_shape(), self.bodys))

    def get_link_sizes(self):
        return list(map(lambda body: body.get_body_size().tolist(), self.bodys))

    def set_pose(self, pose):
        """ Set character's pose

        Inputs:
            pose   np.array of float, should be equal to self.dof
        """
        assert(pose.size == self.dof)
        self._kin_core.setPose(pose)

    def set_vel(self, vel):
        """ Set character's velocity

            *NOTICE* set_pose should be already called

        Inputs:
            vel  numpy array of velocity, shoubld be equal to self.dof
        """
        assert(vel.size == self.dof)
        self._kin_core.setVel(vel)

    def set_heading_vec(self, head):
        """ Set heading vector

        """
        assert(len(head) == 3)
        self._kin_core.setHeadingVec(head)

    def build_state(self, origin_pos=None, origin_rot=None, root_global=True):
        """ Build character state

            Inputs:
                origin_pos      new coordinate's origin (x, y, z)
                origin_rot      reference rotation (w, x, y, z), then a heading direction
                                                is calculated as new coordinate's x-axis

            Outputs:
                state           np.array of float
        """
        if origin_pos is None and origin_rot is None:
            state = self._kin_core.buildState()
            state = np.array(state)
        elif origin_pos is not None and origin_rot is not None:
            state = self._kin_core.buildState(origin_pos, origin_rot, root_global)
        else:
            raise ValueError("not implemented yet")
        return state

    def build_state2(self):
        state = self._kin_core.buildState2()
        return np.array(state)

    def get_com_pos(self):
        """ Return CoM position

            *NOTICE* set_pose should be already called
        """
        return np.array(self._kin_core.getCoMPos())

    def get_com_vel(self):
        """ Return CoM velocity

            *NOTICE* set_pose and set_vel should be already called
        """
        return np.array(self._kin_core.getCoMVel())

    def slerp(self, pose0, pose1, t):
        """ slerp between two poses

            Inputs:
                pose0   np.array of float, start pose
                pose1   np.array of float, end pose
                t       float in [0, 1], interpolating parameter

            Outputs:
                pose_t  np.array of float, interpolated pose
        """

        assert(pose0.size == self.dof)
        assert(pose1.size == self.dof)

        pose_t = self._kin_core.slerp(pose0, pose1, t)
        pose_t = np.array(pose_t)

        return pose_t

    def computeVel(self, pose0, pose1, dt, padding=True):
        """ Compute velocity between two poses

            Inputs:
                pose0   np.array of float, start pose
                pose1   np.array of float, end pose
                dt      float, duraction between two poses

            Outputs:
                avg_vel np.array of float, vel (w/ or w/o padding)
        """
        assert(pose0.size == self.dof)
        assert(pose1.size == self.dof)

        avg_vel = np.zeros_like(pose0)

        root0 = pose0[:3]
        root1 = pose1[:3]
        avg_vel[:3] = (root1 - root0) / dt

        offset = 3

        # root angular velocity is in world coordinate
        dof = self.joint_dof[0]
        quat0 = Quaternion.fromWXYZ(pose0[offset:offset+dof])
        quat1 = Quaternion.fromWXYZ(pose1[offset:offset+dof])
        avg_vel[offset : offset+3] = computeAngVel(quat0, quat1, dt)
        offset += dof

        # other joints
        for i in range(1, self.num_joints):
            dof = self.joint_dof[i]
            if dof == 1:  # revolute
                theta0 = pose0[offset]
                theta1 = pose1[offset]
                avg_vel[offset] = (theta1 - theta0) / dt
            elif dof == 4:  # spherical
                quat0 = Quaternion.fromWXYZ(pose0[offset:offset+dof])
                quat1 = Quaternion.fromWXYZ(pose1[offset:offset+dof])
                avg_vel[offset : offset+3] = computeAngVelRel(quat0, quat1, dt)
            offset += dof

        if padding is False:
            avg_vel = avg_vel[self.comp2pad]

        return avg_vel

    def toLocalFrame(self, pose, vel, ori_pos=None, ori_rot=None):
        """ Convert pose and vel from world frame to local frame,
                the local frame heading direction is rotated x-axis

            Inputs:
                pose        np.array of float, character pose
                vel         np.array of float, character vel w/ padding
                ori_pos     np.array of float, 3 dim, position of local coordinate origin
                ori_rot     np.array of float, 4 dim, (w, x, y, z) quat of local coordinate orientation

            Outputs:
                local_pose
                local_vel
        """
        if ori_pos is None:
            ori_pos = pose[:3]
        if ori_rot is None:
            ori_rot = pose[3:7]

        # heading theta
        inv_ori_rot = self.buildHeadingTrans(ori_rot)

        local_pos = pose.copy()
        local_vel = vel.copy()

        # ignore y difference, because local cooridnate shares xoz plane with world
        local_pos[0] -= ori_pos[0]                      # root x pos
        local_pos[2] -= ori_pos[2]                      # root y pos

        ori_rot = Quaternion.fromWXYZ(ori_rot)
        ori_rot = inv_ori_rot.mul(ori_rot)
        local_pos[3:7] = ori_rot.pos_wxyz()              # root orientation

        local_vel[:3] = inv_ori_rot.rotate(vel[:3])      # root velocity
        local_vel[3:6] = inv_ori_rot.rotate(vel[3:6])    # root angular velocity

        return local_pos, local_vel

    def buildHeadingTrans(self, rot):
        """ Build the rotation that rotate to local coordinate
            rot     np.array of float, 4 dim, (w, x, y, z) quat of coordinate orientation
        """
        theta = self._kin_core.getHeadingTheta(rot)
        inv_rot = Quaternion.fromExpMap(np.array([0, -theta, 0]))
        return inv_rot

    def compressVel(self, vel):
        """ Squeeze velocity from padded to compressed

            Inputs:
                vel     np.array of float, vel w/ padding

            Outputs:
                vel_cmp np.array of float, vel w/o padding
        """
        vel_cmp = vel[self.comp2pad]
        return vel_cmp

    def padVel(self, vel):
        """ Pad velocity from compressed to padded

            Inputs:
                vel     np.array of float, vel w/o padding

            Outputs:
                vel_pad np.array of float, vel w/ padding
        """
        vel_pad = np.zeros(self.dof)
        vel_pad[self.comp2pad] = vel
        return vel_pad

    def action_to_targ_pose(self, action):
        """ Converte action to PD controller target pose

            Inputs:
                action      np.array of float, action which DeepMimicSim can take in

            Outputs:
                pose        np.array of float, pose of character
        """
        assert(action.size == self.dof - 7)

        targ_pose = np.zeros(self.dof)
        targ_pose[3] = 1
        for i in range(1, self.num_joints):
            dof = self.joint_dof[i]
            p_off = self.pos_start[i]
            a_off = self.act_start[i]
            if dof == 1:  # revolute
                targ_pose[p_off] = action[a_off]
            elif dof == 4:  # spherical
                angle = action[a_off]
                axis = action[a_off+1:a_off+4]
                quata = Quaternion.fromAngleAxis(angle, axis)
                targ_pose[p_off : p_off+4] = quata.wxyz()

        assert(np.isfinite(sum(targ_pose))), embed() #(action, targ_pose)
        return targ_pose

    def targ_pose_to_action(self, pose):
        """ Convert desired pose to action

            Inputs:
                pose        np.array of float, pose of character

            Outputs:
                action      np.array of float, action which DeepMimicSim can take in
        """
        assert(pose.size == self.dof)

        action = np.zeros(self.dof-7)
        for i in range(1, self.num_joints):
            dof = self.joint_dof[i]
            p_off = self.pos_start[i]
            a_off = self.act_start[i]
            if dof == 1:  # revolute
                action[a_off] = pose[p_off]
            elif dof == 4:  # spherical
                quata = Quaternion.fromWXYZ(pose[p_off:p_off+4])
                action[a_off : a_off+4] = quata.angaxis()

        assert(np.isfinite(sum(action))), embed() #(action, targ_pose)
        return action

    def action_as_offset(self, pose, action):
        """ Take action as offset of a given reference pose, return standard action

            Inputs:
                pose        np.array of float, pose of character
                action      np.array of float, offset action to the character

            Outputs:
                new_action  np.array of float, standard action which DeepMimicSim can take in
        """
        assert(pose.size == self.dof)
        assert(action.size == self.dof - 7)

        new_action = self._kin_core.actionAsOffset(pose, action.tolist())

        return new_action

    def exp_to_action(self, expmap):
        """ Convert action represented in exponential map to angle-axis action
                which deepmimic_sim can take in

            Inputs:
                expmap      np.array of float, action represented in exponential map

            Outputs:
                action      np.array of float, action for deepmimic environment
        """
        assert(expmap.size == self.expdof)
        action = np.zeros(self.dof-7)

        for i in range(1, self.num_joints):
            dof = self.joint_dof[i]
            e_off = self.exp_start[i]
            a_off = self.act_start[i]
            if dof == 1:  # revolute
                action[a_off] = expmap[e_off]
            elif dof == 4:  # spherical
                quata = Quaternion.fromExpMap(expmap[e_off:e_off+3])
                action[a_off:a_off+4] = quata.angaxis()

        assert(np.isfinite(sum(action))), embed() #(action, targ_pose)
        return action


    def exp_to_targ_pose(self, expmap, padding=True):
        """ Convert action represented in exponential map to angle-axis action
                which deepmimic_sim can take in

            Inputs:
                expmap      np.array of float, action represented in exponential map

            Outputs:
                pose        np.array of float, pose of character
        """
        assert(expmap.size == self.expdof)
        return self._kin_core.expMapToTargetPose(expmap.tolist(), padding)

    def exp_to_targ_pose_old(self, expmap):
        """ Convert action represented in exponential map to angle-axis action
                which deepmimic_sim can take in

            Inputs:
                expmap      np.array of float, action represented in exponential map

            Outputs:
                pose        np.array of float, pose of character
        """
        assert(expmap.size == self.expdof)
        pose = np.zeros(self.dof)

        for i in range(1, self.num_joints):
            dof = self.joint_dof[i]
            e_off = self.exp_start[i]
            p_off = self.pos_start[i]
            if dof == 1:  # revolute
                pose[p_off] = expmap[e_off]
            elif dof == 4:  # spherical
                quata = Quaternion.fromExpMap(expmap[e_off:e_off+3])
                pose[p_off:p_off+4] = quata.wxyz()

        assert(np.isfinite(sum(pose))), embed() #(action, targ_pose)
        return pose

    def targ_pose_to_exp(self, pose):
        """ Convert target pose to exponential map, used as initialization of
                Actor_FDM reference memory

            Inputs:
                pose        np.array of float, pose of character

            Outputs:
                exp         np.array of float, action represented in exponential map
        """
        assert(pose.size == self.dof)

        expmap = np.zeros(self.expdof)
        for i in range(1, self.num_joints):
            dof = self.joint_dof[i]
            p_off = self.pos_start[i]
            e_off = self.exp_start[i]
            if dof == 1:  # revolute
                expmap[e_off] = pose[p_off]
            elif dof == 4:  # spherical
                quata = Quaternion.fromWXYZ(pose[p_off:p_off+4])
                expmap[e_off : e_off+3] = quata.expmap()

        assert(np.isfinite(sum(expmap))), embed() #(action, targ_pose)
        return expmap

    def normalize_target_pose(self, pose):
        """
            normalize given pose to make quaternions norm as 1
        """
        assert(pose.size == self.dof)
        norm_pose = pose.copy()
        for i in range(self.num_joints):
            dof = self.joint_dof[i]
            p_off = self.pos_start[i]
            if dof == 4:  # spherical
                quata = Quaternion.fromWXYZ(pose[p_off:p_off+4])
                norm_pose[p_off : p_off+4] = quata.pos_wxyz()

        assert(np.isfinite(sum(norm_pose))), embed() #(action, targ_pose)
        return norm_pose

    def pose_wxyz_to_xyzw(self, pose):
        pose_new = pose.copy()

        p_off = 3
        quat = pose[p_off:p_off+4]
        pose_new[p_off:p_off+3] = quat[1:]
        pose_new[p_off+3] = quat[0]

        for i in range(1, self.num_joints):
            dof = self.joint_dof[i]
            p_off = self.pos_start[i]
            if dof == 4:
                quat = pose[p_off:p_off+4]
                pose_new[p_off:p_off+3] = quat[1:]
                pose_new[p_off+3] = quat[0]

        return pose_new

    def get_reward(self, pose0, vel0, pose1, vel1):
        assert(len(pose0) == self.dof), "pose 0 size missmatch"
        assert(len(pose1) == self.dof), "pose 1 size missmatch"
        assert(len(vel0) == self.dof), "vel 0 size missmatch"
        assert(len(vel1) == self.dof), "vel 1 size missmatch"
        return self._kin_core.calcReward(pose0, vel0, pose1, vel1)

    def get_reward2(self, pose0, vel0, pose1, vel1):
        assert(len(pose0) == self.dof), "pose 0 size missmatch"
        assert(len(pose1) == self.dof), "pose 1 size missmatch"
        assert(len(vel0) == self.dof), "vel 0 size missmatch"
        assert(len(vel1) == self.dof), "vel 1 size missmatch"
        return self._kin_core.calcReward2(pose0, vel0, pose1, vel1)

    def get_err_vec(self, pose0, vel0, pose1, vel1):
        assert(len(pose0) == self.dof), "pose 0 size missmatch"
        assert(len(pose1) == self.dof), "pose 1 size missmatch"
        assert(len(vel0) == self.dof), "vel 0 size missmatch"
        assert(len(vel1) == self.dof), "vel 1 size missmatch"
        self._kin_core.calcReward(pose0, vel0, pose1, vel1)
        return np.array(self._kin_core.getErrorVec())

    def get_sub_rewards(self, pose0, vel0, pose1, vel1):
        err_vec = self.get_err_vec(pose0, vel0, pose1, vel1)
        scale = np.array([2, 0.1, 40, 5, 10])
        sub_rewards = np.exp(- scale * err_vec)
        return sub_rewards

    def lowest_height(self, pose):
        return self._kin_core.lowestHeight(pose)

    def disturb_pose(self, pose, noise):
        """ uniform disturb pose

            pose
            noise  float, deviation angle in radius
        """
        new_pose = pose.copy()

        noise3d = noise / np.sqrt(3)
        new_pose[:3] += uniform(-noise3d, noise3d, 3)

        for jstar, jdof in zip(self.pos_start, self.joint_dof):
            if jdof == 1:
                new_pose[jstar] += uniform(-noise, noise)
            elif jdof == 4:
                noise_quat = Quaternion.fromExpMap(uniform(-noise3d, noise3d, 3))
                ori_quat = Quaternion.fromWXYZ(pose[jstar:jstar+4])
                new_quat = noise_quat.mul(ori_quat)
                new_pose[jstar:jstar+4] = new_quat.wxyz()
            elif jdof == 0:
                pass
            else:
                assert(False and "not support jdof other than 1 and 4")

        return new_pose

    def disturb_vel(self, vel, noise):
        """ uniform disturb vel

            vel
            noise  float, deviation angle in radius
        """
        new_vel = vel.copy()

        noise3d = noise / np.sqrt(3)
        new_vel[:3] += uniform(-noise3d, noise3d, 3)

        for jstar, jdof in zip(self.pos_start, self.joint_dof):
            if jdof == 1:
                new_vel[jstar] += uniform(-noise, noise)
            elif jdof == 4:
                new_vel[jstar:jstar+3] += uniform(-noise3d, noise3d, 3)
            elif jdof == 0:
                pass
            else:
                assert(False and "not support jdof other than 1 and 4")

        return new_vel
    
    def get_joint_global_pos(self, id):
        return list(self._kin_core.getJointPos(id))
    
    def get_joint_global_quat(self, id):
        return list(self._kin_core.getJointQuat(id))
    
    def get_body_global_pos(self, id):
        return list(self._kin_core.getBodyPos(id))
    
    def get_feature(self):
        return self._kin_core.getFeature()
    
    def inv_feature(self, pos):
        return self._kin_core.invFeature(pos)
