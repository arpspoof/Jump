from ur import URObject
from .scene_object import SceneObject
from utils.humanoid_kin import JointType
from utils.quaternion import Quaternion
import numpy as np

class Character(URObject, SceneObject):
    def __init__(self, skeleton, self_collision):
        self.skeleton = skeleton
        self.self_collision = self_collision
        self.allowed_contacts = []
        self.not_allowed_contacts = []
        self.kpkd_multiplier = 1.0

    def initialize(self):
        self.load_character()
        self.init_spd()
    
    @property
    def link_names(self):
        return self.skeleton.get_link_names() + [
            'belly', 'hipr', 'hipl', 'rj_shoulder', 'lj_shoulder',
    #        'rj_knee', 'lj_knee', 'rj_elbow', 'lj_elbow',
            'j_neck', 'r_clavicle', 'l_clavicle'
        ]
    
    @property
    def link_shapes(self):
        return self.skeleton.get_link_shapes() + [
            'sphere', 'sphere', 'sphere', 'sphere', 'sphere',
    #        'sphere', 'sphere', 'sphere', 'sphere',
            'sphere', 'capsule', 'capsule'
        ]
    
    @property
    def link_sizes(self):
        return self.skeleton.get_link_sizes() + [
            [0.07, 0, 0], [0.045, 0, 0], [0.045, 0, 0], [0.04, 0, 0], [0.04, 0, 0],
    #        [0.035, 0, 0], [0.035, 0, 0], [0.03, 0, 0], [0.03, 0, 0],
            [0.04, 0, 0], [0.045, 0.05, 0.045], [0.045, 0.05, 0.045]
        ]

    def get_link_states(self):
        pose, vel = self.get_pose()
        self.skeleton.set_pose(pose)
        self.skeleton.set_vel(vel)
        
        states = self.get_link_states_by_ids(self.skeleton.get_link_ids())
        
        pos_array = []
        rot_array = []
        for s in states:
            pos_array.append(s[0:3])
            rot_array.append(s[3:7])

        # belly
        #pos_root = np.array(pos_array[0])
        #pos_chest = np.array(pos_array[1])
        #pos_belly = 0.5*(pos_root + pos_chest)
        #states.append([pos_belly[0], pos_belly[1], pos_belly[2], 0, 0, 0, 1])
        root_pos = np.array(self.skeleton.get_joint_global_pos(0))
        root_rot = Quaternion.fromXYZW(rot_array[0])
        root_pos += root_rot.rotate(np.array([0,0.205,0]))
        states.append(root_pos.tolist() + [0, 0, 0, 1])
        
        # joints
        states.append(self.skeleton.get_joint_global_pos(3) + [0, 0, 0, 1])
        states.append(self.skeleton.get_joint_global_pos(9) + [0, 0, 0, 1])
        states.append(self.skeleton.get_joint_global_pos(6) + [0, 0, 0, 1])
        states.append(self.skeleton.get_joint_global_pos(12) + [0, 0, 0, 1])
    #    states.append(self.skeleton.get_joint_global_pos(4) + [0, 0, 0, 1])
    #    states.append(self.skeleton.get_joint_global_pos(10) + [0, 0, 0, 1])
    #    states.append(self.skeleton.get_joint_global_pos(7) + [0, 0, 0, 1])
    #    states.append(self.skeleton.get_joint_global_pos(13) + [0, 0, 0, 1])

        # neck
        neck_pos = np.array(self.skeleton.get_joint_global_pos(2))
        head_rot = Quaternion.fromXYZW(rot_array[2])
        neck_pos += head_rot.rotate(np.array([0,0.02,0]))
        states.append(neck_pos.tolist() + [0, 0, 0, 1])

        # clavicle
        chest_rot = Quaternion.fromXYZW(rot_array[1])
        j_chest_pos = np.array(self.skeleton.get_joint_global_pos(1))
        r_clavicle_pos = chest_rot.rotate(np.array([-0.011, 0.24, 0.095])) + j_chest_pos
        r_clavicle_rot = chest_rot.mul(Quaternion.fromEulerZYX([-1.64, -0.21, 0.0338])).xyzw()
        states.append(r_clavicle_pos.tolist() + r_clavicle_rot.tolist())
        l_clavicle_pos = chest_rot.rotate(np.array([-0.011, 0.24, -0.095])) + j_chest_pos
        l_clavicle_rot = chest_rot.mul(Quaternion.fromEulerZYX([1.64, 0.21, 0.0338])).xyzw()
        states.append(l_clavicle_pos.tolist() + l_clavicle_rot.tolist())

        return states

    def get_link_states_by_ids(self, link_ids):
        low_states = self.sim_client.getLinkStates(self.object_id, link_ids)
        states = []
        for i in range(len(low_states)):
            state = list(low_states[i][0]) + list(low_states[i][1])
            states.append(state)
        return states

    def load_character(self):
        # load simulation model
        flags = self.sim_client.URDF_MAINTAIN_LINK_ORDER
        if self.self_collision:
            flags = flags | self.sim_client.URDF_USE_SELF_COLLISION

        from presets import preset
        urdf = preset.env.character_urdf
        self.object_id = self.sim_client.loadURDF(
            "./data/urdf/humanoid/%s.urdf" % urdf, [0,0.85,0],
            useFixedBase=False, flags=flags, globalScaling=1.14117647 if urdf == "jumper_scale" else 1.0)

        self.sim_client.changeDynamics(self.object_id, -1, lateralFriction=2.0)
        for j in range (self.sim_client.getNumJoints(self.object_id)):
            self.sim_client.changeDynamics(self.object_id, j, lateralFriction=2.0)

        self.sim_client.changeDynamics(self.object_id, -1, linearDamping=0, angularDamping=0)

        # initialize joints' controllers
        jointFrictionForce = 0
        for j, jtype in enumerate(self.skeleton.joint_types):
            if jtype is JointType.BASE:
                pass
            elif jtype is JointType.FIXED:
                pass
            elif jtype is JointType.REVOLUTE:
                self.sim_client.setJointMotorControl2(self.object_id, j, self.sim_client.POSITION_CONTROL, targetPosition=0, positionGain=0, targetVelocity=0,force=jointFrictionForce)
            elif jtype is JointType.SPHERE:
                self.sim_client.setJointMotorControlMultiDof(self.object_id, j, self.sim_client.POSITION_CONTROL,targetPosition=[0,0,0,1], targetVelocity=[0,0,0], positionGain=0,velocityGain=1,force=[jointFrictionForce,jointFrictionForce,jointFrictionForce])

    def init_spd(self):
        """ Initialize spd settings

        """
        # spd control set up
        spd_joint_ids = []
        spd_joint_dofs = []
        spd_joint_kps = []
        spd_joint_kds = []
        spd_joint_force_limits = []

        index = 7 # start point to read self.skeleton.kp / kd / torque_lim
        for i in range(1, len(self.skeleton.joint_dof)):
            nDof = self.skeleton.joint_dof[i]
            if nDof == 0:
                continue
            spd_joint_ids.append(i)
            spd_joint_dofs.append(nDof)
            spd_joint_kps.append(self.skeleton.kp[index])
            spd_joint_kds.append(self.skeleton.kd[index])
            if nDof == 4:
                spd_joint_force_limits.append([
                        self.skeleton.torque_lim[index + 0],
                        self.skeleton.torque_lim[index + 1],
                        self.skeleton.torque_lim[index + 2]
                ])
            elif nDof == 1:
                spd_joint_force_limits.append([self.skeleton.torque_lim[index + 0]])

            index += nDof

        self._spd_params = {
                "ids": spd_joint_ids,
                "dofs": spd_joint_dofs,
                "kps": spd_joint_kps,
                "kds": spd_joint_kds,
                "force_limits": spd_joint_force_limits
        }

    def get_pose(self):
        """ Get current pose and velocity expressed in general coordinate
            Outputs:
                pose
                vel
        """
        pose = []
        vel = []

        # root position/orientation and vel/angvel
        pos, orn = self.sim_client.getBasePositionAndOrientation(self.object_id)
        linvel, angvel = self.sim_client.getBaseVelocity(self.object_id)
        pose += pos
        if orn[3] < 0:
            orn = [-orn[0], -orn[1], -orn[2], -orn[3]]
        pose.append(orn[3])  # w
        pose += orn[:3] # x, y, z
        vel += linvel
        vel += angvel

        for i in range(self.skeleton.num_joints):
            j_info = self.sim_client.getJointStateMultiDof(self.object_id, i)
            orn = j_info[0]
            if len(orn) == 4:
                pose.append(orn[3])  # w
                pose += orn[:3] # x, y, z
            else:
                pose += orn
            vel += j_info[1]

        pose = np.array(pose)
        vel = self.skeleton.padVel(vel)
        return pose, vel

    def set_pose(self, pose, vel, initBase=True):
        """ Set character state in physics engine
            Inputs:
                pose   np.array of float, self.skeleton.pos_dim, position of base and
                             orintation of joints, represented in local frame
                vel    np.array of float, self.skeleton.vel_dim, velocity of base and
                             angular velocity of joints, represented in local frame

                initBase bool, if set base position/orintation/velocity/angular velocity
                                 as well
        """
        s = self.skeleton
        if initBase:
            pos = pose[:3]
            orn_wxyz = pose[3:7]
            orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
            v   = vel[:3]
            omg = vel[3:6]
            self.sim_client.resetBasePositionAndOrientation(self.object_id, pos, orn)
            self.sim_client.resetBaseVelocity(self.object_id, v, omg)

        for i in range(s.num_joints):
            jtype = s.joint_types[i]
            p_off = s.pos_start[i]
            if jtype is JointType.BASE:
                pass
            elif jtype is JointType.FIXED:
                pass
            elif jtype is JointType.REVOLUTE:
                orn = [pose[p_off]]
                omg = [vel[p_off]]
                self.sim_client.resetJointStateMultiDof(self.object_id, i, orn, omg)
            elif jtype is JointType.SPHERE:
                orn_wxyz = pose[p_off : p_off+4]
                orn = [orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]]
                omg = vel[p_off : p_off+3]
                self.sim_client.resetJointStateMultiDof(self.object_id, i, orn, omg)

    def set_spd_target(self, pose):
        """ Apply target pose on SPD controller
            Inputs:
                target
        """
        if pose is None:
            assert(False and "set target pose as None")

        pose = np.array(pose)
        pybullet_target = self.skeleton.pose_wxyz_to_xyzw(pose)

        self._spd_target_positions = []
        index = 7
        for dof in self._spd_params["dofs"]:
            if dof == 4:
                self._spd_target_positions.append([
                        pybullet_target[index + 0],
                        pybullet_target[index + 1],
                        pybullet_target[index + 2],
                        pybullet_target[index + 3],
                ])
            else :
                self._spd_target_positions.append([pybullet_target[index + 0]])

            index += dof

    def pre_step(self):
        self.apply_spd_torque()

    def apply_spd_torque(self):
        """ Apply spd target pose to step simulation
        """
        self.sim_client.setJointMotorControlMultiDofArray(
            self.object_id,
            self._spd_params["ids"],
            self.sim_client.STABLE_PD_CONTROL,
            targetPositions= self._spd_target_positions,
            positionGains=(np.array(self._spd_params["kps"])*self.kpkd_multiplier).tolist(),
            velocityGains=(np.array(self._spd_params["kds"])*self.kpkd_multiplier).tolist(),
            forces=self._spd_params["force_limits"]
        )

    def __check_scene_obj(self, obj):
        if not isinstance(obj, SceneObject):
            print('obj must be instance of SceneObject')
            assert False
    
    def clear_contact_settings(self):
        self.allowed_contacts.clear()
        self.not_allowed_contacts.clear()

    def set_allowed_fall_contact_with(self, body_ids, other_obj, other_obj_link_id=-2):
        self.__check_scene_obj(other_obj)
        self.allowed_contacts.append((other_obj, other_obj_link_id, body_ids))

    def set_not_allowed_fall_contact_with(self, body_ids, other_obj, other_obj_link_id=-2):
        self.__check_scene_obj(other_obj)
        self.not_allowed_contacts.append((other_obj, other_obj_link_id, body_ids))

    def check_not_allowed_contacts(self, report_bodies=False):
        body_list = []
        other_obj_list = []

        for info in self.allowed_contacts:
            other_obj = info[0].object_id
            other_obj_link_id = info[1]
            body_id_list = info[2]
            pts = self.sim_client.getContactPoints(bodyA=self.object_id, bodyB=other_obj, linkIndexB=other_obj_link_id)
            for p in pts:
                if p[3] not in body_id_list:
                    body_list.append(p[3])
                    other_obj_list.append(info[0])
        for info in self.not_allowed_contacts:
            other_obj = info[0].object_id
            other_obj_link_id = info[1]
            body_id_list = info[2]
            pts = self.sim_client.getContactPoints(bodyA=self.object_id, bodyB=other_obj, linkIndexB=other_obj_link_id)
            for p in pts:
                if p[3] in body_id_list:
                    body_list.append(p[3])
                    other_obj_list.append(info[0])
                    
        if report_bodies:
            return list(dict.fromkeys(body_list)), list(dict.fromkeys(other_obj_list))

        return len(body_list) > 0
    
    def check_self_contact(self):
        pts = self.sim_client.getContactPoints(bodyA=self.object_id, bodyB=self.object_id)
        return len(pts)
