from utils.quaternion import Quaternion
from ..base import BaseEnv
from utils.humanoid_mocap import HumanoidMocap
from utils.motion_recorder import MotionRecorder
import numpy as np

class JumperRunEnv(BaseEnv):
    def __init__(self, **kwargs):
        """ Initialize FDM0 environment
        """
        super().__init__(**kwargs)

        from presets import preset
        settings = preset.env.jumper_run

        self.checkpoint = None
        if "checkpoint" in kwargs:
            self.checkpoint = kwargs["checkpoint"]
            self.recorder = MotionRecorder()

        self._mocap = HumanoidMocap(self._skeleton, settings.motion_file)
        self._min_t = settings.min_t
        self._max_t = settings.max_t
        self.speed_multiplier = settings.speed_multiplier
        self.cut_off = settings.cut_off
        self.name = settings.name

        # contact type
        from utils.contact_manager import load_contact_file
        allowed_contacts = load_contact_file('walk')
        self.character.set_allowed_fall_contact_with(allowed_contacts, self.plane)
        self._skeleton.set_heading_vec([1, 0, 0])

        count, pose, vel = self._mocap.slerp(self._min_t)
        self._skeleton.set_pose(pose)
        self._skeleton.set_vel(vel)
        self.low_y = self._skeleton.lowest_height(pose.tolist())

    def get_state_size(self):
        single_state = self._skeleton.build_state()
        return 20 + single_state.size

    def get_action_size(self):
        return self._skeleton.expdof

    def build_action_bound(self):
        a_min = np.array(self._skeleton.build_a_min())
        a_max = np.array(self._skeleton.build_a_max())

        return a_min, a_max

    def get_state_normalization_exclusion_list(self):
        return list(range(20))

    def reset(self):
        self.t = np.clip(np.random.uniform()*self._max_t, self._min_t, self._max_t)
        if self.checkpoint:
            self.t = self._min_t

        count, pose, vel = self._mocap.slerp(self.t)
        pose[1] -= self.low_y

        pose[0] = 0; pose[2] = 0
        vel[0] *= self.speed_multiplier; vel[2] *= self.speed_multiplier

        self.character.set_pose(pose, vel)
        self.current_obs = self.record_state()

        if self.checkpoint:
            self.recorder.append('character', pose)

        return self.current_obs

    def set_action(self, action):
        phase = self.current_obs[0]
        _, target_pose, _ = self._mocap.slerp(phase)
        off_base = self._skeleton.targ_pose_to_exp(target_pose)

        self.target_pose = self._skeleton.exp_to_targ_pose(action + off_base, True)
        self.character.set_spd_target(self.target_pose)

    def record_info(self):
        info = { "terminate": self.check_terminate(), "wrap_end": False }
        return info

    def post_update(self):
        self.t += self._sim_step

        if self.checkpoint:
            pose, vel = self.character.get_pose()
            self.recorder.append('character', pose)

        if self.checkpoint and self.t > self.cut_off:
            pose, vel = self.character.get_pose()
            pose, vel = self._skeleton.toLocalFrame(pose, vel)
            with open('data/states/%s.npy'%self.name, 'wb') as f:
                np.save(f, pose)
                np.save(f, vel)
            
            frames = self.recorder.get('character')
            last_frame = frames[-1]
            inv_ori_rot = self._skeleton.buildHeadingTrans(np.array(last_frame[3:7]))
            offset = np.array(last_frame[0:3])
            offset[1] = 0
            for frame in frames:
                pos = np.array(frame[0:3])
                rot = Quaternion.fromWXYZ(np.array(frame[3:7]))
                newpos = inv_ori_rot.rotate(pos - offset)
                newrot = inv_ori_rot.mul(rot)
                frame[0:3] = newpos.tolist()
                frame[3:7] = newrot.wxyz().tolist()

            self.recorder.save('data/records/%s.yaml'%self.name)

            print(self.t)
            print(pose)
            print(vel)
            print('finish data saving')
            input()

    def record_state(self):
        """
            state is made up with 2 items:
             - phase
             - sim_state
        """
        # build state using current reference root, and without global info
        pose, vel = self.character.get_pose()
        self._curr_sim_pose = pose
        self._curr_sim_vel = vel
        self._skeleton.set_pose(pose)
        self._skeleton.set_vel(vel)
        ori_pos = pose[:3]
        ori_rot = pose[3:7]
        sim_state = self._skeleton.build_state(ori_pos, ori_rot, True)

        state = np.concatenate(([self.t]*20, sim_state))
        return state

    def calc_reward(self):
        if self.is_fail:
            return 0
            
        count, pose, vel = self._mocap.slerp(self.t)
        self._curr_kin_pose = pose
        self._curr_kin_vel = vel
        local_sim_pose = self._curr_sim_pose.copy()
        local_sim_pose[0] = 0
        local_sim_pose[2] = 0
        local_sim_vel = self._curr_sim_vel
        local_kin_pose = self._curr_kin_pose.copy()
        local_kin_pose[0] = 0
        local_kin_pose[2] = 0
        local_kin_vel = self._curr_kin_vel
        local_kin_vel[0] *= self.speed_multiplier; local_kin_vel[2] *= self.speed_multiplier
        reward = self._skeleton.get_reward(local_sim_pose, local_sim_vel, local_kin_pose, local_kin_vel)

        return reward 

    def check_terminate(self):
        return self.check_not_allowed_contacts() 

    def is_episode_end(self):
        return self.t >= self._max_t

    def check_time_end(self):
        return self.t >= self._max_t

    def start_recorder(self, path):
        pass
