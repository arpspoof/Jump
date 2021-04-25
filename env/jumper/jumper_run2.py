from ..base import BaseEnv
from utils.bell import bell
from utils.humanoid_mocap import HumanoidMocap
from utils.motion_recorder import MotionRecorder
from utils.quaternion import Quaternion
import numpy as np

class JumperRun2Env(BaseEnv):
    def __init__(self, **kwargs):
        """ Initialize FDM0 environment
        """
        super().__init__(**kwargs)

        from presets import preset
        self.preset = preset.env.jumper_run2

        self.checkpoint = None
        if "checkpoint" in kwargs:
            self.checkpoint = kwargs["checkpoint"]
            self.motion_recorder = MotionRecorder('data/records/%s.yaml'%self.preset.p1_name)

        self._mocap = HumanoidMocap(self._skeleton, self.preset.motion_file)
        
        self._max_t = self.preset.max_t
        self.init_t = self.preset.min_t

        self.character.set_allowed_fall_contact_with([5, 11], self.plane)
        self._skeleton.set_heading_vec([1, 0, 0])

        count, pose, vel = self._mocap.slerp(0)
        self._skeleton.set_pose(pose)
        self._skeleton.set_vel(vel)
        self.low_y = self._skeleton.lowest_height(pose.tolist())

        with open('data/states/%s.npy'%self.preset.p1_name, 'rb') as f:
            self.init_pose = np.load(f)
            self.init_vel = np.load(f)

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
        self.t = self.init_t

        self.character.set_pose(self.init_pose, self.init_vel)
        self.current_obs = self.record_state()

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
            self.motion_recorder.append('character', pose)

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

        state = np.concatenate(([self.t - self.init_t]*20, sim_state))
        return state

    def calc_reward(self):
        if self.is_fail:
            return 0
            
        count, pose, vel = self._mocap.slerp(self.t)

        vel[0] *= self.preset.speed_multiplier; vel[2] *= self.preset.speed_multiplier

        self._curr_kin_pose = pose
        self._curr_kin_vel = vel
        
        local_sim_pose, local_sim_vel = self._skeleton.toLocalFrame(self._curr_sim_pose, self._curr_sim_vel)
        local_kin_pose, local_kin_vel = self._skeleton.toLocalFrame(self._curr_kin_pose, self._curr_kin_vel)
        reward = self._skeleton.get_reward(local_sim_pose, local_sim_vel, local_kin_pose, local_kin_vel)

        if self.is_episode_end():
            pose, vel = self.character.get_pose()
            pose, vel = self._skeleton.toLocalFrame(pose, vel)

            bonus = 10.0*self._skeleton.get_reward2(local_sim_pose, local_sim_vel, local_kin_pose, local_kin_vel)
            bonus *= np.exp(-1.0 * np.mean(np.abs(vel[6:] - local_kin_vel[6:])))
            bonus *= np.exp(-1.0 * np.mean(np.abs(vel[3:6] - self.preset.angular_v)))

            bonus *= bell(vel[2], self.preset.linear_v_z, 1.0)

            if self.checkpoint:
                print('v[2]', vel[2])
                print(pose)
                print(vel)
                print('writing to %s ...' % (self.checkpoint + '.npy'))
                with open(self.checkpoint + '.npy', 'wb') as f:
                    np.save(f, pose)
                    np.save(f, vel)
                print('processing recording, please wait ...')
                frames = self.motion_recorder.get('character')
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
                self.motion_recorder.save(self.checkpoint + '.yaml')
                print('writing done')
            
            return bonus + reward

        return reward 

    def check_terminate(self):
        return self.check_not_allowed_contacts() 

    def is_episode_end(self):
        return self.t >= self._max_t

    def check_time_end(self):
        return self.t >= self._max_t

    def start_recorder(self, path):
        pass

    def update_param(self, name, value):
        pass
    #    setattr(self.preset, name, value)
