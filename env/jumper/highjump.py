from utils.motion_recorder import MotionRecorder
from utils.quaternion import Quaternion
from utils.sigmoid import sigmoid_norm
from utils.bell import bell
from ..vae_env import VaeEnv
from .wall import Wall, BarStock
from .block import Block
import numpy as np

class HighJumpEnv(VaeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.checkpoint = None
        self.jump_only = False

        if "checkpoint" in kwargs:
            self.checkpoint = kwargs["checkpoint"]
            if "jump_only" in kwargs:
                self.jump_only = True
                self.motion_recorder = MotionRecorder()
            else:
                self.motion_recorder = MotionRecorder(self.initial_state_path[:-4] + '.yaml')
        
        self.link_ids = self._skeleton.get_link_ids()

        if self.test_time:
            self.wall = BarStock(self.wall_rotation, self.wall_distance, self.initial_wall_height, vis_offset=self.vis_offset)
        else:
            self.wall = Wall(self.wall_rotation, self.wall_distance, self.initial_wall_height, vis_offset=self.vis_offset)
        self.block = Block(self.wall_rotation, self.wall_distance, self.initial_wall_height, 0.7, vis_offset=self.vis_offset, real=self.test_time)
        self._engine.add_object(self.wall)
        self._engine.add_object(self.block)
        if self._enable_draw:
            self._ur_renderer.register_object("wall", self.wall)
            self._ur_renderer.register_object("block", self.block)

        self.character.clear_contact_settings()
        self.character.set_allowed_fall_contact_with([], self.wall)
        self.character.set_allowed_fall_contact_with([], self.block)
        self.character.set_allowed_fall_contact_with([5], self.plane)

        self._skeleton.set_heading_vec([1, 0, 0])

        if self.initial_state_path:
            with open(self.initial_state_path, 'rb') as f:
                self.init_pose = np.load(f)
                self.init_vel = np.load(f)
            self.init_pose, self.init_vel = self._skeleton.toLocalFrame(self.init_pose, self.init_vel)

        self._skeleton.set_pose(self.init_pose)
        self._skeleton.set_vel(self.init_vel)

        low_y = self._skeleton.lowest_height(self.init_pose.tolist())
        self.init_pose[1] -= low_y

        self.num_step = 0

        self.update_offset_max()
        self.recording = False

        if self.jump_only:
            self.round = 0
    
    def init_parameters(self):
        from presets import preset
        jumper_settings = preset.env.highjump

        self.preset = jumper_settings

        self.initial_state_path = jumper_settings.initial_state
        self.q_bases = jumper_settings.q_bases
        self.wall_distance = jumper_settings.wall_distance
        self.wall_rotation = jumper_settings.wall_rotation
        self.initial_wall_height = jumper_settings.initial_wall_height
        self.max_wall_height = jumper_settings.max_wall_height
        self.episode_length = jumper_settings.episode_length
        self.offset_penalty_coeff = jumper_settings.offset_init
        self.angv_penalty_coeff = jumper_settings.angv_penalty
        self.offset_end = jumper_settings.offset_end
        self.vis_offset = jumper_settings.vis_offset
        self.test_time = jumper_settings.test_time

    def reset(self):
        self.status = 0
        self.num_step = 0
        self.near_success = False
        self.head_contact = False
        self.offset_penalty = 0
        self.angv_penalty = 0
        self.character.kpkd_multiplier = 1
        self.success_contact = -2
        self.com_closest_dist = 100
        self.root_quat = self.init_pose[3:7]
        self.character.set_pose(self.init_pose, self.init_vel)
        if self.recording:
            self.action_record = []
        if self.test_time:
            self.wall.reset_bar()
        if self.jump_only:
            self.round += 1

        return self.record_state()
    
    def step(self, action):
        self.num_step += 1
        self.action = action

        if self.recording:
            self.action_record.append(action)
        
        self.set_action()

        _,vel = self.character.get_pose()
        self.angv_penalty += np.linalg.norm(vel[3:6])

        for i in range(self._nsubsteps):
            self.penalty_method(remove_y_list=[])
            self.update()
            if self.checkpoint:
                pose, vel = self.character.get_pose()
                self.motion_recorder.append('character', pose)
                if self.test_time:
                    self.motion_recorder.append('bar', np.array(self.wall.get_link_states()[2]))
            if self.status == 0:
                self.status = self.check_status()
            if self.status != 0 and self._mode == 0:
                break

        r = self.calc_reward()
        done = self.check_terminate() or self.is_episode_end()
        info = self.record_info()

        if self.recording and r > 0:
            self.recorder.write(np.array(self.action_record))

        if done and self._mode == 1:
            self.character.kpkd_multiplier = 0.1
            self.num_step -= 1

        self.current_obs = self.record_state()

        com_pos = self._skeleton.get_com_pos()
        com_dist = np.abs(self.wall.point_to_plane_distance(com_pos))
        if com_dist < self.com_closest_dist:
            self.com_closest_dist = com_dist
            self.root_quat = self.current_pose[3:7]
        
        if self.checkpoint and done and (not self.jump_only or self.round >= 10):
            import yaml
            with open(self.checkpoint + '.result.yaml', 'w') as f:
                yaml.safe_dump({'quat': self.root_quat.tolist()}, f)
            print('processing motion recording, please wait ...')
            self.motion_recorder.save(self.checkpoint + '.yaml')
            print('recording saved')

        return self.current_obs, r, done, info

    def set_action(self):
        self.target_pose, offset = self.get_target_pose(self.action, 2.0)
        self.offset_penalty += np.linalg.norm(offset, ord=1)

    def record_info(self):
        info = { "terminate": self.check_terminate(), "wrap_end": False, "logs": {
            "offset": (self.offset_penalty / self.num_step) if self.status == 1 else 0, 
            "angv": (self.angv_penalty / self.num_step) if self.status == 1 else 0,
            "freq": self.control_frequency if self.status != 0 else 0,
            "o-norm": self.max_offset_norm if self.status != 0 else 0 }}
        return info

    def get_state_size(self):
        single_state = self._skeleton.build_state2()
        return single_state.size 
    
    def get_state_normalization_exclusion_list(self):
        return []

    def record_state(self):
        self.current_pose, self.current_vel = self.character.get_pose()
        self._skeleton.set_pose(self.current_pose)
        self._skeleton.set_vel(self.current_vel)
        return self._skeleton.build_state2()
    
    def generalized_falling_rwd(self):
        avg_penalty = self.offset_penalty / self.num_step
        rwd = 1.0 - np.clip(np.power(avg_penalty / self.offset_penalty_coeff, 2), 0, 1)

        avg_ang_penalty = self.angv_penalty / self.num_step
        rwd *= np.exp(-self.angv_penalty_coeff * avg_ang_penalty)

        if rwd > 0:
            q = Quaternion.fromWXYZ(self.root_quat)
            min_angle = np.pi
            for qb in self.q_bases:
                q_base = Quaternion.fromWXYZ(np.array(qb))
                q_diff = q.mul(q_base.conjugate())
                angle = np.abs(q_diff.angle())
                min_angle = np.min([min_angle, angle])
            rwd *= np.clip(min_angle/(np.pi/2), 0.01, 1)
            #print(min_angle)
            #print(self.root_quat)

            if self.head_contact:
                rwd *= 0.7

        return rwd

    def calc_reward(self):
        return self.generalized_falling_rwd() if self.status == 1 else 0

    def is_episode_end(self):
        return self.status == 1

    def check_terminate(self):
        return self.status < 0

    def check_status(self):
        # 0: undermined, 1: success, -1: bad contact, -2: timeout
        if self.num_step >= self.episode_length:
            return -2

        if self.test_time or self.near_success:
            bodies, other = self.character.check_not_allowed_contacts(report_bodies=True)
            if len(bodies) == 0:
                return 0
            for o in other:
                if o != self.block:
                    return -1 if not self.test_time else 0
            if 2 in bodies:
                self.head_contact = True
            if 0 in bodies or 1 in bodies or 2 in bodies or len(bodies) >= 4:
                self.success_contact = bodies
                return 1
            return 0
            
        if self.check_not_allowed_contacts():
            return -1
        link_states = self.character.get_link_states_by_ids(self.link_ids)
        for state in link_states:
            pos = state[0:3]
            if self.wall.point_to_plane_distance(pos) < 0.2:
                return 0
        
        self.near_success = True
        return 0
    
    def increase_wall_height(self):
        self.wall.height += 0.01
        self.block.height += 0.01
        self.update_offset_max()
    
    def decrease_wall_height(self):
        self.wall.height -= 0.1
        self.block.height -= 0.1
        self.update_offset_max()
    
    def current_wall_height(self):
        return self.wall.height
    
    def reach_max_wall_height(self):
        return self.wall.height + 0.001 > self.max_wall_height
    
    def update_offset_max(self):
        ratio = np.clip(self.wall.height - 0.6, 0, 1)
        init = self.preset.offset_init
        dist = init - self.offset_end
        self.offset_penalty_coeff = init - ratio * dist

        ratio = np.clip((self.wall.height - 0.5)/0.5, 0, 1)
        self.update_control_frequency(10 + 20*ratio)

        ratio = np.clip((self.wall.height - 0.5)/0.5, 0, 1)
        self.max_offset_norm = self.max_offset_norm_begin + ratio*(self.max_offset_norm_end - self.max_offset_norm_begin)

        ratio = np.clip((self.wall.height - 0.5)/0.5, 0, 1)
        if self.wall_distance > 0.8:
            self.wall.wallDistance = 0.8 + ratio*(self.wall_distance - 0.8)
            self.block.wallDistance = 0.8 + ratio*(self.wall_distance - 0.8)

    def start_recorder(self, path):
        from ..sample_recorder import SampleRecorder
        self.recorder = SampleRecorder(path)
        self.recording = True
