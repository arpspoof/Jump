from .base import BaseEnv
from .joint_limit import AngleLimit, SphericalLimit, Limit
import numpy as np

class VaeEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_joint_limit()

        from presets import preset
        self.max_offset_norm = preset.vae.max_offset_norm_begin
        self.max_offset_norm_begin = preset.vae.max_offset_norm_begin
        self.max_offset_norm_end = preset.vae.max_offset_norm_end
    
    def init_joint_limit(self):
        self.joint_names = ["chest", "neck", "rhip", "rknee", "rankle", "rshoulder", "relbow",
            "lhip", "lknee", "lankle", "lshoulder", "lelbow"]
        self.jointdofs = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]
        self.jointlimits = [
            SphericalLimit( # chest
                Limit(-0.43, 0.43, -0.1, 0.1),
                Limit(-0.32, 0.32, -0.05, 0.05),
                Limit(-0.25, 0.25, -0.05, 0.05)
            ),
            SphericalLimit( # neck
                Limit(-0.65, 0.65, -0.35, 0.35),
                Limit(-0.6, 0.6, -0.3, 0.3),
                Limit(-0.75, 0.75, -0.35, 0.35)
            ),
            SphericalLimit( # right hip
                Limit(-0.52, -0.05, -0.3, -0.1),
                Limit(-0.35, 1.57, -0.15, 0.75),
                Limit(-0.2, 0.2, -0.05, 0.05)
            ),
            AngleLimit( # right knee
                Limit(-2, 0, -1, 0)
            ),
            SphericalLimit( # right ankle
                Limit(-0.2, 0.4, -0.1, 0.2),
                Limit(-0.69, 0.69, -0.4, 0.4),
                Limit(-0.2, 0.44, -0.05, 0.1)
            ),
            SphericalLimit( # right shoulder
                Limit(-1.44, 0.2, -0.6, 0.02),
                Limit(-0.1, 2.96, 0, 1.8),
                Limit(-0.58, 0.3, -0.15, 0.1)
            ),
            AngleLimit( # right elbow
                Limit(0, 2.44, 0, 1)
            ),
            SphericalLimit( # left hip
                Limit(0.05, 0.52, 0.1, 0.3),
                Limit(-0.35, 1.57, -0.15, 0.75),
                Limit(-0.2, 0.2, -0.05, 0.05)
            ),
            AngleLimit( # left knee
                Limit(-2, 0, -1, 0)
            ),
            SphericalLimit( # left ankle
                Limit(-0.4, 0.2, -0.2, 0.1),
                Limit(-0.69, 0.69, -0.4, 0.4),
                Limit(-0.44, 0.2, -0.1, 0.05)
            ),
            SphericalLimit( # left shoulder
                Limit(-0.2, 1.44, -0.02, 0.6),
                Limit(-0.1, 2.96, 0, 1.8),
                Limit(-0.3, 0.58, -0.1, 0.15)
            ),
            AngleLimit( # left elbow
                Limit(0, 2.44, 0, 1)
            )
        ]

    def get_action_size(self):
        return self.latent_dim + 28

    def build_action_bound(self):
        a_min = np.array([-1] * self.get_action_size())
        a_max = np.array([1] * self.get_action_size())
        return a_min, a_max

    def get_target_pose(self, action, offset_scale=1.0):
        quatpose =[0,0,0,1,0,0,0] + list(self._skeleton.inv_feature(action[0:180].tolist()))
        exppose = self._skeleton.targ_pose_to_exp(np.array(quatpose))
        offset = action[180:] * offset_scale
        offset_norm = np.linalg.norm(offset, 1)
        if offset_norm > self.max_offset_norm:
            offset *= self.max_offset_norm / offset_norm
        return list(self._skeleton.exp_to_targ_pose(exppose + offset, True)), offset
    
    def penalty_method(self, remove_y_list=[]):
        self.target_pose_penalty = self.target_pose.copy()
        self.current_pose, self.current_vel = self.character.get_pose()
        idx = 7
        for i in range(len(self.jointdofs)):
            dof = self.jointdofs[i]
            if dof == 1:
                self.target_pose_penalty[idx] = self.jointlimits[i].adjust_target(
                    self.current_pose[idx],
                    self.target_pose[idx]
                )
            else:
                remove_y = i in remove_y_list 
                self.target_pose_penalty[idx:idx+4] = self.jointlimits[i].adjust_target(
                    self.current_pose[idx:idx+4],
                    self.target_pose[idx:idx+4],
                    remove_y=remove_y
                )
            idx += dof
        self.character.set_spd_target(self.target_pose_penalty)
