from abc import ABC, abstractmethod
import numpy as np
import gym
import gym.spaces

from utils.humanoid_kin import HumanoidSkeleton
from ur import UnityRenderer
from sim import engine_builder

class BaseEnv(ABC, gym.Env):
    def __init__(self, seed=0, **kwargs):
        from presets import preset

        if "new_process" in kwargs and kwargs["new_process"] == True:
            preset.override(kwargs["forked_preset"])
        
        self.init_parameters()

        env_settings = preset.env

        self.latent_dim = preset.vae.latent_dim

        self._rand = np.random.RandomState(seed)

        self.sim_frequency = env_settings.simulation_frequency
        self.control_frequency = env_settings.control_frequency

        self._sim_step = 1.0 / self.sim_frequency
        self._nsubsteps = int(self.sim_frequency / self.control_frequency)

        # initialize kinematic parts
        char_file = "data/characters/%s.txt" % env_settings.character
        ctrl_file = "data/controllers/%s.txt" % env_settings.character_ctrl
        self._skeleton = HumanoidSkeleton(char_file, ctrl_file)
            
        # initialize simulation parts
        self._engine = engine_builder(env_settings.physics_engine, self._sim_step)

        from sim import Plane
        from sim import Character
        self.plane = Plane()
        self.character = Character(self._skeleton, env_settings.enable_self_collision)
        
        self._engine.add_object(self.plane)
        self._engine.add_object(self.character)

        self._enable_draw = env_settings.enable_rendering
        if self._enable_draw:
            self.__init_render()

        self.a_min, self.a_max = self.build_action_bound()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, [self.get_state_size()], np.float32)
        self.action_space = gym.spaces.Box(self.a_min.astype(np.float32), self.a_max.astype(np.float32), dtype=np.float32)

        self._mode = 0   # 0 for train and 1 for test
    
    def init_parameters(self):
        pass

    def update_control_frequency(self, f):
        self.control_frequency = f
        self._nsubsteps = int(self.sim_frequency / f)

    def __init_render(self):
        self._ur_renderer = UnityRenderer(self._sim_step)
        self._ur_renderer.register_object("character", self.character)

    def __get_link_states(self):
        low_states = self._engine.get_link_states(self._skeleton.get_link_ids())
        states = []
        for i in range(len(low_states)):
            state = list(low_states[i][0]) + list(low_states[i][1])
            states.append(state)
        return states

    @abstractmethod
    def get_state_size(self):
        raise NotImplementedError

    @abstractmethod
    def get_action_size(self):
        raise NotImplementedError

    @abstractmethod
    def build_action_bound(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_state_normalization_exclusion_list(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """ Step environment

            Inputs:
                action

            Outputs:
                ob
                r
                done
                info
        """
        self.set_action(action)

        # if not terminated during simulation
        self.is_fail = False
        for i in range(self._nsubsteps):
            self.update()
            if self.check_terminate():
                self.is_fail = True
                break

        self.current_obs = self.record_state()
        r = self.calc_reward()
        done = self.is_fail or self.is_episode_end()
        info = self.record_info()

        return self.current_obs, r, done, info

    @abstractmethod
    def set_action(self, action):
        raise NotImplementedError

    def update(self):
        if self._enable_draw:
            self._ur_renderer.tick()

        self._engine.step_sim()
        self.post_update()

    @abstractmethod
    def record_info(self):
        raise NotImplementedError

    def post_update(self):
        pass

    @abstractmethod
    def record_state(self):
        raise NotImplementedError

    @abstractmethod
    def calc_reward(self):
        raise NotImplementedError

    @abstractmethod
    def is_episode_end(self):
        raise NotImplementedError

    @abstractmethod
    def check_terminate(self):
        raise NotImplementedError

    def check_not_allowed_contacts(self):
        return self.character.check_not_allowed_contacts()

    def set_mode(self, mode):
        # 0 for train, 1 for test
        assert(mode >= 0 and mode < 2)
        self._mode = mode

    def close(self):
        self._engine.close()
