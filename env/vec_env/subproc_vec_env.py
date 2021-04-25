import numpy as np
from multiprocessing import Process, Pipe
from .vec_env import VecEnv, CloudpickleWrapper

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    info["end_ob"] = ob
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'call':
                fn_name = data[0]
                args = data[1]
                kwargs = data[2]
                remote.send(getattr(env, fn_name)(*args, **kwargs))
            elif cmd == 'property':
                property_name = data
                remote.send(getattr(env, property_name))
            else:
                raise NotImplementedError
    finally:
        env.close()

class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, **kwargs):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.exp_id = kwargs["exp_id"]
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                     for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        def call_any(func_name, *args, **kwargs):
            self.remotes[0].send(("call", [func_name, args, kwargs]))
            return self.remotes[0].recv()
        
        def call_all(func_name, *args, **kwargs):
            for remote in self.remotes:
                remote.send(("call", [func_name, args, kwargs]))
            results = []
            for remote in self.remotes:
                results.append(remote.recv())
            return results
        
        def property_any(property_name):
            self.remotes[0].send(("property", property_name))
            return self.remotes[0].recv()
        
        def property_all(property_name):
            for remote in self.remotes:
                remote.send(("property", property_name))
            results = []
            for remote in self.remotes:
                results.append(remote.recv())
            return results

        from munch import munchify
        self.any = munchify({
            "call": call_any,
            "property": property_any
        })
        self.all = munchify({
            "call": call_all,
            "property": property_all
        })

        VecEnv.__init__(self, len(env_fns))
        
        self.observation_space = self.any.property("observation_space")
        self.action_space = self.any.property("action_space")
        self.state_normalization_exclusion_list = self.any.call("get_state_normalization_exclusion_list")

        self.any.call("start_recorder", 'results/%s/records.npy' % self.exp_id)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

def _flatten_obs(obs):
    assert isinstance(obs, list) or isinstance(obs, tuple)
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        import collections
        assert isinstance(obs, collections.OrderedDict)
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

