import numpy as np
import torch
from utils.gpu import USE_GPU_MODEL

class Runner:
    """ Given environment, policy, sample given number of samples

    """
    def __init__(self, env, s_norm, policy, sample_size, exp_rate=1.0):
        """
            Inputs:
                env       gym.env_vec, vectorized environment, need to have following funcs:
                                        env.num_envs
                                        env.observation_space
                                        env.set_sample_prob(prob)
                                        obs = env.reset()
                                        obs, rwds, news, infos = env.step(acs)

                s_norm    torch.nn, normalizer for input states, need to have following funcs:
                                        obst_normed = s_norm(obst)
                                        s_norm.record(obt)
                                        s_norm.update()

                policy     torch.nn, policy model, need to have following funcs: act_stochastic

                sample_size   int, number of samples per run()

                exp_rate  float, probability to take stochastic action
        """
        self.env = env
        self.s_norm = s_norm
        self.policy = policy
        self.nenv = nenv = env.num_envs
        self.obs = np.zeros((nenv, env.observation_space.shape[0]), dtype=np.float32)
        self.obs[:] = env.reset()
        if USE_GPU_MODEL:
            self.toTorch = lambda x: torch.cuda.FloatTensor(x)
        else:
            self.toTorch = lambda x: torch.FloatTensor(x)
        self.sample_size = sample_size
        self.news = [True for _ in range(nenv)]
        self.exp_rate = exp_rate
        self.attached_modules = []

        from scheduler import get_schedulers
        for scheduler_lambda in get_schedulers():
            scheduler = scheduler_lambda()
            scheduler.attach_runner(self, env)
            self.attached_modules.append(scheduler)
        
        from presets import preset
        self.enable_vae = preset.vae.enable
        self.latent_dim = preset.vae.latent_dim

        if self.enable_vae:
            from utils.vae import VAE
            self.vae = VAE(self.latent_dim, use_gpu=USE_GPU_MODEL)

            vae_path = 'results/vae/models/%s.pth' % preset.vae.model
            self.vae.load_state_dict(torch.load(vae_path))

            with open(vae_path + '.norm.npy', 'rb') as f:
                self.vae_mean = np.load(f)
                self.vae_std = np.load(f)

        self.iter = 0
        self.writer = None

        self.test_avg_rwd = 0
        self.test_avg_step = 0

        self.total_samples = 0

    def set_exp_rate(self, exp_rate):
        self.exp_rate = exp_rate
    
    def set_iter(self, iter):
        self.iter = iter
    
    def set_writer(self, writer):
        self.writer = writer

    def __prev_run(self):
        self.end_rwds = []
    
    def get_infos(self, name):
        return [self.infos[i][name] for i in range(self.nenv)]
    
    def get_test_infos(self, name):
        return [self.test_infos[i][name] for i in range(self.nenv)]
    
    def get_infos_logs(self, name):
        return [self.infos[i]["logs"][name] for i in range(self.nenv)]
    
    def get_test_infos_logs(self, name):
        return [self.test_infos[i]["logs"][name] for i in range(self.nenv)]

    def run(self):
        """ run policy and get data for PPO training

            ==========================================================
            steps     0       1       2       ...     t       t+1
            -----------------------------------------------------------
            new       True    False   True    ...     False   True
            ob        s0      s1      s2      ...     s_t     s_{t+1}
            pose      p0      p1      p2      ...     p_t     p_{t+1}
            exp       True    True    False   ...     True    True
            ac        a0      a1      a2      ...     a_t     a_{t+1}
            --------------------------------------------------------- >> where yeild happens
            rwd       r0      r1      r2      ...     r_t     r_{t+1}
            fail      False   False   False   ...     True    False
            vpred     v(s0)   v(s1)   v(s2)   ...     v(s_t)  v(s_{t+1})
            alogp     lp(a0)  lp(a1)  lp(a2)  ...     lp(a_t) lp(a_{t+1})
                                                |                       |
                                                v                       v
            end_obs   -       ep0_end -       ...     ep1_end -
            end_vpreds-       ep0_val -       ...     ep1_val -

            for vectorized env, all data are stored in mb_${var}s

            besides new, ob, exp, ac, rwd, fail, there is information of networks:
                vpred   predicted value of current state s_t
                alogp   log_prob of choosing current action a_t
        """

        # clear record of fail phase
        self.__prev_run()

        # estimated number of steps env_vec need to take
        self.nsteps = int(self.sample_size / self.nenv) + 1

        self.mb_news = []
        self.mb_obs  = []
        self.mb_exps = []
        self.mb_acs  = []
        self.mb_rwds = []
        self.mb_alogps = []
        self.mb_ends   = []  # if the episode ends here
        self.mb_fails  = []  # if the episode ends here cause of failure
        self.mb_wraps  = []  # if the episode ends here cause of succeed to the end
        self.mb_ob_ends= []
        self.mb_logs = {}

        # update normalizer, then freeze until finish training
        self.s_norm.update()
        if USE_GPU_MODEL:
            self.s_norm.cpu()
            self.policy.cpu()

        for _ in range(self.nsteps):
            obst = torch.FloatTensor(self.obs)
            self.s_norm.record(obst)
            obst_norm = self.s_norm(obst)

            with torch.no_grad():
                exp = True # deprecated: np.random.rand() < self.exp_rate
                acs, alogps = self.policy.act_stochastic(obst_norm, withLogP=True)
                if self.enable_vae:
                    decoded = self.vae.decode(self.toTorch(acs[:,0:self.latent_dim]*self.vae_std + self.vae_mean)).cpu().numpy()

            self.mb_news.append(self.news.copy())
            self.mb_obs.append(self.obs.copy())
            self.mb_exps.append([exp]*self.nenv)
            self.mb_acs.append(acs)
            self.mb_alogps.append(alogps)

            if self.enable_vae:
                self.obs[:], rwds, self.news, self.infos = self.env.step(np.concatenate((decoded, acs[:,self.latent_dim:]), axis=1))
            else:
                self.obs[:], rwds, self.news, self.infos = self.env.step(acs)

            self.mb_rwds.append(rwds)

            # classify those stop by timer to be success, and those stoped by contacting
            # ground or torque exceeding limit as fail
            fails = self.get_infos("terminate")
            wraps = self.get_infos("wrap_end")

            logs = {}
            if "logs" in self.infos[0]:
                for key in self.infos[0]["logs"]:
                    if key not in self.mb_logs:
                        self.mb_logs[key] = []
                    self.mb_logs[key].append(self.get_infos_logs(key))
                    
            self.mb_fails.append(fails)
            self.mb_wraps.append(wraps)

            ends = np.zeros(self.nenv)
            for i, done in enumerate(self.news):
                if done:
                    ob_end = self.infos[i]["end_ob"]
                    self.mb_ob_ends.append(ob_end)
                    ends[i] = 1

            self.mb_ends.append(ends)

            # record end reward
            for i, done in enumerate(self.news):
                if done:
                    self.end_rwds.append(rwds[i])
            
            for module in self.attached_modules:
                module.on_sample_step()

        if USE_GPU_MODEL:
            self.s_norm.cuda()
            self.policy.cuda()

        self.mb_end_rwds = np.array(self.end_rwds)

        self.mb_news = np.asarray(self.mb_news,   dtype=np.bool)
        self.mb_obs  = np.asarray(self.mb_obs,    dtype=np.float32)
        self.mb_exps = np.asarray(self.mb_exps,   dtype=np.bool)
        self.mb_acs  = np.asarray(self.mb_acs,    dtype=np.float32)
        self.mb_rwds = np.asarray(self.mb_rwds,   dtype=np.float32)
        self.mb_alogps=np.asarray(self.mb_alogps, dtype=np.float32)
        self.mb_fails= np.asarray(self.mb_fails,  dtype=np.bool)
        self.mb_wraps= np.asarray(self.mb_wraps,  dtype=np.bool)
        self.mb_ends = np.asarray(self.mb_ends,  dtype=np.bool)
        self.mb_ob_ends= np.asarray(self.mb_ob_ends, dtype=np.float32)

        for key in self.mb_logs:
            self.mb_logs[key] = np.asarray(self.mb_logs[key], dtype=np.float32)

        keys = ["news", "obs", "exps", "acs", "rwds", "fails", "a_logps"]
        contents = map(self.sf01, (self.mb_news, self.mb_obs, self.mb_exps, self.mb_acs, self.mb_rwds, self.mb_fails, self.mb_alogps))

        self.data = {}
        for key, cont in zip(keys, contents):
            self.data[key] = cont

        self.data["samples"] = self.data["news"].size
        self.data["explores"] = self.data["exps"].sum()
        self.data["end_rwds"] = self.mb_end_rwds

        # logging interested variables
        N = np.clip(self.data["news"].sum(), a_min=1, a_max=None) # prevent divding 0
        self.avg_rwd = self.data["rwds"].sum()/N
        
        avg_logs = {}
        for key in self.mb_logs:
            avg_logs[key] = self.mb_logs[key].sum()/N
            
        avg_step = self.data["samples"]/N
        rwd_per_step = self.avg_rwd / avg_step
        # examine end_rwd, fail and phases
        rwd_end = self.data["end_rwds"]
        avg_rwd_end = sum(rwd_end) / len(rwd_end) if len(rwd_end) > 0 else -1
        fail_rate = sum(self.data["fails"])/N
        self.total_samples += self.data["samples"]
        self.writer.add_scalar("sample/avg_rwd",       self.avg_rwd,         self.iter)
        for key in self.mb_logs:
            self.writer.add_scalar("sample/avg_"+key,  avg_logs[key],        self.iter)
        self.writer.add_scalar("sample/avg_step",      avg_step,             self.iter)
        self.writer.add_scalar("sample/rwd_per_step",  rwd_per_step,         self.iter)
        self.writer.add_scalar("sample/end_rwd",       avg_rwd_end,          self.iter)
        self.writer.add_scalar("anneal/total_samples", self.total_samples,   self.iter)
        self.writer.add_scalar("debug/fail_rate",      fail_rate,            self.iter)
        
        print("avg_rwd       = %f" % self.avg_rwd)
        for key in self.mb_logs:
            print("{:14s}= {:f}".format("avg_"+key, avg_logs[key]))
        print("avg_step      = %f" % avg_step)
        print("rwd_per_step  = %f" % rwd_per_step)
        print("test_rwd      = %f" % self.test_avg_rwd)
        print("test_step     = %f" % self.test_avg_step)
        print("fail_rate     = %f" % fail_rate)
        print("end_rwd       = %f" % avg_rwd_end)
        print("total_samples = %d" % self.total_samples)

        for module in self.attached_modules:
            module.on_sample_collected()

    def get_data(self):
        return self.data

    def test(self):
        """ Test current policy with unlimited timer

            Outputs:
                avg_step
                avg_rwd
        """
        alive = np.array([True for _ in range(self.nenv)])
        any_alive = True
        acc_rwd = np.zeros(self.nenv)
        acc_step = np.zeros(self.nenv)
        acc_logs = {}

        # prepare environment, set mode to TEST
        self.env.all.call("set_mode", 1)
        self.obs = self.env.reset()
        self.news = [True for _ in range(self.nenv)]
        if USE_GPU_MODEL:
            self.s_norm.cpu()
            self.policy.cpu()
        while any_alive:
            obst = torch.FloatTensor(self.obs)
            self.s_norm.record(obst)
            obst_norm = self.s_norm(obst)

            with torch.no_grad():
                # with probability exp_rate to act stochastically
                acs = self.policy.act_deterministic(obst_norm)
                if self.enable_vae:
                    decoded = self.vae.decode(self.toTorch(acs[:,0:self.latent_dim]*self.vae_std + self.vae_mean)).cpu().numpy()
            if self.enable_vae:
                self.obs[:], rwds, self.news, self.test_infos = self.env.step(np.concatenate((decoded, acs[:,self.latent_dim:]), axis=1))
            else:
                self.obs[:], rwds, self.news, self.test_infos = self.env.step(acs)

            # record the rwd and step for alive agents
            acc_rwd += rwds * alive
            if "logs" in self.test_infos[0]:
                for key in self.test_infos[0]["logs"]:
                    if key not in acc_logs:
                        acc_logs[key] = np.zeros(self.nenv)
                    acc_logs[key] += np.array(self.get_test_infos_logs(key)) * alive
            acc_step += alive

            # decide which are alive, since timer is set to max, so using self.news as fails
            alive = np.logical_and(alive, np.logical_not(self.news))

            # decide if any are alive
            any_alive = np.any(alive)
            
            for module in self.attached_modules:
                module.on_test_step()

        if USE_GPU_MODEL:
            self.s_norm.cuda()
            self.policy.cuda()

        self.test_avg_step = np.mean(acc_step)
        self.test_avg_rwd  = np.mean(acc_rwd)

        test_avg_logs = {}
        for key in acc_logs:
            test_avg_logs[key] = np.mean(acc_logs[key])

        # turn mode back to train IMPORTANT
        self.env.all.call("set_mode", 0)
        self.obs = self.env.reset()

        self.writer.add_scalar("sample/test_step", self.test_avg_step, self.iter)
        self.writer.add_scalar("sample/test_rwd",  self.test_avg_rwd,  self.iter)
        for key in test_avg_logs:
            self.writer.add_scalar("sample/test_"+key,  test_avg_logs[key],  self.iter)

        for module in self.attached_modules:
            module.on_test_completed()

        return self.test_avg_step, self.test_avg_rwd

    # save extra info to checkpoint here
    def save_info(self, data):
        return data
    
    def sf01(self, arr):
        """
        swap and then flatten axes 0 and 1
        """
        s = arr.shape
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
