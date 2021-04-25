import numpy as np
import torch

# presets/default/algorithm
# presets/default/algorithm/gae
class GAE:
    """ Given environment, actor and critic, sample given number of samles

    """
    def __init__(self, critic):
        """
            Inputs:
                critic    torch.nn, critic model, need to have following funcs:
                                        vpreds = critic(obst_norm)

                gamma     float, discount factor of reinforcement learning

                lam       float, lambda for GAE algorithm
        """
        from presets import preset
        algorithm_settings = preset.algorithm
        self.critic = critic
        self.lam = algorithm_settings.gae.lam
        self.use_end_value_prediction = algorithm_settings.gae.use_end_value_prediction
        self.gamma = algorithm_settings.gamma
        self.v_min = 0
        max_value = algorithm_settings.max_value
        self.v_max = 1.0 / (1.0 - self.gamma) if max_value is None else max_value

    def collect_sample_and_compute_GAE(self, runner):
        runner.run()

        evp = self.use_end_value_prediction

        # evaluate vpred and vends separately
        with torch.no_grad():
            obst = torch.Tensor(runner.mb_obs)
            obst_norm = runner.s_norm(obst)
            mb_vpreds = self.critic(obst_norm)
            dim0, dim1, dim2 = mb_vpreds.shape
            mb_vpreds = mb_vpreds.reshape(dim0, dim1)
            mb_vpreds = mb_vpreds.cpu().data.numpy()

            mb_vends = np.zeros(runner.mb_ends.shape)
            if len(runner.mb_ob_ends) > 0:
                obst = torch.Tensor(runner.mb_ob_ends)
                obst_norm = runner.s_norm(obst)
                vends = self.critic(obst_norm)
                mb_vends[runner.mb_ends] = vends.cpu().view(-1)

        with torch.no_grad():
            obst = runner.toTorch(runner.obs)
            obst_norm = runner.s_norm(obst)
            last_vpreds = self.critic(obst_norm).cpu().view(-1).numpy()

            fail_end = np.logical_and(runner.news, runner.mb_fails[-1])
            succ_end = np.logical_and(runner.news, np.logical_not(runner.mb_fails[-1]))
            wrap_end = np.logical_and(runner.news, runner.mb_wraps[-1])
            last_vpreds[fail_end] = self.v_min
            last_vpreds[succ_end] = mb_vends[-1][succ_end] if evp else 0
            last_vpreds[wrap_end] = self.v_max if evp else 0

        mb_vtargs= np.zeros_like(runner.mb_rwds)
        mb_advs  = np.zeros_like(runner.mb_rwds)

        mb_nextvalues = mb_advs
        mb_nextvalues[:-1] = mb_vpreds[1:]
        fail_end = np.logical_and(runner.mb_news[1:], runner.mb_fails[:-1])
        succ_end = np.logical_and(runner.mb_news[1:], np.logical_not(runner.mb_fails[:-1]))
        wrap_end = np.logical_and(runner.mb_news[1:], runner.mb_wraps[:-1])
        mb_nextvalues[:-1][fail_end] = self.v_min
        mb_nextvalues[:-1][succ_end] = mb_vends[:-1][succ_end] if evp else 0
        mb_nextvalues[:-1][wrap_end] = self.v_max if evp else 0

        mb_nextvalues[-1] = last_vpreds

        mb_delta = mb_advs
        mb_delta = runner.mb_rwds + self.gamma * mb_nextvalues - mb_vpreds

        lastgaelam = 0
        for t in reversed(range(runner.nsteps)):
            if t == runner.nsteps - 1:
                 nextnonterminal = 1.0 - runner.news
            else:
                 nextnonterminal = 1.0 - runner.mb_news[t+1]
            mb_advs[t] = lastgaelam = mb_delta[t] + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_vtargs = mb_advs + mb_vpreds

        keys = ["advs", "vtargs"]
        contents = map(runner.sf01, (mb_advs, mb_vtargs))

        data = runner.get_data()
        for key, cont in zip(keys, contents):
            data[key] = cont

        return data

    def save_info(self, data):
        return data
