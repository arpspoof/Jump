from algorithm.runner import Runner
from algorithm.GAE import GAE
from .data_tools import PPO_Dataset

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from tensorboardX import SummaryWriter

def calc_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad == True:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1./2)
    return total_norm

def PPO_vec():
    from presets import preset
    algorithm_settings = preset.algorithm
    experiment_settings = preset.experiment
    # parameters
    # presets/default/experiment
    exp_id = experiment_settings.name
    extra_config = experiment_settings.extra_config
    # presets/default/algorithm
    gamma = algorithm_settings.gamma
    max_iterations = algorithm_settings.max_iterations
    # presets/default/algorithm/ppo
    ppo_settings = algorithm_settings.ppo
    sample_size = ppo_settings.runner_sample_size
    epoch_size = ppo_settings.epoch_size
    batch_size = ppo_settings.batch_size
    checkpoint_every_iteration = ppo_settings.checkpoint_every_iteration
    test_every_iteration = ppo_settings.test_every_iteration
    clip_threshold = ppo_settings.clip_threshold
    actor_lr = ppo_settings.actor_lr
    actor_wdecay = ppo_settings.actor_wdecay
    actor_momentum = ppo_settings.actor_momentum
    critic_lr = ppo_settings.critic_lr
    critic_wdecay = ppo_settings.critic_wdecay
    critic_momentum = ppo_settings.critic_momentum
    max_grad_norm = ppo_settings.max_grad_norm
    # presets/default/vae
    enable_vae = preset.vae.enable
    vae_dim = preset.vae.latent_dim

    import time
    stamp = time.strftime("%Y-%b-%d-%H%M%S", time.localtime())
    if preset.experiment.use_timestamp:
        exp_id = "%s-%s" % (exp_id, stamp) 

    save_dir = "results"
    save_path = "%s/%s" % (save_dir, exp_id)
    writer = SummaryWriter(save_path)

    import subprocess
    with open('%s/git.commit.txt' % save_path, 'w') as f:
        subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=f)
    with open('%s/git.diff.patch' % save_path, 'w') as f:
        subprocess.Popen(['git', 'diff'], stdout=f)

    from env import spawn_vec_env
    vec_env = spawn_vec_env(exp_id=exp_id)

    obs_space = vec_env.observation_space
    ac_space =vec_env.action_space

    from model import actor_critic
    s_norm = actor_critic.Normalizer(obs_space.shape[0], vec_env.state_normalization_exclusion_list)
    actor = actor_critic.Actor(obs_space.shape[0], ac_space.shape[0], ac_space.low, ac_space.high)
    
    max_value = algorithm_settings.max_value
    v_max = 1.0 / (1.0 - gamma) if max_value is None else max_value
    critic= actor_critic.Critic(obs_space.shape[0], 0, v_max)

    checkpoint_file = experiment_settings.checkpoint
    if (checkpoint_file is not None):
        try:
            checkpoint = torch.load(checkpoint_file)
            actor.load_state_dict(checkpoint["actor"])
            critic.load_state_dict(checkpoint["critic"])
            s_norm.load_state_dict(checkpoint["s_norm"])
            print("load from %s" % checkpoint_file)

        except:
            print("fail to load from %s" % checkpoint_file)
            assert(False)

    from utils.gpu import USE_GPU_MODEL

    actor_optim = optim.SGD(actor.parameters(), actor_lr, momentum=actor_momentum, weight_decay=actor_wdecay)
    critic_optim = optim.SGD(critic.parameters(), critic_lr, momentum=critic_momentum, weight_decay=critic_wdecay)

    # set up environment and data generator
    runner = Runner(vec_env, s_norm, actor, sample_size, exp_rate=1.0)
    gae = GAE(critic)

    if USE_GPU_MODEL:
        T = lambda x: torch.cuda.FloatTensor(x)
    else:
        T = lambda x: torch.FloatTensor(x)
    runner.set_writer(writer)

    total_sample = 0

    config_sample_rwds = []
    config_test_rwds = []
    config_best_test_rwd = 0

    result = {}
    
    for it in range(max_iterations + 1):
        # sample data with gae estimated adv and vtarg
        print("\n===== iter %d ====="% it)
        runner.set_iter(it)
        data = gae.collect_sample_and_compute_GAE(runner)
        dataset = PPO_Dataset(data)

        atarg = dataset.advantage
        atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-5) # trick: standardized advantage function

        adv_clip_rate = np.mean(np.abs(atarg) > 4)
        adv_max = np.max(atarg)
        adv_min = np.min(atarg)
        val_min = 0
        val_max = v_max
        vtarg = dataset.vtarget
        vtarg_clip_rate = np.mean(np.logical_or(vtarg < val_min, vtarg > val_max))
        vtd_max = np.max(vtarg)
        vtd_min = np.min(vtarg)

        atarg = np.clip(atarg, -4, 4)
        vtarg = np.clip(vtarg, val_min, val_max)

        dataset.advantage = atarg
        dataset.vtarget = vtarg

        if extra_config == 'j':
            config_sample_rwds.append(runner.avg_rwd)
            if np.sum(config_sample_rwds) > 0:
                config_sample_rwds = []
            elif len(config_sample_rwds) >= 10:
                config_sample_rwds = []
                runner.env.all.call('decrease_wall_height')
                new_height = runner.env.any.call('current_wall_height')
                print('decreasing height to', new_height)
                if new_height < 0.199:
                    result['fail'] = True
                    break
        if extra_config == 'lj':
            config_sample_rwds.append(runner.avg_rwd)
            if np.sum(config_sample_rwds) > 0:
                config_sample_rwds = []
            elif len(config_sample_rwds) >= 20:
                config_sample_rwds = []
                result['fail'] = True
                break

        if (it % test_every_iteration == 0):
            runner.test()

        print("adv_clip_rate = %f, (%f, %f)" % (adv_clip_rate, adv_min, adv_max))
        print("vtd_clip_rate = %f, (%f, %f)" % (vtarg_clip_rate, vtd_min, vtd_max))

        writer.add_scalar("debug/adv_clip",     adv_clip_rate,      it)
        writer.add_scalar("debug/vtarg_clip",   vtarg_clip_rate,    it)

        # start training
        pol_loss_avg    = 0
        pol_surr_avg    = 0
        pol_abound_avg  = 0
        vf_loss_avg     = 0
        clip_rate_avg   = 0

        actor_grad_avg  = 0
        critic_grad_avg = 0

        for epoch in range(epoch_size):
            #print("iter %d, epoch %d" % (it, epoch))

            for bit, batch in enumerate(dataset.batch_sample(batch_size)):
                # prepare batch data
                ob, ac, atarg, tdlamret, log_p_old = batch
                ob = T(ob)
                ac = T(ac)
                atarg = T(atarg)
                tdlamret = T(tdlamret).view(-1, 1)
                log_p_old = T(log_p_old)

                # clean optimizer cache
                actor_optim.zero_grad()
                critic_optim.zero_grad()

                # calculate new log_pact
                ob_normed = s_norm(ob)
                vpred = critic(ob_normed)
                log_pact = actor.logp(ob_normed, ac)
                if log_pact.dim() == 2:
                    log_pact = log_pact.sum(dim=1)

                # PPO object, clip advantage object
                ratio = torch.exp(log_pact - log_p_old)
                surr1 = ratio * atarg
                surr2 = torch.clamp(ratio, 1.0 - clip_threshold, 1.0 + clip_threshold) * atarg
                pol_surr = -torch.mean(torch.min(surr1, surr2))

                if (surr2.mean() > 12):
                    assert(False)
                    from IPython import embed; embed()

                # action bound penalty, normalized
                if enable_vae:
                    violation_min = torch.clamp(ac - actor.a_min, max=0)[:,vae_dim:] / actor.a_std[vae_dim:]
                    violation_max = torch.clamp(ac - actor.a_max, min=0)[:,vae_dim:] / actor.a_std[vae_dim:]
                else:
                    violation_min = torch.clamp(ac - actor.a_min, max=0) / actor.a_std
                    violation_max = torch.clamp(ac - actor.a_max, min=0) / actor.a_std
                violation = torch.sum(torch.pow(violation_min, 2) + torch.pow(violation_max, 2), dim=1)
                pol_abound = 0.5 * torch.mean(violation)

                pol_loss = pol_surr + pol_abound    # trick: add penalty for violation of bound

                pol_surr_avg += pol_surr.item()
                pol_abound_avg += pol_abound.item()
                pol_loss_avg += pol_loss.item()


                # critic vpred loss
                vf_criteria = nn.MSELoss()
                vf_loss = vf_criteria(vpred, tdlamret) / (critic.v_std**2) # trick: normalize v loss

                vf_loss_avg += vf_loss.item()

                if (not np.isfinite(pol_loss.item())):
                    print("pol_loss infinite")
                    assert(False)
                    from IPython import embed; embed()

                if (not np.isfinite(vf_loss.item())):
                    print("vf_loss infinite")
                    assert(False)
                    from IPython import embed; embed()

                pol_loss.backward(retain_graph=True)
                vf_loss.backward(retain_graph=True)

                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

                actor_grad_avg += calc_grad_norm(actor)
                critic_grad_avg+= calc_grad_norm(critic)

                # for debug use
                clip_rate = (torch.abs(ratio - 1.0) > clip_threshold).float()
                clip_rate = torch.mean(clip_rate)
                clip_rate_avg += clip_rate.item()

                actor_optim.step()
                critic_optim.step()


        batch_num = (sample_size // batch_size)
        pol_loss_avg    /= batch_num
        pol_surr_avg    /= batch_num
        pol_abound_avg  /= batch_num
        vf_loss_avg     /= batch_num
        clip_rate_avg   /= batch_num

        writer.add_scalar("debug/clip_rate", clip_rate_avg,     it)
        writer.add_scalar("train/pol_loss",  pol_loss_avg,      it)
        writer.add_scalar("train/pol_surr",  pol_surr,          it)
        writer.add_scalar("train/ab_loss",   pol_abound,        it)
        writer.add_scalar("train/vf_loss",   vf_loss_avg,       it)

        print("pol_loss      = %f" % pol_loss_avg)
        print("pol_surr      = %f" % pol_surr)
        print("ab_loss       = %f" % pol_abound)
        print("vf_loss       = %f" % vf_loss_avg)
        print("clip_rate     = %f" % clip_rate_avg)

        actor_grad_avg /= batch_num
        critic_grad_avg/= batch_num
        writer.add_scalar("train/actor_grad", actor_grad_avg,   it)
        writer.add_scalar("train/critic_grad", critic_grad_avg, it)
        print("actor_grad    = %f" % actor_grad_avg)
        print("critic_grad   = %f" % critic_grad_avg)

        # save checkpoint
        if (it % checkpoint_every_iteration == 0):
            print("save check point ...")
            actor.cpu()
            critic.cpu()
            s_norm.cpu()
            data = {"actor": actor.state_dict(),
                            "critic": critic.state_dict(),
                            "s_norm": s_norm.state_dict()}
            if USE_GPU_MODEL:
                actor.cuda()
                critic.cuda()
                s_norm.cuda()

            data = runner.save_info(data)
            data = gae.save_info(data)

            torch.save(data, "%s/%s/checkpoint_%d.tar" % (save_dir, exp_id, it))

            if extra_config == 'p2':
                config_test_rwds.append(runner.test_avg_rwd)

                if config_test_rwds[-1] > 11.3:
                    break
                elif runner.test_avg_rwd - config_best_test_rwd > 0.1:
                    config_test_rwds = []
                elif len(config_test_rwds) >= 5:
                    break

                config_best_test_rwd = np.maximum(config_best_test_rwd, runner.test_avg_rwd)
            
            elif extra_config == 'p2l':
                config_test_rwds.append(runner.test_avg_rwd)

                if config_test_rwds[-1] > 16.0:
                    break
                elif runner.test_avg_rwd - config_best_test_rwd > 0.1:
                    config_test_rwds = []
                elif len(config_test_rwds) >= 5:
                    break

                config_best_test_rwd = np.maximum(config_best_test_rwd, runner.test_avg_rwd)

            elif extra_config == 'j':
                reach_best = runner.env.any.call('reach_max_wall_height')
                if reach_best:
                    break

            elif extra_config == 'lj':
                reach_best = runner.env.any.call('reach_max_slices')
                if reach_best:
                    break
    
    if extra_config == 'j':
        result['final_height'] = float(runner.env.any.call('current_wall_height'))
    if extra_config == 'lj':
        result['final_slices'] = int(runner.env.any.call('current_slices'))
    if extra_config == 'p2' or extra_config == 'p2l':
        result['final_rwd'] = float(runner.test_avg_rwd)

    with open('%s/result.yaml'%save_path, 'w') as f:
        yaml.safe_dump(result, f)
