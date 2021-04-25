# command line arguments
# [preset names, ...] checkpoint file

import numpy as np

def test_model(env, model, record=False):
    from presets import preset
    enable_vae = preset.vae.enable

    env.set_mode(1)

    s_norm = model["s_norm"]
    actor = model["policy"]

    import torch
    T = lambda x: torch.FloatTensor(x)

    if enable_vae:
        from utils.vae import VAE
        vae = VAE(preset.vae.latent_dim, use_gpu=False)

        vae_path = 'results/vae/models/%s.pth' % preset.vae.model
        vae.load_state_dict(torch.load(vae_path))

        with open(vae_path + '.norm.npy', 'rb') as f:
            vae_mean = np.load(f)
            vae_std = np.load(f)

    while True:
        ob = env.reset()
        while True:
            with torch.no_grad():
                obt = T(ob)
                obt_normed = s_norm(obt)
                ac = actor.act_deterministic(obt_normed)
                if enable_vae:
                    decoded = vae.decode(T(ac[0:preset.vae.latent_dim]*vae_std + vae_mean)).cpu().numpy()
                    ac = np.concatenate((decoded, ac[preset.vae.latent_dim:]))
                
            ob, rwd, done, info = env.step(ac)

            if done:
                break

if __name__=="__main__":
    import sys
    
    from presets import preset
    preset.load_default()

    exp_settings = preset.experiment
    env_settings = preset.env
    env_settings.enable_rendering = True

    for i in range(1, len(sys.argv) - 1):
        preset.load_custom(sys.argv[i])
        
    preset.load_env_override()
    
    checkpoint = sys.argv[-1]

    from env import get_env
    test_env = get_env(exp_settings.env)(seed=0, evaluate=True)

    from algorithm import algorithm_bundle_dict
    algorithm_name = preset.experiment.algorithm
    model_class = algorithm_bundle_dict[algorithm_name].model_class

    from model import model_dict
    load_model = model_dict[model_class].loader

    model = load_model(checkpoint)
    test_model(test_env, model, exp_settings.record_motion)
