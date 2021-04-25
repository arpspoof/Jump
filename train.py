# command line arguments:
# primary_preset_name, [auxiliary_preset_names, ...]

if __name__=="__main__":
    import sys
    
    from presets import preset
    preset.load_default()

    primary_preset = sys.argv[1]
    preset.load_custom(primary_preset)

    if preset.experiment.name is None:
        preset.experiment.name = primary_preset

    for i in range(2, len(sys.argv)):
        preset_name = sys.argv[i]
        preset.load_custom(preset_name)
    
    preset.load_env_override()

    exp_settings = preset.experiment

    import torch
    import utils.gpu as gpu
    gpu.USE_GPU_MODEL = exp_settings.use_gpu and torch.cuda.is_available()
    if gpu.USE_GPU_MODEL:
        print("experiment settings: enable GPU model")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    from algorithm import algorithm_bundle_dict
    algorithm_bundle = algorithm_bundle_dict[exp_settings.algorithm]
    algorithm_bundle.algorithm()
