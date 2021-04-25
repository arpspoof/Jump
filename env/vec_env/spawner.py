def spawn_vec_env(**kwargs):
    from presets import preset
    exp_settings = preset.experiment

    from env import get_env
    task_env_bundle = get_env(exp_settings.env, bundle=True)

    from scheduler import register_scheduler
    schedulers = task_env_bundle.schedulers
    for i in range(len(schedulers)):
        register_scheduler((task_env_bundle.env, i), schedulers[i])

    def make_env(seed, new_process=False):
        return lambda: task_env_bundle.env(seed=seed, **{"new_process": new_process, "forked_preset": preset})

    num_workers = exp_settings.num_workers
    if num_workers == "default":
        import multiprocessing
        num_cpu = multiprocessing.cpu_count()
        num_workers = num_cpu
    
    import time
    from .subproc_vec_env import SubprocVecEnv
    return SubprocVecEnv([make_env(i + time.process_time_ns() % 1000, new_process=True) for i in range(num_workers)], **kwargs)
