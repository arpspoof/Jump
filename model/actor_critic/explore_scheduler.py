import numpy as np
from scheduler import AbstractScheduler

class ExploreScheduler(AbstractScheduler):
    def __init__(self, actor):
        self.actor = actor
    
    def initialize(self):
        from presets import preset
        actor_settings = preset.model.actor

        self.actor_std_begin = actor_settings.std_begin
        self.actor_std_end = actor_settings.std_end

        self.max_anneal_samples = actor_settings.anneal_samples
        self.anneal_samples = actor_settings.anneal_samples_past
    
    def update_explore_schedule(self):
        ratio = np.clip(self.anneal_samples / self.max_anneal_samples, 0, 1)
        std = ratio * (self.actor_std_end - self.actor_std_begin) + self.actor_std_begin
        self.actor.set_actor_noise_level(std)
        print("new std       = %f" % std)

    def on_sample_collected(self):
        data = self.runner.data
        self.anneal_samples += data["explores"]
        self.update_explore_schedule()
