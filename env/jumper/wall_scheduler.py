from scheduler import AbstractScheduler

class JumperWallScheduler(AbstractScheduler):
    def initialize(self):
        jumper_settings = self.vec_env.any.property('preset')

        self.reward_threshold = jumper_settings.reward_threshold
        self.wall_height = jumper_settings.initial_wall_height
        self.max_wall_height = jumper_settings.max_wall_height

        self.cum_rwd = 0
    
    def update_task_schedule(self):
        if self.wall_height < self.max_wall_height:
            self.vec_env.all.call("increase_wall_height")
            self.wall_height = self.vec_env.any.call("current_wall_height")

    def on_sample_collected(self):
        self.cum_rwd += self.runner.avg_rwd
        if self.cum_rwd > self.reward_threshold:
            self.cum_rwd = 0
            self.update_task_schedule()

        self.runner.writer.add_scalar("jumper/wall",  self.wall_height,  self.runner.iter)
        print("wall height   = %f" % self.wall_height)
        print("cum rwd       = %f" % self.cum_rwd)
