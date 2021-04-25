class AbstractScheduler:
    def attach_runner(self, runner, vec_env):
        self.runner = runner
        self.vec_env = vec_env
        self.initialize()
    
    def initialize(self):
        pass

    def on_sample_step(self):
        pass

    def on_sample_collected(self):
        pass

    def on_test_step(self):
        pass

    def on_test_completed(self):
        pass
