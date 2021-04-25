from abc import ABC, abstractmethod

class SceneObject:
    def __init__(self):
        self.engine = None
        self.sim_client = None
        self.object_id = None
    
    @abstractmethod
    def initialize(self):
        # engine and sim_client are available before this call
        # please initialize object_id here
        raise NotImplementedError

    @abstractmethod
    def pre_step(self):
        raise NotImplementedError
    