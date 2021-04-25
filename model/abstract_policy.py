from abc import ABC, abstractmethod

class AbstractPolicy(ABC):
    @abstractmethod
    def act_deterministic(self, x):
        raise NotImplementedError

    @abstractmethod
    def act_stochastic(self, x, withLogP=False):
        raise NotImplementedError
    
    def logp(self, x, ac):
        raise NotImplementedError
