from abc import ABC, abstractmethod

class URObject:
    @property
    @abstractmethod
    def link_names(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def link_shapes(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def link_sizes(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_link_states(self):
        raise NotImplementedError
