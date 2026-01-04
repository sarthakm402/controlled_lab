from abc import ABC, abstractmethod
class BaseEnv(ABC):
    @abstractmethod
    def __init__(self):
        """initialise env specific state:
        -must set up initial state
        -must have constraints like done flag,step counter etc."""
    @abstractmethod
    def reset(self):
        """re-initialise to start step along with resetting step counter """
        pass

    @abstractmethod
    def step(self,action):
        """guarantees state updated if action valid, error returned if invalid, increments step counter, updates done flag"""
        pass

    @abstractmethod
    def get_state(self):
        """get the state u r currently in """
        pass

