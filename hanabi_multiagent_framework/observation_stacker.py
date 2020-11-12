import abc
from typing import Tuple, Union, List
from numpy import ndarray

class ObservationStacker(abc.ABC):
    """
    """
    
    @abc.abstractmethod
    def add_observation(self, 
                        observation) -> None:
        """
        """
        
    @abc.abstractmethod
    def get_current_obs(self):
        """
        """
    
    @abc.abstractmethod    
    def get_observation_stack_t(self):
        """
        """
    
    @abc.abstractmethod
    def get_observation_stack_tm1(self):
        """
        """
        
    @abc.abstractmethod
    def reset(self, 
              indices) -> None:
        """
        """
        
    @abc.abstractmethod
    def reset_history(self, 
                      indices) -> None:
        """
        """
        
    @abc.abstractproperty
    def history_size(self) -> int:
        """
        """
        
    @abc.abstractproperty
    def observation_size(self) -> int:
        """
        """
        
    @abc.abstractproperty
    def size(self) -> ndarray:
        """
        """