"""
This file defines an abstract class for hanabi agents.
All agents which shall be managed by the HanabiParallelSession, have to comply with this interface.

Glossary:
    observations -- List of HanabiObservation for each game state (default). If the agent requests
                    the encoded observation, a tuple containing one-hot encoded observations and
                    legal moves is appended. See requires_vectorized_observation() method.
                    The encoded observations have a shape (n_states x encoded_observation_len),
                    and encoded legal moves have a shape (n_states x max_moves).

    actions -- List of HanabiMove, or a list of move UIDs, which are zero-based encodings of all
               possible moves for the game of Hanabi. In case of list of move UIDs the shape is
               assumed to be (n_states x 1)

    rewards -- List with one reward per state. Shape (n_states x 1)
    step_type -- List of dm_env.StepType. Describes the current status of each state.
                 Status:
                    initial (=StepType.FIRST): state right after the reset, no action were taken yet.
                    terminal (=StepType.LAST): the game has ended, no more actions are allowed.
                    intermediate (=StepType.MID): all other (normal) states

                 Shape (n_states x 1)
"""
import abc
from typing import Tuple, Union, List
from numpy import ndarray
from hanabi_learning_environment.pyhanabi_pybind import HanabiMove, HanabiObservation, RewardShaper, RewardShapingParams, ShapingType
from .observation_stacker import ObservationStacker

class HanabiAgent(abc.ABC):
    """
    Abstract class which defines and describes functions required by the ParallelSession.
    """

    @abc.abstractmethod
    def explore(self,
                observations: Union[List[HanabiObservation],
                                    Tuple[List[HanabiObservation], Tuple[ndarray, ndarray]]]
                ) -> Union[ndarray, List[HanabiMove]]:
        """Produce an action using the exploration policy. This function is called when it's agent's
        turn to make a move, i.e. once per round.

        Args:
            observations -- List of HanabiObservation for each game state (default).

        Returns actions
        """

    @abc.abstractmethod
    def exploit(self, observations: Union[Tuple[ndarray], ndarray]) -> ndarray:
        """Same as explore, but use the exploitation policy instead.

        Args:
            observations -- List of HanabiObservation for each game state (default).

        Returns actions
        """

    @abc.abstractmethod
    def add_experience_first(self,
                             observations: ndarray,
                             step_types: ndarray) -> None:
        """Add initial observations to the agent's buffers. All states are passed for completeness,
        but only the one with step_type == dm_env.StepType.FIRST should be processed.

        Args:
            observations -- List of HanabiObservation for each game state (default).
            step_types   -- state status.
        """

    @abc.abstractmethod
    def add_experience(self, 
                       observations_tm1, 
                       actions_tm1, 
                       rewards_t, 
                       observations_t, 
                       term_t) -> None:
        """Add observations to the agent's buffers. All states are passed for completeness,
        but the ones with step_type == dm_env.StepType.FIRST should be ignored (they should have
        already been processed in add_experience_first(...) function).

        Args:
            observations -- new states.
            actions      -- actions taken reach this states.
            rewards      -- rewards for taking this actions.
            step_types   -- state status.
        """
        
    @abc.abstractmethod
    def shape_rewards(self, 
                      observations, 
                      moves) -> Union[ndarray, int]:
        """
        """
        
    @abc.abstractmethod
    def create_stacker(self, 
                       obs_len, 
                       n_states) -> ObservationStacker: 
        """
        """

    @abc.abstractmethod
    def update(self):
        """Trigger an update of the agent, i.e make it perform training step(s)."""

    @abc.abstractmethod
    def requires_vectorized_observation(self):
        """Whether the observations supplied to this agent shall be vectorized."""
