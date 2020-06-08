"""
This file defines an abstract class for hanabi agents.
All agents which shall be managed by the HanabiParallelSession, have to comply with this interface.
"""
import abc
from numpy import ndarray

class HanabiAgent(abc.ABC):
    """
    Abstract class which defines and describes functions required by the ParallelSession.
    """

    @abc.abstractmethod
    def explore(self, observations: ndarray, legal_moves: ndarray) -> ndarray:
        """Produce an action using the exploration policy. This function is called when it's agent's
        turn to make a move, i.e. once per round.

        Args:
            observations (n_states x observation_len) -- one-hot encoded batch observation.
            legal_moves (n_states x max_moves) -- one-hot encoded indices of the legal moves.
        Returns numpy array (n_states, 1) with generated actions.
        """

    @abc.abstractmethod
    def exploit(self, observations: ndarray, legal_moves: ndarray) -> ndarray:
        """Same as explore, but use the exploitation policy instead.

        Args:
            observations (n_states x observation_len) -- one-hot encoded batch observation.
            legal_moves (n_states x max_moves) -- one-hot encoded indices of the legal moves.
        Returns numpy array (n_states, 1) with generated actions.
        """

    @abc.abstractmethod
    def add_experience_first(self,
                             observations: ndarray,
                             legal_moves: ndarray,
                             step_types: ndarray) -> None:
        """Add initial observations to the agent's buffers. All states are passed for completeness,
        but only the one with step_type == dm_env.StepType.LAST should be processed.

        Args:
            observations (n_states x observation_len) -- one-hot encoded observations.
            legal_moves  (n_states x max_moves)       -- one-hot encoded actions that can be taken.
            step_types   (n_states, 1) -- state types (see dm_env.StepType).
        """

    @abc.abstractmethod
    def add_experience(self,
                       observations: ndarray,
                       legal_moves: ndarray,
                       actions: ndarray,
                       rewards: ndarray,
                       step_types: ndarray) -> None:
        """Add observations to the agent's buffers. All states are passed for completeness,
        but the ones with step_type == dm_env.StepType.LAST should be ignored (they should have
        already been processed in add_experience_first(...) function).

        Args:
            observations (n_states x observation_len) -- one-hot encoded observations.
            legal_moves  (n_states x max_moves)       -- one-hot encoded actions that can be taken.
            actions      (n_states, 1)  -- action taken reach this state.
            rewards      (n_states, 1)  -- reward for taking this action.
            step_types   (n_states, 1)  -- state types (see dm_env.StepType).
        """

    @abc.abstractmethod
    def update(self):
        """Trigger an update of the agent, i.e make it perform training step(s).
        """
