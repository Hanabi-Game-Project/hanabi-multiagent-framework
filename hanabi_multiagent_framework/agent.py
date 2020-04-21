"""
This file defines an abstract class for hanabi agents.
All agents which shall be managed by the HanabiParallelSession, have to comply with this interface.
"""
from numpy import ndarray
from abc import ABC

class HanabiAgent(ABC):
    """
    Abstract class which defines and describes functions required by the ParallelSession.
    """

    def explore(self, observations: ndarray, legal_moves: ndarray) -> ndarray:
        """Produce an action using the exploration policy. This function is called when it's agent's
        turn to make a move, i.e. once per round.

        Args:
            observations (batch_size x observation_len) -- one-hot encoded batch observation.
            legal_moves (batch_size x max_moves) -- one-hot encoded indices of the legal moves.
        Returns numpy array (batch_size, 1) with generated legal (!) actions.
        """
        raise NotImplementedError("When implementing HanabiAgent interface, "
                                  "make sure to implement <explore> function.")

    def exploit(self, observations: ndarray, legal_moves: ndarray) -> ndarray:
        """Same as explore, but use the exploitation policy instead.

        Args:
            observations (batch_size x observation_len) -- one-hot encoded batch observation.
            legal_moves (batch_size x max_moves) -- one-hot encoded indices of the legal moves.
        Returns numpy array (batch_size, 1) with generated legal (!) actions.
        """
        raise NotImplementedError("When implementing HanabiAgent interface, "
                                  "make sure to implement <exploit> function.")

    def train(self,
              obs_tm1: ndarray,
              act_tm1: ndarray,
              obs_t: ndarray,
              lm_t: ndarray,
              r_t: ndarray,
              term_t: ndarray) -> None:
        """Train the agent using the experience.

        Args:
            obs_tm1 (batch_size x observation_len) -- one-hot encoded source observations.

            act_tm1 (batch_size, 1)  -- action taken from source to destination state.
            obs_t (batch_size, observation_len)  -- destination observation.
            lm_t (batch_size, max_moves) -- actions that can be taken from destination
                                            observation.
            r_t (batch_size, 1) -- reward for getting from source to destination state.
            term_t (batch_size, 1) -- flag showing whether the destination state is terminal.
        """
        raise NotImplementedError("When implementing HanabiAgent interface, "
                                  "make sure to implement <train> function.")
