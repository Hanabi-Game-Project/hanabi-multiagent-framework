"""
This file defines an abstract class for hanabi agents.
All agents which shall be managed by the HanabiGameManager, have to comply with this interface.
"""

from hanabi_learning_environment.rl_env import Agent

class HanabiAgent(Agent):
    """
    Abstract class from which all agents participating in game of hanabi should inherit.
    A notable difference to "traditional" agents is that consuming an observation and producing
    an action have to be split in this agent implementation, because the agent observes
    the game more frequently, than it makes a move.
    """
    def act(self, legal_actions):
        """Produce an action. This function is called when it's agent's turn to make a move,
        i.e. once per round.
        The agent is supposed to take care of observations internally.
        See also self.consume_observation(observation).

        Args:
            legal_moves (list) -- list with indices of the legal moves.
        Returns (int) a legal action.
        """
        raise NotImplementedError("When implementing HanabiAgent interface, "
                                  "make sure to implement <act> function.")

    def consume_observation(self, observation, reward):
        """Receive an observation and accociated reward and handle it
        according to agent's internals.

        Args:
            observation -- a vectorized observation which uses one-hot encoding.
            reward      -- a reward.
        """
        raise NotImplementedError("When implementing HanabiAgent interface, "
                                  "make sure to implement <consume_observation> function.")
