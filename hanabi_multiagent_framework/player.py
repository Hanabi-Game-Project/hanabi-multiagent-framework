"""
This file defines an abstract class for hanabi agents.
All agents which shall be managed by the HanabiGameManager, have to comply with this interface.
"""
import numpy as np

class HanabiPlayer:
    """
    HanabiPlayer is a wrapper class for the Hanabi game. Its main objective is to accumulate
    the training samples while multiple parallel games are being executed.
    """
    def __init__(self, pipe, agent_config, n_players, n_moves):
        """Constructor
        Args:
            agent                      -- an agent.
            n_players            (int) -- number of players in the game.
            observe_whole_round (bool) -- whether only one last observation, or all observations
                                          collected in the last round should be fed into the agent.
        """
        self._pipe = pipe
        self.n_moves = n_moves
        self.n_players = n_players
        self.observe_round = agent_config["consumes_round_observations"]
        self.current_observations = []
        self.legal_moves_template = np.full((self.n_moves,), -np.inf)

    def act(self, legal_moves):
        """Produce an action. This function is called when it's agent's turn to make a move,
        i.e. once per round.

        The agent is required to have the "act" method which consumes observation(s) and produces
        a list with probabilities of all actions. These probabilities are then filtered by selecting
        the most probable of only legal actions. The agent should still be able to explore by
        manipulating the action probabilities.

        Args:
            legal_moves (list) -- list with indices of the legal moves.
        Returns (int) a legal action.
        """
        legal_moves_np = np.full((self.n_moves,), -np.inf)
        legal_moves_np[legal_moves] = 0.0
        if self.observe_round:
            self._pipe.send((self.current_observations[-self.n_players:], legal_moves_np))
        else:
            #  print(getsizeof(self.current_observations[-1][0]))
            self._pipe.send((self.current_observations[-1], legal_moves_np))
        self.current_observations.clear()
        action = self._pipe.recv()
        #  illegal_moves = [move_id for move_id in range(len(actions)) if move_id not in legal_moves]
        #  actions = actions[legal_moves]
        #  actions[illegal_moves] = -1
        return action

    def consume_observation(self, observation, reward):
        """Receives an observation and accociated reward and stores it in the training_data member
        variable. After the game(s) training_data can/should be collected for further usage.

        When an agent is requested to act, it receives the last observation(s) stored in the
        training_data.

        Args:
            observation -- a vectorized observation which uses one-hot encoding.
            reward      -- a reward.
        """
        self.current_observations.append((observation, reward))

    def train_agent(self):
        """Train the agent.
        """
        raise NotImplementedError("When implementing HanabiAgent interface, "
                                  "make sure to implement <train_agent> function.")
