"""
This file defines a HanabiGame which facilitates an interaction between agents and
the environment.
"""

import random

from hanabi_framework.agent import HanabiAgent
from hanabi_framework.environment import HanabiEnvironment


class HanabiGameManager:
    """
    A game of hanabi. It takes a hanabi environment and a list of agents and
    makes sure that the game works flawlessly.
    """
    def __init__(self, env_config, agents, reward_func=None):
        self.env_config = env_config
        self.env = HanabiEnvironment(env_config)
        if not 0 < len(agents) < 6:
            raise ValueError(f"Wrong number of players: Expected between 2 and 5, "
                             "but got {len(agents)}.")
        if not all([isinstance(a, HanabiAgent) for a in agents]):
            raise TypeError("Not all Agents inherit from HanabiAgent")
        self.agents = list(enumerate(agents))

        self.last_full_observation = None

        # A way to assign individual rewards. Default: pass reward to the agents as is.
        self.reward_func = reward_func or (lambda reward, _: reward)

    def reset_game(self):
        """Reset a game.

        Returns observation
        """
        self.last_full_observation = self.env.reset()

    def reverse_agents(self):
        """
        Reverse the agent sequence.
        """
        self.agents.reverse()

    def reorder_agents(self):
        """
        Shuffle the agent sequence.
        """
        random.shuffle(self.agents)

    def round(self):
        """
        Let the agents play one round of the game.
        """
        for player_id, agent in self.agents:
            legal_moves = self.last_full_observation[player_id].legal_moves()
            legal_moves = [self.env.game.get_move_uid(move) for move in legal_moves]
            action = agent.act(legal_moves)
            self.last_full_observation, reward, done = self.env.step(action)
            reward = [self.reward_func(reward, player_id) for _ in self.agents]
            # each agent consumes its observation of the turn.
            for consumer_id, consumer in self.agents:
                consumer.consume_observation(
                    self.last_full_observation[consumer_id],
                    reward[consumer_id])
            if done:
                return False
        return True

    def play_game(self):
        """Play a game of hanabi until the end.
        """
        n_steps = 0
        while self.round():
            # TODO: handle the case when the round is interrupted in the middle.
            n_steps += len(self.agents)
        return n_steps

    def play_fork(self, n_steps):
        """Play a game, but forking at each step
        """
        raise NotImplementedError()
