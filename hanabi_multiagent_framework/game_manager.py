"""
This file defines a HanabiGame which facilitates an interaction between agents and
the environment.
"""

import random

from hanabi_multiagent_framework.player import HanabiPlayer
from hanabi_multiagent_framework.environment import HanabiEnvironment


class HanabiGameManager:
    """
    A game of hanabi. It takes a hanabi environment and a list of agents and
    makes sure that the game works flawlessly.
    """
    def __init__(self, env_config, agent_infos, pipes, reward_func=None):
        self.env_config = env_config
        self.env = HanabiEnvironment(env_config)
        if not 0 < len(pipes) < 6:
            raise ValueError(f"Wrong number of players: Expected between 2 and 5, "
                             "but got {len(agents)}.")
        self.n_players = len(pipes)
        self.players = list(enumerate(
            [HanabiPlayer(pipe, agent_config, self.n_players, self.env.num_moves())
             for pipe, agent_config in zip(pipes, agent_infos)]))

        self.last_full_observation = None

        # A way to assign individual rewards. Default: pass reward to the agents as is.
        self.reward_func = reward_func or (lambda reward, _: reward)

    def __del__(self):
        for _, player in self.players:
            player._pipe.send((("Done", None), None))

    def reset_game(self):
        """Reset a game.

        Returns observation
        """
        self.last_full_observation = self.env.reset()
        #  for consumer_id, consumer in self.players:
        #      consumer.consume_observation(
        #          self.last_full_observation[consumer_id],
        #          0)
        self.distribute_observations([0 for _ in self.players])

    def distribute_observations(self, rewards):
        """Distribute last observation between players.

        Args:
            reward (list) -- individual rewards for players.
        """
        for consumer_id, consumer in self.players:
            consumer.consume_observation(
                self.env.observation_encoder.encode(self.last_full_observation[consumer_id]),
                rewards[consumer_id])


    def reverse_players(self):
        """
        Reverse the agent sequence.
        """
        self.players.reverse()

    def reorder_players(self):
        """
        Shuffle the agent sequence.
        """
        random.shuffle(self.players)

    def round(self):
        """
        Let the players play one round of the game.

        Returns:
            Id of the player who made the last move if the move has lead to a terminal state,
            or number of players = number of steps otherwise.
        """
        for player_id, agent in self.players:
            legal_moves = self.last_full_observation[player_id].legal_moves()
            legal_moves = [self.env.game.get_move_uid(move) for move in legal_moves]
            action = agent.act(legal_moves)
            self.last_full_observation, reward, done = self.env.step(action)
            reward = [self.reward_func(reward, player_id) for _ in self.players]
            # each agent consumes its observation of the turn.
            self.distribute_observations(reward)
            #  for consumer_id, consumer in self.players:
            #      consumer.consume_observation(
            #          self.last_full_observation[consumer_id],
            #          reward[consumer_id])
            if done:
                return player_id
        return self.n_players

    def play_game(self):
        """Play a game of hanabi until the end.
        """
        n_steps = 0
        cur_steps = self.round()
        while cur_steps == self.n_players:
            n_steps += cur_steps
            cur_steps = self.round()
        # add steps from the last round, which is the last player id + 1
        return n_steps + cur_steps + 1

    def play_fork(self, n_steps):
        """Play a game, but forking at each step
        """
        raise NotImplementedError()

    #  def collect_training_data(self):
    #      return [(player_id, player.training_data) for player_id, player in self.players]
