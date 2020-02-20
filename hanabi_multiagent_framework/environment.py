"""
This file defines a wrapper for HanabiEnv which overrides and extends the functionality of the
rl_env.HanabiEnv.
"""

from hanabi_learning_environment import rl_env
from hanabi_learning_environment import pyhanabi

class HanabiEnvironment(rl_env.HanabiEnv):
    """Hanabi environment wrapper for use with HanabiGameManager.
    """
    #  def __init__(self, env_config):
    #      super(HanabiEnvironment, self).__init__(env_config)

    def step(self, action_id):
        """Take one step in the game
        Overrides the step() method from rl_env.HanabiEnv.
        Breaking changes:
          -- Observations contain only vectorized representations.
          -- Action has to be an integer (i.e. action_id).
        """

        # fetch a move corresponding to action id.
        action = self.game.get_move(action_id)
        last_score = self.state.score()
        # Apply the action to the state.
        self.state.apply_move(action)

        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        observation = self._make_observation_all_players()
        #  done = self.state.is_terminal()
        # Reward is score differential. May be large and negative at game end.
        reward = self.state.score() - last_score
        #  info = {}

        return observation, reward, self.state.is_terminal()

    def _make_observation_all_players(self):
        """Make observation for all players.
        Returns:
        dict, containing observations for all players.
        """
        #  obs = {}
        player_observations = [self.state.observation(pid) for pid in range(self.players)]
        #  obs["player_observations"] = player_observations
        #  obs["current_player"] = self.state.cur_player()
        return player_observations

    def encode_observation(self, observation):
        """Encode an observation.

        Args:
            observation (pyhanabi.Observation) -- an observation to encode.

        Returns and encoded observation.
        """
        return self.observation_encoder.encode(observation)

    def reset(self):
        """Resets the environment for a new game.
        Returns:
            observation: vectorized observation
        """
        self.state = self.game.new_initial_state()
        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()
        return self._make_observation_all_players()
