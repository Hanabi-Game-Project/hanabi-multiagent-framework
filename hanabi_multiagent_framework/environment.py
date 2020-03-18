"""
This file defines a wrapper for HanabiEnv which overrides and extends the functionality of the
rl_env.HanabiEnv.
"""

from hanabi_learning_environment import rl_env
from hanabi_learning_environment import pyhanabi
import dm_env
from dm_env import specs
import numpy as np

class HanabiParallelEnvironment:
    """Hanabi environment wrapper for use with HanabiGameManager.
    """
    def __init__(self, env_config: dict, n_parallel: int):
        self._parallel_env = pyhanabi.HanabiParallelEnv(env_config, n_parallel, True)

    def step(self, action_ids: list, agent_id: int) -> tuple:
        """Take one step in all game states.
        Args:
            action_ids -- list with ids of the actions for each state.
            agent_id -- id of the agent taking the actions.
        Return:
            a tuple consisting of:
              - observation tuple (vectorized observation, legal moves)
              - reward array
              - done (always False)
              - info (always None)
        """

        last_score = self._parallel_env.last_observation.rewards.copy()
        # Apply the action to the state.
        self._parallel_env.apply_batch_move(action_ids, agent_id)

        #  done = self.state.is_terminal()
        # Reward is score differential. May be large and negative at game end.
        reward = self._parallel_env.last_observation.rewards - last_score
        #  info = {}

        return ((self._parallel_env.last_observation.batch_observation.copy(),
                 self._parallel_env.last_observation.legal_moves.copy()),
                reward.copy(),
                False, # the enviroment resets automatically.
                None) # no additional info

    def reset(self) -> tuple:
        """Resets the environment for a new game.
        Returns:
            observation: (vectorized observation, legal moves)
        """
        self._parallel_env.new_initial_observation()
        return (self._parallel_env.last_observation.batch_observation.copy(),
                self._parallel_env.last_observation.legal_moves.copy())

class HanabiDMEnv(dm_env.Environment):
    """Hanabi environment wrapper for use with HanabiGameManager.
    """
    def __init__(self, config):
        assert isinstance(config, dict), "Expected config to be of type dict."
        self.game = pyhanabi.HanabiGame(config)
        self.observation_encoder = pyhanabi.ObservationEncoder(
            self.game, pyhanabi.ObservationEncoderType.CANONICAL)
        self.n_players = self.game.num_players()
        self.state = None
        #  super(HanabiEnvironment, self).__init__(env_config)

    def step(self, action_id: int) -> dm_env.TimeStep:
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

        #  hanabi_observations = self._make_observation_all_players()
        cur_hanabi_observation = self.state.observation(self.state.cur_player())
        #  done = self.state.is_terminal()
        # Reward is score differential. May be large and negative at game end.
        reward = float(self.state.score() - last_score)
        #  info = {}

        if self.state.is_terminal():
            return dm_env.termination(reward, self._observation(cur_hanabi_observation))
        return dm_env.transition(reward, self._observation(cur_hanabi_observation))

    def reset(self) -> dm_env.TimeStep:
        """Resets the environment for a new game.
        Returns:
            observation: vectorized observation
        """
        self.state = self.game.new_initial_state()
        while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            self.state.deal_random_card()
        return dm_env.restart(self._observation(self.state.observation(self.state.cur_player())))

    def observation_spec(self) -> dict:
        """Returns the observation spec."""
        return {"observation" : specs.BoundedArray(shape=self.observation_encoder.shape(),
                                                   dtype=np.float32,
                                                   name="current_player_observation",
                                                   minimum=0.0, maximum=1.0),
                "legal_moves": specs.BoundedArray(shape=(self.game.max_moves(),), dtype=np.int,
                                                  name="legal_moves", minimum=0,
                                                  maximum=self.game.max_moves())}

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(num_values=self.game.max_moves(), dtype=np.int, name="action")

    def _make_observation_all_players(self):
        """Make observation for all players.
        Returns:
        dict, containing observations for all players.
        """
        player_observations = [self.state.observation(pid) for pid in range(self.n_players)]
        #  obs["player_observations"] = player_observations
        #  obs["current_player"] = self.state.cur_player()
        return player_observations

    def _observation(self, hanabi_observation):
        #  lmove_uids = [self.game.get_move_uid(move) for move in hanabi_observation.legal_moves()]
        #  legal_moves = np.full((self.game.max_moves(),), -np.inf, dtype=np.float32)
        #  legal_moves[lmove_uids] = 0.0
        return {"observation": self._encode_observation(hanabi_observation),
                "legal_moves": [self.game.get_move_uid(move)
                                for move in hanabi_observation.legal_moves()]}

    def _encode_observation(self, hanabi_observation) -> np.ndarray:
        """Encode an observation.

        Args:
            observation (pyhanabi.Observation) -- an observation to encode.

        Returns and encoded observation.
        """
        return np.array(self.observation_encoder.encode(hanabi_observation), dtype=np.float32)
