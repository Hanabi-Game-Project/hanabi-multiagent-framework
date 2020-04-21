from typing import Tuple
import numpy as np

class ExperienceBuffer:
    """ExperienceBuffer stores transitions for training"""

    def __init__(self, observation_len: int, action_len: int, reward_len: int, size: int):
        self._obs_tm1_buf = np.empty((size, observation_len), dtype=np.byte)
        self._act_tm1_buf = np.empty((size, 1), dtype=np.byte)
        self._obs_t_buf = np.empty((size, observation_len), dtype=np.byte)
        self._lms_t_buf = np.empty((size, action_len), dtype=np.byte)
        self._rew_t_buf = np.empty((size, reward_len), dtype=np.float64)
        self._terminal_t_buf = np.empty((size, 1), dtype=bool)
        self._sample_range = np.arange(0, size, dtype=np.int)
        self.full = False
        self.cur_idx = 0
        self.size = size

    def add_transition(self,
                       observation_tm1: np.ndarray,
                       action_tm1: np.ndarray,
                       observation_t: np.ndarray,
                       legal_moves_t: np.ndarray,
                       reward_t: np.ndarray,
                       terminal_t: np.ndarray):
        """Add a transition to buffer.

        Args:
            observation_tm1 -- source observation. batch of shape (batch_size, observation_len)
            action_tm1      -- action taken from source to destination state.
                               batch of shape (batch_size, 1)
            observation_t   -- destination observation. batch of shape (batch_size, observation_len)
            legal_moves_t   -- actions that can be taken from destination observation.
                               batch of shape (batch_size, max_moves)
            reward_t        -- reward for getting from source to destination state.
                               batch of shape (batch_size, 1)
            terminal_t      -- flag showing whether the destination state is terminal.
                               batch of shape (batch_size, 1)
        """
        batch_size = len(observation_tm1)
        if self.cur_idx + batch_size <= self.size:
            self._obs_tm1_buf[self.cur_idx : self.cur_idx + batch_size, :] = observation_tm1
            self._act_tm1_buf[self.cur_idx : self.cur_idx + batch_size, :] = \
                action_tm1.reshape((batch_size, 1))
            self._obs_t_buf[self.cur_idx : self.cur_idx + batch_size, :] = observation_t
            self._lms_t_buf[self.cur_idx : self.cur_idx + batch_size, :] = legal_moves_t
            self._rew_t_buf[self.cur_idx : self.cur_idx + batch_size, :] = \
                reward_t.reshape((batch_size, 1))
            self._terminal_t_buf[self.cur_idx : self.cur_idx + batch_size, :] = \
                terminal_t.reshape((batch_size, 1))
            if self.cur_idx + batch_size == self.size:
                self.full = True
            self.cur_idx = (self.cur_idx + batch_size) % self.size
        else:
            # handle the case when at the end of the buffer
            tail = self.cur_idx + batch_size - self.size
            self._obs_tm1_buf[self.cur_idx:, :] = observation_tm1[:batch_size - tail]
            self._act_tm1_buf[self.cur_idx:, :] = \
                action_tm1[:batch_size - tail].reshape((batch_size - tail, 1))
            self._obs_t_buf[self.cur_idx:, :] = observation_t[:batch_size - tail]
            self._lms_t_buf[self.cur_idx:, :] = legal_moves_t[:batch_size - tail]
            self._rew_t_buf[self.cur_idx:, :] = \
                reward_t[:batch_size - tail].reshape((batch_size - tail, 1))
            self._terminal_t_buf[self.cur_idx:, :] = \
                terminal_t[:batch_size - tail].reshape((batch_size - tail, 1))
            self._obs_tm1_buf[:tail, :] = observation_tm1[-tail:]
            self._act_tm1_buf[:tail, :] = action_tm1[-tail:].reshape((tail, 1))
            self._obs_t_buf[:tail, :] = observation_t[-tail:]
            self._lms_t_buf[:tail, :] = legal_moves_t[-tail:]
            self._rew_t_buf[:tail, :] = reward_t[-tail:].reshape((tail, 1))
            self._terminal_t_buf[:tail, :] = terminal_t[-tail:].reshape((tail, 1))
            self.cur_idx = tail
            self.full = True

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
        """Sample <batch_size> transitions from the ExperienceBuffer.

        Returns (observation{batch_size, observation_len}, action{batch_size, 1},
                 reward{batch_size, 1}, q_vals{batch_size, max_moves})
        """
        indices = np.random.choice(self._sample_range[:self.size if self.full else self.cur_idx],
                                   size=batch_size)
        return (self._obs_tm1_buf[indices], self._act_tm1_buf[indices],
                self._obs_t_buf[indices], self._lms_t_buf[indices],
                self._rew_t_buf[indices], self._terminal_t_buf[indices])
