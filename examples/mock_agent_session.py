"""
An example with a mock agent on how to operate the framework.
"""
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
import numpy as np
from numpy import ndarray
from collections import namedtuple

n_players = 2
n_parallel = 10
env_conf = make_hanabi_env_config('Hanabi-Full-Oracle', n_players)

env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)

class MockAgent(hmf.agent.HanabiAgent):
    """A mock agent which always selects the first legal move.
    """

    def explore(self, observations: ndarray, legal_moves: ndarray) -> ndarray:
        action = np.argmax(legal_moves, axis=1)
        return action

    def exploit(self, observations: ndarray, legal_moves: ndarray) -> ndarray:
        return self.explore(None, legal_moves)

    def add_experience_first(self,
                             observations: ndarray,
                             legal_moves: ndarray,
                             step_types: ndarray) -> None:
        pass

    def add_experience(self,
                       observations: ndarray,
                       legal_moves: ndarray,
                       actions: ndarray,
                       rewards: ndarray,
                       step_types: ndarray) -> None:
        pass

    def update(self):
        pass


agents = [MockAgent() for _ in range(n_players)]

parallel_session = hmf.HanabiParallelSession(env, agents)
parallel_session.reset()

parallel_session.train(n_iter=30000,
                       n_sim_steps=n_players,
                       n_train_steps=1,
                       n_warmup=0,
                       train_batch_size=256)
