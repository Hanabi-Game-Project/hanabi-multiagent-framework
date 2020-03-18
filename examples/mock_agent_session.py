import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
import numpy as np
from collections import namedtuple

n_players = 2
n_parallel = 1024
env_conf, nb_actions, obs_shape = make_hanabi_env_config('Hanabi-Full', n_players)

class MockAgent:

    def act(self, observation, legal_moves):
        action = np.argmax(legal_moves, axis=1)
        return action

agents = [MockAgent() for _ in range(n_players)]

parallel_session = hmf.HanabiParallelSession(agents, env_conf, n_parallel)

parallel_session.run(1000)
