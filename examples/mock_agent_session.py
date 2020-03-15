import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
import numpy as np
from collections import namedtuple

AgentConfig = namedtuple('AgentConfig', "consumes_round_observations")

n_players = 5
n_parallel = 10
env_conf, nb_actions, obs_shape = make_hanabi_env_config('Hanabi-Full', 5)

class MockAgent:

    def step(self, observation, reward, legal_moves):
        action = np.argmax(legal_moves)
        return action

agents = [MockAgent() for _ in range(n_players)]
agent_infos = [{"consumes_round_observations" : False} for _ in agents]
env_configs = [env_conf for _ in range(n_parallel)]

parallel_session = hmf.HanabiParallelSession(agents, agent_infos, env_configs)

parallel_session.run_parallel({"n_games": 10000})
