"""
An example with a mock agent on how to operate the framework.
"""
import random
import numpy as np
from numpy import ndarray
import hanabi_multiagent_framework as hmf
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
from hanabi_multiagent_framework.utils import make_hanabi_env_config

n_players = 2
n_parallel = 100_000
#  n_parallel = 10
env_conf = make_hanabi_env_config('Hanabi-Full-Oracle', n_players)

env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)

class MockAgent(hmf.agent.HanabiAgent):
    """A mock agent which always selects the first legal move.
    """

    def __init__(self, action_spec):
        self.n_actions = action_spec.num_values

    def explore(self, observations: ndarray) -> ndarray:
        #  return np.random.randint(0, self.n_actions + 1, size=len(observations))
        #  action = np.argmax(legal_moves, axis=1)
        #  return action
        #  return [random.choice(o.legal_moves) for o in observations]
        actions = pyhanabi.HanabiMoveVector()
        for o in observations:
            actions.append(o.legal_moves[0])
        return actions

    def exploit(self, observations) -> ndarray:
        return self.explore(observations)

    def add_experience_first(self,
                             observations: ndarray,
                             step_types: ndarray) -> None:
        pass

    def add_experience(self,
                       observations: ndarray,
                       actions: ndarray,
                       rewards: ndarray,
                       step_types: ndarray) -> None:
        pass

    def update(self):
        pass

    def requires_vectorized_observation(self):
        return False

    def requires_vectorized_legal_moves(self):
        return True


agents = [MockAgent(env.action_spec_vec()) for _ in range(n_players)]

parallel_session = hmf.HanabiParallelSession(env, agents)

parallel_session.run_eval()

#  parallel_session.reset()
#  parallel_session.train(n_iter=30000,
#                         n_sim_steps=n_players,
#                         n_train_steps=1,
#                         n_warmup=0)
