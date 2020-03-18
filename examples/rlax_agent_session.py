import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
import numpy as np
from collections import namedtuple
from rlax_dqn import DQNAgent

n_players = 2
n_parallel = 1024
epochs = 2

env_conf, nb_actions, obs_shape = make_hanabi_env_config('Hanabi-Very-Small-Oracle', n_players)

agents = [DQNAgent(obs_shape, nb_actions) for _ in range(n_players)]

parallel_session = hmf.HanabiParallelSession(agents, env_conf, n_parallel)


# eval before
parallel_session_pre = hmf.HanabiParallelSession(agents, env_conf, 10)
_, total_reward_before = parallel_session_pre.run(100)

# train
parallel_session.train(n_epochs=10, n_steps=5, n_warmup=10, train_batch_size=256)

# eval after
del parallel_session_pre
parallel_session_eval = hmf.HanabiParallelSession(agents, env_conf, 10)
_, total_reward = parallel_session_eval.run(100)

print("total_reward before:", total_reward_before)
print("total_reward:", total_reward)
