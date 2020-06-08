import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
import numpy as np
from collections import namedtuple
from hanabi_agents.rlax_dqn import DQNAgent

n_players = 2
n_parallel = 32
epochs = 1_000_000
eval_n_parallel = 1_000
eval_freq = 300

def create_scheduler(eps_start, eps_min, steps1, steps2):
    def scheduler(x):
        if x <= steps1:
            return eps_start
        elif x <= steps2:
            return eps_start / 2
        else:
            return max(eps_min, min(eps_start / (x - steps2) * 10000, eps_start / 2))
    return scheduler

schedule = create_scheduler(0.9, 0.05, 200_000, 2_000_000)

env_conf = make_hanabi_env_config('Hanabi-Full-Oracle', n_players)

env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
eval_env = hmf.HanabiParallelEnvironment(env_conf, eval_n_parallel)
#  env_conf["max_life_tokens"] = 3
#  obs_shape[0] += env_conf["max_life_tokens"] - 1

self_play_agent = DQNAgent(env.observation_spec_vec()[0], env.action_spec(),
                           target_update_period=500,
                           learning_rate=5e-6,
                           epsilon=schedule,
                           layers=[512, 512])

#  agents = [DQNAgent(obs_shape, nb_actions) for _ in range(n_players)]
agents = [self_play_agent for _ in range(n_players)]

parallel_session = hmf.HanabiParallelSession(env, agents)

parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)

print("Game config", parallel_session.parallel_env.game_config)

# eval before
parallel_eval_session.run_eval()

# train
parallel_session.train(n_iter=eval_freq,
                       n_sim_steps=n_players,
                       n_train_steps=2,
                       n_warmup=10,
                       train_batch_size=256)

print("step", 1 * eval_freq)
# eval
parallel_eval_session.run_eval()

for i in range(epochs // eval_freq):
    parallel_session.train(n_iter=eval_freq,
                           n_sim_steps=n_players,
                           n_train_steps=2,
                           n_warmup=0,
                           train_batch_size=256)
    print("step", (i + 2) * eval_freq)
    # eval after
    parallel_eval_session.run_eval()
