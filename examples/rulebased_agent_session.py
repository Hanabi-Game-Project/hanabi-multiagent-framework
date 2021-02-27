"""
An example with a mock agent on how to operate the framework.
"""
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
import numpy as np
from numpy import ndarray
from hanabi_agents.rule_based import RulebasedAgent
from hanabi_agents.rule_based.predefined_rules import piers_rules_adjusted
import time

n_players = 2
n_parallel = 1000
env_conf = make_hanabi_env_config('Hanabi-Small', n_players)

env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)

agents = [RulebasedAgent(piers_rules_adjusted) for _ in range(n_players)]

parallel_session = hmf.HanabiParallelSession(env, agents)

print("Game config", env.game_config)
start_time = time.time()
parallel_session.run_eval()
print(time.time() - start_time)

rule_names = [r.__name__ for r in agents[0].rules] + ['random']
for rule, counter in zip(rule_names, agents[0].histogram):
    print(rule, counter)
