"""
An example with a mock agent on how to operate the framework.
"""
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
import numpy as np
from numpy import ndarray
from hanabi_agents.rule_based import RulebasedAgent
from hanabi_agents.rule_based.predefined_rules import piers_rules, flawed_rules, VanDenBergh

n_players = 2
n_parallel = 100
env_conf = make_hanabi_env_config('Hanabi-Small', n_players)

env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)

agents = [RulebasedAgent(piers_rules) for _ in range(n_players)]

parallel_session = hmf.HanabiParallelSession(env, agents)

print("Game config", env.game_config)
parallel_session.run_eval()
