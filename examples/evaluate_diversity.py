import os
import numpy as np
import gin
import logging
import time

import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, PBTParams 
from hanabi_agents.rlax_dqn import RewardShapingParams
from hanabi_agents.pbt import AgentDQNPopulation
from hanabi_multiagent_framework.utils import eval_pretty_print

import matplotlib.pyplot as plt
import pickle
import sqlite3



# for use on local machine to not overload GPU-memory given jax default setting to occupy 90% of total GPU-Memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"

"""This script loads agents from a specified directory (path_to_agents) one by one and and feeds a set of observations to them
from a specified database (path_to_db). The database was generated with a set of different agents to diversify the set of observations obtained.
Furthermore those agents that generated the observations randomly chose to explore or exploit the current state to further diversify"""
@gin.configurable()
def session(
            self_play: bool = True,
            agent_config_path=None,
            output_dir = "./output",
            hanabi_game_type="Hanabi-Small",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 384,
            n_parallel_eval:int = 1000,
            n_train_steps: int = 1,
            n_sim_steps: int = 2,
            epochs: int = 1,
            eval_freq: int = 500,
            path_to_agents: str = "./pool/agents",
            path_to_db: str = 'obs.db',
            no_obs_test: int = 10
        ):


    def load_agent(env):
        '''Loads a DQNAgent with the given parameters'''
        with gin.config_scope('agent_0'):
            agent_params = RlaxRainbowParams()

        return DQNAgent(
            env.observation_spec_vec_batch()[0],
            env.action_spec_vec(),
            agent_params)

    #loading all reinforcement learning agents in a given directory to evaluate
    def no_agent_in_path(path):
        '''Gathers information on the number of agents in the specified path and also seperates between target and online parameters, as well as non-weights-files'''            
        if not os.path.isdir(path):
            raise Exception('{} is no valid path!'.format(path))
        paths_agents = []
        for file in os.listdir(path):
            #############################################################
            '''Supposes that agent weights have been stored with "target" as some part of their name'''
            if 'target' in str(file):
            #############################################################
                paths_agents.append(file)
        return len(paths_agents), paths_agents


    def _load_db(path, no_rows, requires_vectorized_observation = True):
        '''Loads n = no_rows observatoins from the previously created database'''
        conn = sqlite3.connect(path)
        c = conn.cursor()

        if requires_vectorized_observation:
            c.execute('SELECT obs_vec, obs_act_vec FROM obs LIMIT {}'.format(no_rows))
            query = c.fetchmany(no_rows)
            obs_o = []
            act_vec = []
            for elem in query:
                act_vec.append(np.asarray(pickle.loads(elem[1])))
                obs_o.append(np.asarray(pickle.loads(elem[0])))
            return (None,(np.asarray(obs_o), np.asarray(act_vec)))
        else:
            c.execute('SELECT obs_obj FROM obs LIMIT {}'.format(no_rows))
            query = c.fetchmany(no_rows)
            obs = pickle.loads(query[0])
            return obs

    def mutual_information(self, actions_agent_a, actions_agent_b, no_obs):
        '''Compares both vectors by calculating the mutual information'''
        return 1 - metrics.normalized_mutual_info_score(actions_agent_a, actions_agent_b)

    def simple_match(actions_agent_a, actions_agent_b, no_obs):
        '''Compares both action vectors and calculates the number of matching actions'''
        return (1 - np.sum(actions_agent_a == actions_agent_b)/no_obs)

    def cosine_distance(actions_agent_a, actions_agent_b, no_obs):
        '''Compares agents by cosine distance == simple match'''
        pass

    pop_size, agent_names = no_agent_in_path(path_to_agents)


    print('Evaluating {} agents from the following given files {}'.format(pop_size, agent_names))
    #load dummy-parameters in order to supply Agent-Inits with necessary arguments
    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)

    #load necessary environment stuff
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval * pop_size)

    #variables for later use in diversity measure
    action_matrix = []
    mean_rewards = np.zeros((pop_size, 1))
    div_obs = _load_db(path_to_db, no_obs_test)    

    #creates a dummy agent-object and loads it into the environment
    if self_play:
        self_play_agent = load_agent(eval_env)
        agents = [self_play_agent for _ in range(n_players)]

    #reloads weights from path_to_agents and substitutes weights of dummy-agent
    for i in range(pop_size):
        #for-loop actually not even necessary?! --> self-play
        for agent in agents:
            path_to_agent = os.path.join(path_to_agents, agent_names[i])
            agent.restore_weights(path_to_agent, path_to_agent)
        parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)
        total_reward = parallel_eval_session.run_eval()
        mean_rewards[i] = np.mean(total_reward)

        #feed set of predefined obs to each agent and saves results to list
        action_matrix.append(agent.exploit(div_obs))

        print(action_matrix)

    #compare each agents actions taken for the set of obs to each other by defined metric
    diversity_matrix = [[] for i in range(len(action_matrix))]
    for i, elem in enumerate(action_matrix):
        for vec in action_matrix:
            mut_info = simple_match(elem, vec, no_obs_test)
            diversity_matrix[i].append(mut_info)
    diversity_matrix = np.asarray(diversity_matrix)
    
    """
    Diversity = 1 - measure_specified:

    If two agents are identical: Diversity = 0
    If two agents are 100% diverse: Diversity = 1
    
    Example:
    Actions_taken_Agent_A = [1, 2, 2, 6, 9, 7, 1, 1, 1, 1]
    Actions_taken_Agent_B = [1, 2, 2, 6, 9, 7, 1, 9, 9, 9]
    --> 30% of actions are different (simple_match measure)
    Diversity score ==> 1 - 7/10 = 0.3

    Therefore we fill the diagonal of the confusion matrix (diversity_matrix) with 1's in order to not consider diversity to oneself 

    Actually not necessary at this point but important if we want to select the least diverse agents for PBT --> just maintained same notation
    """
    #read above example
    np.fill_diagonal(diversity_matrix, 1)

    #printout diversity matrix in heatmap
    plt.imshow(diversity_matrix)
    plt.colorbar()
    plt.show()

   
           
if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Train a dm-rlax based rainbow agent.")


    parser.add_argument(
        "--self_play", default=True, action='store_true',
        help="Whether the agent should play with itself, or an independent agent instance should be created for each player.")

    parser.add_argument(
        "--agent_config_path", type=str, default=None,
        help="Path to gin config file for rlax rainbow agent.")

    parser.add_argument(
        "--output_dir", type=str, default="/output",
        help="Destination for storing weights and statistics")
    
    parser.add_argument(
        "--path_to_agents", type=str, default="/pool/agents",
        help="Destination for storing weights and statistics")

    parser.add_argument(
        "--path_to_db", type=str, default="obs.db",
        help="Destination for storing weights and statistics")
    
    parser.add_argument(
        "--no_obs_test", type=int, default=1000,
        help="Number of observations from DB to evaluate diversity on")

    args = parser.parse_args()
  
    session(**vars(args))      
            
            
