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

# for use on local machine to not overload GPU-memory given jax default setting to occupy 90% of total GPU-Memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"

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
            path_to_db: str = 'obs.db'
        ):


    def split_evaluation(total_reward, no_pbt_agents):
        '''Assigns the total rewards from the different parallel states to the respective atomic agent'''
        states_per_agent = int(len(total_reward) / no_pbt_agents)
        mean_reward = np.zeros(no_pbt_agents)
        for i in range(no_pbt_agents):
            mean_score = total_reward[i * states_per_agent: (i + 1) * states_per_agent].mean()
            mean_reward[i] = mean_score
            print('Average score achieved by AGENT_{} = '.format(i), mean_score)
        return mean_reward

    def load_agent(env, no_agents):
        '''Loads a Agent containing n atomic subagents with the given parameters'''
        with gin.config_scope('agent_0'):
            reward_shaping_params = RewardShapingParams()
            population_params = PBTParams()
            custom_parm = population_params._replace(population_size = no_agents)
            population_params._replace(population_size = no_agents)
            agent_params = RlaxRainbowParams()
        return AgentDQNPopulation(
                        env.num_states,
                        env.observation_spec_vec_batch()[0],
                        env.action_spec_vec(),
                        custom_parm,
                        agent_params,
                        reward_shaping_params)

    #loading all reinforcement learning agents in a given directory to evaluate
    def no_agent_in_path(path):            
        if not os.path.isdir(path):
            raise Exception('{} is no valid path!'.format(path))
        paths_agents = []
        for file in os.listdir(path):
            if 'target' in str(file):
                paths_agents.append(file)
        return len(paths_agents), paths_agents



    pop_size, agent_names = no_agent_in_path(path_to_agents)
    print('Evaluating {} agents from the following given files {}'.format(pop_size, agent_names))
    #load dummy-parameters in order to supply Agent-Inits with necessary arguments
    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)

    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval * pop_size)

    if self_play:
        with gin.config_scope('agent_0'):
            self_play_agent = load_agent(eval_env, pop_size)
            agents = [self_play_agent for _ in range(n_players)]
    # # TODO: --later-- non-self-play
    # else:
    #     agent_1 = AgentDQNPopulation()
    #     agent_X = None
    #     agents = [agent_1]

    for agent in agents:
        agent.restore_weights(agent_names, path_to_agents)

    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)
    total_reward = parallel_eval_session.run_eval()
    mean_reward = split_evaluation(total_reward, pop_size)

    for i, agent in enumerate(agents[0].agents):
        print('agent_{} is object {}'.format(i, agent))

    div_matrix = agents[0].mutual_information_eval(path_to_db)

    #printout diversity matrix in heatmap
    plt.imshow(div_matrix)
    plt.colorbar()
    plt.show()

def main(args):
    
    # load configuration from gin file
    if args.agent_config_path is not None:
        gin.parse_config_file(args.agent_config_path)
    
    del args.agent_config_path
    session(**vars(args))
           
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

    args = parser.parse_args()
  
    main(args)         
            
            
