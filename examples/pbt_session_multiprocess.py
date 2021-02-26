import os
import numpy as np
import gin
import logging
import time
import shutil
import ray

import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, PBTParams 
from hanabi_agents.rlax_dqn import RewardShapingParams
from hanabi_agents.pbt import AgentDQNPopulation
from hanabi_multiagent_framework.utils import eval_pretty_print

from multiprocessing import Pool, Process, Queue
import multiprocessing



# for use on local machine to not overload GPU-memory given jax default setting to occupy 90% of total GPU-Memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

# setting only necessary for NI server where cuda is installed via conda-env
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/mnt/antares_raid/home/maltes/miniconda/envs/RL"


"""
This is an example on how to run the PBT approach for training on DQN/Rainbow agents --> One agent interoperating with
environment and distributing/merging obtained observations/actions to the actual agents.
"""

# @gin.configurable(blacklist=['output_dir', 'self_play'])
@gin.configurable(blacklist=[ 'self_play'])
def session(
            input_ = None,
            q = None,
            self_play: bool = True,
            agent_config_path=None,
            output_dir = "./output",
            hanabi_game_type="Hanabi-Small",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 384,
            n_parallel_eval:int = 1200,
            n_train_steps: int = 1,
            n_sim_steps: int = 2,
            epochs: int = 3500,
            eval_freq: int = 500,
    ):

    #load gin-config
    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)

    # load data that is being passed from multiprocess-processes from one to another
    input_dict = input_.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = input_dict['gpu']
    epoch_circle = input_dict['epoch_circle']
    pbt_counter = input_dict['pbt_counter']
    agent_data = input_dict['agent_data']
    restore_weights = input_dict['restore_weights']

    #load gin-config with special focus for self-play for now
    with gin.config_scope('agent_0'):
        population_params = PBTParams()
    #load necessary params for the PBT-algorithm -- populations_size in half for 2 sub-processes
    population_size = int(population_params.population_size / 2)
    discard_perc = population_params.discard_percent
    lifespan = population_params.life_span
    pbt_epochs = int(epochs / population_params.generations)

    # make directories etc. just during very first generation
    if epoch_circle == 0:
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "weights"))
        os.makedirs(os.path.join(output_dir, "stats"))
        for i in range(n_players):
            os.makedirs(os.path.join(output_dir, "weights", "pos_" + str(i)))
            for j in range(population_size):
                os.makedirs(os.path.join(output_dir, "weights","pos_" + str(i), "agent_" + str(j)))
        #set pbt_counter such that after first generation agent will be discarded already
        pbt_counter = np.zeros(population_size) + (2 * pbt_epochs + 2)

        assert n_parallel % population_size == 0, 'n_parallel has to be multiple of pop_size'
        assert n_parallel_eval % population_size == 0, 'n_parallel_eval has to be multiple of pop_size'


    def load_agent(env):
        '''load agent with parameters from gin config -- self-play for now'''
        with gin.config_scope('agent_0'):
            agent_params = RlaxRainbowParams()
            reward_shaping_params = RewardShapingParams()
            population_params = PBTParams()
        population_params = population_params._replace(population_size = int(population_params.population_size/2))
        return AgentDQNPopulation(
                        env.num_states,
                        env.observation_spec_vec_batch()[0],
                        env.action_spec_vec(),
                        population_params,
                        agent_params,
                        reward_shaping_params,
                        eval_run = False)

    def create_exp_decay_scheduler(val_start, val_min, inflection1, inflection2):
        def scheduler(step):
            if step <= inflection1:
                return val_start
            elif step <= inflection2:
                return val_start / 2
            else:
                return max(val_min, min(val_start / (step - inflection2) * 10000, val_start / 2))
        return scheduler

    def create_linear_scheduler(val_start, val_end, interscept):
        def scheduler(step):
            return min(val_end, val_start + step * interscept)
        return scheduler


    def split_evaluation(total_reward, no_pbt_agents, prev_rew):
        '''Assigns the total rewards from the different parallel states to the respective atomic agent'''
        states_per_agent = int(len(total_reward) / no_pbt_agents)
        print('Splitting evaluations for {} states and {} agents!'.format(len(total_reward), no_pbt_agents))
        mean_reward = np.zeros(no_pbt_agents)
        for i in range(no_pbt_agents):
            mean_score = total_reward[i * states_per_agent: (i + 1) * states_per_agent].mean()
            mean_reward[i] = mean_score
            print('Average score achieved by AGENT_{} = {} & reward over past runs = {}'.format(i, mean_score, np.average(prev_rew, axis=1)[i]))
        return mean_reward

    def add_reward(x, y):
        '''Add reward to reward matrix by pushing prior rewards back'''
        x = np.roll(x, -1)
        x[:,-1] = y
        return x

    #environment configurations & initialization
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)
    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)

    #initialize agents before starting generation
    if self_play:
        with gin.config_scope('agent_0'):
            self_play_agent = load_agent(env)
            self_play_agent.pbt_counter = pbt_counter
            if epoch_circle == 0:
                if restore_weights is not None:
                    self_play_agent.restore_weights(restore_weights)
            if epoch_circle > 0:
                self_play_agent.restore_characteristics(agent_data)
            agents = [self_play_agent for _ in range(n_players)]
    # TODO: --later-- non-self-play
    # else:
        # agent_1 = AgentDQNPopulation()
        # agent_X = None
        # ...
        # agents = [agent_1]

    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()
    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)
    print("Game config", parallel_session.parallel_env.game_config)


    # eval before starting training
    mean_reward_prev = np.zeros((population_size, population_params.n_mean))
    total_reward = parallel_eval_session.run_eval()
    mean_reward = split_evaluation(total_reward, population_size, mean_reward_prev)
    start_time = time.time()
    
    # train with warmup before very first generation
    if epoch_circle == 0:
        parallel_session.train(
            n_iter=eval_freq,
            n_sim_steps=n_sim_steps,
            n_train_steps=n_train_steps,
            n_warmup=int(256 * 5 * n_players / n_sim_steps))

        print("step", 1 * eval_freq * n_train_steps)
        # eval
        mean_reward_prev = add_reward(mean_reward_prev, mean_reward)
        total_reward = parallel_eval_session.run_eval(dest=os.path.join(output_dir, "stats_0"))
        mean_reward= split_evaluation(total_reward, population_size, mean_reward_prev)

        if self_play:
            agents[0].save_weights(
                os.path.join(output_dir, "weights","pos_0"), mean_reward)
        else:
            for aid, agent in enumerate(agents):
                agent.save_weights(
                    os.path.join(output_dir, "weights","pos_" + str(aid)), mean_reward)
        print('Epoch took {} seconds!'.format(time.time() - start_time))

    # train
    for epoch in range(pbt_epochs):
        start_time = time.time()

        agents[0].increase_pbt_counter()

        parallel_session.train(
            n_iter=eval_freq,
            n_sim_steps=n_sim_steps,
            n_train_steps=n_train_steps,
            n_warmup=0)
        print("step", (epoch_circle * pbt_epochs + (epoch + 2)) * eval_freq * n_train_steps)
        
        # eval after train-step
        mean_reward_prev = add_reward(mean_reward_prev, mean_reward)
        total_reward = parallel_eval_session.run_eval(
            dest=os.path.join(
                output_dir,
                "stats", str(epoch_circle * pbt_epochs +(epoch + 1)))
            )
        mean_reward = split_evaluation(total_reward, population_size, mean_reward_prev)

        if self_play:
            agents[0].save_weights(
                os.path.join(output_dir, "weights", "pos_0"), mean_reward)
        else:
            for aid, agent in enumerate(agents):
                agent.save_weights(
                    os.path.join(output_dir, "weights", "pos_" + str(aid)), mean_reward)
                #TODO: Questionable for non-selfplay --> just one agent?
        print('Epoch {} took {} seconds!'.format((epoch + pbt_epochs * epoch_circle), time.time() - start_time))

    epoch_circle += 1
    mean_reward_prev = add_reward(mean_reward_prev, mean_reward)
    q.put([[agents[0].save_characteristics()], 
            epoch_circle, agents[0].pbt_counter, 
            mean_reward_prev])


@gin.configurable(blacklist=['self_play'])
def evaluation_session(input_,
            output_,
            self_play: bool = True,
            agent_config_path=None,
            output_dir = "./output",
            hanabi_game_type="Hanabi-Small",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 384,
            n_parallel_eval:int = 12000,
            n_train_steps: int = 1,
            n_sim_steps: int = 2,
            epochs: int = 1,
            eval_freq: int = 500,
        ):

    def concatenate_agent_data(data_lists):
        '''Prepares splitted Agent-Data from several divided sub-process for training to evaluate alltogether'''
        all_agents = {'online_weights' : [], 'trg_weights' : [],
            'opt_states' : [], 'experience' : [], 'parameters' : [[],[],[],[],[],[]]}
        for elem in data_lists:
            all_agents['online_weights'].extend(elem['online_weights'])
            all_agents['trg_weights'].extend(elem['trg_weights'])
            all_agents['opt_states'].extend(elem['opt_states'])
            all_agents['experience'].extend(elem['experience'])
            all_agents['parameters'][0].extend(elem['parameters'][0])
            all_agents['parameters'][1].extend(elem['parameters'][1])
            all_agents['parameters'][2].extend(elem['parameters'][2])
            all_agents['parameters'][3].extend(elem['parameters'][3])
            all_agents['parameters'][4].extend(elem['parameters'][4])
            all_agents['parameters'][5].extend(elem['parameters'][5])
        return all_agents
    
    def separate_agent(agent, split_no = 2):
        """Split the sigle dictionary back to several to then distribute on different GPUs"""
        agent_data = agent.save_characteristics()
        return_data = [{'online_weights' : [], 'trg_weights' : [],
            'opt_states' : [], 'experience' : [], 'parameters' : [[],[],[],[],[],[]]} for i in range(split_no)]
        length = int(len(agent_data['online_weights'])/split_no)
        for i in range(split_no):
            return_data[i]['online_weights'].extend(agent_data['online_weights'][i*length : (i+1)*length])
            return_data[i]['trg_weights'].extend(agent_data['trg_weights'][i*length : (i+1)*length])
            return_data[i]['opt_states'].extend(agent_data['opt_states'][i*length : (i+1)*length])
            return_data[i]['experience'].extend(agent_data['experience'][i*length : (i+1)*length])
            return_data[i]['parameters'][0].extend(agent_data['parameters'][0][i*length : (i+1)*length])
            return_data[i]['parameters'][1].extend(agent_data['parameters'][1][i*length : (i+1)*length])
            return_data[i]['parameters'][2].extend(agent_data['parameters'][2][i*length : (i+1)*length])
            return_data[i]['parameters'][3].extend(agent_data['parameters'][3][i*length : (i+1)*length])
            return_data[i]['parameters'][4].extend(agent_data['parameters'][4][i*length : (i+1)*length])
            return_data[i]['parameters'][5].extend(agent_data['parameters'][5][i*length : (i+1)*length])
        return return_data
    
    def choose_fittest(mean_reward, discard_perc, agent):
        """Chosses the fittest agents after evaluation run and overwrites all the other agents with weights + permutation of lr + buffersize"""
        no_fittest = mean_reward.shape[0] - int(mean_reward.shape[0] * discard_perc)
        index_loser = np.argpartition(mean_reward, no_fittest)[:no_fittest]
        index_survivor = np.argpartition(-mean_reward, no_fittest)[:no_fittest]
        agent.survival_fittest(index_survivor, index_loser)

    def split_evaluation(total_reward, no_pbt_agents):
        '''Assigns the total rewards from the different parallel states to the respective atomic agent'''
        states_per_agent = int(len(total_reward) / no_pbt_agents)
        mean_reward = np.zeros(no_pbt_agents)
        for i in range(no_pbt_agents):
            mean_score = total_reward[i * states_per_agent: (i + 1) * states_per_agent].mean()
            mean_reward[i] = mean_score
            print('Average score achieved by AGENT_{} = '.format(i), mean_score)
        return mean_reward

    def load_agent(env):
        '''load agent with parameters from gin config -- self-play for now'''
        with gin.config_scope('agent_0'):
            reward_shaping_params = RewardShapingParams()
            population_params = PBTParams()
            agent_params = RlaxRainbowParams()
        print(agent_params)
        return AgentDQNPopulation(
                        env.num_states,
                        env.observation_spec_vec_batch()[0],
                        env.action_spec_vec(),
                        population_params,
                        agent_params,
                        reward_shaping_params,
                        eval_run = True)

    def moving_average(mean_rewards):
        '''Averages over the 2D Matrix of Means of the past runs'''
        rewards = np.average(mean_rewards, axis = 1)
        return rewards


    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)

    #initialize agents before starting generation
    input_dict = input_.get()
    agent_data = input_dict['agent_data']
    epoch_circle = input_dict['epoch_circle']
    pbt_counter = input_dict['pbt_counter']
    mean_rewards = moving_average(input_dict['mean_rewards'])
    db_path = input_dict['db_path']
    pbt_history = input_dict['pbt_history']
    pbt_history_params = input_dict['pbt_history_params']

    #environment configurations & initialization
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)

    all_agent_data = concatenate_agent_data(agent_data)

    if self_play:
        with gin.config_scope('agent_0'):
            self_play_agent = load_agent(eval_env)
            self_play_agent.restore_characteristics(all_agent_data)
            agents = [self_play_agent for _ in range(n_players)]
    # # TODO: --later-- non-self-play
    # else:
    #     agent_1 = AgentDQNPopulation()
    #     agent_X = None
    #     agents = [agent_1]


    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)

    with gin.config_scope('agent_0'):
        population_params = PBTParams()
    population_size = population_params.population_size
    discard_perc = population_params.discard_percent
    lifespan = population_params.life_span
    total_reward = parallel_eval_session.run_eval(dest=os.path.join(output_dir, "pbt_{}".format(epoch_circle)))

    #start PBT-logic
    agents[0].pbt_counter = pbt_counter
    agents[0].pbt_history = pbt_history
    agents[0].pbt_history_params = pbt_history_params


    agents[0].pbt_eval(mean_rewards, output_dir)
    agents[0].save_pbt_log(output_dir, epoch_circle)
    
    for i, agent in enumerate(agents[0].agents):
        print('agent_{} is object {}'.format(i, agent))

    #return data to continue training in divided sub-processes
    return_data = separate_agent(agents[0])
    pbt_counter = agents[0].pbt_counter
    output_.put((return_data, 
                pbt_counter,
                agents[0].pbt_history, 
                agents[0].pbt_history_params))


def training_run(agent_data = [], 
                epoch_circle = None,
                pbt_counter = [],
                restore_weights = None):
    '''Function that administers the use of Multiprocessing Sub-Processes for training'''
    print('IN TRAINING', args.agent_config_path)
    input_ = Queue()
    output = Queue()
    processes = []
    
    #divide training into n=2 subprocesses
    for i in range(2):
        input_data = {'agent_data' : agent_data[i], 
                    'epoch_circle' : epoch_circle, 
                    'pbt_counter' : pbt_counter[i],
                    'gpu' : str(i),
                    'restore_weights' : restore_weights
                    }

        # Share data between processes via Queue-Object and start processes
        input_.put(input_data)
        output_dir = (os.path.join(args.output_dir,'over_agent_{}'.format(i)))
        p = Process(target=session, args=(input_, 
                                        output, 
                                        args.self_play, 
                                        args.agent_config_path, 
                                        output_dir))
        processes.append(p)
        p.start()
    
    #collect results from subprocesses
    agent_data = []
    pbt_counter_2 = []
    mean_rewards = []
    for p in processes:
        ret = output.get() # will block
        agent_data.append(ret[0][0])
        pbt_counter_2.append(ret[2])
        mean_rewards.append(ret[3])

    for p in processes:
        p.join()
    pbt_counter_2 = np.concatenate(pbt_counter_2)
    mean_rewards = np.concatenate(mean_rewards, axis = 0)
    return agent_data, ret[1], pbt_counter_2, mean_rewards

def evaluation_run(agent_data = [], 
                epoch_circle = None,
                pbt_counter = None,
                mean_rewards = None,
                db_path = None,
                pbt_history = [],
                pbt_history_params = []):
    '''Function that administers the use of Multiprocessing-Process for evaluation (Subprocess because of CUDA init issues otherwise)'''
    input_ = Queue()
    output = Queue()
    processes = []

    #share necessary data with sub-process
    input_data = {'agent_data' : agent_data, 
                'pbt_counter' : pbt_counter,
                'epoch_circle' : epoch_circle,
                'mean_rewards' : mean_rewards,
                'db_path' : db_path,
                'pbt_history' : pbt_history,
                'pbt_history_params' : pbt_history_params
                }
    input_.put(input_data)
    # output_dir = os.path.join(args.output_dir, 'best_agents')
    output_dir = args.output_dir

    #make directory for all processes to use
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    p = Process(target=evaluation_session, args=(input_, 
                                    output, 
                                    args.self_play, 
                                    args.agent_config_path, 
                                    output_dir))
    processes.append(p)
    p.start()
    
    #collect results from sub-processes
    eval_data = output.get() # will block
    agent_data = eval_data[0]
    pbt_counter = eval_data[1]
    pbt_history = eval_data[2]
    pbt_history_params = eval_data[3]
    p.join()


    return agent_data, pbt_counter, pbt_history, pbt_history_params




def main(args):
    # load configuration from gin file and args from parser
    if args.agent_config_path is not None:
        gin.parse_config_file(args.agent_config_path)
    db_path = args.db_path
    with gin.config_scope('agent_0'):
        pbtparams = PBTParams()
    agent_data = [[],[]]
    pbt_counter = np.zeros(pbtparams.population_size)
    pbt_history = []
    pbt_history_params = []
    # run PBT-algorithm in generations
    epoch_circle = 0
    for gens in range(pbtparams.generations):
        agent_data, epoch_circle, pbt_counter, mean_rewards = training_run(agent_data, epoch_circle, np.split(pbt_counter, 2), args.restore_weights)
        print('pbt_counter after training {}'.format(pbt_counter))
        time.sleep(5)
        agent_data, pbt_counter, pbt_history, pbt_history_params = evaluation_run(agent_data, 
                                                                                    epoch_circle, 
                                                                                    pbt_counter, 
                                                                                    mean_rewards, 
                                                                                    db_path, 
                                                                                    pbt_history, 
                                                                                    pbt_history_params)
        print('pbt_counter before training {}'.format(pbt_counter))
        time.sleep(5)


            
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
        "--output_dir", type=str, default="./output",
        help="Destination for storing weights and statistics")
    
    parser.add_argument(
        "--db_path", type=str, default=None,
        help="Path to the DB that contains observations for diversity measure"
    )
    parser.add_argument(
        "--restore_weights", type=str, default=None,
        help="Path pickle file with agent weights"
    )


    args = parser.parse_args()


 
main(args)         
        
