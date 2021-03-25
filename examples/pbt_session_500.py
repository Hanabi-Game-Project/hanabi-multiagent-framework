import os
import numpy as np
import gin
import logging
import time
import shutil
import random
import math

import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, PBTParams, DQNParallel
from hanabi_agents.rlax_dqn import RewardShapingParams
from hanabi_agents.pbt import AgentDQNPopulation
from hanabi_multiagent_framework.utils import eval_pretty_print

from multiprocessing import Pool, Process, Queue
import multiprocessing





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
            hanabi_game_type="Hanabi-Full",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 256,
            n_parallel_eval:int = 4096,
            n_train_steps: int = 4,
            n_sim_steps: int = 2,
            epochs: int = 3500,
            eval_freq: int = 1000,
    ):

    def load_agent(env, agent_data, num_gpus):
        '''load agent with parameters from gin config -- self-play for now'''
        with gin.config_scope('agent_0'):
            agent_params = RlaxRainbowParams()
            reward_shaping_params = RewardShapingParams()
            population_params = PBTParams()
            # buffersizes, lrs, alphas = [], [], []
            # for i in range(len(agent_data['buffersize'])):
            #     buffersizes.append(agent_data['buffersize'][i])
            #     lrs.append(agent_data['lr'][i])
            #     alphas.append(agent_data['alpha'][i])
        return DQNParallel(
                        env.observation_spec_vec_batch()[0],
                        env.action_spec_vec(),
                        agent_data['buffersize'],
                        agent_data['lr'],
                        agent_data['alpha'],
                        agent_params)

    def split_evaluation(total_reward, no_pbt_agents, prev_rew):
        '''Assigns the total rewards from the different parallel states to the respective atomic agent'''
        states_per_agent = int(len(total_reward) / no_pbt_agents)
        print('Splitting evaluations for {} states and {} agents!'.format(len(total_reward), no_pbt_agents))
        mean_reward = np.zeros(no_pbt_agents)
        for i in range(no_pbt_agents):
            mean_score = total_reward[i * states_per_agent: (i + 1) * states_per_agent].mean()
            mean_reward[i] = mean_score
            print('Average score achieved by AGENT_{} = {:.2f} & reward over past runs = {}'.format(i, mean_score, np.average(prev_rew, axis=1)[i]))
        return mean_reward

    def add_reward(x, y):
        '''Add reward to reward matrix by pushing prior rewards back'''
        x = np.roll(x, -1)
        x[:,-1] = y
        return x

    #load gin-config
    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)

    # load data that is being passed from multiprocess-processes from one to another
    input_dict = input_.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = input_dict['gpu']
    epoch_circle = input_dict['epoch_circle']
    agent_data = input_dict['agent_data']
    num_gpus = input_dict['num_gpus']


    #load gin-config with special focus for self-play for now
    with gin.config_scope('agent_0'):
        population_params = PBTParams()
        agent_params = RlaxRainbowParams()
    #load necessary params for the PBT-algorithm -- populations_size in half for 2 sub-processes
    population_size = int(population_params.population_size / num_gpus)
    lifespan = population_params.life_span
    pbt_epochs = lifespan
    # make directories etc. just during very first generation

    output_dir = os.path.join(output_dir, ('generation_' + str(int(input_dict['gpu'])*1000 + epoch_circle)))
    os.makedirs(output_dir)

    #environment configurations & initialization
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)
    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)

    def sample_buffersize(pbt_params, agent_params):
        exp_factor = math.log(agent_params.experience_buffer_size, 2)
        buffer_sizes_start = [2**i for i in range(int(exp_factor) - pbt_params.buffersize_start_factor, \
                                                    int(exp_factor) + pbt_params.buffersize_start_factor)]                                     
        return random.choice(buffer_sizes_start)

    def sample_init_lr(pbt_params):
        return random.choice(np.linspace(pbt_params.lr_min, pbt_params.lr_max, pbt_params.lr_sample_size))
    
    def sample_init_alpha(pbt_params):
        return random.choice(np.linspace(pbt_params.alpha_min, pbt_params.alpha_max, pbt_params.alpha_sample_size))
        
    #sample params
    
    if epoch_circle == 0:
        agent_data = {'buffersize': [], 'lr': [], 'alpha': []}
        for i in range(population_params.population_size):
            agent_data['buffersize'].append(sample_buffersize(population_params, agent_params))
            agent_data['lr'].append(sample_init_lr(population_params))
            agent_data['alpha'].append(sample_init_alpha(population_params))



    #initialize agents before starting generation
    if self_play:
        with gin.config_scope('agent_0'):
            self_play_agent = load_agent(env, agent_data, num_gpus)
            # self_play_agent.pbt_counter = pbt_counter
            agents = [self_play_agent for _ in range(n_players)]
    else:
        raise NotImplementedError()

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
    # if epoch_circle == 0:
    parallel_session.train(
        n_iter=eval_freq,
        n_sim_steps=n_sim_steps,
        n_train_steps=n_train_steps,
        n_warmup=int(256 * 5 * n_players / n_sim_steps),
        optimize_for_parallel = True)

    print("step", 1 * eval_freq * n_train_steps)
    # eval
    mean_reward_prev = add_reward(mean_reward_prev, mean_reward)
    total_reward = parallel_eval_session.run_eval(dest=os.path.join(output_dir, "stats_epoch_0"))
    mean_reward= split_evaluation(total_reward, population_size, mean_reward_prev)

    print('Epoch took {:.2f} seconds!'.format(time.time() - start_time))

    # train
    for epoch in range(pbt_epochs):
        start_time = time.time()

        # agents[0].increase_pbt_counter()

        parallel_session.train(
            n_iter=eval_freq,
            n_sim_steps=n_sim_steps,
            n_train_steps=n_train_steps,
            n_warmup=0,
            optimize_for_parallel= True)
        print("step", (epoch_circle * pbt_epochs + (epoch + 2)) * eval_freq * n_train_steps)
        
        # eval after train-step
        mean_reward_prev = add_reward(mean_reward_prev, mean_reward)
        total_reward = parallel_eval_session.run_eval(
            dest=os.path.join(output_dir, ("stats_epoch_" + str(epoch + 1))))
            
        mean_reward = split_evaluation(total_reward, population_size, mean_reward_prev)

        print('Epoch {} took {:.2f} seconds!'.format((epoch + pbt_epochs * epoch_circle), time.time() - start_time))

    if self_play:
        os.makedirs(os.path.join(output_dir, 'weights'))
        agents[0].save_attributes(
            os.path.join(output_dir, "weights"), 'params')
        agents[0].save_weights(os.path.join(output_dir, 'weights'), 'parallel_agents')
    else:
        for aid, agent in enumerate(agents):
            agent.save_weights(
                os.path.join(output_dir, "weights", "pos_" + str(aid)), mean_reward)
            #TODO: Questionable for non-selfplay --> just one agent?

    epoch_circle += 1
    mean_reward_prev = add_reward(mean_reward_prev, mean_reward)

    q.put([agents[0].save_min_characteristics(), 
            epoch_circle, 
            mean_reward_prev])


def training_run(agent_data = [], 
                epoch_circle = None,
                restore_weights = None,
                num_gpus = 1,
                which_gpu = 0,
                hanabi_game_type = 'Hanabi-Small'):
    '''Function that administers the use of Multiprocessing Sub-Processes for training'''
    print('IN TRAINING', args.agent_config_path)
    input_ = Queue()
    output = Queue()
    processes = []
    
    #divide training into n=2 subprocesses for 2 GPUs
    for i in range(args.num_gpus):
        input_data = {'agent_data' : agent_data[i], 
                    'epoch_circle' : epoch_circle,
                    'gpu' : str(which_gpu),
                    'restore_weights' : restore_weights,
                    'num_gpus' : num_gpus
                    }

        # Share data between processes via Queue-Object and start processes
        input_.put(input_data)
        output_dir = args.output_dir
        p = Process(target=session, args=(input_, 
                                        output, 
                                        args.self_play, 
                                        args.agent_config_path, 
                                        output_dir,
                                        hanabi_game_type))
        processes.append(p)
        p.start()
    
    #collect results from subprocesses
    agent_data = []
    mean_rewards = []
    for p in processes:
        ret = output.get() # will block
        agent_data.append(ret[0])
        mean_rewards.append(ret[2])

    for p in processes:
        p.join()

    mean_rewards = np.concatenate(mean_rewards, axis = 0)
    return agent_data, ret[1], mean_rewards

class Book_keeper():

    def __init__(self,
                pbt_params = None,
                agent_params = None,
                num_gpus = 1):

        # scores achieved with prior parameter sets
        self.highest_score = np.zeros(int(pbt_params.population_size*(1-pbt_params.discard_percent)+1))
        self.best_sets = [{} for i in range(int(pbt_params.population_size*(1-pbt_params.discard_percent)+1))]
        self.pbt_params = pbt_params
        self.agent_params = agent_params
        # parameter_sets that have been explored already
        self.explored_sets = []
        self.num_gpus = num_gpus

        


    # evolve parameter_set or make completely new
    def sample_new_params(self, set_p, new_sample = False):

        def sample_buffersize():
            exp_factor = math.log(self.agent_params.experience_buffer_size, 2)
            buffer_sizes_start = [2**i for i in range(int(exp_factor) - self.pbt_params.buffersize_start_factor, \
                                                        int(exp_factor) + self.pbt_params.buffersize_start_factor)]                                     
            return int(random.choice(buffer_sizes_start))

        def sample_init_lr():
            return random.choice(np.linspace(self.pbt_params.lr_min, self.pbt_params.lr_max, self.pbt_params.lr_sample_size))
        
        def sample_init_alpha():
            return random.choice(np.linspace(self.pbt_params.alpha_min, self.pbt_params.alpha_max, self.pbt_params.alpha_sample_size))
        

        if new_sample:
            new_set = {}
            for key in set_p.keys():
                if str(key) == 'buffersize':
                    set_p[key] = sample_buffersize()
                elif str(key) == 'alpha':
                    set_p[key] = sample_alpha()
                elif str(key) == 'lr':
                    set_p[key] = sample_init_lr()
                else:
                    raise NameError()
        else:
            new_set = {}
            for key in set_p.keys():
                if str(key) == 'buffersize':
                    factor = self.pbt_params.buffersize_factor
                    new_set[key] = int(random.choice([set_p[key]*factor, set_p[key], set_p[key]/factor]))
                elif str(key) == 'alpha':
                    factor = self.pbt_params.factor_alpha
                    new_set[key] = random.choice([set_p[key]*factor, set_p[key], set_p[key]/factor])
                elif str(key) == 'lr':
                    factor = self.pbt_params.factor_alpha
                    new_set[key] = random.choice([set_p[key]*factor, set_p[key], set_p[key]/factor])
                else:
                    raise NameError()
            return new_set

    #checks if the current parameter set has been explored already
    def already_used(self, set_):
        known_set = False
        for prev_set in self.explored_sets:
            if set_ == prev_set:
                known_set = True
                return known_set
        return known_set
    
    def make_set(self, macro_set, index):
        new_set = {}
        for key in macro_set.keys():
            new_set[key] = macro_set[key][index]
        return new_set

    def evaluation_run(self, agent_data = None,
                    mean_rewards = None,
                    pbt_history = [],
                    pbt_history_params = []):

    
        # merge different dicts
        parameters = {}
        for key in agent_data[0].keys():
            # parameters[key] = [b[key] for b in agent_data]
            parameters.setdefault(key, []).extend([b[key] for b in agent_data][0])
            
        for i in range(self.pbt_params.population_size):
            self.explored_sets.append(self.make_set(parameters, i))
        
        mean_rewards = np.mean(mean_rewards, axis = 1)

        # number of agents to keep possibly and their index
        no_fittest = int(self.pbt_params.population_size * self.pbt_params.discard_percent)
        index_best_scoring = np.argpartition(-mean_rewards, no_fittest)[:no_fittest]

        # compare current run with previous results and just keep globally best results
        index_global_best = []
        for i in range(no_fittest):
            j = i+1
            if mean_rewards[index_best_scoring[-j]] >= np.min(self.highest_score):
                index_prev_min = np.argmin(self.highest_score)
                print(self.highest_score)
                mean_reward = mean_rewards[index_best_scoring[-j]]
                self.highest_score[index_prev_min] = mean_reward
                index_global_best.append(index_best_scoring[-j])
                # substitute current set into best_performing sets
                # print(index_prev_min, index_prev_min[0])
                self.best_sets[index_prev_min] = self.make_set(parameters, index_best_scoring[-j])

        print('Explored sets: >>>>', self.explored_sets)
        print('\n')
        print('Currently best sets >>>>>>>>>>>>{}'.format(self.best_sets))
        print('\n')        
        print('Past best scores reached   >>>>>{}'.format(self.highest_score))
        print('This Generation best score >>>>>{}'.format(mean_rewards[index_best_scoring]))
        print('Chosen indices to substitute >>>{}'.format(index_global_best))
        print('Respective parameter sets >>>>>>{}'.format(parameters))
        

        # sample new parameter sets given globally best sets so far
        new_parameters = []
        if index_global_best:  
            for i in range(int(len(index_best_scoring)*1.5)):
                while len(new_parameters) < self.pbt_params.population_size:
                    _set = self.make_set(parameters, random.choice(index_global_best))
                    known_set = True
                    j = 0
                    while known_set == True and j <= 10:
                        j+=1
                        new_set = self.sample_new_params(_set)
                        known_set = self.already_used(new_set)
                    if known_set == False:
                        new_parameters.append(new_set)
        print('len new parameters>>>>>>>>>>', new_parameters)
        # try to change some of the previous best sets
        for i in range(3):
            while len(new_parameters) < self.pbt_params.population_size:
                _set = random.choice(self.best_sets)
                known_set = True
                j = 0
                while known_set == True and j < 2:
                    j+=1
                    new_set = self.sample_new_params(_set)
                    known_set = False
                    for prev_set in self.explored_sets:
                        if new_set == prev_set:
                            known_set = True
                    known_set = self.already_used(new_set)
                if known_set == False:
                    new_parameters.append(new_set)
        print(len(new_parameters))
        # for the remaining sets to complete population_size resample completely new params
        for i in range(self.pbt_params.population_size - len(new_parameters)):
            j = 0
            known_set == True
            while known_set == True:
                j+=1
                new_set = sample_new_params(new_parameters[0], True)
                known_set = self.already_used(new_set)
            new_parameters.append(new_set)

        prep_params = []
        len_p_agent = int(len(new_parameters)/self.num_gpus)
        for i in range(self.num_gpus):
            prep_params.append(new_parameters[i*len_p_agent:(i+1)*len_p_agent])

        # track who spends parameters -- not necessary for now

        print('New parameter sets >>>>>>{}'.format(new_parameters))
        print(prep_params)
        prep_prepped = [{'buffersize': [], 'lr': [], 'alpha': []} for i in range(self.num_gpus)]
        for i, elem in enumerate(prep_params):
            print('<>>>>>>>>>>>>>><<<<<<<<<<', elem)
            for _set in elem:
                print(_set)
                for key in _set.keys():
                    print(_set)
                    print(key)
                    print(prep_prepped[i][key])
                    prep_prepped[i][key].append(_set[key])

        print(prep_prepped)
        return prep_prepped




def main(args):
    # load configuration from gin file and args from parser

    if args.num_gpus == 1:
        # for use on local machine to not overload GPU-memory given jax default setting to occupy 90% of total GPU-Memory
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
    else:
        # setting only necessary for NI server where cuda is installed via conda-env
        os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/mnt/antares_raid/home/maltes/miniconda/envs/RL"



    if args.agent_config_path is not None:
        gin.parse_config_file(args.agent_config_path)
    db_path = args.db_path
    with gin.config_scope('agent_0'):
        pbt_params = PBTParams()
        agent_params = RlaxRainbowParams()
    agent_data = [[] for i in range(args.num_gpus)]
    pbt_counter = np.zeros(pbt_params.population_size)
    pbt_history = []
    pbt_history_params = []

    keeper = Book_keeper(pbt_params, agent_params)

    # run PBT-algorithm in generations
    epoch_circle = 0
    for gens in range(pbt_params.generations):
        agent_data, epoch_circle, mean_rewards = training_run(agent_data, epoch_circle, args.restore_weights, args.num_gpus, args.which_gpu, args.hanabi_game_type)
        print('pbt_counter after training {}'.format(pbt_counter))
        # time.sleep(5)
        agent_data = keeper.evaluation_run(agent_data, 
                    mean_rewards)
        print('pbt_counter before training {}'.format(pbt_counter))
        # time.sleep(5)


            
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
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Define the Machine to run on by installed GPUs"
    )
    parser.add_argument(
        "--which_gpu", type=int, default=0,
        help="Define the GPU if more than 1 available"
    )
    parser.add_argument(
        "--hanabi_game_type", type=str, default='Hanabi-Small',
        help="Define the Hanabi Game to be played"
    )


    args = parser.parse_args()


 
main(args)         
        
