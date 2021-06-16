import os
import numpy as np
import gin
import logging
import time
import shutil
import random
import math
import jax.numpy as jnp

import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, PBTParams, DQNParallel
from hanabi_agents.rlax_dqn import RewardShapingParams
from hanabi_agents.pbt import AgentDQNPopulation
from hanabi_multiagent_framework.utils import eval_pretty_print


import haiku as hk
from haiku._src.data_structures import to_mutable_dict, FlatMapping
import pickle

import math
import faulthandler
import shutil

"""
This is an example on how to run the PBT approach for training on DQN/Rainbow agents --> One agent interoperating with
environment and distributing/merging obtained observations/actions to the actual agents.
"""

# @gin.configurable(blacklist=['output_dir', 'self_play'])
@gin.configurable(blacklist=[ 'self_play'])
def session(
            self_play: bool = True,
            agent_config_path=None,
            output_dir = "./output",
            hanabi_game_type="Hanabi-Small",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 400,  ## has to be multiple of population_size
            n_parallel_eval:int = 2000, ## has to be multiple of population_size
            n_train_steps: int = 4,
            n_sim_steps: int = 2,
            epochs: int = 3500,
            eval_freq: int = 500,
            db_path: str = None,
            restore_weights: str = None, 
            restore_state: str = None, 
            restore_parameters: str = None,
            one_agent_only: bool = False
    ):
    max_score = 15
    saver_epochs = [5, 10, 25, 50, 100, 200, 500, 1000]

    def load_agent(env):
        '''load agent with parameters from gin config -- self-play for now'''
        def sample_buffersize(pbt_params, agent_params):
            exp_factor = math.log(agent_params.experience_buffer_size, 2)
            buffer_sizes_start = [2**i for i in range(int(exp_factor) - pbt_params.buffersize_start_factor, \
                                                        int(exp_factor) + pbt_params.buffersize_start_factor)]                                     
            return random.choice(buffer_sizes_start)
            
        def sample_init_lr(pbt_params):
            return random.choice(np.linspace(pbt_params.lr_min, pbt_params.lr_max, pbt_params.lr_sample_size))

        def sample_init_alpha(pbt_params):
            return random.choice(np.linspace(pbt_params.alpha_min, pbt_params.alpha_max, pbt_params.alpha_sample_size))

        with gin.config_scope('agent_0'):
            agent_params = RlaxRainbowParams()
            reward_shaping_params = RewardShapingParams()
            pbt_params = PBTParams()

        return DQNParallel(
                        env.observation_spec_vec_batch()[0],
                        env.action_spec_vec(),
                        [sample_buffersize(population_params, agent_params) for n in range(pbt_params.population_size)],
                        [sample_init_lr(population_params) for n in range(pbt_params.population_size)],
                        [sample_init_alpha(population_params) for n in range(pbt_params.population_size)],
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

    # fullfill PBT logic after N epochs
    def evolution():

        def evolve_x():
            pass
        pass

    def multiplace_agent(weights):
        weights_dict = to_mutable_dict(weights)
        objective = {}

        for key in weights_dict.keys():
            intermediate_dict = {}
            for sub_key in weights_dict[key].keys():
                # objective[key][sub_key] = []
                device_array = jnp.array([np.array(weights[key][sub_key]) for n in range(population_params.population_size)])
                # objective[key][sub_key] = device_array
                intermediate_dict[sub_key] = device_array
            objective[key] = FlatMapping(intermediate_dict)

        return FlatMapping(objective)
 
    def define_couples(index_loser, index_survivor):
        a = np.arange(len(index_survivor)).reshape(-1, 1)
        b = np.random.choice(len(index_survivor), len(index_survivor), replace = True).reshape(-1, 1)
        return np.hstack([a, b])

    def perform_pbt(agent, hyperparams, couples, pbt_params, index_loser, index_survivor):
        weights = agent.trg_params
        opt_states = agent.opt_state[0]


        weights_dict = to_mutable_dict(weights)
    #     opt_states_dict = to_mutable_dict(opt_states)
        objective_weights = {}

        # print(weights_dict['noisy_mlp/~/noisy_linear_0']['b_mu'])
        for key in weights_dict.keys():
            sub_dict = weights_dict[key]
            intermediate_dict = {}
            for sub_key in sub_dict.keys():

                root_array = np.array(sub_dict[sub_key])
                # print('original array is', type(root_array), root_array)
                for pair in couples:
                    # print('changing these two')
                    # print(root_array[index_survivor[pair[1]]], '>>>>>>>>>>>>')
                    root_array[index_loser[pair[0]]] = root_array[index_survivor[pair[1]]]
                device_array = jnp.array(root_array)

                
                # print('new device array is', device_array)
                # time.sleep(10)
                # objective[key][sub_key] = device_array
                intermediate_dict[sub_key] = device_array
            objective_weights[key] = FlatMapping(intermediate_dict)

        # print(objective_weights['noisy_mlp/~/noisy_linear_0']['b_mu'])

        for field in opt_states._fields:
            # print('>>>>>>>>.', opt_states)
            # print(field)
            # print(type(getattr(opt_states, field)))
            if str(type(getattr(opt_states, field))) == "<class 'haiku._src.data_structures.FlatMapping'>":
                # print('FlatMapping here we are')
    #             print(getattr(opt_states, field))
                obj = to_mutable_dict(getattr(opt_states, field))
                
                for key in obj.keys():
                    intermediate_dict = {}
                    for sub_key in obj[key].keys():
                        array = np.array(obj[key][sub_key])
                        for pair in couples:
                            array[index_loser[pair[0]]] = array[index_survivor[pair[1]]]
                        device_array = jnp.array(array)
                        # objective[key][sub_key] = device_array
                        intermediate_dict[sub_key] = device_array
                    obj[key] = FlatMapping(intermediate_dict)
                # print(str(field))
                opt_states = opt_states._replace(**{field  : FlatMapping(obj)})
    #             setattr(opt_states, field, FlatMapping(obj)) 
                # print('<<<<<<<<<<<<<,,,,,', opt_states)

        for pair in couples:
            if 'lr' in hyperparams:
                lrs = agent.lr
                # print('lrs before >>>>>>>', lrs)
                choices = [lrs[index_survivor[pair[1]]] * pbt_params.lr_factor, lrs[index_survivor[pair[1]]], lrs[index_survivor[pair[1]]]/pbt_params.lr_factor]
                lrs[index_loser[pair[0]]] = random.choice(choices)
                # print('lrs after >>>>>>>>', lrs)
                agent.lr = lrs

            if 'buffersize' in hyperparams:
                buffersizes = agent.buffersizes
                # print('buffer before >>>>>', buffersizes)
                choices = [buffersizes[index_survivor[pair[1]]] * pbt_params.buffersize_factor, buffersizes[index_survivor[pair[1]]], buffersizes[index_survivor[pair[1]]]/pbt_params.buffersize_factor]
                choice = int(random.choice(choices))
                if choice <= 512:
                    choice = 512
                elif choice >= 2**20:
                    choice = 2**20
                buffersizes[index_loser[pair[0]]] = choice 
                agent.buffersizes = buffersizes
                agent.experience[index_loser[pair[0]]].change_size(buffersizes[index_survivor[pair[1]]])
                # print('buffer after >>>>>>', buffersizes)

            if 'alpha' in hyperparams:
                alphas = agent.alphas
                # print('alphas before >>>>>', alphas)
                choices = [alphas[index_survivor[pair[1]]] * pbt_params.factor_alpha, alphas[index_survivor[pair[1]]], alphas[index_survivor[pair[1]]]/pbt_params.factor_alpha]
                alphas[pair[0]] = random.choice(choices)
                agent.alphas = alphas
                agent.experience[index_loser[pair[0]]].alpha = alphas[index_survivor[pair[1]]]
                # print('alphas after >>>>>', alphas)

        return FlatMapping(objective_weights), [opt_states]

    def restore_param(agent, res_params, agent_params):

        agent.lr = np.asarray(res_params['lr'])
        agent.buffersizes = res_params['buffersize']

        if agent_params.use_priority == True:
            
            
            agent.alphas = res_params['alphas']

            for i, buffer in enumerate(agent.experience):
                buffer.alpha = agent.alphas[i]
        
        for i, buffer in enumerate(agent.experience):
            buffer.change_size(agent.buffersizes[i])
        
        return agent
    
    def save_intermediate(output_dir, agent, epoch, epoch_alive, mean_reward_max, agent_params, keep_records = True):
        
        int_dir = output_dir
        
        if os.path.isdir(int_dir):
            if keep_records is not True:
                for file in os.listdir(int_dir):
                    os.remove(os.path.join(int_dir, file))
        else:
            os.makedirs(int_dir)

        #save weights
        agent.save_weights(int_dir, 'ckpt_at_epoch_{}'.format(epoch))
        agent.save_states(int_dir, 'states_at_epoch_{}'.format(epoch))

        parameters = {'lr': agent.lrs, 'buffersize' : agent.buffersizes, 'alphas' : agent.alphas, 'epoch': epoch, 'epoch_alive' : epoch_alive, 'mean_reward_max' : mean_reward_max}         
        with open(os.path.join(int_dir, "rlax_rainbow_" + 'parameters' + ".pkl"), 'wb') as of:
            pickle.dump(parameters, of)


    #load gin-config
    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)

    #load gin-config with special focus for self-play for now
    with gin.config_scope('agent_0'):
        population_params = PBTParams()
        agent_params = RlaxRainbowParams()

    #load necessary params for the PBT-algorithm 
    population_size = int(population_params.population_size) 
    pbt_epochs = population_params.life_span
    epochs_alive = np.ones(population_size) + population_params.life_span
    epoch_ = 0
    mean_reward_max = 9

    # make directories etc. just during very first generation
    os.makedirs(output_dir)

    #environment configurations & initialization
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)
    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)


    #initialize agents before starting generation
    if self_play:
        if restore_weights == None:
            os.makedirs(os.path.join(output_dir, 'weights'))
        with gin.config_scope('agent_0'):
            self_play_agent = load_agent(env)

            if restore_weights != None:
                with open(restore_weights, 'rb') as iwf:
                    weights = pickle.load(iwf)
                with open(restore_state, 'rb') as iwf:
                    states = pickle.load(iwf)
                with open(restore_parameters, 'rb') as iwf:
                    restored_parameters = pickle.load(iwf)
                
                if one_agent_only == True:
                    print('ERROR')
                    new_weights = multiplace_agent(weights)
                    opt_states = multiplace_agent(states) ### needs alteration because of states different construction
                else:
                    new_weights = weights
                    opt_states = states
                # print(opt_states)
                # time.sleep(15)
                print(self_play_agent.lr, type(self_play_agent.lr))
                
                self_play_agent.trg_params = new_weights
                self_play_agent.online_params = new_weights
                self_play_agent.opt_state = states
                self_play_agent = restore_param(self_play_agent, restored_parameters, agent_params)
                epochs_alive = restored_parameters['epoch_alive']
                epoch_ = restored_parameters['epoch']
                if restored_parameters['mean_reward_max'] >= 9:
                    mean_reward_max = restored_parameters['mean_reward_max']
                print(self_play_agent.lr, type(self_play_agent.lr))
                

                # print(new_weights)
                # time.sleep(10)

                # print(self_play_agent.lr)
                # time.sleep(10)
            # self_play_agent.pbt_counter = pbt_counter
            agents = [self_play_agent for _ in range(n_players)]
    else:
        raise NotImplementedError()

    #load one agent for all agents
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

    # for generation in generations:
        # train


    for epoch in range(epochs):
        
        print('EPOCH {}'.format(epoch_), epochs)
        start_time = time.time()

        parallel_session.train(
            n_iter=eval_freq,
            n_sim_steps=n_sim_steps,
            n_train_steps=n_train_steps,
            n_warmup=0,
            optimize_for_parallel= True)
        # print("step", (epoch_circle * pbt_epochs + (epoch + 2)) * eval_freq * n_train_steps)
        
        # eval after train-step
        mean_reward_prev = add_reward(mean_reward_prev, mean_reward)
        total_reward = parallel_eval_session.run_eval(
            dest=os.path.join(output_dir, ("stats_epoch_" + str(epoch_ + 1))))
            
        mean_reward = split_evaluation(total_reward, population_size, mean_reward_prev)

        print('Epoch {} took {:.2f} seconds!'.format((epoch_), time.time() - start_time))
      
        epochs_alive += 1
        print(epochs_alive)

        if self_play:

            if epoch_ in saver_epochs:
                agents[0].save_weights(os.path.join(output_dir, 'weights'), 'Rainbow_diversity_at_{}_epochs'.format(epoch_))

        if np.max(mean_reward) > mean_reward_max:
            # save best agent
            save_intermediate(os.path.join(output_dir, 'best_performing_agents'), agents[0], epoch_, epochs_alive, mean_reward_max, agent_params)   
            mean_reward_max = np.max(mean_reward)

        if epoch_ % population_params.generations == 0 and epoch_ > 0:
            
            print('>>>>>>>>>>>>>>>>>>>>>>. PBT <<<<<<<<<<<<<<<<<<<<')

            def mutual_information(self, actions_agent_a, actions_agent_b):
                '''Compares both vectors by calculating the mutual information'''
                return 1 - metrics.normalized_mutual_info_score(actions_agent_a, actions_agent_b)

            def simple_match(actions_agent_a, actions_agent_b):
                '''Compares both action vectors and calculates the number of matching actions'''
                return (1 - np.sum(actions_agent_a == actions_agent_b)/actions_agent_a.shape[0])

            def cosine_distance(actions_agent_a, actions_agent_b):
                pass

            # 1. run_eval to generate observations -- done in rlax_agent
            obs_diversity = np.vstack([agents[0].past_obs for i in range(population_params.population_size)])
            lms = np.vstack([agents[0].past_lms for i in range(population_params.population_size)])
            # print(obs_diversity.shape)
            # 2. feed to current agents to get action vectors and diversity -- done
                    # 2b. generate action vectors from path agents

            action_vecs = agents[0].exploit([[], [obs_diversity, lms]], eval = True).reshape(population_params.population_size, -1)

            # generate diversity matrix
            diversity_matrix = [[] for i in range(action_vecs.shape[0])]
            for i, elem in enumerate(action_vecs):
                for vec in action_vecs:
                    mut_info = simple_match(elem, vec)
                    diversity_matrix[i].append(mut_info)
            diversity_matrix = np.asarray(diversity_matrix)
            np.fill_diagonal(diversity_matrix, 1)

            # retrieve min values
            div_minima = np.min(diversity_matrix, axis = 1)

            # combine diversities with mean_rewards
            # pbt_score = mean_reward +  1/(1 + np.exp(-(mean_reward - 2/3*max_score))) * population_params.w_diversity * div_minima
            pbt_score = mean_reward
            readiness = np.array(epochs_alive >= population_params.life_span)
            helper_index = np.array(np.where(readiness))[0]
            no_agents_pbt = np.sum(readiness) - int(np.sum(readiness) * population_params.discard_percent)

            test1 = np.argpartition(pbt_score[readiness], no_agents_pbt)
            test = helper_index[np.argpartition(pbt_score[readiness], no_agents_pbt)]

            index_loser = helper_index[np.argpartition(pbt_score[readiness], no_agents_pbt)[:no_agents_pbt]]
            index_survivor = helper_index[np.argpartition(-pbt_score[readiness], no_agents_pbt)[:no_agents_pbt]]
            couples = define_couples(index_loser, index_survivor)
            print(index_loser, 'losers')
            print(index_survivor, 'survivors')
            print('These are the agents to be exchanged (loser/winner) {}'.format(couples))


            epochs_alive[index_loser] = 0
            new_weights, new_states = perform_pbt(agents[0], ['lr', 'alpha', 'buffersize'], couples, population_params, index_loser, index_survivor)
            # print(agents[0].trg_params)

            agents[0].trg_params = new_weights
            agents[0].online_params = new_weights
            agents[0].opt_state = new_states
            # print('>>>>>>>>>>>>>>>>>.')
            # print(agents[0].trg_params)



            # parallel_session = hmf.HanabiParallelSession(env, agents)
            # parallel_session.reset()
            # parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)  
            # parallel_eval_session.reset()

            print('EVAL AFTER PBT')
            reward_pbt_test = parallel_eval_session.run_eval()
            split_evaluation(reward_pbt_test, population_size, mean_reward_prev)

            
         
        save_intermediate(os.path.join(output_dir, 'restore'), agents[0], epoch_, epochs_alive, mean_reward_max, agent_params, keep_records = False)   
        epoch_ += 1
            # 3. perform PBT
            # 3a. weights overwriting -- done
            # 3b. buffer switch etc.
            # 3bb. sample new parameters
            # 3c. save couples that switched -- workaround done
            


    mean_reward_prev = add_reward(mean_reward_prev, mean_reward)
    
            
if __name__ == "__main__":
    import argparse
    import json
    faulthandler.enable()
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
        "--restore_state", type=str, default=None,
        help="Path pickle file with agent optimizer state"
    )
    parser.add_argument(
        "--restore_parameters", type=str, default=None,
        help="Path pickle file with agent parameters from before"
    )
    parser.add_argument(
        "--one_agent_only", type=bool, default=False,
        help="Restore a copy of a single agent multiple times or many agents"
    )
    
    # parser.add_argument(
    #     "--num_gpus", type=int, default=1,
    #     help="Define the Machine to run on by installed GPUs"
    # )
    # parser.add_argument(
    #     "--which_gpu", type=int, default=0,
    #     help="Define the GPU if more than 1 available"
    # )
    parser.add_argument(
        "--hanabi_game_type", type=str, default='Hanabi-Small',
        help="Define the Hanabi Game to be played"
    )


    args = parser.parse_args()

    # for use on local machine to not overload GPU-memory given jax default setting to occupy 90% of total GPU-Memory
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
    # setting only necessary for NI server where cuda is installed via conda-env
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/mnt/antares_raid/home/maltes/miniconda/envs/RL"

 
    session(**vars(args))     

