import os
import numpy as np
import gin
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, AgentType
from hanabi_agents.rlax_dqn import RewardShapingParams, RewardShaper
#from hanabi_agents.rule_based import RulebasedParams, RulebasedAgent
#from hanabi_agents.rule_based.predefined_rules import piers_rules, piers_rules_adjusted
import logging
import time

from pympler.tracker import SummaryTracker

def load_agent(env):
    
    # load reward shaping infos
    reward_shaping_params = RewardShapingParams()
    if reward_shaping_params.shaper:
        reward_shaper = RewardShaper(reward_shaping_params)
    else:
        reward_shaper = None
    
    # load agent based on type
    agent_type = AgentType()
    
    if agent_type.type == 'rainbow':
        agent_params = RlaxRainbowParams()
        return DQNAgent(env.observation_spec_vec_batch()[0],
                        env.action_spec_vec(),
                        agent_params,
                        reward_shaper)
        
    elif agent_type.type == 'rulebased':        
        agent_params = RulebasedParams()
        return RulebasedAgent(agent_params.ruleset)


@gin.configurable(blacklist=['output_dir', 'self_play'])
def session(
            #agent_config_path=None,
            hanabi_game_type="Hanabi-Full",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 32,
            n_parallel_eval:int = 1_000,
            n_train_steps: int = 4,
            n_sim_steps: int = 2,
            epochs: int = 1_000_000,
            epoch_offset = 0,
            eval_freq: int = 500,
            self_play: bool = True,
            output_dir = "/output",
            start_with_weights=None,
            n_backup = 500, 
            restore_weights = None
    ):
    
    print(epochs, n_parallel, n_parallel_eval)
    #tracker = SummaryTracker()
    
    # create folder structure
    os.makedirs(os.path.join(output_dir, "weights"))
    os.makedirs(os.path.join(output_dir, "stats"))
    for i in range(n_players):
        os.makedirs(os.path.join(output_dir, "weights", "agent_" + str(i)))
    
    #logger
    logger = logging.getLogger('Training_Log')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, 'debug.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create hanabi environment configuration
    # env_conf = make_hanabi_env_config('Hanabi-Small-CardKnowledge', n_players)
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    
    if max_life_tokens is not None:
            env_conf["max_life_tokens"] = str(max_life_tokens)
    logger.info('Game Config\n' + str(env_conf))
            
    # create training and evaluation parallel environment
    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, n_parallel_eval)
            
    # get agent and reward shaping configurations
    if self_play:
        
        with gin.config_scope('agent_0'):
            
            agent = load_agent(env)
            if restore_weights is not None:
                agent.restore_weights(restore_weights, restore_weights)
            agents = [agent for _ in range(n_players)]
            logger.info("self play")
            logger.info("Agent Config\n" + str(agent))
            logger.info("Reward Shaper Config\n" + str(agent.reward_shaper))
    
    else:
        
        agents = []
        logger.info("multi play")
          
        for i in range(n_players):
            with gin.config_scope('agent_'+str(i)): 
                agent = load_agent(env)
                logger.info("Agent Config " + str(i) + " \n" + str(agent))
                logger.info("Reward Shaper Config\n" + str(agent.reward_shaper))
                agents.append(agent)
                
    # load previous weights            
    if start_with_weights is not None:
        print(start_with_weights)
        for aid, agent in enumerate(agents):
            if "agent_" + str(aid) in start_with_weights:
                agent.restore_weights(*(start_with_weights["agent_" + str(aid)]))
    
    # start parallel session for training and evaluation          
    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()
    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)
    
    print("Game config", parallel_session.parallel_env.game_config)
    
    # evaluate the performance before training
    mean_reward_prev = parallel_eval_session.run_eval().mean()
    
    # calculate warmup period
    n_warmup = int(350 * n_players / n_parallel) + n_players

    # start time
    start_time = time.time()

    # activate store_td
    for a in agents:
      if epoch_offset < 50:
        a.store_td=True
      else:
        a.store_td=False
    print('store TD', agents[0].store_td)
    
    # start training
    for epoch in range(epoch_offset+4, epochs + epoch_offset, 5):
        
        # train
        parallel_session.train(n_iter=eval_freq,
                               n_sim_steps=n_sim_steps,
                               n_train_steps=n_train_steps,
                               n_warmup=n_warmup)
        
        # no warmup after epoch 0
        n_warmup = 0
        
        # print number of train steps
        print("step", agents[0].train_step)
        #if self_play:
        #    print("step", (epoch + 1) * eval_freq * n_train_steps * n_players)
        #else:
        #    print("step", (epoch + 1) * eval_freq * n_train_steps)
        
        # evaluate
        output_path = os.path.join(output_dir, "stats", str(epoch))
        mean_reward = parallel_eval_session.run_eval(
            dest=output_path,
            store_steps=False,
            store_moves=False
            ).mean()
            
        stochasticity = agents[0].get_stochasticity()
        np.save(output_path + "_stochasticity.npy", stochasticity)

        #drawn_td = agents[0].get_drawn_tds(deactivate=False)
        #np.save(output_path + "_drawn_tds.npy", drawn_td)
        
        if (epoch +1) % 50 == 0:
            buffer_td = agents[0].get_buffer_tds()
            np.save(output_path + "_buffer_tds.npy", buffer_td)
        
        if (epoch+1) < 50:
            drawn_td = agents[0].get_drawn_tds(deactivate=False)
            np.save(output_path + "_drawn_tds.npy", drawn_td)
        elif (epoch+1) == 50:
            drawn_td = agents[0].get_drawn_tds(deactivate=True)
            np.save(output_path + "_drawn_tds.npy", drawn_td)

        # compare to previous iteration and store checkpoints
        if (epoch + 1) % n_backup == 0:
            
            print('save weights', epoch)
            only_weights = False#True if (epoch + 1) < epochs + epoch_offset else False
            
            if self_play:
                agents[0].save_weights(
                    os.path.join(output_dir, "weights", "agent_0"), 
                    "ckpt_" + str(agents[0].train_step),
                    only_weights=only_weights)
                
            else:
                for aid, agent in enumerate(agents):
                    agent.save_weights(
                        os.path.join(output_dir, "weights", "agent_" + str(aid)), 
                        "ckpt_" + str(agent.train_step),
                        only_weights=only_weights)
                    
        # store the best network
        if mean_reward_prev < mean_reward:
            
            if self_play:
                agents[0].save_weights(
                    os.path.join(output_dir, "weights", "agent_0"), "best") 
                
            else:
                for aid, agent in enumerate(agents):
                    agent.save_weights(
                        os.path.join(output_dir, "weights", "agent_" + str(aid)), "best") 
                    
            mean_reward_prev = mean_reward

        # logging
        logger.info("epoch {}: duration={}s    reward={}".format(epoch, time.time()-start_time, mean_reward))
        start_time = time.time()

        #tracker.print_diff()
        
        
def linear_schedule(val_start, val_end, n_steps):
    
    def schedule(step):
        increase = (val_end - val_start) / n_steps
        if val_end > val_start:
            return min(val_end, val_start + step * increase)
        else:
            return max(val_end, val_start + step * increase)
    
    return schedule


def exponential_schedule(val_start, val_end, decrease):
    
    def schedule(step):
        return max(val_start * (decrease**step), val_end)
    
    return schedule


def ramp_schedule(val_start, val_end, n_steps):
    
    def schedule(step):
        return val_start if step < n_steps else val_end
    
    return schedule


@gin.configurable
def schedule_beta_is(value_start, value_end, steps):
    return linear_schedule(value_start, value_end, steps)

@gin.configurable
def schedule_epsilon(value_start=1, value_end=0, steps=50*2000):
    return linear_schedule(value_start, value_end, steps)

@gin.configurable
def schedule_tau(value_start=1, value_end=0.0001, decrease=0.99995):
    return exponential_schedule(value_start, value_end, decrease)
        
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
        "--self_play", default=False, action='store_true',
        help="Whether the agent should play with itself, or an independent agent instance should be created for each player.")
    parser.add_argument(
        "--restore_weights", type=str, default=None,
        help="Path to weights of pretrained agent.")
    parser.add_argument(
        "--agent_config_path", type=str, default=None,
        help="Path to gin config file for rlax rainbow agent.")

    parser.add_argument(
        "--output_dir", type=str, default="/output",
        help="Destination for storing weights and statistics")
    parser.add_argument(
        "--start_with_weights", type=json.loads, default=None,
        help="Initialize the agents with the specified weights before training. Syntax: {\"agent_0\" : [\"path/to/weights/1\", ...], ...}")

    args = parser.parse_args()

    #main(**vars(args))  
    main(args) 
            
