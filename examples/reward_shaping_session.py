import os
import numpy as np
import gin
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams
from hanabi_agents.rlax_dqn import RewardShapingParams, RewardShaper
import logging
import time


def load_agent(env):
    
    reward_shaping_params = RewardShapingParams()
    reward_shaper = RewardShaper(reward_shaping_params)
    
    agent_params = RlaxRainbowParams()
    return DQNAgent(env.observation_spec_vec_batch()[0],
                    env.action_spec_vec(),
                    agent_params,
                    reward_shaper)


@gin.configurable(blacklist=['output_dir', 'self_play'])
def session(
            #agent_config_path=None,
            hanabi_game_type="Hanabi-Small",
            n_players: int = 2,
            max_life_tokens: int = None,
            n_parallel: int = 32,
            n_parallel_eval:int = 1_000,
            n_train_steps: int = 4,
            n_sim_steps: int = 2,
            epochs: int = 1_000_000,
            eval_freq: int = 500,
            self_play: bool = True,
            output_dir = "/output",
            n_backup = 500
    ):
    
    print(epochs, n_parallel, n_parallel_eval)
    
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
            agents = [agent for _ in range(n_players)]
            logger.info("self play")
            logger.info("Agent Config\n" + str(agent))
    
    else:
        
        agents = []
          
        for i in range(n_players):
            with gin.config_scope('agent_'+str(i)): 
                
                agent = load_agent(env)
                agents.append(agent)
    
    # start parallel session for training and evaluation          
    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()
    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)
    
    print("Game config", parallel_session.parallel_env.game_config)
    
    # evaluate the performance before training
    mean_reward_prev = parallel_eval_session.run_eval().mean()
    
    # calculate warmup period
    n_warmup = int(350 * n_players / n_parallel)

    # start time
    start_time = time.time()
    
    # start training
    for epoch in range(epochs):
        
        # train
        parallel_session.train(n_iter=eval_freq,
                               n_sim_steps=n_sim_steps,
                               n_train_steps=n_train_steps,
                               n_warmup=n_warmup)
        
        # no warmup after epoch 0
        n_warmup = 0
        
        # print number of train steps
        print("step", (epoch + 1) * eval_freq * n_train_steps)
        
        # evaluate
        mean_reward = parallel_eval_session.run_eval(
            dest=os.path.join(output_dir, "stats", str(epoch))
            ).mean()

        # compare to previous iteration and store checkpoints
        if (epoch + 1) % n_backup == 0:
            
            if self_play:
                agents[0].save_weights(
                    os.path.join(output_dir, "weights", "agent_0"), 
                    "ckpt_" + str(agents[0].train_step))
                
            else:
                for aid, agent in enumerate(agents):
                    agent.save_weights(
                        os.path.join(output_dir, "weights", "agent_" + str(aid)), 
                        "ckpt_" + str(agents[0].train_step))
                    
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
        
        
def linear_schedule(val_start, val_end, n_steps):
    
    def schedule(step):
        increase = (val_end - val_start) / n_steps
        return min(val_end, val_start + step * increase)
    
    return schedule


def ramp_schedule(val_start, val_end, n_steps):
    
    def schedule(step):
        return val_start if step < n_steps else val_end
    
    return schedule


@gin.configurable
def schedule_beta_is(value_start, value_end, steps):
    return linear_schedule(value_start, value_end, steps)


@gin.configurable
def schedule_risk_penalty(value_start, value_end, steps):
    return ramp_schedule(value_start, value_end, steps)
     
        
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

#     parser.add_argument(
#         "--hanabi_game_type", type=str, default="Hanabi-Small-Oracle",
#         help='Can be "Hanabi-{VerySmall,Small,Full}-{Oracle,CardKnowledge}"')
#     parser.add_argument("--n_players", type=int, default=2, help="Number of players.")
#     parser.add_argument(
#         "--max_life_tokens", type=int, default=None,
#         help="Set a different number of life tokens.")
# #     parser.add_argument(
# #         "--n_parallel", type=int, default=32,
# #         help="Number of games run in parallel during training.")
    parser.add_argument(
        "--self_play", default=False, action='store_true',
        help="Whether the agent should play with itself, or an independent agent instance should be created for each player.")
#     parser.add_argument(
#         "--n_train_steps", type=int, default=4,
#         help="Number of training steps made in each iteration. One iteration consists of n_sim_steps followed by n_train_steps.")
#     parser.add_argument(
#         "--n_sim_steps", type=int, default=2,
#         help="Number of environment steps made in each iteration.")
#     parser.add_argument(
#         "--epochs", type=int, default=1_000_000,
#         help="Total number of rotations = epochs * eval_freq.")
# #     parser.add_argument(
# #         "--eval_n_parallel", type=int, default=1_000,
# #         help="Number of parallel games to use for evaluation.")
#     parser.add_argument(
#         "--eval_freq", type=int, default=500,
#         help="Number of iterations to perform between evaluations.")
    parser.add_argument(
        "--agent_config_path", type=str, default=None,
        help="Path to gin config file for rlax rainbow agent.")

    parser.add_argument(
        "--output_dir", type=str, default="/output",
        help="Destination for storing weights and statistics")

    args = parser.parse_args()

    #main(**vars(args))  
    main(args)         
            
            