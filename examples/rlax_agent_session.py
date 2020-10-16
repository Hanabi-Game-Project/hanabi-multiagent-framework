import os
import numpy as np
import gin
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams

def main(
        agent_config_path=None,
        hanabi_game_type="Hanabi-Small-Oracle",
        n_players=2,
        max_life_tokens=None,
        n_parallel=64,
        self_play=True,
        n_train_steps=5,
        n_sim_steps=2,
        epochs=1_000_000,
        eval_n_parallel=1_000,
        eval_freq=500,
        output_dir="/output",
        start_with_weights=None
    ):

    os.makedirs(os.path.join(output_dir, "weights"))
    os.makedirs(os.path.join(output_dir, "stats"))
    for i in range(n_players):
        os.makedirs(os.path.join(output_dir, "weights", "agent_" + str(i)))

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

    eps_schedule = create_exp_decay_scheduler(0.5, 0.01, 12000, 20000)
    beta_is_schedule = create_linear_scheduler(0.0, 1.0, 25e-7 / 4)

    #  env_conf = make_hanabi_env_config('Hanabi-Full-Oracle', n_players)
    #  env_conf = make_hanabi_env_config('Hanabi-Full-CardKnowledge', n_players)
    #  env_conf = make_hanabi_env_config('Hanabi-Small-Oracle', n_players)
    #  env_conf = make_hanabi_env_config('Hanabi-Small-CardKnowledge', n_players)
    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)

    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, eval_n_parallel)

    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)


    agent_params = RlaxRainbowParams(
            #  train_batch_size=512,
            #  target_update_period=500,
            #  learning_rate=2.5e-5,
            #  epsilon=lambda x: 0.2,
            #  beta_is=lambda x: 0.2,
            #  layers=[512, 512]
    )
    
    
    print('params', agent_params)
    
    if self_play:
        self_play_agent = DQNAgent(
            env.observation_spec_vec_batch()[0],
            env.action_spec_vec(),
            agent_params)

        agents = [self_play_agent for _ in range(n_players)]
    else:
        agents = [DQNAgent(env.observation_spec_vec_batch()[0], env.action_spec_vec(),
                           agent_params) for _ in range(n_players)]

    if start_with_weights is not None:
        print(start_with_weights)
        for aid, agent in enumerate(agents):
            if "agent_" + str(aid) in start_with_weights:
                agent.restore_weights(*(start_with_weights["agent_" + str(aid)]))
    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()

    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)

    print("Game config", parallel_session.parallel_env.game_config)

    # eval before
    mean_reward_prev = parallel_eval_session.run_eval().mean()
    #mean_reward = parallel_eval_session.run_eval().mean()

    # train
    parallel_session.train(
        n_iter=eval_freq,
        n_sim_steps=n_sim_steps,
        n_train_steps=n_train_steps,
        n_warmup=int(agent_params.train_batch_size * 5 * n_players / n_sim_steps / n_parallel))

    print("step", 1 * eval_freq * n_train_steps)
    # eval
    #mean_reward_prev = mean_reward
    mean_reward = parallel_eval_session.run_eval(
        dest=os.path.join(
            output_dir,
            "stats", "0")
        ).mean()
    if self_play:
        agents[0].save_weights(
            os.path.join(output_dir, "weights", "agent_0"), "ckpt_" + str(agents[0].train_step))
    else:
        for aid, agent in enumerate(agents):
            agent.save_weights(
                os.path.join(output_dir, "weights", "agent_" + str(aid)), "ckpt_" + str(agents[0].train_step))
    if mean_reward_prev < mean_reward:
        if self_play:
            agents[0].save_weights(
                os.path.join(output_dir, "weights", "agent_0"), "best")
        else:
            for aid, agent in enumerate(agents):
                agent.save_weights(
                    os.path.join(output_dir, "weights", "agent_" + str(aid)), "best")
                
        mean_reward_prev = mean_reward

    for epoch in range(epochs):
        parallel_session.train(
            n_iter=eval_freq,
            n_sim_steps=n_sim_steps,
            n_train_steps=n_train_steps,
            n_warmup=0)
        print("step", (epoch + 2) * eval_freq * n_train_steps)
        # eval after
        #mean_reward_prev = mean_reward
        mean_reward = parallel_eval_session.run_eval(
            dest=os.path.join(
                output_dir,
                "stats", str(epoch + 1))
            ).mean()
            
        # TODO check how it is done
        #if epoch % (100_000 // eval_freq) == 0:
        if self_play:
            agents[0].save_weights(
                os.path.join(output_dir, "weights", "agent_0"), "ckpt_" + str(agents[0].train_step))
        else:
            for aid, agent in enumerate(agents):
                agent.save_weights(
                    os.path.join(output_dir, "weights", "agent_" + str(aid)), "ckpt_" + str(agents[0].train_step))
        
        if mean_reward_prev < mean_reward:
            if self_play:
                agents[0].save_weights(
                    os.path.join(output_dir, "weights", "agent_0"), "best")
            else:
                for aid, agent in enumerate(agents):
                    agent.save_weights(
                        os.path.join(output_dir, "weights", "agent_" + str(aid)), "best")
                    
            mean_reward_prev = mean_reward

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Train a dm-rlax based rainbow agent.")

    parser.add_argument(
        "--hanabi_game_type", type=str, default="Hanabi-Small-Oracle",
        help='Can be "Hanabi-{VerySmall,Small,Full}-{Oracle,CardKnowledge}"')
    parser.add_argument("--n_players", type=int, default=2, help="Number of players.")
    parser.add_argument(
        "--max_life_tokens", type=int, default=None,
        help="Set a different number of life tokens.")
    parser.add_argument(
        "--n_parallel", type=int, default=32,
        help="Number of games run in parallel during training.")
    parser.add_argument(
        "--self_play", default=False, action='store_true',
        help="Whether the agent should play with itself, or an independent agent instance should be created for each player.")
    parser.add_argument(
        "--n_train_steps", type=int, default=4,
        help="Number of training steps made in each iteration. One iteration consists of n_sim_steps followed by n_train_steps.")
    parser.add_argument(
        "--n_sim_steps", type=int, default=2,
        help="Number of environment steps made in each iteration.")
    parser.add_argument(
        "--epochs", type=int, default=1_000_000,
        help="Total number of rotations = epochs * eval_freq.")
    parser.add_argument(
        "--eval_n_parallel", type=int, default=1_000,
        help="Number of parallel games to use for evaluation.")
    parser.add_argument(
        "--eval_freq", type=int, default=500,
        help="Number of iterations to perform between evaluations.")

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

    main(**vars(args))
