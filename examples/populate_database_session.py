import os
import numpy as np
import gin
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams
from hanabi_agents.pbt import DataGenerationAgent





def main(
        agent_config_path=None,
        hanabi_game_type="Hanabi-Small",
        n_players=2,
        max_life_tokens=None,
        n_parallel=64,
        self_play=True,
        n_train_steps=4,
        n_sim_steps=2,
        epochs=1_0,
        eval_n_parallel=1_000,
        eval_freq=500,
        output_dir="/output",
        start_with_weights=None
    ):

    env_conf = make_hanabi_env_config('Hanabi-Small-CardKnowledge', n_players)

    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)

    eval_env = hmf.HanabiParallelEnvironment(env_conf, eval_n_parallel)

    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)


    agent_params = RlaxRainbowParams()
    
    print('params', agent_params)
    
    if self_play:
        self_play_agent = DataGenerationAgent(
            eval_n_parallel,
            eval_env.observation_spec_vec_batch()[0],
            eval_env.action_spec_vec(),
            agent_params)

        agents = [self_play_agent for _ in range(n_players)]
    # else:
    #     agents = [DQNAgent(eval_env.observation_spec_vec_batch()[0], eval_env.action_spec_vec(),
    #                        agent_params) for _ in range(n_players)]

    if start_with_weights is not None:
        # print(start_with_weights)
        agents[0].restore_weights(start_with_weights)

    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)

    print("Game config", parallel_eval_session.parallel_env.game_config)

    # eval before
    mean_reward_prev = parallel_eval_session.run_eval()

    # eval
    #mean_reward_prev = mean_reward
    mean_reward = parallel_eval_session.run_eval(
        dest=os.path.join(
            output_dir,
            "stats", "0")
        ).mean()


    for epoch in range(epochs):
       
        print("step", (epoch + 2) * eval_freq * n_train_steps)
        # eval after

        mean_reward = parallel_eval_session.run_eval(
            dest=os.path.join(
                output_dir,
                "stats", str(epoch + 1))
            ).mean()
            


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
        "--self_play", default=True, action='store_true',
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
        "--output_dir", type=str, default="./output",
        help="Destination for storing weights and statistics")

    parser.add_argument(
        "--start_with_weights", type=str, default=None,
        help="Initialize the agents with the specified weights before training. Syntax: {\"agent_0\" : [\"path/to/weights/1\", ...], ...}")

    args = parser.parse_args()

    main(**vars(args))
