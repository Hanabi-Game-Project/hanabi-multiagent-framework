import numpy as np
import gin
import hanabi_multiagent_framework as hmf
from hanabi_multiagent_framework.utils import make_hanabi_env_config
from hanabi_agents.neuroevo import NeuroEvoPopulation, NeuroEvoParams, Mutation, Crossover

def main(
        agent_config_path=None,
        hanabi_game_type="Hanabi-Small-Oracle",
        n_players=2,
        max_life_tokens=None,
        n_parallel=10000,
        self_play=True,
        n_train_steps=1,
        n_sim_steps=20,
        epochs=1_000_000,
        eval_n_parallel=1_000,
        eval_freq=500,
    ):


    env_conf = make_hanabi_env_config(hanabi_game_type, n_players)
    if max_life_tokens is not None:
        env_conf["max_life_tokens"] = str(max_life_tokens)

    env = hmf.HanabiParallelEnvironment(env_conf, n_parallel)
    eval_env = hmf.HanabiParallelEnvironment(env_conf, eval_n_parallel)

    if agent_config_path is not None:
        gin.parse_config_file(agent_config_path)
    agent_params = NeuroEvoParams(
        population_size=100,
        chromosome_init_layers=[16, 16],
        chromosome_n_seeds=1000,
        crossover_attempts=1,
        extinction_period=1,
        n_survivors=10
    )

    def fitness_func(observations, actions, rewards):
        return rewards

    mutation = Mutation(
        seed_mutation_proba=0.001,
        layer_size_mutation_proba=0.1,
        layer_number_mutation_proba=0.01
    )

    crossover = Crossover()

    if self_play:
        self_play_agent = NeuroEvoPopulation(
            env.observation_spec_vec_batch()[0],
            env.action_spec_vec(),
            fitness_func,
            mutation,
            crossover,
            agent_params)

        agents = [self_play_agent for _ in range(n_players)]
    else:
        agents = [
            NeuroEvoPopulation(
                env.observation_spec_vec()[0],
                env.action_spec(),
                fitness_func,
                mutation,
                crossover,
                agent_params)
            for _ in range(n_players)
        ]

    parallel_session = hmf.HanabiParallelSession(env, agents)
    parallel_session.reset()

    parallel_eval_session = hmf.HanabiParallelSession(eval_env, agents)

    print("Game config", parallel_session.parallel_env.game_config)

    # eval before
    parallel_eval_session.run_eval()

    for i in range(epochs):
        parallel_session.train(
            n_iter=eval_freq,
            n_sim_steps=n_sim_steps,
            n_train_steps=n_train_steps,
            n_warmup=0)
        print("step", (i + 1) * eval_freq * n_train_steps)
        # eval after
        parallel_eval_session.run_eval()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a neuroevolutionary population of agents.")

    parser.add_argument(
        "--hanabi_game_type", type=str, default="Hanabi-Small-Oracle",
        help='Can be "Hanabi-{VerySmall,Small,Full}-{Oracle,CardKnowledge}"')
    parser.add_argument("--n_players", type=int, default=2, help="Number of players.")
    parser.add_argument(
        "--max_life_tokens", type=int, default=None,
        help="Set a different number of life tokens.")
    parser.add_argument(
        "--n_parallel", type=int, default=10000,
        help="Number of games run in parallel during training.")
    parser.add_argument(
        "--self_play", type=bool, default=False,
        help="Whether the agent should play with itself, or an independent agent instance should be created for each player.")
    parser.add_argument(
        "--n_train_steps", type=int, default=1,
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
        help="Path to gin config file for neuroevolutionary population.")

    args = parser.parse_args()

    main(**vars(args))
