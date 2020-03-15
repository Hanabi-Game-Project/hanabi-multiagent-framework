"""
This file defines a class for managing parallel games of hanabi
"""
from sys import getsizeof
import multiprocessing as mp
from collections import namedtuple

from hanabi_multiagent_framework.game_manager import HanabiGameManager

AgentInfo = namedtuple("AgentConfig", "consumes_round_observations")
class HanabiParallelSession:
    """
    A class for instantiating and running parallel game sessions
    """
    def __init__(self, agents, agent_infos, env_configs):
        self.agents = agents
        self.agent_pipes = [[] for _ in agents]
        self.player_pipes = [[] for _ in env_configs]
        for e_id in range(len(env_configs)):
            for a_id in range(len(agents)):
                agent_end, player_end = mp.Pipe()
                self.agent_pipes[a_id].append(agent_end)
                self.player_pipes[e_id].append(player_end)

        self.env_configs = env_configs
        self.agent_infos = agent_infos
        #  self.game_managers = [HanabiGameManager(env_conf, pipes, agent_configs)
        #                        for env_conf, pipes in zip(env_configs, player_pipes)]

    @staticmethod
    def _run_n_games(task):
        """Run hanabi games one after another until the target number of games is reached.

        Args:
        task (tuple) -- combines the following:
            env_conf               (dict) -- environment config.
            agent_infos (list(AgentInfo)) -- information for each agent.
            pipes         (list(mp.Pipe)) -- pipes for communication between agents and players.
            target_n_steps          (int) -- target number of steps.
        """
        print("Starting parallel game")
        env_conf, agent_infos, pipes, target_n_games = task
        game_man = HanabiGameManager(env_conf, agent_infos, pipes)
        n_games = 0
        while n_games < target_n_games:
            game_man.reset_game()
            game_man.play_game()
            n_games += 1
        #  return game_man.collect_training_data()
        return n_games

    @staticmethod
    def _run_n_steps(task):
        """Run hanabi games one after another until the target number of steps over all games
        is reached.

        Args:
        task (tuple) -- combines the following:
            env_conf               (dict) -- environment config.
            agent_infos (list(AgentInfo)) -- information for each agent.
            pipes         (list(mp.Pipe)) -- pipes for communication between agents and players.
            target_n_steps          (int) -- target number of steps.
        """
        env_conf, agent_infos, pipes, target_n_steps = task
        game_man = HanabiGameManager(env_conf, agent_infos, pipes)
        n_steps = 0
        while n_steps < target_n_steps:
            game_man.reset_game()
            game_man.play_game()
            n_steps += 1
        #  return game_man.collect_training_data()
        return n_steps

    #  @staticmethod
    #  def run_game_series(objective, game_man):
    #      """Run hanabi games one after another until an objective (e.g. a total number of steps)
    #      is reached.
    #
    #      Args:
    #          objective (dict) -- a dictionary with ONE of the following keys defined:
    #                              n_steps, n_games.
    #      """
    #      result = {}
    #      if not isinstance(objective, dict):
    #          raise TypeError(f"Objective has to be a dictionary, "
    #                          "but actually of type <{type(objective)}>.")
    #      if "n_steps" in objective:
    #          result["n_steps"] = 0
    #          while result["n_steps"] < objective["n_steps"]:
    #              result["n_step"] += game_man.play_game()
    #              game_man.reset_game()
    #      elif "n_games" in objective:
    #          result["n_games"] = 0
    #          while result["n_games"] < objective["n_games"]:
    #              game_man.play_game()
    #              result["n_games"] += 1
    #              game_man.reset()
    #      else:
    #          raise ValueError(f"Objective has neither n_steps nor n_games keys defined.")

    def consume_pipes(self, n_players):
        print("Consuming...")
        consume_counter = 0
        player_finished_counter = 0
        while True:
            for agent, pipes in zip(self.agents, self.agent_pipes):
                for pipe in pipes:
                    # receive an observation from the player and send an action back.
                    if pipe.poll():
                        (obs, reward), legal_actions = pipe.recv()
                        # abort if received None
                        if isinstance(obs, str) and obs == "Done":
                            #  print("Got 'Done' from player. Player finished all games")
                            player_finished_counter += 1
                            print(f"Players finished: {player_finished_counter}. Players total: {n_players}")
                            if player_finished_counter >= n_players:
                                print("All players finished. Stop consuming")
                                return
                        else:
                            pipe.send(agent.step(obs, reward, legal_actions))
                            consume_counter += 1
                        #  yield
                        #  print("Consumed observation")
                    #  else:
                    #      print("No observations")
        print(f"Consumed {consume_counter} observations")

    @staticmethod
    def agent_consume_pipes(agent, pipes):
        #  agent, pipes = task
        print(f"Agent consuming...")
        games_finished_counter = 0
        #  agent = self.agents[agent_id]
        while True:
            for pipe in pipes:
                (obs, reward), legal_actions = pipe.recv()
                # abort if received "Done"
                print("Agent received observation")
                if isinstance(obs, str) and obs == "Done":
                    #  print("Got 'Done' from player. Player finished all games")
                    games_finished_counter += 1
                    print(f"Agent: Games finished: {games_finished_counter}. Games total: {len(pipes)}")
                    if games_finished_counter >= len(pipes):
                        print("All players finished. Stop consuming")
                        return
                else:
                    pipe.send(agent.step(obs, reward, legal_actions))
                    #  consume_counter += 1


    def run_parallel(self, objective):
        """Run hanabi games one after another until an objective (e.g. a total number of steps)
        is reached.

        Args:
            objective (dict) -- a dictionary with ONE of the following keys defined:
                                n_steps, n_games.
        """

        if not isinstance(objective, dict):
            raise TypeError(f"Objective has to be a dictionary, "
                            "but actually of type <{type(objective)}>.")
        if "n_steps" in objective:
            target = objective["n_steps"]
            run_games_func = self._run_n_steps
        elif "n_games" in objective:
            target = objective["n_games"]
            run_games_func = self._run_n_games
        else:
            raise ValueError(f"Objective has neither n_steps nor n_games keys defined.")

        self.agent_processes = [mp.Process(target=self.agent_consume_pipes, args=(agent, pipes))
            for agent, pipes in zip(self.agents, self.agent_pipes)]
        print("Dispatching agents")
        for ap in self.agent_processes:
            ap.start()
        # distribute tasks
        local_targets = [target // len(self.env_configs) for _ in self.env_configs]
        local_targets[0] += target - local_targets[0] * len(self.env_configs)
        with mp.Pool(len(self.env_configs)) as pool:
            print("Dispatching games")
            #  res = pool.apply_async(self.consume_pipes, (sum([env_conf["players"] for env_conf in self.env_configs]),))
            tasks = zip(self.env_configs, [self.agent_infos for _ in local_targets],
                        self.player_pipes, local_targets)
            res_game = pool.map_async(run_games_func, tasks)
            res = pool.map(
                    self.agent_consume_pipes,
                    ((agent, pipes) for agent, pipes in zip(self.agents, self.agent_pipes)))

            #  for env_config, pipes, trg in zip(self.env_configs, self.player_pipes, local_targets):
            #
            #  a, b = res_game.get(), res.get()
            #  self.consume_pipes(sum([env_conf["players"] for env_conf in self.env_configs]))
            #  res.get()
            res_game.get()

        for ap in self.agent_processes:
            ap.join()
