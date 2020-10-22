"""
This file defines a class for managing parallel games of hanabi and agents

Throughout this file you will find suffixes _t and _tm1. It designates temporal correspondence:
t stands for "at time t" and tm1 stands for "at time t - 1"
"""
from typing import List, Dict, Tuple
import numpy as np
from dm_env import StepType
from .agent import HanabiAgent
from .environment import HanabiParallelEnvironment
from .experience_buffer import ExperienceBuffer
from .utils import eval_pretty_print
from hanabi_agents.rlax_dqn import RewardShaper
from _cffi_backend import typeof
import timeit
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi

class HanabiParallelSession:
    """
    A class for running parallel game sessions
    """

    class AgentRingQueue:
        """Class which keeps track of agents' turns"""

        def __init__(self, agents: List[HanabiAgent]):
            self.agents = agents
            self._len = len(agents)
            self.cur_agent_id = None
            self.reset()

        def reset(self):
            """Restart counting the agents"""
            self.cur_agent_id = -1

        def next(self) -> Tuple[int, HanabiAgent]:
            """Get the agent, whose turn it is to play"""
            self.cur_agent_id = (self.cur_agent_id + 1) % self._len
            return self.cur_agent_id, self.agents[self.cur_agent_id]

        def __len__(self) -> int:
            return self._len


    def __init__(self,
                 env: HanabiParallelEnvironment,
                 agents: List[HanabiAgent]):
        """Constructor.
        Args:
            env        -- hanabi parallel environment.
            agents     -- list with instances of agents.
            exp_buffer_size -- size of the experience buffer.
        """
        assert len(agents) == env.num_players
        self.agents = HanabiParallelSession.AgentRingQueue(agents)
        self.parallel_env = env
        self.n_states = env.num_states
        self.obs_len = self.parallel_env.observation_len
        self.max_moves = self.parallel_env.max_moves
        self._cur_obs = None
        self.reset()
        # variables to preserve the agents' rewards between runs
        self.agent_cum_rewards, self.agent_terminal_states = None, None

    def reset(self):
        """Reset the session, i.e. reset the all states and start from agent 0."""
        self.agents.reset()
        self._cur_obs = self.parallel_env.reset()
        self.agent_cum_rewards = np.zeros((len(self.agents), self.n_states, 1))
        self.agent_contiguous_states = np.full((len(self.agents), self.n_states), True)

    def run_eval(self, dest: str = None, print_intermediate: bool = True) -> np.ndarray:
        """Run each state until the end and return the final scores.
        Args:
            print_intermediate -- Flag indicating whether each step of evaluation should be printed.
        """
        self.reset()
        print("Agents", self.agents.agents)
        #  print("Running evaluation")
        total_reward = np.zeros((self.n_states,))
        total_play_moves = np.zeros((self.n_states,))
        total_discard_moves = np.zeros((self.n_states,))
        total_reveal_moves = np.zeros((self.n_states,))
        total_risky_moves = np.zeros((self.n_states,))
        step_rewards = []
        step_types = self.parallel_env.step_types

        step = 0
        done = np.full((self.n_states, ), False)
        # run until all states terminate
        while not np.all(done):
            valid_states = np.logical_not(done)
            agent_id, agent = self.agents.next()
            self._cur_obs, step_types = \
                self.parallel_env.reset_states(
                    np.nonzero(step_types == StepType.LAST)[0], agent_id)

            obs = self.preprocess_obs_for_agent(self._cur_obs, agent)
            actions = agent.exploit(obs)

            self._cur_obs, reward, step_types = \
                    self.parallel_env.step(actions, agent_id)
                    
            # convert moves
            moves = self.parallel_env.get_moves(actions)
            play_moves = [1 if m.move_type == pyhanabi.HanabiMove.Type.kPlay else 0 
                          for m in moves]
#             discard_moves = [1 if m.move_type == pyhanabi.HanabiMove.Type.kDiscard else 0 
#                              for m in moves]
#             reveal_moves = [1 if m.move_type == pyhanabi.HanabiMove.Type.kRevealColor or 
#                             m.move_type == pyhanabi.HanabiMove.Type.kRevealRank else 0 
#                             for m in moves]
                        
            # get shaped rewards
            reward_shaping = agent.shape_rewards(obs, moves)
            risky_moves = reward_shaping < 0
                                
            total_reward[valid_states] += reward[valid_states]
            total_play_moves[valid_states] += np.array(play_moves)[valid_states]
#             total_discard_moves[valid_states] += np.array(discard_moves)[valid_states]
#             total_reveal_moves[valid_states] += np.array(reveal_moves)[valid_states]
            total_risky_moves[valid_states] += risky_moves[valid_states]
                        
            done = np.logical_or(done, step_types == StepType.LAST)
            if print_intermediate:
                step_rewards.append({"terminated": np.sum(done), "rewards" : reward[valid_states]})
            step += 1

        if print_intermediate:
            eval_pretty_print(step_rewards, total_reward)
        if dest is not None:
            np.save(dest + "_step_rewards.npy", step_rewards)
            np.save(dest + "_total_rewards.npy", total_reward)
            np.save(dest + "_move_eval.npy", (total_play_moves, total_risky_moves))
        return total_reward

    def run(self, n_steps: int):
        """Make <n_steps> in each of the parallel game states.
        States, rewards, etc. are preserved between runs.
        """
        total_reward = np.zeros(self.n_states)
        cur_step = 0
        #  step_types = self.parallel_env.step_types

        def handle_terminal_states(step_types, agent_id):
            terminal = step_types == StepType.LAST
            self._cur_obs, step_types = self.parallel_env.reset_states(
                np.nonzero(terminal)[0],
                agent_id)
            obs = self.preprocess_obs_for_agent(self._cur_obs, agent)
            agent.add_experience_first(obs, step_types)

        while cur_step < n_steps:
            # beginning of the agent's turn.
            agent_id, agent = self.agents.next()

            handle_terminal_states(self.parallel_env.step_types, agent_id)

            # agent acts
            obs = self.preprocess_obs_for_agent(self._cur_obs, agent)
            actions = agent.explore(obs)
            
            # convert actions to HanabiMOve objects
            moves = self.parallel_env.get_moves(actions)
            
            # apply actions to the states and get new observations, rewards, statuses.
            self._cur_obs, rewards, step_types = self.parallel_env.step(
                actions, agent_id)
            
            # reward is cumulative over the course of the whole round.
            # i.e. the agents gets a reward for his action as well as for the
            # actions of its co-players.
            self.agent_cum_rewards[self.agent_contiguous_states] += np.broadcast_to(
                rewards.reshape((-1, 1)),
                self.agent_cum_rewards.shape)[self.agent_contiguous_states]
                
            # set the terminal states as not contiguous for all agents
            self.agent_contiguous_states[:, step_types == 2] = False
            # reset contigency of all states for current agent
            self.agent_contiguous_states[agent_id, :] = True
            
            # call reward shaping function
            add_rewards = agent.shape_rewards(obs, moves).reshape(-1, 1)
            shaped_rewards = self.agent_cum_rewards[agent_id] + add_rewards
            
            # add new experiences to the agent
            obs = self.preprocess_obs_for_agent(self._cur_obs, agent)
            agent.add_experience(
                obs,
                actions,
                shaped_rewards,#self.agent_cum_rewards[agent_id] + add,
                step_types)

            # reset the cumulative reward for the current agent
            self.agent_cum_rewards[agent_id, :] = 0

            total_reward += rewards

            cur_step += 1
        return cur_step, total_reward


    def train(self,
              n_iter: int,
              n_sim_steps: int,
              n_train_steps: int,
              n_warmup: int):
        """Train agents.

        Args:
            n_iter -- number of training iteration.
            n_sim_steps -- number of game steps to run in each training iteration.
            n_train_steps -- number of agents' training updates per training iteration.
            n_warmup -- number of steps to run before the training starts
                        (e.g. to fill the experience buffer)
        """
        self.run(n_warmup)
        for _ in range(n_iter):
            self.run(n_sim_steps)
            for _ in range(n_train_steps):
                for agent in self.agents.agents:
                    #print(repr(agent))
                    agent.update()
                    

    def preprocess_obs_for_agent(self, obs, agent):
        if agent.requires_vectorized_observation():
            vobs = np.array(self.parallel_env._parallel_env.encoded_observations)
            #print(vobs)
            vlms = np.array(self.parallel_env._parallel_env.encoded_legal_moves)
            return (obs, (vobs, vlms))
        return obs
