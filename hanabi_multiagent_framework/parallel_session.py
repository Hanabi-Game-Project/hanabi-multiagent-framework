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
        self._cur_obs, self._cur_lm = None, None
        self.reset()
        # variables to preserve the agents' rewards between runs
        self.agent_cum_rewards, self.agent_terminal_states = None, None

    def reset(self):
        """Reset the session, i.e. reset the all states and start from agent 0."""
        self.agents.reset()
        self._cur_obs, self._cur_lm = self.parallel_env.reset()
        self.agent_cum_rewards = np.zeros((len(self.agents), self.n_states, 1))
        self.agent_contiguous_states = np.full((len(self.agents), self.n_states), True)

    def run_eval(self, print_intermediate: bool = True) -> np.ndarray:
        """Run each state until the end and return the final scores.
        Args:
            print_intermediate -- Flag indicating whether each step of evaluation should be printed.
        """
        self.reset()
        #  print("Running evaluation")
        total_reward = np.zeros((self.n_states,))
        step_rewards = []
        step_types = self.parallel_env.step_types

        step = 0
        done = np.full((self.n_states, ), False)
        # run until all states terminate
        while not np.all(done):
            valid_states = np.logical_not(done)
            agent_id, agent = self.agents.next()
            (self._cur_obs, self._cur_lm), step_types = \
                self.parallel_env.reset_states(
                    np.nonzero(step_types == StepType.LAST)[0], agent_id)

            actions = agent.exploit(self._cur_obs, self._cur_lm)

            (self._cur_obs, self._cur_lm), reward, step_types = \
                    self.parallel_env.step(actions, agent_id)
            total_reward[valid_states] += reward[valid_states]
            done = np.logical_or(done, step_types == StepType.LAST)
            if print_intermediate:
                step_rewards.append({"terminated": np.sum(done), "rewards" : reward[valid_states]})
            step += 1

        if print_intermediate:
            eval_pretty_print(step_rewards, total_reward)
        return total_reward

    def run(self, n_steps: int):
        """Make <n_steps> in each of the parallel game states.
        """
        dummy_obs = np.zeros((self.n_states, self.obs_len))
        dummy_lms = np.ones((self.n_states, self.max_moves))
        total_reward = np.zeros(self.n_states)
        cur_step = 1
        agent_obs_tm1 = np.full(((len(self.agents)), self.n_states, self.obs_len), np.nan)
        agent_act_tm1 = np.empty(((len(self.agents)), self.n_states, 1))
        agent_obs_t = np.full(((len(self.agents)), self.n_states, self.obs_len), np.nan)
        agent_act_t = np.empty(((len(self.agents)), self.n_states, 1))
        agent_lms_t = np.empty(((len(self.agents)), self.n_states, self.max_moves))
        agent_reward_tm1 = np.zeros((len(self.agents), self.n_states, 1))
        agent_reward_t = np.zeros((len(self.agents), self.n_states, 1))
        def handle_terminal_states(terminal, reward, agent_id):
            terminal = terminal == 1
            agent_reward_t[:, terminal] = 0
            self.experience[agent_id].add_transition(
                agent_obs_t[agent_id, terminal, :],
                agent_act_t[agent_id, terminal, :],
                dummy_obs[terminal, :],
                dummy_lms[terminal, :],
                reward[terminal].reshape((-1, 1)),
                np.full((np.sum(terminal),), True))
            agent_obs_tm1[:, terminal, :] = np.nan
            self._cur_obs, self._cur_lm = self.parallel_env.reset_terminal_states(
                (agent_id + 1) % len(self.agents))


        while cur_step < n_steps:
            # beginning of the agent's turn.
            agent_id, agent = self.agents.next()


            # collect the outcome of the last round.
            # it includes the agents last observation_t (not current!),
            # corresponding action_t, and reward_t.
            # agent also has to know the observation_tm1 and action_tm1

            valid_transitions = np.logical_not(np.any(np.isnan(agent_obs_tm1[agent_id]), axis=1))
            self.experience[agent_id].add_transition(
                agent_obs_tm1[agent_id, valid_transitions, :],
                agent_act_tm1[agent_id, valid_transitions, :],
                agent_obs_t[agent_id, valid_transitions, :],
                agent_lms_t[agent_id, valid_transitions, :],
                agent_reward_tm1[agent_id, valid_transitions, :],
                np.full((np.sum(valid_transitions),), False))
                #  agent_terminal_t[agent_id, valid_transitions])

            # record new observation and legal actions
            agent_obs_tm1[agent_id] = agent_obs_t[agent_id]
            agent_obs_t[agent_id] = self._cur_obs
            agent_lms_t[agent_id] = self._cur_lm

            # agent acts
            _, actions = agent.explore(self._cur_obs, self._cur_lm)
            # generate actions again if the agent produced illegal actions.
            while not np.all(self._cur_lm[range(len(actions)), actions]):
                print("Illegal action detected. Re-generating actions.")
                _, actions = agent.explore(self._cur_obs, self._cur_lm)

            # record agent's actions.
            agent_act_tm1[agent_id] = agent_act_t[agent_id]
            agent_act_t[agent_id, :, 0] = actions

            # apply actions to the states and get new observations, rewards, statuses.
            (self._cur_obs, self._cur_lm), rewards, done, _, = self.parallel_env.step(
                list(actions), agent_id)

            #  rewards = np.clip(rewards, 0, None)
            #  rewards = np.abs(rewards)

            handle_terminal_states(done, rewards, agent_id)
            contiguous_states = done == 0

            # reward is cumulative over the curse of the whole round.
            # i.e. the agents gets a reward for his action as well as for the
            # actions of its co-players.

            # agent's reward should not be updated if the state has reached its terminus
            # during the last round
            agent_reward_tm1[agent_id] = agent_reward_t[agent_id]
            agent_reward_t[agent_id, :] = 0
            agent_reward_t[:, contiguous_states] += np.broadcast_to(
                rewards.reshape((-1, 1)),
                agent_reward_t.shape)[:, contiguous_states]

            cur_step += 1
        return cur_step, total_reward


    def train(self,
              n_iter: int,
              n_sim_steps: int,
              n_train_steps: int,
              n_warmup: int,
              train_batch_size: int):
        """Train agents.

        Args:
            n_epochs -- number of training iteration.
            n_steps -- number of steps to run in each iteration.
            n_warmup -- number of steps to run before the training starts
                        (to fill the experience buffer)
            train_batch_size -- training batch size.
        """
        self.run(n_warmup)
        for _ in range(n_iter):
            self.run(n_sim_steps)
            for _ in range(n_train_steps):
                for agent_id, agent in enumerate(self.agents.agents):
                    exp = self.experience[agent_id].sample(train_batch_size)
                    agent.train(*exp)
