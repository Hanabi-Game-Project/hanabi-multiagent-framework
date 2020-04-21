"""
This file defines a class for managing parallel games of hanabi
"""
import numpy as np

from hanabi_multiagent_framework.environment import HanabiParallelEnvironment

class AgentRingQueue:
    """Class which keeps track of agents' turns"""

    def __init__(self, agents: list):
        self.agents = agents
        self._len = len(agents)
        self.cur_agent_id = -1

    def next(self) -> tuple:
        """Get the agent, whose turn it is to play"""
        self.cur_agent_id = (self.cur_agent_id + 1) % self._len
        return self.cur_agent_id, self.agents[self.cur_agent_id]

    def __len__(self) -> int:
        return self._len


class HanabiParallelSession:
    """
    A class for instantiating and running parallel game sessions
    """
    def __init__(self, agents: list, env_config: dict, n_states: int):
        assert len(agents) == env_config['players']
        self.agents = AgentRingQueue(agents)
        self.parallel_env = HanabiParallelEnvironment(env_config, n_states)
        self.env_config = env_config
        self.n_states = n_states
        _, self.obs_len, self.max_moves = self.parallel_env._parallel_env.get_shapes()
        self.experience = [ExperienceBuffer(self.obs_len, self.max_moves, 1, 1000000)
                           for _ in agents]
        self._cur_obs, self._cur_lm = self.parallel_env.reset()

    def run(self, n_steps: int):
        """Make <n_steps> in each of the parallel game states.
        """
        total_reward = np.zeros(self.n_states)
        cur_step = 0
        agent_rewards = np.zeros((len(self.agents), self.n_states))
        agent_qs = [None for _ in range(len(self.agents))]
        agent_last_action = [None for _ in range(len(self.agents))]
        agent_next_observation = [None for _ in range(len(self.agents))]
        while cur_step < n_steps:
            agent_id, agent = self.agents.next()
            self.experience[agent_id].add_transition(agent_next_observation[agent_id],
                                                     agent_last_action[agent_id],
                                                     agent_rewards[agent_id],
                                                     agent_qs[agent_id])
            q_vals, actions = agent.act(self._cur_obs, self._cur_lm)
            agent_qs[agent_id] = q_vals
            agent_last_action[agent_id] = actions
            (self._cur_obs, self._cur_lm), rewards, _, _, = self.parallel_env.step(
                list(actions), agent_id)
            agent_next_observation[agent_id] = self._cur_obs
            # reward is cumulative over the curse of the whole round.
            # i.e. the agents gets a reward for his action as well as for the
            # actions of its co-players.
            agent_rewards[agent_id, :] = 0
            agent_rewards += rewards
            total_reward += rewards
            cur_step += 1
        return cur_step, total_reward

    def train(self, n_epochs: int, n_steps: int, n_warmup: int, train_batch_size: int):
        """Train agents.

        Args:
            n_epochs -- number of training iteration.
            n_steps -- number of steps to run in each iteration.
            n_warmup -- number of steps to run before the training starts
                        (to fill the experience buffer)
            train_batch_size -- training batch size.
        """
        self.run(n_warmup)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("epoch", epoch)
            self.run(n_steps)
            for agent_id, agent in enumerate(self.agents.agents):
                exp = self.experience[agent_id].sample(train_batch_size)
                agent.train(*exp)
