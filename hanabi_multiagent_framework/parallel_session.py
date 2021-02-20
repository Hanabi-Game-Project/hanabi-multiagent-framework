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
from hanabi_agents.rlax_dqn import RewardShaper, ShapingType
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

        # variables to preserve the agents' rewards between runs
        self.agent_cum_rewards, self.agent_terminal_states = None, None
        
        # create stacker objects
        self.stacker = [a.create_stacker(self.obs_len, self.n_states) for a in agents]
        self.stacker_eval = [a.create_stacker(self.obs_len, self.n_states) for a in agents]
        
        # create caches
        self.last_actions = [None for i in range(self.agents.__len__())]
        self.last_step_types = [np.zeros((self.n_states)) for i in range(self.agents.__len__())]
        self.last_observations = [None for i in  range(self.agents.__len__())]
        
        self.reset()

    def reset(self):
        """Reset the session, i.e. reset the all states and start from agent 0."""
        self.agents.reset()
        self._cur_obs = self.parallel_env.reset()
        self.agent_cum_rewards = np.zeros((len(self.agents), self.n_states, 1))
        self.agent_contiguous_states = np.full((len(self.agents), self.n_states), True)
        for stack in self.stacker_eval:
            if stack is not None:
                stack.reset()

    def run_eval(self, dest: str = None, print_intermediate: bool = True) -> np.ndarray:
        """Run each state until the end and return the final scores.
        Args:
            print_intermediate -- Flag indicating whether each step of evaluation should be printed.
        """
        self.reset()
        print("Agents", self.agents.agents)
        #  print("Running evaluation")
        total_reward = np.zeros((self.n_states,))
        total_shaped_reward = np.zeros((self.n_states,))
        total_play_moves = np.zeros((self.n_states,))
        total_discard_moves = np.zeros((self.n_states,))
        total_reveal_moves = np.zeros((self.n_states,))
        total_risky_moves = np.zeros((self.n_states,))
        total_bad_discards = np.zeros((self.n_states))
        step_rewards = []
        playability = [[] for i in range(self.n_states)]
        move_eval = [[] for i in range(self.n_states)]
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

            obs = self.preprocess_obs_for_agent(self._cur_obs, agent, self.stacker_eval[agent_id])
            actions = agent.exploit(obs)

            if agent.requires_vectorized_observation():
                moves = self.parallel_env.get_moves(actions)
            else:
                moves = actions
                
            # get shaped rewards
            reward_shaping, shape_type = agent.shape_rewards(obs, moves)
            
            risky_moves = shape_type == ShapingType.RISKY
            bad_discards = shape_type == ShapingType.DISCARD_LAST_OF_KIND

            # playability
            counter = 0
            step_playability = []
            for o, m in zip(self._cur_obs, moves):
                if m.move_type == pyhanabi.HanabiMove.Type.kPlay and valid_states[counter]:
                    try:
                        prob = o.playable_percent()[m.card_index]
                        playability[counter].append(prob)
                        step_playability.append(prob)
                    except IndexError:
                        pass
                counter += 1
                
            # reveal options
            reveal_options = [1 for o in self._cur_obs if o.information_tokens>0]

            # moves
            for idx, a in enumerate(actions):
                if valid_states[idx]:
                    move_eval[idx].append(a)
            
            # get new observation based on action
            self._cur_obs, reward, step_types = \
                    self.parallel_env.step(actions, agent_id)
                    
            shaped_reward = reward + reward_shaping

            # convert moves
            play_moves = [1 if m.move_type == pyhanabi.HanabiMove.Type.kPlay else 0
                          for m in moves]
            discard_moves = [1 if m.move_type == pyhanabi.HanabiMove.Type.kDiscard else 0
                             for m in moves]
            reveal_moves = [1 if m.move_type == pyhanabi.HanabiMove.Type.kRevealColor or
                            m.move_type == pyhanabi.HanabiMove.Type.kRevealRank else 0
                            for m in moves]

            total_reward[valid_states] += reward[valid_states]
            total_shaped_reward[valid_states] += shaped_reward[valid_states]
            total_play_moves[valid_states] += np.array(play_moves)[valid_states]
            total_discard_moves[valid_states] += np.array(discard_moves)[valid_states]
            total_reveal_moves[valid_states] += np.array(reveal_moves)[valid_states]
            total_risky_moves[valid_states] += risky_moves[valid_states]
            total_bad_discards[valid_states] += bad_discards[valid_states]

            done = np.logical_or(done, step_types == StepType.LAST)
            if print_intermediate:
                step_rewards.append({"terminated": np.sum(done),
                    "risky": np.sum(risky_moves[valid_states]),
                    "play": np.sum(np.array(play_moves)[valid_states]),
                    "bad_discards":  np.sum(bad_discards[valid_states]),
                    "discard": np.sum(np.array(discard_moves)[valid_states]), 
                    "reveal": np.sum(np.array(reveal_moves)[valid_states]),
                    "reveal_options": np.sum(np.array(reveal_options)[valid_states]),
                    "rewards" : reward[valid_states],
                    "playability": step_playability,
                    "agent_id": agent_id})

            step += 1

        if print_intermediate:
            eval_pretty_print(step_rewards, total_reward)
            print(playability[:2])
        if dest is not None:
            np.save(dest + "_step_rewards.npy", step_rewards)
            np.save(dest + "_total_rewards.npy", total_reward)
            np.save(dest + "_total_shaped_rewards.npy", total_shaped_reward)
            np.save(dest + "_move_eval.npy", {"play": total_play_moves,
                "risky": total_risky_moves,
                "bad_discard": total_bad_discards,
                "discard": total_discard_moves,
                "reveal": total_reveal_moves,
                "playability": playability,
                "moves": move_eval})
        
        # store the average reward as performance parameter in reward shaping
        for agent in self.agents.agents:
            if agent.reward_shaper is not None:
                agent.reward_shaper.performance = np.mean(total_reward)  
        
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

        while cur_step < n_steps:
            
            # beginning of the agent's turn.
            agent_id, agent = self.agents.next()
            handle_terminal_states(self.parallel_env.step_types, agent_id)

            # agent acts
            obs = self.preprocess_obs_for_agent(self._cur_obs, agent, self.stacker[agent_id])
            
            # check if somewhere within last round a final step was reached
            is_last_step = np.zeros((self.n_states), dtype=bool)
            for st in self.last_step_types:
                is_last_step[st==StepType.LAST] = True
                
            # observation is complete
            if self.last_actions[agent_id] is not None:
                
                # shape rewards
                # convert actions to HanabiMOve objects

                if agent.requires_vectorized_observation():
                    last_moves = self.parallel_env.get_moves(self.last_actions[agent_id])
                else:
                    last_moves = self.last_actions[agent_id]

                add_rewards, shape_type = agent.shape_rewards(self.last_observations[agent_id], last_moves)
                shaped_rewards = self.agent_cum_rewards[agent_id] + add_rewards.reshape(-1, 1)

                # add observation to agent
                agent.add_experience(
                    self.last_observations[agent_id],
                    self.last_actions[agent_id],
                    shaped_rewards,
                    obs,
                    is_last_step.reshape(-1, 1))
            
            # clear history for all states that had a last step
            # then only the first state observation should be in stack
            if True in is_last_step:
                if self.stacker[agent_id] is not None:
                    self.stacker[agent_id].reset_history(is_last_step)
                    obs = self.update_obs_for_agent(obs, agent, self.stacker[agent_id])           
            
            actions = agent.explore(obs)
            
            # apply actions to the states and get new observations, rewards, statuses.
            self._cur_obs, rewards, step_types = self.parallel_env.step(
                actions, agent_id)
            
            # store info from this round
            self.last_actions[agent_id] = actions             
            self.last_observations[agent_id] = obs
            self.last_step_types[agent_id] = np.copy(step_types)
            
            # reset the cumulative reward for the current agent
            self.agent_cum_rewards[agent_id, :] = 0
            self.agent_contiguous_states[agent_id, :] = True
            
             # calculate team reward = own reward + reward of co players
            self.agent_cum_rewards[self.agent_contiguous_states] += np.broadcast_to(
                rewards.reshape((-1, 1)),
                self.agent_cum_rewards.shape)[self.agent_contiguous_states]

            self.agent_contiguous_states[:, step_types == 2] = False

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
                    agent.update()
                    
    def preprocess_obs_for_agent(self, obs, agent, stack):
        
        if agent.requires_vectorized_observation():
            vobs = np.array(self.parallel_env._parallel_env.encoded_observations)
            if stack is not None:
                stack.add_observation(vobs)
                vobs = stack.get_current_obs()
            vlms = np.array(self.parallel_env._parallel_env.encoded_legal_moves)
            return (obs, (vobs, vlms))
        return obs
    
    def update_obs_for_agent(self, obs, agent, stack):
        
        if agent.requires_vectorized_observation() and stack is not None:
            return (obs[0], (stack.get_current_obs(), obs[1][1]))
        return obs
    
    
