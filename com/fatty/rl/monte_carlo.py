# -*- coding: utf-8 -*-
"""
Created on 20-1-10

@author: caoweiquan
"""
import numpy as np
from com.fatty.rl.environment import Environment
from com.fatty.rl.control import Controller
from com.fatty.rl.policy import Policy, LookupBasedPolicy


class MonteCarlo(Controller):
    def __init__(self, policy: Policy, discount=1.0, learn_rate=0.5, greedy_epsilon=0.01):
        super(MonteCarlo, self).__init__(policy)
        self.discount = discount
        self.learn_rate = learn_rate
        self.greedy_epsilon = greedy_epsilon

    def get_action(self, state):
        return self.policy.get_action(state, self.greedy_epsilon)

    def feed_episode(self, rewards, states, actions):
        """
        Train controller with a episode.
        Episodes are organized as a list of (r,s,term), a, (r,s,term), a ..., a, (r,s,term).
        :param rewards:
        :param states:
        :param actions:
        :return:
        """
        if len(rewards) < 2:
            raise ValueError('The #rewards(%d) is too short' % len(rewards))
        if len(rewards) != len(states):
            raise ValueError('The #rewards(%d) must equal to #states(%d)' % (len(rewards), len(states)))
        if len(rewards) != len(actions) + 1:
            raise ValueError('The #rewards(%d) must equal to #actions(%d)+1' % (len(rewards), len(actions)))
        self.do_feed_episode(rewards, states, actions)

    def do_feed_episode(self, rewards, states, actions):
        """
        Train controller with a episode.
        Episodes are organized as a list of (r,s,term), a, (r,s,term), a ..., a, (r,s,term).
        :param rewards:
        :param states:
        :param actions:
        :return:
        """
        raise RuntimeError('You should implement %s yourself' % 'do_feed_episode')

    def learn(self, env: Environment, num_rounds=1, round_max_itr=0):
        """
        通过环境来尝试学习，例如更新policy
        :param env:
        :param rounds:
        :param round_max_itr:
        :return:
        """
        for _ in range(num_rounds):
            itr = 0
            # List of (r,s,term), a, (r,s,term), a ..., a, (r,s,term).
            rewards = []
            states = []
            actions = []
            (reward, state, term) = env.reset()
            rewards.append(reward)
            states.append(state)
            while round_max_itr <= 0 or itr < round_max_itr:
                action = self.get_action(state)
                actions.append(action)
                (reward, state, term) = env.step(action)
                rewards.append(reward)
                states.append(state)
                if term:
                    break
                itr += 1
            # Feed episode we generated!
            if len(rewards) >= 2:
                self.feed_episode(rewards, states, actions)
        return self.policy


class MonteCarloLookup(MonteCarlo):
    """
    Assume states and actions are both list(tuple) of integers(strings).
    """
    def __init__(self, policy: LookupBasedPolicy, state_encode, state_shape,
                 action_encode, action_shape, discount=1.0, explore_epsilon_N0=100):
        # The learn_rate and greedy_epsilon actually make no sense for Lookup problems!
        super(MonteCarloLookup, self).__init__(policy, discount, learn_rate=0.5, greedy_epsilon=0.01)
        if state_encode is None:
            raise RuntimeError('must provide a convert method from state to state space indices')
        if state_shape is None or not state_shape is tuple:
            raise ValueError('state_shape must be tuple giving shape of the state space')
        if action_encode is None:
            raise RuntimeError('must provide a convert method from action to action space indices')
        if action_shape is None or not action_shape is tuple:
            raise ValueError('action_shape must be tuple giving shape of the action space')
        self.explore_epsilon_N0 = explore_epsilon_N0
        # State/Action space conversion.
        self.state_encode = state_encode
        self.state_shape = state_shape
        self.action_encode = action_encode
        self.action_shape = action_shape
        # Statistics for MonteCarlo
        self.state_count = np.ndarray(state_shape, np.int)
        self.state_action_count = np.ndarray(action_shape, np.int)
        self.Q = np.zeros((*state_shape, *action_shape), np.float)
        self.V = np.zeros(state_shape, np.float)

    def get_action(self, state):
        idx = self.state_encode(state)  # In case of invalid state, such as term state.
        if idx is not None:
            n = self.state_count[idx]
            # explore_epsilon = self.explore_epsilon_N0/(self.explore_epsilon_N0+n)
            return self.policy.get_action(state, self.explore_epsilon_N0/(self.explore_epsilon_N0+n))

    def do_feed_episode(self, rewards, states, actions):
        num_actions = len(actions)
        Gt = np.zeros((num_actions,), np.float)
        Gt[num_actions-1] = rewards[num_actions]
        for i in range(num_actions-2, -1, -1):
            Gt[i] = rewards[i+1] + self.discount * Gt[i+1]
        # Evaluation
        for i in range(num_actions):
            s_idx = self.state_encode(states[i])
            a_idx = self.action_encode(actions[i])
            sa_idx = (*s_idx, *a_idx)
            if s_idx is None or a_idx is None:
                continue
            # State count
            self.state_count[s_idx] += 1
            # State-Action count
            self.state_action_count[sa_idx] += 1
            # Q function
            self.Q[sa_idx] = self.Q[sa_idx] + (1.0/self.state_action_count[sa_idx])*(Gt[i]-self.Q[sa_idx])
        # Improve policy
        a_axis = tuple(range(len(s_idx), len(sa_idx)))
        self.V = np.max(self.Q, axis=a_axis)
        flat_Q = np.reshape(self.Q, (*self.state_shape, np.prod(self.action_shape)))
        flat_pi = np.argmax(flat_Q, axis=-1)
        pi = np.unravel_index(flat_pi, self.action_shape)
        np.unravel_index()
        self.policy.update(np.argmax(self.Q, axis=a_axis))
        for s in self.Q:
            max_action = max(self.Q[s], key=self.Q[s].get)
            self.policy.update_pi(s, max_action)
            self.V[s] = self.Q[s][max_action]
