# -*- coding: utf-8 -*-
"""
Created on 20-1-10

@author: caoweiquan
"""
import numpy as np
from com.fatty.rl.environment import Environment
from com.fatty.rl.control import Controller
from com.fatty.rl.policy import Policy, LookupBasedPolicy


class Sarsa(Controller):
    def __init__(self, policy: Policy, td_lambda=0.5, discount=1.0, learn_rate=0.5, greedy_epsilon=0.01):
        super(Sarsa, self).__init__(policy)
        self.td_lambda = td_lambda
        self.discount = discount
        self.learn_rate = learn_rate
        self.greedy_epsilon = greedy_epsilon

    def get_action(self, state):
        return self.policy.get_action(state, self.greedy_epsilon)

    def clear_eligibility(self):
        raise RuntimeError('You should implement your own clear_eligibility')

    def feed_sars(self, s1, a1, r2, s2, terminated):
        """
        :param s1:
        :param a1:
        :param r2:
        :param s2:
        :param terminated:
        :return:
        """
        a2 = None if terminated else self.get_action(s2)
        self.do_feed_sarsa(s1, a1, r2, s2, a2, terminated)
        return a2

    def do_feed_sarsa(self, s1, a1, r2, s2, a2, terminated):
        """
        :param s1:
        :param a1:
        :param r2:
        :param s2:
        :param a2:
        :param terminated:
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
            (reward, state, term) = env.reset()
            next_action = self.get_action(state)
            while round_max_itr <= 0 or itr < round_max_itr:
                action = next_action
                s1 = state
                (reward, state, term) = env.step(action)
                # Feed sars sequence we generated!
                next_action = self.feed_sars(s1, action, reward, state, term)
                if term:
                    break
                itr += 1
            self.clear_eligibility()
        return self.policy


class SarsaLookup(Sarsa):
    """
    Assume states and actions are both list(tuple) of integers(strings).
    """
    def __init__(self, policy: LookupBasedPolicy, state_encode, state_shape,
                 action_encode, action_shape, td_lambda=0.5, discount=1.0, explore_epsilon_N0=100):
        # The learn_rate and greedy_epsilon actually make no sense for Lookup problems!
        super(SarsaLookup, self).__init__(policy, td_lambda, discount, learn_rate=0.5, greedy_epsilon=0.01)
        if state_encode is None:
            raise RuntimeError('must provide a convert method from state to state space indices')
        if state_shape is None or not isinstance(state_shape, tuple):
            raise ValueError('state_shape must be tuple giving shape of the state space')
        if action_encode is None:
            raise RuntimeError('must provide a convert method from action to action space indices')
        if action_shape is None or not isinstance(action_shape, tuple):
            raise ValueError('action_shape must be tuple giving shape of the action space')
        self.explore_epsilon_N0 = explore_epsilon_N0
        # State/Action space conversion.
        self.state_encode = state_encode
        self.state_shape = state_shape
        self.action_encode = action_encode
        self.action_shape = action_shape
        # Statistics for Sarsa
        self.state_count = np.zeros(state_shape, np.int)
        self.state_action_count = np.zeros((*state_shape, *action_shape), np.int)
        self.Q = np.zeros((*state_shape, *action_shape), np.float64)
        self.V = np.zeros(state_shape, np.float64)
        self.eligibility = np.zeros((*state_shape, *action_shape), np.float64)

    def get_action(self, state):
        idx = self.state_encode(state)  # In case of invalid state, such as term state.
        n = 0.0 if idx is None else self.state_count[idx]
        # explore_epsilon = self.explore_epsilon_N0/(self.explore_epsilon_N0+n)
        return self.policy.get_action(state, self.explore_epsilon_N0 / (self.explore_epsilon_N0 + n))

    def clear_eligibility(self):
        self.eligibility *= 0

    def do_feed_sarsa(self, s1, a1, r2, s2, a2, terminated):
        """
        :param s1:
        :param a1:
        :param r2:
        :param s2:
        :param a2:
        :param terminated:
        :return:
        """
        # Evaluation
        s1_idx = self.state_encode(s1)
        a1_idx = self.action_encode(a1)
        sa1_idx = (*s1_idx, *a1_idx)
        self.state_count[s1_idx] += 1
        self.state_action_count[sa1_idx] += 1
        s2_idx = self.state_encode(s2)
        a2_idx = self.action_encode(a2)
        if terminated or s2_idx is None or a2_idx is None:  # s2/a2 might be illegal during to term.
            delta = r2 - self.Q[sa1_idx]
        else:
            sa2_idx = (*s2_idx, *a2_idx)
            delta = r2 + self.discount * self.Q[sa2_idx] - self.Q[sa1_idx]
        self.eligibility[sa1_idx] += 1.0
        learn_rate = (1.0 / self.state_action_count[sa1_idx])
        self.Q += (learn_rate * delta) * self.eligibility
        self.eligibility *= (self.discount * self.td_lambda)
        # Improve policy
        flat_Q = np.reshape(self.Q, (*self.state_shape, np.prod(self.action_shape)))
        self.V = np.max(flat_Q, axis=-1)
        # print('V.shape=' + str(self.V.shape))
        # print('V=' + str(self.V))
        flat_pi = np.argmax(flat_Q, axis=-1)
        optimal_action_tuple = np.unravel_index(flat_pi, self.action_shape)
        self.policy.update_pi_lookup_tuple(optimal_action_tuple)
