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

    def feed_sars(self, s1, a1, r2, s2):
        """
        :param s1:
        :param a1:
        :param r2:
        :param s2:
        :return:
        """
        a2 = self.get_action(s2)
        self.do_feed_sarsa(s1, a1, r2, s2, a2)
        return a2

    def do_feed_sarsa(self, s1, a1, r2, s2, a2):
        """
        :param s1:
        :param a1:
        :param r2:
        :param s2:
        :param a2:
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
                if term:
                    break
                # Feed sars sequence we generated!
                next_action = self.feed_sars(s1, action, reward, state)
                itr += 1
            self.clear_eligibility()
        return self.policy


class SarsaLookup(Sarsa):
    """
    Assume states and actions are both list(tuple) of integers(strings).
    """
    def __init__(self, policy: LookupBasedPolicy, td_lambda=0.5, discount=1.0, explore_epsilon_N0=100):
        # The learn_rate and greedy_epsilon actually make no sense for Lookup problems!
        super(SarsaLookup, self).__init__(policy, td_lambda, discount, learn_rate=0.5, greedy_epsilon=0.01)
        self.explore_epsilon_N0 = explore_epsilon_N0
        self.state_count = {}
        self.state_action_count = {}
        self.Q = {}
        self.V = {}
        self.eligibility = {}

    def get_action(self, state):
        #state = tuple(state)
        n = (state in self.state_count) and self.state_count[state] or 0
        # explore_epsilon = self.explore_epsilon_N0/(self.explore_epsilon_N0+n)
        return self.policy.get_action(state, self.explore_epsilon_N0/(self.explore_epsilon_N0+n))

    def _initialize_s_table(self, tbl, s):
        if s not in tbl:
            tbl[s] = 0

    def _initialize_sa_table(self, tbl, s, a):
        if s not in tbl:
            tbl[s] = {}
        if a not in tbl[s]:
            tbl[s][a] = 0.0

    def decay_eligibility(self):
        for s in self.eligibility:
            for a in self.eligibility[s]:
                self.eligibility[s][a] *= self.td_lambda

    def clear_eligibility(self):
        for s in self.eligibility:
            for a in self.eligibility[s]:
                self.eligibility[s][a] = 0

    def do_feed_sarsa(self, s1, a1, r2, s2, a2):
        """
        :param s1:
        :param a1:
        :param r2:
        :param s2:
        :param a2:
        :return:
        """
        # Evaluation
        self._initialize_s_table(self.state_count, s1)
        self._initialize_s_table(self.state_count, s2)
        self._initialize_s_table(self.V, s1)
        self._initialize_s_table(self.V, s2)
        self._initialize_sa_table(self.state_action_count, s1, a1)
        self._initialize_sa_table(self.state_action_count, s2, a2)
        self._initialize_sa_table(self.Q, s1, a1)
        self._initialize_sa_table(self.Q, s2, a2)
        self._initialize_sa_table(self.eligibility, s1, a1)
        self.decay_eligibility()

        self.state_count[s1] += 1
        self.state_action_count[s1][a1] += 1
        self.eligibility[s1][a1] += 1
        learn_rate = (1.0/self.state_action_count[s1][a1])
        delta = r2+self.discount*self.Q[s2][a2]-self.Q[s1][a1]
        self.Q[s1][a1] = self.Q[s1][a1] + learn_rate*delta*self.eligibility[s1][a1]
        # Improve policy
        for s in self.Q:
            max_action = max(self.Q[s], key=self.Q[s].get)
            self.policy.update_pi(s, max_action)
            self.V[s] = self.Q[s][max_action]
