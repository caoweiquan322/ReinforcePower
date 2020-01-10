# -*- coding: utf-8 -*-
"""
Created on 20-1-10

@author: caoweiquan
"""
import random
import numpy as np


class Policy:
    """
    策略PI
    """
    def __init__(self):
        pass

    def get_action(self, observed_state, greedy_epsilon=0.0):
        return 0


class LookupBasedPolicy(Policy):
    def __init__(self, action_set: set):
        super(LookupBasedPolicy, self).__init__()
        if action_set is None or len(action_set) == 0:
            raise ValueError('You must provide the full action set')
        self.actions = list(action_set)
        self.determine_pi = {}  # Map state--->action

    def update_pi(self, state, action):
        self.determine_pi[state] = action

    def get_action(self, observed_state, greedy_epsilon=0.0):
        observed_state = tuple(observed_state)
        if observed_state not in self.determine_pi or random.random() < greedy_epsilon:
            return self.actions[random.randint(0, len(self.actions)-1)]
        return self.determine_pi[observed_state]


