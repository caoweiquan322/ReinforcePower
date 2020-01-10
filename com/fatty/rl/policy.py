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
    def __init__(self, state_encode, state_shape,
                 action_decode, action_shape):
        super(LookupBasedPolicy, self).__init__()
        if state_encode is None:
            raise RuntimeError('must provide a convert method from state to state space indices')
        if state_shape is None or not isinstance(state_shape, tuple):
            raise ValueError('state_shape must be tuple giving shape of the state space')
        if action_decode is None:
            raise RuntimeError('must provide a convert method from action space indices to action')
        if action_shape is None or not isinstance(action_shape, tuple):
            raise ValueError('action_shape must be tuple giving shape of the action space')

        self.state_encode = state_encode
        self.state_shape = state_shape
        self.action_decode = action_decode
        self.action_shape = action_shape
        self.optimal_action_tuple = None  # Map state--->action

    def update_pi_lookup_tuple(self, optimal_action_tuple):
        self.optimal_action_tuple = optimal_action_tuple

    def get_action(self, observed_state, greedy_epsilon=0.0):
        s_idx = self.state_encode(observed_state)
        dim = len(self.action_shape)
        optimal_action = [0]*dim

        if observed_state is None or self.optimal_action_tuple is None or random.random() < greedy_epsilon:
            for i in range(dim):
                optimal_action[i] = np.random.randint(0, self.action_shape[i] - 1)
        else:
            for i in range(dim):
                optimal_action[i] = self.optimal_action_tuple[i][s_idx]
        return self.action_decode(tuple(optimal_action))


