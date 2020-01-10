# -*- coding: utf-8 -*-
"""
Created on 20-1-10

@author: caoweiquan
"""
from com.fatty.rl.policy import Policy
from com.fatty.rl.environment import Environment


class Controller:
    """
    控制算法
    """
    def __init__(self, policy: Policy):
        if policy is None:
            raise ValueError('Need a valid policy as initialization')
        self.policy = policy

    def get_policy(self):
        return self.policy

    def get_action(self, action):
        return self.policy.get_action(action, greedy_epsilon=0.0)

    def learn(self, env: Environment):
        """
        通过环境来尝试学习，例如更新policy
        :param env:
        :return:
        """
        return self.policy
