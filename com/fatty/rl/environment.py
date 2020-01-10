# -*- coding: utf-8 -*-
"""
Created on 20-1-10

@author: caoweiquan
"""
import numpy as np
from com.fatty.rl.policy import Policy


class Environment:
    """
    环境类.
    """
    def __init__(self):
        """
        构造环境
        """
        # state的类型没有限制
        self.state = None
        self.seed_num = None

    def seed(self, seed_num):
        if seed_num is not None:
            self.seed_num = seed_num
        np.random.seed(self.seed_num)

    def reset(self):
        """
        重置环境为初始状态，如随机初始化等
        :return:
        """
        return 0, self.state, False  # Reward, Observed state, term

    def step(self, action):
        """
        环境向前仿真一步
        :param action:
        :return:
        """
        return 0, None, False  # Reward, Observed state, Terminated

    def run(self, policy: Policy, need_reset=True, max_itr=0):
        """
        环境运行直至终结，需要给定Policy
        :param policy:
        :param need_reset:
        :param max_itr:
        :return:
        """
        if policy is None:
            raise ValueError('Controller cannot be none!')
        if need_reset:
            self.reset()
        itr = 0
        (reward, state, term) = (0, self.state, False)
        while max_itr <= 0 or itr < max_itr:
            action = policy.get_action(self.state)
            (reward, state, term) = self.step(action)
            if term:
                break
            itr += 1
        return reward, state, term  # Reward, Observed state, Terminated

