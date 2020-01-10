# -*- coding: utf-8 -*-
"""
Created on 20-1-10

@author: caoweiquan
"""
import numpy as np
import random
from com.fatty.rl.policy import Policy
from com.fatty.rl.environment import Environment


class Easy21ManualPolicy(Policy):
    """
    手动请求policy
    """
    def __init__(self):
        super(Easy21ManualPolicy, self).__init__()

    def get_action(self, observed_state):
        print('Your card: %d' % observed_state[0])
        print('Dealer card: %d' % observed_state[1])
        s = input('Do you continue?(y/n)')
        if s == 'y':
            return 0
        else:
            return 1
        # return random.randint(0, 1)


class Easy21Env(Environment):
    ACTION_HIT = 0
    ACTION_STICK = 1
    PLAYER_IDX = 0
    DEALER_IDX = 1

    def __init__(self):
        super(Easy21Env, self).__init__()
        self.state = (1, 1)

    def reset(self):
        self.state = (Easy21Env.draw_init_card(), Easy21Env.draw_init_card())
        return 0, self.state, False  # Reward, Observed state, term

    def step(self, action):
        # Go one step further of the game
        # Action, 0:hit, 1:stick
        if action == Easy21Env.ACTION_HIT:  # Player need one more card
            self.state = (self.state[0]+Easy21Env.draw_card(), self.state[1])
            if Easy21Env.is_bust(self.state[Easy21Env.PLAYER_IDX]):
                return Easy21Env.judge(self.state), self.state, True  # Return reward, state and TERM
        else:
            # The dealer starts!
            while not Easy21Env.is_bust(self.state[Easy21Env.DEALER_IDX]) and self.state[Easy21Env.DEALER_IDX] < 17:
                self.state = (self.state[0], self.state[1]+Easy21Env.draw_card())
            return Easy21Env.judge(self.state), self.state, True  # Return reward, state and TERM
        return 0, self.state, False  # Return reward, state and NOT_TERM

    @staticmethod
    def draw_init_card():
        # Draw a initial black card
        return random.randint(1, 10)

    @staticmethod
    def draw_card():
        # Draw a red card or a black card
        card = random.randint(1, 10)
        is_red = random.random() < 1.0/3.0
        return (-card) if is_red else card

    @staticmethod
    def is_bust(points):
        # Check if the points go bust
        return points < 0.5 or points > 21.5

    @staticmethod
    def judge(state):
        # Give the reward w.r.t. the final states.
        if Easy21Env.is_bust(state[Easy21Env.PLAYER_IDX]):
            return -1
        if Easy21Env.is_bust(state[Easy21Env.DEALER_IDX]) or state[Easy21Env.PLAYER_IDX] > state[Easy21Env.DEALER_IDX]:
            return 1
        if state[Easy21Env.PLAYER_IDX] < state[Easy21Env.DEALER_IDX]:
            return -1
        return 0


def game():
    easy21_env = Easy21Env()
    (reward, state, is_term) = easy21_env.run(Easy21ManualPolicy(), need_reset=True)

    print('********************************')
    print('Your card: %d' % state[0])
    print('Dealer card: %d' % state[1])
    if reward > 0:
        print('You won!')
    elif reward < 0:
        print('You lose!')
    else:
        print('We draw!')


if __name__ == '__main__':
    game()
