# -*- coding: utf-8 -*-
"""
Created on 20-1-10

@author: caoweiquan
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from com.fatty.rl.policy import LookupBasedPolicy
from easy21 import Easy21Env
from com.fatty.rl.monte_carlo import MonteCarloLookup


class Easy21Policy(LookupBasedPolicy):
    def __init__(self):
        super(Easy21Policy, self).__init__({Easy21Env.ACTION_HIT, Easy21Env.ACTION_STICK})


if __name__ == '__main__':
    controller = MonteCarloLookup(Easy21Policy(), discount=1, explore_epsilon_N0=100)
    optimal_policy = controller.learn(Easy21Env(), num_rounds=10000, round_max_itr=0)  # Train for 100 rounds.
    # Visualize the optimal value function.
    optimal_v = controller.V
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    # 定义三维数据
    xx = np.arange(1, 22, 1)
    yy = np.arange(1, 11, 1)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros(X.shape, np.float)
    row, col = X.shape
    for i in range(row):
        for j in range(col):
            Z[i, j] = (X[i, j], Y[i, j]) in optimal_v and optimal_v[(X[i, j], Y[i, j])] or -1
    # 作图
    ax3.plot_surface(X, Y, Z, cmap='rainbow')
    plt.show()
    # Test it!
    env = Easy21Env()
    num_won = 0
    for i in range(100):
        (reward, state, is_term) = env.run(optimal_policy, need_reset=True, max_itr=0)
        # print('********************************')
        # print('Your card: %d' % state[0])
        # print('Dealer card: %d' % state[1])
        if reward > 0:
            num_won += 1
            # print('You won!')
        elif reward < 0:
            pass
            # print('You lose!')
        else:
            pass
            # print('We draw!')
    print('================================')
    print('Won rate: %d/%d' % (num_won, 100))
