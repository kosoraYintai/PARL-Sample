#coding:UTF-8
#绘制平均奖励、平均步数的学习曲线
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
import os
if __name__ == '__main__':
    log_freq=10
    dqnReward= np.loadtxt(os.getcwd()+'/learning_curve_log/naive_dqn_reward.txt')
    dqnStep= np.loadtxt(os.getcwd()+'/learning_curve_log/naive_dqn_step.txt')
    rainbowReward= np.loadtxt(os.getcwd()+'/learning_curve_log/rainbow_reward.txt')
    rainbowStep= np.loadtxt(os.getcwd()+'/learning_curve_log/rainbow_step.txt')
    
    maxShow=512

    plt.title('对比普通DQN与Rainbow的平均奖励')
    plt.xlabel('迭代轮数')
    plt.ylabel('Avg Reward')
    x1=np.arange(0,maxShow)
    x1*=log_freq
    plt.plot(x1,dqnReward[0:maxShow],color='blue',linewidth=2.5,label='普通DQN的平均奖励')
    x2=np.arange(0,maxShow)
    x2*=log_freq
    plt.plot(x2,rainbowReward[0:maxShow],color='red',linewidth=2.5,label='Rainbow的平均奖励')    
    plt.legend()
    
    plt.figure()
    plt.title('对比普通DQN与Rainbow的平均步数')
    plt.xlabel('迭代轮数')
    plt.ylabel('Avg Step')
    x3=np.arange(0,maxShow)
    x3*=log_freq
    plt.plot(x3,dqnStep[0:maxShow],color='blue',linewidth=2.5,label='普通DQN的平均步数')
    x4=np.arange(0,maxShow)
    x4*=log_freq
    plt.plot(x4,rainbowStep[0:maxShow],color='red',linewidth=2.5,label='Rainbow的平均步数')       
    plt.legend()
    plt.show()