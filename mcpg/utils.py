#coding:UTF-8
import numpy as np
from sklearn.preprocessing import StandardScaler

#根据bellman方程计算G(t)并进行归一化
def calc_discount_norm_reward(reward_list, gamma):
    discount_norm_reward = np.zeros_like(reward_list)
    discount_cumulative_reward = 0
    for i in reversed(range(0, len(reward_list))):
        discount_cumulative_reward = (
            gamma * discount_cumulative_reward + reward_list[i])
        discount_norm_reward[i] = discount_cumulative_reward
    scaler = StandardScaler() 
    normReward = scaler.fit_transform(discount_norm_reward.reshape(-1,1))  
    return normReward.ravel()
