#coding:UTF-8
from mcpg.MazeEnv import Maze
import time
import numpy as np
from parl.algorithms import PolicyGradient
from parl.utils import logger
from mcpg.utils import calc_discount_norm_reward
from mcpg.MazeModel import MazeModel
from mcpg.MazeAgent import MazeAgent
import matplotlib.pyplot as plt

OBS_DIM = 2
ACT_DIM = 4
GAMMA = 0.99
LEARNING_RATE = 1e-3

MeanReward=0
ErrorCount=0      
 
def run_train_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)
        nextObs, reward, done, info = env.step(action)      
        global ErrorCount
        infoCode=info['code']
        #OutOfBounds or MeetWall
        if infoCode==-1:
            ErrorCount+=1
        obs=nextObs
        reward_list.append(reward)
        if done:
            break
    return obs_list, action_list, reward_list


def run_evaluate_episode(env, agent):
    obs = env.reset()
    time.sleep(1)
    all_reward = 0
    while True:
        env.render()
        action = agent.predict(obs)
        nextObs, reward, done, info = env.step(action)
        global ErrorCount
        infoCode=info['code']
        #OutOfBounds or MeetWall
        if infoCode==-1:
            ErrorCount+=1
        obs=nextObs
        all_reward += reward
        time.sleep(0.3)
        if done:
            env.render()
            time.sleep(1)
            break
    return all_reward


def trainTest():
    env = Maze()
    model = MazeModel(act_dim=ACT_DIM)
    alg = PolicyGradient(model, hyperparas={'lr': LEARNING_RATE})
    agent = MazeAgent(alg, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    
    beforeTrainReward=[]
    global MeanReward
    print('BeforeTrain:')
    for i in range(1,129):
        obs_list, action_list, reward_list = run_train_episode(env, agent)
        MeanReward=MeanReward+(sum(reward_list)-MeanReward)/i
        beforeTrainReward.append(MeanReward)
        logger.info('Episode:{},nowReward: {:.2f},avgReward:{:.2f}'.format(i,np.sum(reward_list),MeanReward))
    global ErrorCount
    ErrorCountBeforeTrain=ErrorCount
    print('TrainStart!')   
    trainReward=[] 
    MeanReward=0
    
    #MCPG
    #迭代十万个episode
    for i in range(1,100001):
        #采样
        obs_list, action_list, reward_list = run_train_episode(env, agent)
        #使用滑动平均的方式计算奖励的期望
        MeanReward=MeanReward+(sum(reward_list)-MeanReward)/i
        if i%10==0:
            trainReward.append(MeanReward)
            logger.info("Episode:{}, nowReward:{:.2f},avgReward:{:.2f}.".format(i, sum(reward_list),MeanReward))
        if MeanReward>=0 and i>=256:
            break
        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        #通过backup的方式计算G(t)，并进行归一化处理
        batch_reward = calc_discount_norm_reward(reward_list, GAMMA)
        #学习
        agent.learn(batch_obs, batch_action, batch_reward)
    
    print('TrainEnd!')
    input()
    MeanReward=0
    testReward=[]
    ErrorCount=0
    for i in range(1,129):
        all_reward = run_evaluate_episode(env, agent)
        logger.info('Test reward: {:.3f}'.format(all_reward))
        MeanReward=MeanReward+(all_reward-MeanReward)/i
        testReward.append(MeanReward)
        logger.info('Episode:{},nowReward: {:.2f},avgReward:{:.2f}'.format(i,all_reward,MeanReward))
    ErrorCountAfterTrain=ErrorCount
    
    print()
    print('ErrorCountBeforeTrain:{},ErrorCountAfterTrain:{}'.format(ErrorCountBeforeTrain,ErrorCountAfterTrain))
    plt.title('BeforeTrain')
    plt.xlabel('episode')
    plt.ylabel('AvgReward')
    plt.plot(beforeTrainReward)
    
    plt.figure()
    X=np.arange(0,len(trainReward))
    X*=10
    plt.title('Training')
    plt.xlabel('episode')
    plt.ylabel('AvgReward')
    plt.plot(X,trainReward)
    
    plt.figure()
    plt.title('AfterTrain')
    plt.xlabel('episode')
    plt.ylabel('AvgReward')
    plt.plot(testReward)
    
    plt.show()
    
if __name__ == '__main__':
    trainTest()
