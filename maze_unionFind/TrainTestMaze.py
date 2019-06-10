
#coding:UTF-8

import numpy as np
from maze_unionFind.MazeAgent import MazeAgent
from maze_unionFind.MazeModel import MazeModel
from rpm.FcPolicyReplayMemory import Experience,FcPolicyReplayMemory
from parl.algorithms.dqn import DQN
from tqdm import tqdm
from maze_unionFind.MazeEnv import MazeEnv,meetWall
import time
import matplotlib.pyplot as plt

#查看最优路径
def seeBestPath(env,path):
    maze=np.zeros((env.row,env.col))
    for i in range(0,env.row):
        for j in range(0,env.col):
            pos=np.array([i,j])
            if meetWall(env.wallList, pos):
                maze[i][j]=1
    for pos in path:
        maze[pos[0]][pos[1]]=2     
    for i in range(0,env.row):
        for j in range(0,env.col):   
            if maze[i][j]==0:
                print('O',end=' ')
            elif maze[i][j]==2:
                print('●',end=' ')
            else:
                print('X',end=' ')
        print()
        
#------hyper parameters start
#以下超参数均可微调
#需要训练几行几列的迷宫
mazeRow=5
mazeCol=6
#经验池大小
MEMORY_SIZE = int(1e4)
#warm-up
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 4
StateShape=(2,)
#网络学习频率
UPDATE_FREQ = 2
#衰减系数
GAMMA = 0.99
#学习率
LEARNING_RATE = 1e-3
#一共训练多少步
TOTAL=1e5
#batch-size
batchSize=64
#------hyper parameters end

#平均奖励
meanReward=0
#记录训练的episode轮数
trainEp=0
#记录日志的频率
logFreq=10
#学习曲线 记录平均奖励
learning_curve=[]

def run_train_episode(env, agent, rpm):
    global trainEp
    global meanReward
    global learning_curve
    total_reward = 0
    all_cost = []
    state= env.reset()
    step = 0
    trainFlag=False
    while True:
        step += 1
        action = agent.sample(state)
        next_state, reward, isOver,_ = env.step(action)
        rpm.append(Experience(state, action, reward, isOver,next_state))
        if rpm.size() > MEMORY_WARMUP_SIZE:
            trainFlag=True
            if step % UPDATE_FREQ == 0:
                batch_state, batch_action, batch_reward, batch_isOver,batch_next_state = rpm.sample_batch(
                    batchSize)
                cost = agent.learn(batch_state, batch_action, batch_reward,
                                   batch_next_state, batch_isOver)
                all_cost.append(float(cost))
        total_reward += reward
        state = next_state
        if isOver:
            break
    if trainFlag :
        trainEp+=1
        meanReward=meanReward+(total_reward-meanReward)/trainEp
        if trainEp%logFreq==0:
            learning_curve.append(meanReward)
            print('\n trainEpisode:{} total_reward: {:.3f}, meanReward:{:.3f} mean_cost: {:.3f}'.format(trainEp,total_reward, meanReward,np.mean(all_cost)))
    return total_reward, step

def run_test_episode(env, agent, rpm):
    total_reward = 0
    state= env.reset()
    path=[]
    path.append(state.copy())
    while True:
        action = agent.sample(state)
        next_state, reward, isOver,_ = env.step(action)
        total_reward += reward
        state = next_state
        path.append(state.copy())
        if isOver:
            break
    seeBestPath(env, path)
    
def trainTest():
    env = MazeEnv(m=mazeRow,n=mazeCol)
    time.sleep(5)
    rpm = FcPolicyReplayMemory(max_size=MEMORY_SIZE, state_shape=StateShape)
    action_dim = 4
    hyperparas = {
        'action_dim': action_dim,
        'lr': LEARNING_RATE,
        'gamma': GAMMA
    }
    model = MazeModel(act_dim=action_dim)
    
    algorithm = DQN(model, hyperparas)
    agent = MazeAgent(algorithm, action_dim)
    with tqdm(total=MEMORY_WARMUP_SIZE) as pbar:
        while rpm.size() < MEMORY_WARMUP_SIZE:
            __, step = run_train_episode(env, agent, rpm)
            pbar.update(step)
            
    print()
    print('开始训练!')
    total_step = 0
    with tqdm(total=TOTAL) as pbar:
        while total_step <= TOTAL:
                __, step = run_train_episode(env, agent, rpm)
                total_step += step
                if trainEp%logFreq==0:
                    pbar.set_description('totalStep:{},exploration:{:.3f}'.format(total_step,agent.exploration))
                pbar.update(step)
    print()
    print('训练完毕\n')
    time.sleep(5)
    print("随机生成迷宫:")
    print()
    env.render()
    print()
    print("显示最优路径:")
    print()
    run_test_episode(env, agent,rpm)
    
    #平均奖励的学习曲线
    X=np.arange(0,len(learning_curve))
    X*=logFreq
    plt.title('LearningCurve')
    plt.xlabel('TrainEpisode')
    plt.ylabel('AvgReward')
    plt.plot(X,learning_curve)
    plt.show()
        
if __name__ == '__main__':
    trainTest()



