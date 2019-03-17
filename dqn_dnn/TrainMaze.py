
#coding:UTF-8

import numpy as np
from dqn_dnn.MazeAgent import MazeAgent
from dqn_dnn.MazeModel import MazeModel
from dqn_dnn.DNNPolicyReplayMemory import DNNPolicyReplayMemory,Experience
from parl.algorithms.dqn import DQN
from tqdm import tqdm
from dqn_dnn.MazeEnv import MazeEnv
from parl.utils import logger
import os
def stateToArray(observation):
    return np.array([observation//5,observation%5]).astype(np.float32)

def seeMaze(env):
    for i in range(0,5):
        for j in range(0,5):
            pos=i*5+j
            if pos==0:
                print('S',end=' ')
            elif pos==env.row*env.col-1:
                print('T',end=' ')
            elif pos not in env.wallList:
                print('O',end=' ')
            else:
                print('X',end=' ')
        print()

def seeAction(env,agent):
    for i in range(0,5):
        for j in range(0,5):
            pos=i*5+j
            if pos not in env.wallList and pos!=env.row*env.col-1:
                d=agent.predict(stateToArray(pos))
                if d==0:
                    print('↑',end=' ')
                elif d==1:
                    print('↓',end=' ')
                elif d==2:
                    print('←',end=' ')
                else:
                    print('→',end=' ')
            elif pos==env.row*env.col-1:
                print('O',end=' ')
            else:
                print('X',end=' ')
        print()

MEMORY_SIZE = int(5e3)
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 5
StateShape=(2,)
UPDATE_FREQ = 2
GAMMA = 0.99
LEARNING_RATE = 1e-3 
TOTAL=5e4
batchSize=64
meanReward=0
trainEp=0

def run_train_episode(env, agent, rpm):
    global trainEp
    global meanReward
    total_reward = 0
    all_cost = []
    state= env.reset2()
    step = 0
    while True:
        step += 1
        action = agent.sample(stateToArray(state))
        next_state, reward, isOver,_ = env.step(action)
        rpm.append(Experience(stateToArray(state), action, reward, isOver,stateToArray(next_state)))
        # start training
        if rpm.size() > MEMORY_WARMUP_SIZE:
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
    if all_cost:
        trainEp+=1
        meanReward=meanReward+(total_reward-meanReward)/trainEp
        print('trainEpisode:',trainEp)
        print('total_reward: {:.3f}, meanReward:{:.3f} mean_cost: {:.3f}'.format(total_reward, meanReward,np.mean(all_cost)))
    return total_reward, step

    
def train():
    env = MazeEnv()
    rpm = DNNPolicyReplayMemory(max_size=MEMORY_SIZE, state_shape=StateShape)
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

    # train
    print('TrainStart!')
    total_step = 0
    while True:
        # start epoch
        __, step = run_train_episode(env, agent, rpm)
        total_step += step
        print('totalStep:{},exploration:{:.3f}'.format(total_step,agent.exploration))
        print()
        if total_step >= TOTAL:
            break
    print()
    print("训练完毕,每个位置的最佳动作:")
    print()
    seeMaze(env)
    print()
    seeAction(env, agent)
    save(agent)

def save(agent):
    learnDir = os.path.join(logger.get_dir(),'learn_01')
    predictDir = os.path.join(logger.get_dir(),'predict_01')
    agent.save_params(learnDir,predictDir)

def restore(agent):
    learnDir = os.path.join(logger.get_dir(),'learn_01')
    predictDir = os.path.join(logger.get_dir(),'predict_01')   
    logger.info('restore model from {}'.format(learnDir))
    agent.load_params(learnDir,predictDir)
        
def test():

    env = MazeEnv()
    action_dim = 4
    hyperparas = {
        'action_dim': action_dim,
        'lr': LEARNING_RATE,
        'gamma': GAMMA
    }
    model = MazeModel(act_dim=action_dim)
    algorithm = DQN(model, hyperparas)
    agent = MazeAgent(algorithm, action_dim)
    restore(agent) 
    print("\n再次加载,结果不一致:")
    print()
    seeMaze(env)
    print()
    seeAction(env, agent)
    
if __name__ == '__main__':
    train()
#    test()



