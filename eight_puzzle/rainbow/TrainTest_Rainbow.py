#coding:UTF-8

import numpy as np
from eight_puzzle.rainbow.PrioritizedAgent import PrioritizedAgent
from eight_puzzle.rainbow.PuzzleDuelingModel import PuzzleDuelingModel
from eight_puzzle.EightPuzzleEnv import EightPuzzleEnv,A_star_dist
from rpm.FcPolicyReplayMemory import PriorityReplayMemory,Experience
from eight_puzzle.rainbow.pddqn import PDDQN
from tqdm import tqdm
from eight_puzzle.TestCase import testCase
import os

#状态的维度
StateShape=(EightPuzzleEnv.DefaultRow*EightPuzzleEnv.DefaultCol,)

#------hyper parameters start
#以下超参数均可微调

#经验池大小
MEMORY_SIZE = int(1e5)

#warm-up
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 5

#梯度更新频率
UPDATE_FREQ = 4

#衰减系数
GAMMA = 0.99

#学习率
LEARNING_RATE = 2e-4

#一共训练多少步
TOTAL=4e6

#batch-size
batchSize=64

#记录日志的频率
logFreq=10
#------hyper parameters end

#-----辅助变量start
#记录平均奖励
meanReward=0
#记录平均步数
meanEpStep=0
#记录训练的episode轮数
trainEp=0
#移动的箭头
moveDirection={0:'↑',1:'↓',2:'←',3:'→'}
#平均奖励的学习曲线
reward_curve=[]
#平均步数的学习曲线
step_curve=[]
#打印日志的频率
print_freq=10
#-----辅助变量end

#训练一个episode
def run_train_episode(env, agent, rpm):
    global trainEp
    global meanReward
    global meanEpStep
    global step_curve
    global reward_curve
    total_reward = 0
    all_cost = []
    #重置环境
    state= env.reset()
    step = 0
    #二级缓存L2Cache，存储本轮的数据
    L2Cache=[]
    trainFlag=False
    #循环每一步
    while True:
        #用ε-greedy的策略选一个动作
        action = agent.sample(state.ravel())
        #执行动作
        next_state, reward, isOver,_ = env.step(action)
        step += 1
        #将五元组(s,a,r,terminal,s')加入二级缓存
        L2Cache.append(Experience(state.ravel(), action, reward, isOver,next_state.ravel()))
        if rpm.size() > MEMORY_WARMUP_SIZE:
            trainFlag=True
            if step % UPDATE_FREQ == 0:
                #从replay_buffer中进行随机采样,比普通的DQN的五元组(s,a,r,terminal,s')多了权重weight以及SumTree的索引tree_index
                batch_state, batch_action, batch_reward, batch_isOver,batch_next_state,batch_weight,tree_index= rpm.sample_batch(batchSize)
                #执行SGD,训练参数θ
                cost,newTd = agent.learn(batch_state, batch_action, batch_reward,batch_next_state, batch_isOver,batch_weight)
                #通过新的TD-Error更新采样优先级
                rpm.updatePriority(tree_index,newTd)
                all_cost.append(float(cost))
        total_reward += reward
        state = next_state
        if isOver or step>=env._max_episode_steps:
            break
    
    #过滤掉步数大于_max_episode_steps的episode,并将二级缓存中的数据复制到经验池
    if L2Cache[-1].isOver==True:
        for exp in L2Cache:
            rpm.append(exp)
    
    if trainFlag :
        trainEp+=1
        #以滑动平均的方式记录并输出平均奖励、平均步数
        meanEpStep=meanEpStep+(step-meanEpStep)/trainEp
        meanReward=meanReward+(total_reward-meanReward)/trainEp
        if trainEp%logFreq==0:
            step_curve.append(meanEpStep)
            reward_curve.append(meanReward)
            if len(all_cost)>0:
                meanCost=np.mean(all_cost)
            else:
                meanCost=0
            if trainEp%print_freq==0:
                print('\n trainEpisode:{:},epStepCnt:{:},meanEpStep:{:.3f},total_reward:{:.3f},meanReward:{:.3f},mean_cost:{:.3f},beta:{:.3f}'\
                .format(trainEp,step,meanEpStep,total_reward, meanReward,meanCost,rpm.memory.beta))
                print()
    return total_reward, step

#测试一个episode
def run_test_episode(env,agent,board,correctCnt):
    total_reward = 0
    state= env.reset2(board).copy()
    env.render()
    #输出即时A*距离
    print('A*Dist:',A_star_dist(env.target, env.state.ravel()))
    cnt=0
    while True:
        #100%argmax的方式进行预测
        action = agent.predict(state.ravel())
        next_state, reward, isOver,_ = env.step(action)
        #输出方向
        print(moveDirection[action])
        env.render()
        print('A*Dist:',A_star_dist(env.target, env.state.ravel()))
        cnt+=1
        total_reward += reward
        state = next_state
        if isOver:
            break
    #判断答案是否正确
    if cnt!=correctCnt:
        print('Wrong Answer!')
        result=0
    else:
        print('Accepted!')
        result=1
    print('=================')
    return result

#保存模型参数
def save(agent):
    learnDir =os.getcwd()+'/models/learn'
    predictDir = os.getcwd()+'/models/predict'
    agent.save_params(learnDir,predictDir)

#恢复模型
def restore(agent):
    learnDir =os.getcwd()+'/models/learn'
    predictDir = os.getcwd()+'/models/predict'
    print('restore model from {}'.format(learnDir))
    agent.load_params(learnDir,predictDir)

#初始化rainbow模型,包括:环境-environment、神经网络-model、算法-algorithm、智能体-agent、优先级经验池-PrioritizedReplayBuffer
def init_rainbow():
    #构建环境
    env = EightPuzzleEnv()
    #动作维度
    action_dim = 4
    #超参数
    hyperparas = {
        'action_dim': action_dim,
        #学习率
        'lr': LEARNING_RATE,
        #奖励衰减系数
        'gamma': GAMMA
    }
    #model表现为Dueling-DQN
    model = PuzzleDuelingModel(act_dim=action_dim)
    #Double-DQN 的功能在algorithm中实现
    #Prioritized_Replay_Buffer的功能由algorithm+agent+ReplayMemory共同组成，缺一不可
    algorithm = PDDQN(model, hyperparas)
    agent = PrioritizedAgent(algorithm, action_dim)
    rpm =PriorityReplayMemory(max_size=MEMORY_SIZE, state_shape=StateShape)
    return env,agent,rpm

def train():
    env,agent,rpm=init_rainbow()
    #warm_up
    with tqdm(total=MEMORY_WARMUP_SIZE) as pbar:
        while rpm.size() < MEMORY_WARMUP_SIZE:
            lastSize=rpm.size()
            __, step = run_train_episode(env, agent, rpm)
            nowSize=rpm.size()
            if nowSize>lastSize:
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

    save(agent)
    print('训练结束!')
    #保存学习曲线
    nowDir = os.path.dirname(__file__)
    parentDir = os.path.dirname(nowDir)
    np.savetxt(parentDir+'/learning_curve_log/rainbow_reward.txt',reward_curve,fmt='%.3f')
    np.savetxt(parentDir+'/learning_curve_log/rainbow_step.txt',step_curve,fmt='%.3f')
    x,correctList=testCase()
    okCnt=0
    for i in range(len(x)):
        board=x[i]
        correctCnt=correctList[i]
        isOk=run_test_episode(env, agent,board,correctCnt)   
        if isOk:
            okCnt+=1
    print('准确率:{:.1f}%'.format(okCnt/len(x)*100)) 

def test():
    env,agent,_=init_rainbow()
    restore(agent)
    lst,correctList=testCase()
    okCnt=0
    for i in range(len(lst)):
        board=lst[i]
        correctCnt=correctList[i]
        isOk=run_test_episode(env, agent,board,correctCnt)   
        if isOk:
            okCnt+=1
    print('准确率:{:.1f}%'.format(okCnt/len(lst)*100)) 

if __name__ == '__main__':
    print("train or test ?")
    mode=input()
    print(mode)
    if mode=='train':
        train()
    elif mode=='test':
        test()
    else:
        print('Invalid input!')


