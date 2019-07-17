#coding:utf-8
#使用随机数进行打表,生成所有最优解,并自动写代码
#同时观察数据的分布情况
from eight_puzzle.naive_dqn.TrainTestPuzzle import init_environment,restore
from eight_puzzle.EightPuzzleEnv import NDist
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('whitegrid')
np.set_printoptions(suppress=True, linewidth=500, edgeitems=8,precision=4)
def putMap(dic,cnt,lst):
    key=""
    for i in lst:
        key+=str(i)
    if not (key in dic):
        dic[key]=cnt
        
def fac(n):
    '''
    阶乘
    '''
    if n==1:
        return 1
    else:
        return n*fac(n-1)

def write_ac_code(dic):
    '''
    自动写AC代码
    '''
    print()
    print('''def boardToKey(board):
    key=""
    for array in board:
        for i in array:
            key+=str(i)
    return key
    ''')
    
    print('''class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:''')
    print('        d={}')
    for entry in dic.items():
        print('        d["'+entry[0]+'"]='+str(entry[1]))
    print('        key=boardToKey(board)')
    print('''        if key in d:
            return d[key]
        else:
            return -1''')
        
if __name__ == '__main__':
    dic={}
    dic['123450']=0
    env,agent=init_environment()
    restore(agent)
    MAXITER=65535
    Total=fac(6)>>1
    stepList=[0]
    for i in range(MAXITER):
        state=env.reset()
        initState=state.copy()
        cnt=0
        while True:
            action = agent.predict(state.ravel())
            next_state, reward, isOver,_ = env.step(action)
            cnt+=1
            state = next_state
            if isOver or cnt>=env._max_episode_steps:
                break
        stepList.append(cnt)
        if i%50==0:
            print('执行次数:',i,'完成度:{:.2f}%'.format(len(dic)/Total*100))
        putMap(dic, cnt, initState.ravel())
        if len(dic)>=Total:
            print('end!',i)
            break
    write_ac_code(dic)

    #绘制数据分布图
    print('\n\n')
    print('min:',np.min(stepList))
    print('mean:',np.mean(stepList))
    print('median:',np.median(stepList))
    print('max:',np.max(stepList))
    print('std:',np.std(stepList))
    print('猜测数据服从正态分布')
    max_value=32
    XTick=np.linspace(0,32,17)
    plt.hist(stepList,bins=32,alpha=0.5,color='red',edgecolor='red',density=True,range=(0,max_value))
    x=np.linspace(0,max_value,64)
    y=NDist(x,np.median(stepList),np.std(stepList))
    plt.xticks(XTick)
    plt.plot(x,y,color='blue',lw=2)
    plt.show()