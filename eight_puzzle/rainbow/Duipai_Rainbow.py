#coding:UTF-8
#使用标准的A*算法与Rainbow进行对拍，检测正确率
from eight_puzzle.rainbow.TrainTest_Rainbow import restore
from eight_puzzle.rainbow.TrainTest_Rainbow import init_rainbow
from eight_puzzle.StandardAStarSolve import AStarSolution
from eight_puzzle.Permutation import Solution as Perm
from eight_puzzle.EightPuzzleEnv import arrayToMatrix,checkReversePair
import time
from tqdm import tqdm
import numpy as np

def putMap(dic,cnt,lst):
    key=""
    for i in lst:
        key+=str(i)
    if not (key in dic):
        dic[key]=cnt


if __name__ == '__main__':
    rainbowDic={}
    aStarDic={}
    #记录Rainbow的结果
    rainbowDic['123450']=0
    #记录A*的结果
    aStarDic['123450']=0
    env,agent,_=init_rainbow()
    stdSolve=AStarSolution()
    restore(agent)
    perm=Perm()
    nums=[1,2,3,4,5,0]
    permList=perm.permuteUnique(nums)
    timeDQN=0
    timeStd=0
    TOTAL=len(permList)
    idx=0
    #若步数超过阈值,则认为智能体进入了死循环
    step_threshold=256
    
    with tqdm(total=TOTAL) as pbar:
        while idx < TOTAL:
            lst=permList[idx]
            state=env.reset2(arrayToMatrix(lst, 2, 3))
            idx+=1
            #过滤掉逆序对数为奇数的状态以及终止状态
            if not checkReversePair(lst) or np.array_equal(lst, env.target):
                pbar.update(1)
                continue
            initState=state.copy()
            cnt=0
            start = time.time()
            while True:
                action = agent.predict(state.ravel())
                next_state, reward, isOver,_ = env.step(action)
                cnt+=1
                state = next_state
                if isOver or cnt>=step_threshold:
                    break
            putMap(rainbowDic, cnt, initState.ravel())
            end =time.time()
            timeDQN+=end-start
            start = time.time()
            putMap(aStarDic, stdSolve.slidingPuzzle(initState), initState.ravel())
            end=time.time()
            timeStd+=end-start      
            pbar.update(1)

    print()
    TOTAL>>=1
    assert len(rainbowDic)==TOTAL and len(rainbowDic)==len(aStarDic)
    okCnt=0
    errCnt=0
    for entry in rainbowDic.items():
        if aStarDic.get(entry[0])==entry[1]:
            okCnt+=1
        
        if entry[1]>=step_threshold:
            errCnt+=1
    print('死循环个数:',errCnt)
    print('准确率:{:.2f}%'.format(okCnt/TOTAL*100))
    print('强化学习平均耗时: {:.3f} 秒\nA*算法平均耗时: {:.3f} 秒'.format(timeDQN/TOTAL,timeStd/TOTAL))
    