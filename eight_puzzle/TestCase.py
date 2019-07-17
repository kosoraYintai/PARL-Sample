#coding:UTF-8
import numpy as np
def testCase():
    '''
    创建一些测试数据
    '''
    #棋盘的初始状态作为特征
    x=[]
    x.append(np.array([[1,2,3],[4,0,5]]))
    x.append(np.array([[4,1,2],[5,0,3]]))
    x.append(np.array([[1,3,2],[5,4,0]]))
    x.append(np.array([[1,2,4],[5,0,3]]))
    x.append(np.array([[3,5,4],[2,0,1]]))
    x.append(np.array([[3,0,4],[2,5,1]]))
    x.append(np.array([[0,5,2],[4,3,1]]))
    x.append(np.array([[1,3,4],[0,2,5]]))
    x.append(np.array([[3,0,1],[4,5,2]]))
    x.append(np.array([[1,4,3],[5,2,0]]))
    #最小步数作为标签
    label=[1,5,16,11,13,14,15,14,12,14]
    return x,label