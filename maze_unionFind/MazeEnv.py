#coding:UTF-8
import gym
import numpy as np
from maze_unionFind.CreateMaze_With_UnionFind import generateMaze
np.set_printoptions(suppress=True, linewidth=500, edgeitems=8)
dRow=[-1,1,0,0]
dCol=[0,0,-1,1]
def checkBounds(i,j,m,n):
    if i>=0 and i<m and j>=0 and j<n:
        return True
    else:
        return False
    
def meetWall(wallList,pos):
    for arr in wallList:
        if pos[0]==arr[0] and pos[1]==arr[1]:
            return True
    return False

class MazeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self,m=5,n=5):
        self.viewer = None
        self.row=m
        self.col=n
        self.size=self.row*self.col
        self.pos=np.array([0,0])
        self.wallList=generateMaze(self.row, self.col)
    
    def reset(self,position=np.array([0,0])):
        self.pos=position
        return self.pos
       
    def step(self, a):
        nowRow=self.pos[0]
        nowCol=self.pos[1]
        nextRow=nowRow+dRow[a]
        nextCol=nowCol+dCol[a]
        if not checkBounds(nextRow, nextCol,self.row,self.col) :
            return self.pos, -2.0, False, {'info':-1,'MSG':'OutOfBounds!'}
        nextPos=np.array([nextRow,nextCol])
        if meetWall(self.wallList, nextPos):
            return self.pos, -10.0, False, {'info':-1,'MSG':'MeetWall!'}
        self.pos=nextPos
        re=self.reward(self.pos)
        if self.pos[0]==self.row-1 and self.pos[1]==self.col-1:
            return self.pos, re, True, {'info':1,'MSG':'Finish!'}
        return self.pos, re, False, {'info':1,'MSG':'NormalWork!'}

    def reward(self, s):
        if s[0]==self.row-1 and s[1]==self.col-1:
            return 10.0
        else:
            return -0.5
                
    def render(self, mode='human', close=False):
        for i in range(0,self.row):
            for j in range(0,self.col):
                pos=np.array([i,j])
                if i==0 and j==0:
                    print('S',end=' ')
                elif i==self.row-1 and j==self.col-1:
                    print('T',end=' ')
                elif not meetWall(self.wallList, pos):
                    print('O',end=' ')
                else:
                    print('X',end=' ')
            print()

  
        