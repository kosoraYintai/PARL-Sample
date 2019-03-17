#coding:UTF-8
import gym
import numpy as np
np.set_printoptions(suppress=True, linewidth=500, edgeitems=8)
dRow=[-1,1,0,0]
dCol=[0,0,-1,1]
def checkBounds(i,j):
    if i>=0 and i<5 and j>=0 and j<5:
        return True
    else:
        return False
def meetWall(wallList,pos):
    if pos in wallList:
        return True
    else:
        return False
class MazeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self):
        self.viewer = None
        self.row=5
        self.col=5
        self.size=self.row*self.col
        self.pos=0
        self.wallList=[4,6,13,22]
    
    def reset(self,position=0):
        self.pos=position
        return self.pos
    
    def reset2(self):
        for _ in range(0,1024):
            k=np.random.randint(self.size-1)
            if k not in self.wallList:
                self.pos=k
                break
        return self.pos   
       
    def step(self, a):
        try:
            nowRow=self.pos//self.row
            nowCol=self.pos%self.row
            nextRow=nowRow+dRow[a]
            nextCol=nowCol+dCol[a]
            if not checkBounds(nextRow, nextCol) :
                return self.pos, -2.0, False, {'info':-1,'MSG':'OutOfBounds!'}
            nextPos=self.row*nextRow+nextCol
            if meetWall(self.wallList, nextPos):
                return self.pos, -10.0, False, {'info':-1,'MSG':'MeetWall!'}
            self.pos=nextPos
            re=self.reward(self.pos)
            if self.pos==self.row*self.col-1:
                return self.row*self.col-1, re, True, {'info':1,'MSG':'Finish!'}
            return self.pos, re, False, {'info':1,'MSG':'NormalWork!'}
        except Exception as ex:
            print(ex)
            exit()
     
    def reward(self, s):
        if s == self.row*self.col-1:
            return 10.0
        else:
            return -0.5
                
    def render(self, mode='human', close=False):
        pass
  
        