#coding:UTF-8
#迷宫寻宝环境
import numpy as np
import gym
from gym.envs.classic_control import rendering
import time
np.set_printoptions(suppress=True, linewidth=500, edgeitems=8)

#定义四个动作
action_dim=4
dRow=[0,0,-1,1]
dCol=[1,-1,0,0]

#检查是否越界
def checkBounds(i,j):
    if i>=0 and i<5 and j>=0 and j<5:
        return True
    else:
        return False
#是否撞墙
def meetWall(wallList,pos):
    if pos in wallList:
        return True
    else:
        return False
class Maze(gym.Env):
    outBoundsCNT=0
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 5
    }

    def __init__(self):
        self.viewer = None
        self.row=5
        self.col=5
        self.size=self.row*self.col
        self.pos=0
        self.wallList=[(2,0),(3,2),(1,3),(4,4)]
        self.start=(0,4)
        self.end=(4,0)
    
    def reset(self):
        for _ in range(0,1024):
            i=np.random.randint(self.row)
            j=np.random.randint(self.col)
            if (i,j) not in self.wallList and (i,j)!=self.end:
                self.pos=(i,j)
                break
        return self.pos   
    
    def reset2(self,position=(0,4)):
        self.pos=position
        return self.pos      
     
    def step(self, a):
        try:
            nowRow=self.pos[0]
            nowCol=self.pos[1]
            nextRow=nowRow+dRow[a]
            nextCol=nowCol+dCol[a]
            if not checkBounds(nextRow, nextCol) :
                #越界
                return self.pos, -5.0, False, {'code':-1,'MSG':'OutOfBounds!'}
            nextPos=(nextRow,nextCol)
            if meetWall(self.wallList, nextPos):
                #撞墙
                return self.pos, -10.0, False, {'code':-1,'MSG':'MeetWall!'}
            self.pos=nextPos
            re=self.reward(self.pos)
            if self.pos==self.end:
                return self.end, re, True, {'code':1,'MSG':'Finish!'}
            return self.pos, re, False, {'code':1,'MSG':'CorrectStep!'}
        except Exception as ex:
            print(ex)
            exit()
     
    def reward(self, s):
        #到达终点
        if s == self.end:
            return 10.0
        #其他格子
        else:
            return -1.0

    def addWall(self,i,j):
        rList=np.arange(50,550,100)
        cList=np.arange(50,550,100)
        wall = rendering.FilledPolygon([(rList[i]-50,cList[j]+50),(rList[i]-50,cList[j]-50),\
        (rList[i]+50,cList[j]-50),(rList[i]+50,cList[j]+50)])
        circletrans = rendering.Transform()
        wall.add_attr(circletrans)
        wall.set_color(0,0,0)
        self.viewer.add_geom(wall)    
        
    def getPositionByPos(self,position):   
        rList=np.arange(50,550,100)
        cList=np.arange(50,550,100)
        nowRow=position[0]
        nowCol=position[1]
        return cList[nowRow],rList[nowCol]
                
    def render(self, mode='human', close=False):
        screen_width = 550
        screen_height = 550
        
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #创建网格世界
            for y in range(0,600,100):
                line=rendering.Line((0,y),(500,y))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)
            for x in range(0,600,100):
                line=rendering.Line((x,0),(x,500))
                line.set_color(0, 0, 0)
                self.viewer.add_geom(line)
            
            #墙    
            for pos in self.wallList:
                self.addWall(pos[0], pos[1])
            
            #宝藏
            target = rendering.make_circle(50)
            circletrans = rendering.Transform(translation=(450,50))
            target.add_attr(circletrans)
            target.set_color(.8,.8,.2)
            self.viewer.add_geom(target)    
            
            #机器人
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(255,0,0)
            self.viewer.add_geom(self.robot)
                   
        newX,newY=self.getPositionByPos(self.pos)
        self.robotrans.set_translation(newX,newY)                    
        return self.viewer.render(return_rgb_array=True)

if __name__ == '__main__':
    env=Maze()
    while True:
        env.reset2()
        env.render()
        time.sleep(1)
        