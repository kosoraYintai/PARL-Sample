#coding:UTF-8

#将flappy-bird游戏封装成标准的gym环境
#原始版本参考:https://github.com/yenchenlin/DeepLearningFlappyBird

import random
import pygame
import gym
from flappy_bird.game import flappy_bird_utils
from pygame.locals import *
from itertools import cycle

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

class BirdEnv(gym.Env):
    
    def beforeInit(self):
        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        pygame.display.set_caption('PaddlePaddle-Flappy-Bird')
        IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
        self.IMAGES=IMAGES
        self.HITMASKS=HITMASKS
        self.SOUNDS=SOUNDS
        PLAYER_WIDTH = IMAGES['player'][0].get_width()
        self.PLAYER_WIDTH=PLAYER_WIDTH
        PLAYER_HEIGHT = IMAGES['player'][0].get_height()
        self.PLAYER_HEIGHT=PLAYER_HEIGHT
        PIPE_WIDTH = IMAGES['pipe'][0].get_width()
        self.PIPE_WIDTH=PIPE_WIDTH
        PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
        self.PIPE_HEIGHT=PIPE_HEIGHT
        BACKGROUND_WIDTH = IMAGES['background'].get_width()
        self.BACKGROUND_WIDTH=BACKGROUND_WIDTH
    
    def __init__(self):
        if not hasattr(self,'IMAGES'):
            print('InitGame!')
            self.beforeInit()
        
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.3)
        self.playery = int((SCREENHEIGHT - self.PLAYER_HEIGHT) / 2.25)
        self.basex = 0
        self.baseShift = self.IMAGES['base'].get_width() - self.BACKGROUND_WIDTH

        newPipe1 = getRandomPipe(self.PIPE_HEIGHT)
        newPipe2 = getRandomPipe(self.PIPE_HEIGHT)
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1.1  # players downward accleration
        self.playerFlapAcc =  -1.2   # players speed on flapping
        self.playerFlapped = False # True when player flaps
        
    def reset(self,mode='train'):
        self.__init__()
        self.mode=mode
        action0 = 1
        observation, reward, isOver,_ = self.step(action0)
        return observation,reward,isOver
            
    def render(self):
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.FPSCLOCK.tick(FPS)
        return image_data
    
    def step(self, input_action=0):
        pygame.event.pump()
        #飞行一段距离,奖励+0.1
        reward = 0.1
        terminal = False
        if input_action == 1:
            if self.playery > -2 * self.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
#                if self.mode=='test':
#                    self.SOUNDS['wing'].play()

        # check for score
        playerMidPos = self.playerx + self.PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            #穿过一个柱子奖励加1
            pipeMidPos = pipe['x'] + self.PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
#                if self.mode=='test':
#                    self.SOUNDS['point'].play()                
                self.score += 1
                reward = self.reward(1)

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - self.PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe(self.PIPE_HEIGHT)
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
        self.upperPipes, self.lowerPipes,self.IMAGES,self.PIPE_WIDTH,self.PIPE_HEIGHT,self.HITMASKS)
        if isCrash:
            #撞到边缘或者撞到柱子,结束,并且奖励为-1
            terminal = True
            reward = self.reward(-1)
#            if self.mode=='test':
#                self.SOUNDS['hit'].play()
#                self.SOUNDS['die'].play()
        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        showScore(self.score,self)
        self.SCREEN.blit(self.IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data=self.render()
        return image_data, reward, terminal,{}
    
    def reward(self,r):
        return r
    
def getRandomPipe(PIPE_HEIGHT):
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score,obj):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += obj.IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        obj.SCREEN.blit(obj.IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += obj.IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes,IMAGES,PIPE_WIDTH,PIPE_HEIGHT,HITMASKS):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False