#coding:UTF-8
from contextlib import contextmanager
import cv2
import time

#捕获当前屏幕并resize成(84*84*1)的灰度图
def resizeBirdrToAtari(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return observation

@contextmanager
def trainTimer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST_Time:{}'.format(name, end - start))

