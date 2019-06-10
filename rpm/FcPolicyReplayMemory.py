
import numpy as np
from collections import  namedtuple
from rpm.SegmentTree import Memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'isOver','nextState'])

class FcPolicyReplayMemory(object):
    '''
    用于存储全链接神经网络的经验池,区别于卷积神经网络网络
    '''
    def __init__(self, max_size, state_shape):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.state = np.zeros((self.max_size, ) + self.state_shape, dtype='float32')
        self.action = np.zeros((self.max_size, ), dtype='int32')
        self.reward = np.zeros((self.max_size, ), dtype='float32')
        self.isOver = np.zeros((self.max_size, ), dtype='bool')
        self.nextState = np.zeros((self.max_size, ) + self.state_shape, dtype='float32')
        self._curr_pos = 0
        print()
        
    def append(self, exp):
        index = self._curr_pos % self.max_size
        self._assign(index, exp)
        self._curr_pos += 1

    def sample(self, idx):
        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        isOver = self.isOver[idx]
        nextObs=self.nextState[idx]
        return state, reward, action, isOver,nextObs

    def size(self):
        return self._curr_pos
    
    def __len__(self):
        if self._curr_pos>self.max_size:
            return self.max_size
        else:
            return self._curr_pos

    def _assign(self, pos, exp):
        self.state[pos] = exp.state
        self.reward[pos] = exp.reward
        self.action[pos] = exp.action
        self.isOver[pos] = exp.isOver
        self.nextState[pos]=exp.nextState

    def sample_batch(self, batch_size):
        if self._curr_pos > self.max_size:
            sample_index = np.random.choice(self.max_size, size=batch_size)
        else:
            sample_index = np.random.choice(self._curr_pos, size=batch_size)
        batch_exp = [self.sample(i) for i in sample_index]
        return self._process_batch(batch_exp)

    def _process_batch(self, batch_exp):
        state = np.asarray([e[0] for e in batch_exp], dtype='float32')
        reward = np.asarray([e[1] for e in batch_exp], dtype='float32')
        action = np.asarray([e[2] for e in batch_exp], dtype='int8')
        isOver = np.asarray([e[3] for e in batch_exp], dtype='bool')
        nextObs = np.asarray([e[4] for e in batch_exp], dtype='float32')
        return [state, action, reward, isOver,nextObs]

class PriorityReplayMemory(object):
    '''
    优先级队列经验池
    '''
    def __init__(self, max_size,state_shape):
        self.max_size=max_size
        self.memory = Memory(capacity=max_size,beta_increment=1e-6)
        self.state_shape = state_shape
        self._curr_pos = 0
    
    def updatePriority(self,tree_idx,abs_errors):
        self.memory.batch_update(tree_idx, abs_errors) 
        
    def append(self, exp):
        self.memory.store(exp)
        self._curr_pos +=1
        
    def size(self):
        return self._curr_pos
    
    def __len__(self):
        if self._curr_pos>self.max_size:
            return self.max_size
        else:
            return self._curr_pos


    def sample_batch(self, batch_size):
        tree_index, batch_memory, old_td = self.memory.sample(batch_size)
        state = np.zeros((batch_size, ) + self.state_shape, dtype='float32')
        action = np.zeros((batch_size, ), dtype='int32')
        reward = np.zeros((batch_size, ), dtype='float32')
        isOver = np.zeros((batch_size, ), dtype='bool')
        nextState = np.zeros((batch_size, ) + self.state_shape, dtype='float32')      
        td = np.zeros((batch_size, ), dtype='float32')
        for i in range(0,len(batch_memory)):
            exp=batch_memory[i]
            state[i]=exp.state
            action[i]=exp.action
            reward[i]=exp.reward
            isOver[i]=exp.isOver
            nextState[i]=exp.nextState
            td[i]=old_td[i]
        return [state, action, reward, isOver,nextState,td,tree_index]
