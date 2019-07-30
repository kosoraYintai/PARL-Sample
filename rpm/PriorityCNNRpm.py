#coding:UTF-8
#CNN版优先级经验池
import numpy as np
from collections import  namedtuple,deque
from rpm.SegmentTree import Memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'isOver','nextState'])

class PriorityCNNRpm(object):
    def __init__(self, max_size,state_shape,context_len):
        self.max_size=max_size
        self.memory = Memory(capacity=max_size,beta_increment=2e-6)
        self.state_shape = state_shape
        self._curr_pos = 0
        self.context_len = int(context_len)
        self._context = deque(maxlen=context_len)
    
    def addQueue(self,obs):
        self._context.append(obs)
    
    def clearQueue(self):
        self._context.clear()
        
    def initQueue(self,initObs):
        self.clearQueue()
        for _ in range(self.context_len-1):
            self.addQueue(np.zeros_like(initObs))
        self.addQueue(initObs)
    
    def recent_state(self):
        lst = list(self._context)
        states = [np.zeros(self.state_shape, dtype='uint8')]*0
        states.extend([k for k in lst])
        return states    
    
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
        tree_index, batch_memory, oldTd = self.memory.sample(batch_size)
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
            td[i]=oldTd[i]
        return [state, action, reward, isOver,nextState,td,tree_index]