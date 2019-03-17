
import numpy as np
from collections import  namedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'isOver','nextState'])

class DNNPolicyReplayMemory(object):
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

