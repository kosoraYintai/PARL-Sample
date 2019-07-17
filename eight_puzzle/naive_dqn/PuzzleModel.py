#coding:UTF-8
#普通DQN模型
from parl.framework.model_base import Model
import parl.layers as layers

class PuzzleModel(Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim
        #神经网络的层数、宽度均可微调
        self.fc0=layers.fc(size=64,act='tanh') 
        self.fc1=layers.fc(size=64,act='tanh')
        self.fc = layers.fc(size=act_dim)
        
    def value(self, obs):
        out = self.fc0(obs)
        out = self.fc1(out)
        out=self.fc(out)
        return out
