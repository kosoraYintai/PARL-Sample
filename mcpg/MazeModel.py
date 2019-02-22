#coding:UTF-8
#神经网络模型
import parl.layers as layers
from parl.framework.model_base import Model

class MazeModel(Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim
        hid1_size = 32
        hid2_size = 32
        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=hid2_size, act='tanh')
        self.fcOut = layers.fc(size=act_dim,act='softmax')

    def policy(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.fcOut(out)
        return out
