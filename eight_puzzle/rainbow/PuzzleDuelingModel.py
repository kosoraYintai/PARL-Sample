#coding:UTF-8
#Dueling-DQN模型
from parl.framework.model_base import Model
import paddle.fluid as fluid
import parl.layers as layers
class PuzzleDuelingModel(Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim
        
        self.fc0=layers.fc(size=64,act='tanh') 
        self.fc1=layers.fc(size=64,act='tanh')
        #状态值函数V
        self.valueFc = layers.fc(size=1)
        #优势函数A
        self.advantageFc = layers.fc(size=act_dim)
        
    def value(self, obs):
        out = self.fc0(obs)
        out = self.fc1(out)
        V=self.valueFc(out)
        advantage=self.advantageFc(out)
        #计算优势函数的均值,用于归一化
        advMean=fluid.layers.reduce_mean(advantage, dim=1, keep_dim=True)
        #状态行为值函数Q=V+A
        Q = advantage + (V - advMean)
        return Q
