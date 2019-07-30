#coding:utf-8
import parl.layers as layers
from parl.framework.model_base import Model
import paddle.fluid as fluid
class BirdDuelingModel(Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim
        #padding方式为valid
        p_valid=0
        self.conv1 = layers.conv2d(
            num_filters=32, filter_size=8, stride=4, padding=p_valid, act='relu')
        self.conv2 = layers.conv2d(
            num_filters=64, filter_size=4, stride=2, padding=p_valid, act='relu')
        self.conv3 = layers.conv2d(
            num_filters=64, filter_size=3, stride=1, padding=p_valid, act='relu')
        self.fc0=layers.fc(size=512)
        self.fc1 = layers.fc(size=act_dim)
        self.valueFc = layers.fc(size=1)
        self.advantageFc = layers.fc(size=act_dim)
        
    def value(self, obs):
        #输入归一化
        obs = obs / 255.0
        out = self.conv1(obs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = layers.flatten(out, axis=1)
        out = self.fc0(out)
        out = self.fc1(out)
        V=self.valueFc(out)
        advantage=self.advantageFc(out)
        #计算优势函数的均值,用于归一化
        advMean=fluid.layers.reduce_mean(advantage, dim=1, keep_dim=True)
        #状态行为值函数Q=V+A
        Q = advantage + (V - advMean)
        return Q
