#coding:UTF-8
#Double_Prioritized_DQN 继承自PARL官方的Algorithm
import paddle.fluid as fluid
from parl.framework.algorithm_base import Algorithm
import parl.layers as layers
import copy

__all__ = ['PDDQN']

def fluid_argmax(x):
    _, max_index = fluid.layers.topk(x, k=1)
    return max_index

class PDDQN(Algorithm):
    def __init__(self, model, hyperparas):
        Algorithm.__init__(self, model, hyperparas)
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.action_dim = hyperparas['action_dim']
        self.gamma = hyperparas['gamma']
        self.lr = hyperparas['lr']

    def define_predict(self, obs):
        return self.model.value(obs)

    def define_learn(self, obs, action, reward, next_obs, terminal,weight):
        #Q(s,a|θ)
        pred_value = self.model.value(obs) 
        #Q(s',a'|θ')
        targetQ_predict_value = self.target_model.value(next_obs) 
        #Q(s',a'|θ)
        next_s_predcit_value = self.model.value(next_obs) 
        #argMax[Q(s',a'|θ)]
        greedy_action = fluid_argmax(next_s_predcit_value) 
        predict_onehot = fluid.layers.one_hot(greedy_action, self.action_dim) 
        #Q(s',argMax[Q(s',a'|θ)]|θ')
        best_v = fluid.layers.reduce_sum(
            fluid.layers.elementwise_mul(predict_onehot, targetQ_predict_value),
            dim=1)
        best_v.stop_gradient = True
        #TD目标: R+γ*Q(s',argMax[Q(s',a'|θ)]|θ')
        target = reward + (
            1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * best_v
        
        action_onehot = layers.one_hot(action, self.action_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)
        
        #计算新的TD-Error
        newTd = layers.abs(target - pred_action_value)
        cost = layers.square_error_cost(pred_action_value, target)
        #weight表示样本的权重，影响cost的更新幅度
        cost=weight*cost
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(self.lr, epsilon=1e-3)
        optimizer.minimize(cost)
        return cost,newTd

    def sync_target(self, gpu_id):
        self.model.sync_params_to(self.target_model, gpu_id=gpu_id)
