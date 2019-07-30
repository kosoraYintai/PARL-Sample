#coding:utf-8
import numpy as np
import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.agent_base import Agent

IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4

#智能体
class BirdPriorityAgent(Agent):
    def __init__(self, algorithm, action_dim):
        super(BirdPriorityAgent, self).__init__(algorithm)
        
        self.action_dim = action_dim
        self.global_step = 0
        
        #每训练多少步更新target网络,超参数可调
        self.update_target_steps = 5000
        #初始探索概率ε,超参数可微调
        self.exploration = 0.8
        #每步探索的衰减程度,超参数可微调
        self.exploration_dacay=2e-7
        #最小探索概率,超参数可微调
        self.min_exploration=0.1

    def build_program(self):
        self.learn_programs = []
        self.predict_programs=[]
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs',
                shape=[CONTEXT_LEN, IMAGE_SIZE[0], IMAGE_SIZE[1]],
                dtype='float32')
            self.value = self.alg.define_predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs',
                shape=[CONTEXT_LEN, IMAGE_SIZE[0], IMAGE_SIZE[1]],
                dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs',
                shape=[CONTEXT_LEN, IMAGE_SIZE[0], IMAGE_SIZE[1]],
                dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            weight=layers.data(name='weight', shape=[], dtype='float32') 
            self.cost,self.newTd = self.alg.define_learn(obs, action, reward, next_obs,terminal,weight)
        self.learn_programs.append(self.learn_program)
        self.predict_programs.append(self.pred_program)
    
    #ε-greedy
    def sample(self, obs):
        sample = np.random.random()
        if sample < self.exploration:
            act = np.random.randint(self.action_dim)
        else:
            obs = np.expand_dims(obs, axis=0)
            pred_Q = self.fluid_executor.run(
                    self.pred_program,
                    feed={'obs': obs.astype('float32')},
                    fetch_list=[self.value])[0]
            pred_Q = np.squeeze(pred_Q, axis=0)
            act = np.argmax(pred_Q)
        self.exploration = max(self.min_exploration, self.exploration - self.exploration_dacay)
        return act
    
    #预测
    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)
        return act
    
    #学习
    def learn(self, obs, act, reward, next_obs, terminal,weight):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target(self.gpu_id)
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal,
            'weight':weight
        }
        cost,newTd = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost,self.newTd])
        return cost[0],newTd
    
    #保存模型
    def save_params(self, learnDir,predictDir):
        fluid.io.save_params(
                executor=self.fluid_executor,
                dirname=learnDir,
                main_program=self.learn_programs[0])   
        fluid.io.save_params(
                executor=self.fluid_executor,
                dirname=predictDir,
                main_program=self.predict_programs[0])     
       
    #加载模型
    def load_params(self, learnDir,predictDir): 
        fluid.io.load_params(
                    executor=self.fluid_executor,
                    dirname=learnDir,
                    main_program=self.learn_programs[0])  
        fluid.io.load_params(
                    executor=self.fluid_executor,
                    dirname=predictDir,
                    main_program=self.predict_programs[0])    