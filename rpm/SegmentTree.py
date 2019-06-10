#coding:UTF-8
import numpy as np
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        #容量
        self.capacity = capacity  
        #线段树容器
        self.tree = np.zeros(2 * capacity - 1)
        #数组容器
        self.data = np.zeros(capacity, dtype=object) 
        
    def add(self, p, data):
        '''
        #添加一个节点
        '''
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  
        self.update(tree_idx, p)
        self.data_pointer += 1
        #超出容量时 替换旧的数据
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        '''
                 从叶子向根节点回溯，更新一条完整路径 O(logN)
        '''
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
                 寻找前缀和小于v的最大下标i,即:
        sum(tree[0] + tree[1] + ... + tree[i - i]) <= v
        """
        return self.recursion(v, 0)
    
    def recursion(self,v,parent):
        '''
                 递归算法
        '''
        left = (parent<<1) + 1
        right = left + 1
        if left >= len(self.tree):  
            #叶子节点 结束递归
            leaf = parent   
            data_idx = leaf - self.capacity + 1
            return leaf, self.tree[leaf], self.data[data_idx]    
        else:
            if v <= self.tree[left]:
                #递归左子树
                return self.recursion(v, left)
            else:
                #递归右子树
                v -= self.tree[left]
                return self.recursion(v, right)      
    
    @property
    def total_p(self):
        return self.tree[0]  


class Memory(object):
    #存储五元组(state, action, reward, terminal,next_state)
    #微小扰动，防止除0错误
    epsilon = 0.01
    #将TD-Error转化为priority
    alpha = 0.6
    #超参数,用于重要性采样,初始值小于1,最后逐渐增长到1
    beta = 0.3
    #clip Td_Error
    abs_err_upper = 1.0

    def __init__(self, capacity,beta_increment=2e-6):
        self.tree = SumTree(capacity)
        self.beta=Memory.beta
        #超参数，beta的增长速率
        self.beta_increment=beta_increment

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        #设置新添加的数据的权重为最大值
        self.tree.add(max_p, transition)

    def sample(self, n):
        '''
                  采样
        '''
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [], np.empty((n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/(min_prob+1e-4), -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        '''
        批量更新权重
        '''
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            

