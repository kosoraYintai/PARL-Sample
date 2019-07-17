#coding:utf-8   
#使用DFS生成所有的全排列 
#https://leetcode.com/problems/permutations-ii/
def swapArray(nums,i,j):
    '''
       交换
    '''
    t=nums[i]
    nums[i]=nums[j]
    nums[j]=t

def addResult(nums,result):
    '''
    添加一个结果
    '''
    lst=[]
    for k in nums:
        lst.append(k)
    result.append(lst)

def canSwap(nums,level,i):
    '''
    检测[level,i)区间是否存在和i相同的数字
    '''
    for j in range(level,i):
        if nums[j]==nums[i]:
            return False
    return True

def dfs(nums,result,n,level):
    '''
    对解空间进行深度优先搜索
    '''
    if level==n-1 :
        #到达叶子结点
        addResult(nums, result)
    else:
        dfs(nums, result, n, level+1)
        #生成所有的符合条件的子节点
        for i in range(level+1,n):
            if canSwap(nums, level, i):
                swapArray(nums, level, i)
                dfs(nums, result, n, level+1)
                #回溯
                swapArray(nums, level, i)

class Solution:
    def permuteUnique(self, nums):
        result=[]
        if nums==None or len(nums)==0:
            return result
        nums.sort()
        dfs(nums, result, len(nums), 0)
        return result
if __name__ == '__main__':
    s=Solution()
    lst=[2,3,1]
    result=s.permuteUnique(lst)
    print(result)