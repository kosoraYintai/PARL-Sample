#coding:UTF-8
#O(NlogN) 时间内求逆序对数  
#https://www.lintcode.com/problem/reverse-pairs/description
def merge(array,other,leftStart,leftEnd,rightEnd):
    '''
    归并两个有序数组
    '''
    if array[leftEnd]<=array[leftEnd+1]:
        return 0
    cnt=0
    i=leftEnd
    j=rightEnd
    k=rightEnd
    while i>=leftStart and j>leftEnd:
        if array[i]<=array[j]:
            other[k]=array[j]
            j-=1
        else:
            #i位置的值必然大于整个(leftEnd,j]区间
            cnt+=j-leftEnd
            other[k]=array[i]
            i-=1
        k-=1
    while i>=leftStart:
        other[k]=array[i]
        i-=1
        k-=1
    while(j>leftEnd):
        other[k]=array[j]
        j-=1
        k-=1
    array[leftStart:rightEnd+1]=other[leftStart:rightEnd+1]
    return cnt;
    
def postOrder(array,other,start,end):
    if start==end:
        return 0;
    #取中点
    mid=start+end>>1;
    #递归处理左半区间
    leftCnt=postOrder(array, other, start, mid);
    #递归处理右半区间
    rightCnt=postOrder(array, other, mid+1, end);
    #归并
    mergeCnt=merge(array, other, start, mid, end);
    return leftCnt+rightCnt+mergeCnt;
 
def reversePairs_mergeSort(array):
    '''
            使用归并排序求逆序对数
    '''
    if len(array)==0:
        return 0
    other=[]
    length=len(array)
    for _ in range(length):
        other.append(0)
    return postOrder(array, other, 0, length-1)

class Solution:
    def reversePairs(self, A):
        return reversePairs_mergeSort(A)