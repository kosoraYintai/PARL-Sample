#coding:UTF-8
#标准的A*算法
import heapq
import numpy as np
from eight_puzzle.TestCase import testCase
from eight_puzzle.EightPuzzleEnv import swap,checkBounds,checkReversePair,findZeroPos,dRow,dCol
#解空间节点
class Node:
    def __init__(self,row,col,gn,hn,board,parent):
        self.row = row
        self.col = col
        #g(n)表示当前已经走了多少步
        self.gn = gn
        #h(n)表示对剩余距离进行预估
        self.hn = hn
        self.board = board
        self.parent = parent
        #总的估价函数:f(n)=g(n)+h(n)
        self.fn=gn+hn
    
    #重载运算符，用于排序
    def __lt__(self,other):
        return self.fn<other.fn

class AStarSolution:
    m=2
    n=3
    def matrixEquals(self,m1,m2):
        return np.array_equal(m1.ravel(), m2.ravel())
    
    #计算有多少棋子未摆放正确
    def getHn(self,target,now):
        cnt=0
        for i in range(0,AStarSolution.m):
            for j in range(0,AStarSolution.n):
                if  now[i][j]!=target[i][j]:
                    cnt+=1
        return cnt
    
    #回溯至根节点依次检测,防止出现回路
    def checkPath(self,child):
        p=child.parent
        while p!=None :
            if self.matrixEquals(p.board, child.board):
                return False
            p=p.parent
        return True
    
    def slidingPuzzle(self, board):
        board=np.array(board)
        if len(board)!=AStarSolution.m or len(board[0])!=AStarSolution.n:
            return -1
        #如果逆序对为奇数，则无法走到终点
        if not checkReversePair(board.ravel()):
            return -1
        zeroPos=findZeroPos(board, AStarSolution.m, AStarSolution.n)
        startRow=zeroPos[0]
        startCol=zeroPos[1]
        m=AStarSolution.m
        n=AStarSolution.n
        #终止状态
        target=np.zeros([m,n],dtype=np.int16)
        for i in range(0,m):
            for j in range(0,n):
                target[i,j]=i*n+j+1
        target[m-1][n-1]=0
        #优先队列,用于存放活节点,使用最小堆实现
        heap=[]
        #根节点加入队列
        heapq.heappush(heap,Node(startRow, startCol, 0, self.getHn(target, board), board.copy(), None))
        while len(heap)>0:
            p=heapq.heappop(heap)
            pBoard=p.board
            #走到终点
            if self.matrixEquals(target, pBoard):
                return p.gn
            #生成四个子节点
            for k in range(0,len(dRow)):
                chRow=p.row+dRow[k]
                chCol=p.col+dCol[k]
                #若子节点不越界、且未被访问，则加入到活节点队列
                if checkBounds(chRow, chCol,m,n):
                    chBoard=pBoard.copy()
                    swap(chBoard, chRow, chCol, p.row, p.col)
                    ch=Node(chRow, chCol, p.gn+1, self.getHn(target, chBoard), chBoard, p)
                    if self.checkPath(ch):
                        heapq.heappush(heap,ch)
        return -1     
if __name__ == '__main__':
    s=AStarSolution()
    x,label=testCase()
    for i in range(len(x)):
        board=x[i]
        correctCnt=label[i]
        cntStep=s.slidingPuzzle(board)
        assert cntStep==correctCnt
    print("Finish!")