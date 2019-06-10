#coding:UTF-8
#并查集创建随机迷宫
import numpy as np

dRow=[-1,1,0,0]
dCol=[0,0,-1,1]

def find(uf,i,j):
    if uf[i][j][0]==-1:
        return np.array([i,j])
    else:
        root=find(uf,uf[i][j][0],uf[i][j][1]);
        uf[i][j][0]=root[0]
        uf[i][j][1]=root[1]
        return root

def rootEquals(root1,root2):
    if root1[0]==root2[0] and root1[1]==root2[1]:
        return True
    else:
        return False

def union(uf,i1,j1,i2,j2):
    root1=find(uf, i1, j1)
    root2=find(uf, i2, j2)
    if not rootEquals(root1, root2):
        uf[root2[0]][root2[1]][0]=root1[0]
        uf[root2[0]][root2[1]][1]=root1[1]
        
def checkBounds(m,n,i,j):
    if i>=0 and i<m and j>=0 and j<n:
        if i==0 and j==0:
            return False
        elif i==m-1 and j==n-1:
            return False
        else:
            return True
    else:
        return False
    
def checkBounds2(m,n,i,j):
    if i>=0 and i<m and j>=0 and j<n:
        return True
    else:
        return False
    
def printMaze(maze,m,n):
    for i in range(0,m):
        for j in range(0,n):
            if i==0 and j==0:
                print('S',end=' ')
            elif i==m-1 and j==n-1:
                print('T',end=' ')
            elif maze[i][j]==0:
                print('O',end=' ')
            else:
                print('X',end=' ')
        print()
       
def generateMaze(m,n):
    maze=np.ones((m,n))
    uf=np.full([m,n,2],-1)
    maze[0][0]=0;
    maze[m-1][n-1]=0;
    while(True):
        if rootEquals(find(uf, 0, 0), find(uf, m-1, n-1)):
            break
        for _ in range(1024):
            row=np.random.randint(m)
            col=np.random.randint(n)
            if checkBounds(m, n, row, col) and maze[row][col]==1:
                maze[row][col]=0
                for i in range(4):
                    #使用破圈法
                    otherRow=row+dRow[i]
                    otherCol=col+dCol[i]
                    if checkBounds2(m, n, otherRow, otherCol) and maze[otherRow][otherCol]==0:
                        union(uf, row, col, otherRow, otherCol)
                break
    printMaze(maze,m,n)
    wallList=[]
    for i in range(0,m):
        for j in range(0,n):
            if maze[i][j]==1:
                wallList.append(np.array([i,j]))
    return wallList
    
if __name__ == '__main__':
    generateMaze(5,5)