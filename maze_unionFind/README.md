主要有两个步骤：

1. 使用并查集+破圈法创建随机迷宫

2. 使用DQN进行训练

这样做的好处是：此算法框架对于任意M*N的迷宫都是适用的！
比如本例子中的4*4、4*5、5*4、5*5、5*6、6*5、6*6，使用的都是同一套environment、algorithm、agent、model，
甚至学习率、经验池大小、batchSize等等超参数都相同。若要训练更大的迷宫，则可微调超参数，或者增加神经网络的深度与宽度。

平均奖励的学习曲线:
![img](/maze_unionFind/log_dir/learningCurve.png) 
4*4迷宫:

![img](/flappy_bird/maze_unionFind/4X4.jpg)    

4*5迷宫:
![img](/flappy_bird/maze_unionFind/4X5.jpg)    
