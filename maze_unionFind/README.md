主要有两个步骤：

1. 使用并查集+破圈法创建随机迷宫

2. 使用DQN进行训练

这样做的好处是：此算法框架对于任意M*N的迷宫都是适用的！

比如本例子中的4X4、4X5、5X4、5X5、5X6、6X5、6X6，使用的都是同一套environment、algorithm、agent、model，
甚至学习率、经验池大小、batchSize等等超参数都相同。若要训练更大的迷宫，则可微调超参数，或者增加神经网络的深度与宽度。

平均奖励的学习曲线:
![img](/maze_unionFind/log_dir/learningCurve.png) 

4*4迷宫:

![img](/maze_unionFind/log_dir/4X4.jpg)    

4*5迷宫:

![img](/maze_unionFind/log_dir/4X5.jpg)    

5*4迷宫:

![img](/maze_unionFind/log_dir/5X4.jpg)    

5*5迷宫:

![img](/maze_unionFind/log_dir/5X5.jpg)   

5*6迷宫:

![img](/maze_unionFind/log_dir/5X6.jpg)   

6*5迷宫:

![img](/maze_unionFind/log_dir/6X5.jpg)   

6*6迷宫:

![img](/maze_unionFind/log_dir/6X6.jpg)   