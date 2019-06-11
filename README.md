# PARL-Sample
游戏智能+少儿编程，使用深度强化学习框架PARL，包含PG、DQN、Rainbow等算法

## [策略梯度算法走迷宫](/mcpg/)

![img](/mcpg/result-output/result.gif)

* 论文地址：[PARL源码走读——使用策略梯度算法求解迷宫寻宝问题](https://mp.weixin.qq.com/s/D8Z2F-tOjmwl8UpmGee4Bg)

## [并查集+DQN版通用迷宫](/maze_unionFind/)

![img](/maze_unionFind/log_dir/result.jpg)

## [Nature-2015-CNN版Flappy-Bird](/flappy_bird/)

![img](/flappy_bird/log_dir/birdTest01.gif)    ![img](/flappy_bird/log_dir/birdTest02.gif)

* 论文地址：[PaddlePaddle版Flappy-Bird—使用DQN算法实现游戏智能](https://mp.weixin.qq.com/s/R7gV0j5RMg9080d7qvvn2g)

* B站4分钟完整版: [DQN for Flappy-Bird](https://www.bilibili.com/video/av49282860/)

* 百度ai-studio博客地址: [paddlepaddle版Flappy-Bird](https://aistudio.baidu.com/aistudio/#/projectdetail/51092)


## [使用Rainbow模型刷ACM题(todo)](/eight_puzzle/)

整体架构：

![img](/eight_puzzle/imgs/architecture.jpg) 

## 依赖库

* Sklearn
* numpy
* gym
* paddlepaddle
* parl
* opencv-python
* pygame
* tqdm

## 提示
* [parl's setup.py for windows](/setup-for-windows/)

在windows下使用 'pip install parl' 命令可能会出现错误，这时候建议使用 'python setup.py install' 命令进行本地安装，并使用本项目的setup.py代替parl默认的setup.py

