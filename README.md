# PARL-Sample
深度强化学习，使用百度PARL框架
* [策略梯度算法走迷宫](/mcpg/)

![img](/mcpg/result-output/result.gif)

* [并查集+DQN版通用迷宫](/maze_unionFind/)

![img](/maze_unionFind/log_dir/result.jpg)

* [Nature-2015-CNN版Flappy-Bird](/flappy_bird/)

![img](/flappy_bird/log_dir/birdTest01.gif)    ![img](/flappy_bird/log_dir/birdTest02.gif)

B站4分钟完整版: [DQN for Flappy-Bird](https://www.bilibili.com/video/av49282860/)

百度ai-studio博客地址: [paddlepaddle版Flappy-Bird](https://aistudio.baidu.com/aistudio/#/projectdetail/51092)


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

(If you meet some exceptions by using 'pip install parl' command for windows, please use 'python setup.py install' in your local file system
and use the setup.py in my project instead of parl's default setup.py.)
