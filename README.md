# PARL-Sample
simple examples for baidu parl, easy to understand
* [maze using MCPG](/mcpg/)

![img](/mcpg/result-output/result.gif)

* [maze using union-find(to do)](/dqn_dnn/)

![img](/dqn_dnn/log_dir/train.jpg)

* [Flappy-Bird with Nature-2015-CNN](/flappy_bird/)

![img](/flappy_bird/log_dir/birdTest01.gif)    ![img](/flappy_bird/log_dir/birdTest02.gif)

4 mins version in bilibili: [DQN for Flappy-Bird](https://www.bilibili.com/video/av49282860/)

my blog in baidu ai-studio: [paddlepaddle版Flappy-Bird](https://aistudio.baidu.com/aistudio/#/projectdetail/51092)

* [parl's setup.py for windows](/setup-for-windows/)
If you meet some exceptions in using 'pip install parl' command for windows, please use 'python setup.py install' in your local file system
and use the setup.py in my project instead of parl's default setup.py.

(在windows下使用 'pip install parl' 命令可能会出现错误，这时候建议使用 'python setup.py install' 命令进行本地安装，并使用本项目的setup.py代替parl默认的setup.py)

## Dependencies

* Sklearn
* numpy
* gym
* paddlepaddle
* parl
* opencv-python
* pygame
* tqdm