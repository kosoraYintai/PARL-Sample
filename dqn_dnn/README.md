
* 训练完成并save模型。
* 第一张图，左上角的S表示入口，右下角的T表示出口,O表示可走，X表示障碍物
* 第二张图，箭头表示每个位置的最优动作，此时没有任何撞墙的箭头

![img](/dqn_dnn/log_dir/train.jpg)

* 重新load模型，箭头完全乱了，不知道什么原因

![img](/dqn_dnn/log_dir/test.jpg)


* agent层代码：
`
    def save_params(self, learnDir,predictDir):
        fluid.io.save_params(
                executor=self.fluid_executor,
                dirname=learnDir,
                main_program=self.learn_programs[0])   
        fluid.io.save_params(
                executor=self.fluid_executor,
                dirname=predictDir,
                main_program=self.predict_programs[0])        
    
    def load_params(self, learnDir,predictDir): 
        fluid.io.load_params(
                    executor=self.fluid_executor,
                    dirname=learnDir,
                    main_program=self.learn_programs[0])  
        fluid.io.load_params(
                    executor=self.fluid_executor,
                    dirname=predictDir,
                    main_program=self.predict_programs[0])  
`

* train层代码：
`
	def save(agent):
		learnDir = os.path.join(logger.get_dir(),'learn_01')
		predictDir = os.path.join(logger.get_dir(),'predict_01')
		agent.save_params(learnDir,predictDir)

	def restore(agent):
		learnDir = os.path.join(logger.get_dir(),'learn_01')
		predictDir = os.path.join(logger.get_dir(),'predict_01')   
		logger.info('restore model from {}'.format(learnDir))
		agent.load_params(learnDir,predictDir)
`