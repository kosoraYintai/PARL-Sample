
* 训练完成并save模型

![img](/dqn_dnn/log_dir/train.jpg)

* 重新load模型

![img](/dqn_dnn/log_dir/test.jpg)


* agent层代码:

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

* train层代码:

	def save(agent):
		learnDir = os.path.join(logger.get_dir(),'learn_01')
		predictDir = os.path.join(logger.get_dir(),'predict_01')
		agent.save_params(learnDir,predictDir)

	def restore(agent):
		learnDir = os.path.join(logger.get_dir(),'learn_01')
		predictDir = os.path.join(logger.get_dir(),'predict_01')   
		logger.info('restore model from {}'.format(learnDir))
		agent.load_params(learnDir,predictDir)