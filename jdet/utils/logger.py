from .registry import HOOKS 
import time 

@HOOKS.register_module()
class TextLogger:
    def __init__(self):
        work_dir = get_cfg().work_dir
        file_name = "log_"+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+".txt"
        f = os.path.join(work_dir,file_name)
        self.log_file = open(log_file,"a")

    def log(self,data):
        self.log_file.write(data)
        self.log_file.flush()

@HOOKS.register_module()
class TensorboardLogger:
    def __init__(self):
        pass 

    def log(self,data):
        pass

@HOOKS.register_module()
class RunLogger:
    def __init__(self,log_interval,loggers=["TextLogger","TensorboardLogger"]):
        self.log_interval = log_interval
        self.loggers = [build_from_cfg(l,HOOKS) for l in loggers]
        self.step = 0

    def iter(self,data):
        for logger in self.loggers:
            if self.step % self.log_interval==0:
               logger.log(data)
        
        self.step+=1