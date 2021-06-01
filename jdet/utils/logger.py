from .registry import HOOKS,build_from_cfg 
from jdet.config.config import get_cfg
import time 
import os
import logging
from tensorboardX import SummaryWriter

@HOOKS.register_module()
class TextLogger:
    def __init__(self,work_dir):
        work_dir = get_cfg().work_dir
        os.makedirs(os.path.join(work_dir,"textlog"),exist_ok=True)
        file_name = "textlog/log_"+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+".txt"
        f = os.path.join(work_dir,file_name)
        self.log_file = open(f,"a")

    def log(self,data):
        msg = ",".join([f"{k}={d}" for k,d in data.items()])
        msg+="\n"
        self.log_file.write(msg)
        self.log_file.flush()

@HOOKS.register_module()
class TensorboardLogger:
    def __init__(self,work_dir):
        tensorboard_dir = os.path.join(work_dir,"tensorboard")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self,data):
        step = data["iter"]
        for k,d in data.items():
            if k in ["iter","epoch","batch_idx","times"]:
                continue
            if isinstance(d,str):
                continue
            self.writer.add_scalar(k,d,global_step=step)

@HOOKS.register_module()
class RunLogger:
    def __init__(self,work_dir,loggers=["TextLogger","TensorboardLogger"]):
        self.loggers = [build_from_cfg(l,HOOKS,work_dir=work_dir) for l in loggers]

    def log(self,data):
        data = {k:d.item() if hasattr(d,"item") else d for k,d in data.items()}
        for logger in self.loggers:
            logger.log(data)
        self.print_log(data)
    
    def print_log(self,msg):
        if isinstance(msg,dict):
            msg = ",".join([f"{k}={d:.4f} " if isinstance(d,float) else f"{k}={d} "  for k,d in msg.items()])
        print(msg)