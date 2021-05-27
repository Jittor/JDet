import logging

from jdet.config.config import get_cfg
from jdet.utils.registry import build_from_cfg,META_ARCHS,OPTIMS,DATASETS


class Runner:
    def __init__(self,mode="whole"):
        cfg = get_cfg()
        self.work_dir = os.path.abspath(cfg.work_dir)
        self.max_epoch = cfg.epoch 
        self.max_iter = cfg.max_iter
        self.log_interval = cfg.log_interval
        self.save_interval = cfg.save_interval
        self.resume_path = cfg.resume
    
        self.model = build_from_cfg(cfg.model,META_ARCHS)
        self.optimizer = build_from_cfg(cfg.optim,OPTIMS)
        self.scheduler = build_from_cfg(cfg.solver,SOLVERS)
        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
        self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
        self.test_dataset = build_from_cfg(cfg.dataset.test,DATASETS)
    
    def prepare(self):
        os.makedirs(self.work_dir,exists=True)
        
        save_config_file = os.path.join(self.work_dir,"config.yaml")
        write_cfg(save_config_file)

        self.logger = build_from_cfg(cfg.logger,HOOKS)
        self.checkpointer = build_from_cfg("Checkpointer",model=self.model,optimizer=self.optimizer,scheduler = self.scheduler)
        self.iter = 0
        self.epoch = 0

    def save_checkpoint(self):
        save_file = os.path.join(self.work_dir,f"/ckpt_{self.epoch}.pkl")
        self.checkpointer.save(save_file)
    
    def resume(self):
        self.iter,self.epoch = self.checkpointer.load(self.resume_path)
        
    def display(self):
        pass 
    
    def run(self):
        print("running") 
        while self.epoch < self.max_epoch:
            self.train()
            if self.epoch % self.save_interval == 0:
                self.save_checkpoint()
            if self.epoch % self.val_interval == 0:
                self.val()
            self.epoch +=1
        self.test()

    def train(self):
        self.model.train()
        for batch_idx,(images,targets) in enumerate(self.train_dataset):
            results,losses = self.model(images,targets)
            self.optimizer.step(losses["loss"])
            self.scheduler.step()
            self.logger.log(losses)
    
    def run_on_images(self,img_files,save_dir=None):
        pass

    @jt.no_grad()
    def val(self):
        self.model.val() 
    
    @jt.no_grad()
    def test(self):
        self.model.val()
    

    