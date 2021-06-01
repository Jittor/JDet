from genericpath import isfile
import logging
import jittor as jt
import os 
import cv2 
import glob
import time
import warnings
import jdet
from jdet.config.config import get_cfg,save_cfg
from jdet.utils.registry import build_from_cfg,META_ARCHS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.config.constant import COCO_CLASSES
from jdet.utils.visualization import visualize_results,visual_gts

class Runner:
    def __init__(self,mode="whole"):
        cfg = get_cfg()
        self.cfg = cfg
        self.work_dir = os.path.abspath(cfg.work_dir)
        self.max_epoch = cfg.max_epoch 
        self.max_iter = cfg.max_iter
        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
    
        self.model = build_from_cfg(cfg.model,META_ARCHS)
        self.optimizer = build_from_cfg(cfg.optim,OPTIMS,params=self.model.parameters())
        self.scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=self.optimizer)
        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
        self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
        self.test_dataset = build_from_cfg(cfg.dataset.test,DATASETS)
        self.logger = build_from_cfg(cfg.logger,HOOKS)

        self.initialize()
    
    def initialize(self):
        assert (self.max_iter is None)^(self.max_epoch is None),"You must set max_iter or max_epoch"

        os.makedirs(self.work_dir,exist_ok=True)
        save_config_file = os.path.join(self.work_dir,"config.yaml")
        save_cfg(save_config_file)

        self.iter = 0
        self.epoch = 0

        if self.resume_path:
            self.resume()

    @property
    def finish(self):
        if self.max_epoch:
            return self.epoch<self.max_epoch
        else:
            return self.iter<self.max_iter

    @property
    def time(self):
        return time.asctime( time.localtime(time.time()))
    
    @property
    def is_main(self):
        return not jt.in_mpi or jt.rank==0

    def sync_data(self,data,reduce_mode="sum"):
        def sync(d):
            if jt.in_mpi:
                d = d.mpi_all_reduce(reduce_mode)
            return d.numpy()
        
        if isinstance(data,jt.Var):
            data = sync(data)
        elif isinstance(data,list):
            data = [sync(d) if isinstance(d,jt.Var) else d for d in data]
            return data 
        elif isinstance(data,dict):
            data = {k:sync(d) if isinstance(d,jt.Var) else d for k,d in data.items() }
        return data 
    
    def run(self):
        self.logger.print_log("Start running")
        while not self.finish:
            self.train()
            if self.eval_interval is not None and self.epoch % self.eval_interval ==0:
                self.val()
            if self.checkpoint_interval is None or self.epoch % self.checkpoint_interval!=0:
                self.save()
        self.test()

    def train(self):
        self.model.train()
        for batch_idx,(images,targets) in enumerate(self.train_dataset):
            losses = self.model(images,targets)
            all_loss = sum(losses.values())
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)

            if self.iter % self.log_interval ==0:
                lr = self.optimizer.param_groups[0].get("lr")
                data = dict(
                    times=self.time,
                    lr = lr,
                    iter = self.iter,
                    epoch = self.epoch,
                    batch_idx = batch_idx,
                    total_loss = all_loss,
                )
                data.update(losses)
                if self.is_main:
                    self.logger.log(self.sync_data(data))
            
            self.iter+=1
            if self.finish:
                break

        self.epoch +=1

   
    @jt.no_grad()
    def run_on_images(self,img_files,save_dir=None):
        self.model.eval()
        dataset = build_from_cfg("ImageDataset",DATASETS,img_files=img_files)
        for i,(images,targets) in enumerate(dataset):
            results = self.model(images,targets)
            results = results["outs"]
            if save_dir:
                visualize_results(results[0],COCO_CLASSES,[t["img_file"] for t in targets],save_dir)
                # visual_gts(targets,save_dir)                

    @jt.no_grad()
    def val(self):
        if self.val_dataset is None:
            warnings.warn("Please set Val dataset")
        else:
            self.logger.print_log("Validating....")
            self.model.eval()
            results = []
            for batch_idx,(images,targets) in enumerate(self.val_dataset):
                result = self.model(images,targets)
                results.append(result)
            results_save_file = os.path.join(self.work_dir,f"val_{self.epoch}.json")
            eval_results = self.val_dataset.evaluate(results,results_save_file)
            self.logger.print_log(eval_results)


    @jt.no_grad()
    def test(self):
        if self.test_dataset is None:
            warnings.warn("Please set Test dataset")
        else:
            self.model.eval()
            results = []
            for batch_idx,(images,targets) in enumerate(self.val_dataset):
                result = self.model(images,targets)
                results.append(result)


    def save(self):
        if not self.is_main:
            return
        # multi gpus need to sync before save
        if jt.in_mpi:
            jt.sync_all()
        
        checkpoint_dir = os.path.join(self.work_dir,"checkpoints")
        os.makedirs(checkpoint_dir,exist_ok=True)
        save_file = os.path.join(checkpoint_dir,f"ckpt_{self.epoch}.pkl")
        save_data = {
            "meta":{
                "jdet_version": jdet.__version__,
                "epoch": self.epoch,
                "iter": self.iter,
                "max_iter": self.max_iter,
                "max_epoch": self.max_epoch,
                "trained_time":self.time,
                "config": self.cfg.dump()
            },
            "model":self.model.parameters(),
            "scheduler": self.scheduler.parameters(),
            "optimizer": self.optimizer.parameters()
        }
        jt.save(save_data,save_file)
    

    def resume(self):
        if not os.path.exists(self.resume_path):
            warnings.warn(f"{self.resume_path} is not exists")
            return
        if not os.path.isfile(self.resume_path):
            warnings.warn(f"{self.resume_path} must be a file")
            return 
        resume_path = os.path.abspath(self.resume_path)
        
        resume_data = jt.load(resume_path)
        
        meta = resume_data.get("meta",dict())
        self.epoch = meta.get("epoch",self.epoch)
        self.iter = meta.get("iter",self.iter)
        self.max_iter = meta.get("max_iter",self.max_iter)
        self.max_epoch = meta.get("max_epoch",self.max_epoch)

        self.scheduler.load_paramters(resume_data.get("scheduler",dict()))
        self.optimizer.load_paramters(resume_data.get("optimizer",dict()))
        self.model.load_parameters(resume_data.get("model",dict()))

        self.logger.print_log(f"Loading model parameters from {self.resume_path}")