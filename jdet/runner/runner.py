import logging
import jittor as jt
import os 
import cv2 

import jdet
from jdet.config.config import get_cfg,save_cfg
from jdet.utils.registry import build_from_cfg,META_ARCHS,SCHEDULERS,DATASETS,HOOKS
from jdet.utils.checkpointer import Checkpointer


class Runner:
    def __init__(self,mode="whole"):
        cfg = get_cfg()
        self.cfg = cfg
        self.work_dir = os.path.abspath(cfg.work_dir)
        self.max_epoch = cfg.max_epoch 
        self.max_iter = cfg.max_iter
        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.resume_path = cfg.resume_path

        os.makedirs(self.work_dir,exist_ok=True)
        save_config_file = os.path.join(self.work_dir,"config.yaml")
        save_cfg(save_config_file)

    
        self.model = build_from_cfg(cfg.model,META_ARCHS)
        self.optimizer = build_from_cfg(cfg.optim,OPTIMS,params=self.model.parameters())
        self.scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=self.optimizer)
        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
        self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
        self.test_dataset = build_from_cfg(cfg.dataset.test,DATASETS)
        
        self.logger = build_from_cfg(cfg.logger,HOOKS)
        self.checkpointer = Checkpointer(model=self.model,optimizer=self.optimizer,scheduler = self.scheduler)
        self.iter = 0
        self.epoch = 0
    
    def run(self):
        print("running") 
        while self.epoch < self.max_epoch:
            self.train()
            self.val()
            self.save()
            self.epoch +=1
        self.test()

    def train(self):
        self.model.train()
        for batch_idx,(images,targets) in enumerate(self.train_dataset):
            results,losses = self.model(images,targets)
            self.optimizer.step(losses)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)
            self.logger.log({"losses":losses.item()})
            self.iter+=1
   
    @jt.no_grad()
    def run_on_images(self,img_files,save_dir=None):
        self.model.val()
        dataset = build_from_cfg("ImageDataset",img_files=img_files)
        for i,(images,targets) in enumerate(dataset):
            results = self.model(images,targets)

    @jt.no_grad()
    def val(self):
        if self.eval_interval is not None and self.epoch % eval_interval ==0:
            if self.val_dataset is None:
                warnings.warn("Please set Val dataset")
            else:
                self.model.val()
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
            print("Please set test dataset")
            return
        self.model.val()

    def save(self):
        if self.checkpoint_interval is None or self.epoch % self.checkpoint_interval!=0:
            return
        save_file = os.path.join(self.work_dir,f"/ckpt_{self.epoch}.pkl")
        self.checkpointer.save(save_file)
    
    def resume(self):
        self.iter,self.epoch = self.checkpointer.load(self.resume_path)
        
    def display(self,images,targets):
        for image,target in zip(images,targets):
            mean = target["mean"]
            std = target["std"]
            to_bgr = target["to_bgr"]
            if to_bgr:
                image = image[::-1]
            image *=255.
            image = image*std+mean
            image = image[::-1]
            image = image.transpose(1,2,0)
            
            classes = [target["classes"][i-1] for i in target["labels"]]
            draw_boxes(image,target["bboxes"],classes)

    def meta(self):
        states = {
            "jdet_version": jdet.__version__,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "iter": self.iter,
            "trained_time":self.now_time,
            "config": self.cfg.dump()
        }
        return states
