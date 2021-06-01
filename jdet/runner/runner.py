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
from jdet.utils.checkpointer import Checkpointer
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
        test_files = list(glob.glob("/home/lxl/workspace/JDet/coco128/images/train2017/*.jpg"))[:10]
        while self.epoch < self.max_epoch:
            self.train()
            self.val()
            self.save()
            self.epoch +=1
            if self.epoch%5==0:
                self.run_on_images(test_files,"exp/images")
        self.test()

    def train(self):
        self.model.train()
        for batch_idx,(images,targets) in enumerate(self.train_dataset):
            losses = self.model(images,targets)
            all_loss = sum(losses.values())
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)
            lr = self.optimizer.param_groups[0].get("lr")
            if self.iter % self.log_interval ==0:
                data = dict(
                    times=time.asctime( time.localtime(time.time())),
                    lr = lr,
                    iter = self.iter,
                    epoch = self.epoch,
                    batch_idx = batch_idx,
                    total_loss = all_loss,
                )
                data.update(losses)
                self.logger.log(data)

            self.iter+=1
   
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
        if self.eval_interval is not None and self.epoch % self.eval_interval ==0:
            if self.val_dataset is None:
                warnings.warn("Please set Val dataset")
            else:
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
            print("Please set test dataset")
            return
        self.model.eval()

    def save(self):
        if self.checkpoint_interval is None or self.epoch % self.checkpoint_interval!=0:
            return
        save_file = os.path.join(self.work_dir,f"ckpt_{self.epoch}.pkl")
        save_data = {
            "meta":{
                "jdet_version": jdet.__version__,
                "lr_scheduler": self.scheduler.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "iter": self.iter,
                "trained_time":time.asctime( time.localtime(time.time())),
                "config": self.cfg.dump()
            },
            "model":self.model.parameters()
        }
        jt.save(save_data,save_file)
    
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