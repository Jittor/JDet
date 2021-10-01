from genericpath import isfile
import time
from jdet.models.boxes.box_ops import flip_result
import jittor as jt
from tqdm import tqdm
import numpy as np
import jdet
import pickle
import datetime
from jdet.config import get_cfg,save_cfg
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.config import COCO_CLASSES
from jdet.utils.visualization import draw_rboxes, visualize_results,visual_gts
from jdet.utils.general import build_file, current_time, sync,check_file,build_file,check_interval,parse_losses,search_ckpt
from jdet.data.devkits.data_merge import data_merge_result
import os
import shutil

from jittor_utils import auto_diff
import sys

class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        self.flip_test = [] if cfg.flip_test is None else cfg.flip_test
        self.work_dir = cfg.work_dir

        self.max_epoch = cfg.max_epoch 
        self.max_iter = cfg.max_iter
        assert (self.max_iter is None)^(self.max_epoch is None),"You must set max_iter or max_epoch"

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
    
        self.model = build_from_cfg(cfg.model,MODELS)
        if (cfg.parameter_groups_generator):
            params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=self.model.named_parameters(), model=self.model)
        else:
            params = self.model.parameters()
        self.optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
        self.scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=self.optimizer)
        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi)
        self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
        self.test_dataset = build_from_cfg(cfg.dataset.test,DATASETS)
        
        self.logger = build_from_cfg(self.cfg.logger,HOOKS,work_dir=self.work_dir)

        save_file = build_file(self.work_dir,prefix="config.yaml")
        save_cfg(save_file)

        self.iter = 0
        self.epoch = 0

        if self.max_epoch:
            if (self.train_dataset):
                self.total_iter = self.max_epoch * len(self.train_dataset)
            else:
                self.total_iter = 0
        else:
            self.total_iter = self.max_iter

        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=True)
        
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    @property
    def finish(self):
        if self.max_epoch:
            return self.epoch>=self.max_epoch
        else:
            return self.iter>=self.max_iter
    
    def run(self):
        self.logger.print_log("Start running")
        while not self.finish:
            self.train()
            # TODO open evaluation
            if False and check_interval(self.epoch,self.eval_interval):
                # TODO: need remove this
                self.model.eval()
                self.val()
            if check_interval(self.epoch,self.checkpoint_interval):
                self.save()
        self.test()

    def train(self):

        self.model.train()

        start_time = time.time()

        ### Test Begin
        # hook = auto_diff.Hook("gliding2")
        # hook.hook_module(self.model)
        # hook.hook_optimizer(self.optimizer)
        ### Test End

        for batch_idx,(images,targets) in enumerate(self.train_dataset):

            losses = self.model(images,targets)
            all_loss,losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)
    
            if check_interval(self.iter,self.log_interval):
                batch_size = len(targets)*jt.world_size
                ptime = time.time()-start_time
                fps = batch_size*(batch_idx+1)/ptime
                eta_time = (self.total_iter-self.iter)*ptime/(batch_idx+1)
                eta_str = str(datetime.timedelta(seconds=int(eta_time)))
                data = dict(
                    name = self.cfg.name,
                    lr = self.optimizer.cur_lr(),
                    iter = self.iter,
                    epoch = self.epoch,
                    batch_idx = batch_idx,
                    batch_size = batch_size,
                    total_loss = all_loss,
                    fps=fps,
                    eta=eta_str
                )
                data.update(losses)
                data = sync(data)
                # is_main use jt.rank==0, so it's scope must have no jt.Vars
                if jt.rank==0:
                    self.logger.log(data)
            
            self.iter+=1
            if self.finish:
                break

        self.epoch +=1


    @jt.no_grad()
    @jt.single_process_scope()
    def run_on_images(self,img_files,save_dir=None):
        self.model.eval()
        dataset = build_from_cfg("ImageDataset",DATASETS,img_files=img_files)
        for i,(images,targets) in tqdm(enumerate(dataset)):
            results = self.model(images,targets)
            if save_dir:
                visualize_results(results,COCO_CLASSES,[t["img_file"] for t in targets],save_dir)
                # visual_gts(targets,save_dir)                

    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        if self.val_dataset is None:
            self.logger.print_log("Please set Val dataset")
        else:
            self.logger.print_log("Validating....")
            # TODO: need move eval into this function
            # self.model.eval()
            if self.model.is_training():
                self.model.eval()
            results = []
            for batch_idx,(images,targets) in tqdm(enumerate(self.val_dataset),total=len(self.val_dataset)):
                result = self.model(images,targets)
                results.extend([(r,t) for r,t in zip(sync(result),sync(targets))])

            eval_results = self.val_dataset.evaluate(results,self.work_dir,self.epoch,logger=self.logger)

            self.logger.log(eval_results,iter=self.iter)

    @jt.no_grad()
    @jt.single_process_scope()
    def test(self):
        if self.test_dataset is None:
            self.logger.print_log("Please set Test dataset")
        else:
            self.logger.print_log("Testing...")
            self.model.eval()
            results = []
            for batch_idx,(images,targets) in tqdm(enumerate(self.test_dataset),total=len(self.test_dataset)):
                result = self.model(images,targets)
                # for mode in self.flip_test:
                #     images_flip = images.copy()
                #     if (mode == 'H'):
                #         images_flip = images_flip[:, :, :, ::-1]
                #     elif (mode == 'V'):
                #         images_flip = images_flip[:, :, ::-1, :]
                #     elif (mode == 'HV'):
                #         images_flip = images_flip[:, :, ::-1, ::-1]
                #     else:
                #         assert(False)
                #     result_ = self.model(images_flip,targets)
                #     for k in range(len(targets)):
                #         out = flip_result(result_[k], mode, targets[k])
                #         result[k][0] = jt.concat([result[k][0], out], 0)
                #         result[k][1] = jt.concat([result[k][1], result_[k][1]], 0)
                        
                results.extend([(r,t) for r,t in zip(sync(result),sync(targets))])

            save_file = build_file(self.work_dir,f"test/test_{self.epoch}.pkl")
            pickle.dump(results,open(save_file,"wb"))
            if (self.cfg.dataset.dataset_type):
                dataset_type=self.cfg.dataset.dataset_type
            else:
                dataset_type=self.cfg.dataset.train.type
            data_merge_result(save_file,self.work_dir,self.epoch,self.cfg.name,dataset_type)

    @jt.single_process_scope()
    def save(self):
        save_data = {
            "meta":{
                "jdet_version": jdet.__version__,
                "epoch": self.epoch,
                "iter": self.iter,
                "max_iter": self.max_iter,
                "max_epoch": self.max_epoch,
                "save_time":current_time(),
                "config": self.cfg.dump()
            },
            "model":self.model.state_dict(),
            "scheduler": self.scheduler.parameters(),
            "optimizer": self.optimizer.parameters()
        }

        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{self.epoch}.pkl")
        jt.save(save_data,save_file)
    
    def load(self, load_path, model_only=False):
        resume_data = jt.load(load_path)
        
        if (not model_only):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.iter = meta.get("iter",self.iter)
            self.max_iter = meta.get("max_iter",self.max_iter)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)

            self.scheduler.load_parameters(resume_data.get("scheduler",dict()))
            self.optimizer.load_parameters(resume_data.get("optimizer",dict()))
        if ("model" in resume_data):
            self.model.load_parameters(resume_data["model"])
        elif ("state_dict" in resume_data):
            self.model.load_parameters(resume_data["state_dict"])
        else:
            self.model.load_parameters(resume_data)

        self.logger.print_log(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)