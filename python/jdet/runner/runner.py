from genericpath import isfile
import time
import jittor as jt
from tqdm import tqdm
import jdet
import pickle
from jdet.config import get_cfg,save_cfg
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.config import COCO_CLASSES
from jdet.utils.visualization import visualize_results,visual_gts
from jdet.utils.general import build_file, current_time, sync,check_file,build_file,check_interval,parse_losses

class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        self.work_dir = cfg.work_dir

        self.max_epoch = cfg.max_epoch 
        self.max_iter = cfg.max_iter
        assert (self.max_iter is None)^(self.max_epoch is None),"You must set max_iter or max_epoch"

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
    
        self.model = build_from_cfg(cfg.model,MODELS)
        self.optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=self.model.parameters())
        self.scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=self.optimizer)
        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi)
        self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
        self.test_dataset = build_from_cfg(cfg.dataset.test,DATASETS)
        
        self.logger = build_from_cfg(self.cfg.logger,HOOKS,work_dir=self.work_dir)

        save_file = build_file(self.work_dir,prefix="config.yaml")
        save_cfg(save_file)

        self.iter = 0
        self.epoch = 0

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
            if check_interval(self.epoch,self.eval_interval):
                # TODO: need remove this
                self.model.eval()
                self.val()            
            if check_interval(self.epoch,self.checkpoint_interval):
                self.save()
        self.test()

    def train(self):
        self.model.train()
        start_time = time.time()
        img_num = 0
        for batch_idx,(images,targets) in enumerate(self.train_dataset):
            losses = self.model(images,targets)
            all_loss,losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)

            batch_size = len(targets)*jt.mpi.world_size()
            img_num +=batch_size

            if check_interval(self.iter,self.log_interval):
                fps = img_num/(time.time()-start_time)
                data = dict(
                    lr = self.optimizer.cur_lr(),
                    iter = self.iter,
                    epoch = self.epoch,
                    batch_idx = batch_idx,
                    batch_size = batch_size,
                    total_loss = all_loss,
                    fps=fps
                )
                data.update(losses)
                data = sync(data)
                # is_main use jt.rank==0, so it's scope must have no jt.Vars
                if jt.rank==0:
                    self.logger.log(data)
            
            self.iter+=1
            if self.finish:
                break
            
            exit(0)

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
                results.extend(sync(result))

            save_file = build_file(self.work_dir,prefix=f"detections/val_{self.epoch}.json")
            self.val_dataset.save_results2json(results,save_file)
            eval_results = self.val_dataset.evaluate(save_file,logger=self.logger)

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
                results.extend(sync(result))
            
            save_file = build_file(self.work_dir,f"test/test_{self.epoch}.pkl")
            pickle.dump(results,open(save_file,"wb"))


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
    

    def resume(self):
        resume_data = jt.load(self.resume_path)
        
        meta = resume_data.get("meta",dict())
        self.epoch = meta.get("epoch",self.epoch)
        self.iter = meta.get("iter",self.iter)
        self.max_iter = meta.get("max_iter",self.max_iter)
        self.max_epoch = meta.get("max_epoch",self.max_epoch)

        self.scheduler.load_parameters(resume_data.get("scheduler",dict()))
        self.optimizer.load_parameters(resume_data.get("optimizer",dict()))
        self.model.load_parameters(resume_data.get("model",dict()))

        self.logger.print_log(f"Loading model parameters from {self.resume_path}")