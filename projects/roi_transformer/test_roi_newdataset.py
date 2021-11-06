import jdet
import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.ops.bbox_transfomrs import mask2poly_single
from jdet.data.custom import get_mask_from_bbox
from jdet.data.devkits.dota_utils import polygonToRotRectangle
import argparse
import os
import pickle as pk
import numpy as np
import random

def fake_argsort(x, dim=0, descending=False):
    return jt.index(x)[0], x

def fake_argsort2(x, dim=0, descending=False):
    x_ = x.data
    if (descending):
        x__ = -x_
    else:
        x__ = x_
    index_ = np.argsort(x__, axis=dim, kind="stable")
    y_ = x_[index_]
    index = jt.array(index_)
    y = jt.array(y_)
    return index, y

def fake_sort2(x):
    x_ = x.data
    y_ = np.sort(x_, kind="stable")
    y = jt.array(y_)
    return y

def main():
    jt.sort = fake_sort2
    jt.argsort = fake_argsort2
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    args = parser.parse_args()

    jt.flags.use_cuda=1
    jt.set_global_seed(223)
    np.random.seed(0)
    random.seed(0)
    init_cfg("configs/faster_rcnn_RoITrans_r50_fpn_1x_dota_dota.py")
    cfg = get_cfg()
    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi)
    iter = 0
    model = build_from_cfg(cfg.model,MODELS)
    model.load(cfg.pretrained_weights)
    if (cfg.parameter_groups_generator):
        params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=model.named_parameters(), model=model)
    else:
        params = model.parameters()
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
    scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=optimizer)
    model.train()

    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
    imagess = []
    targetss = []
    loss_list = []

    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
    for batch_idx,(images,targets) in enumerate(train_dataset):
        print("batch_idx=" + str(batch_idx))
        imagess.append(jdet.utils.general.sync(images))
        targetss.append(jdet.utils.general.sync(targets))

        losses = model(images,targets)
        all_loss,losses = parse_losses(losses)
        loss_list.append(all_loss.item())
        if (batch_idx > 10):
            break
        optimizer.step(all_loss)
        scheduler.step(iter,0,by_epoch=True)
        iter+=1

    return

    print("success!")
    print(train_dataset[0])
if __name__ == "__main__":
    main()