from math import cos, sin
from unicodedata import name

from pycocotools.coco import COCO
import jdet
import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.ops.bbox_transfomrs import mask2poly_single
#from jdet.data.custom import get_mask_from_bbox
from jdet.data.devkits.dota_utils import polygonToRotRectangle
import argparse
import os
import pickle as pk
import numpy as np
import random
import cv2
from tqdm import tqdm
from jdet.ops.bbox_transfomrs import mask2poly, obb2poly_single
from jdet.models.boxes.box_ops import rotated_box_to_poly_np, rotated_box_to_bbox_np

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
def get_mask_from_bbox(gt_bbox, w, h):
    return jt.code([h, w], jt.uint8, [jt.array(gt_bbox)],
    cpu_header=r'''
    #include<cmath>
    #include<algorithm>
    @alias(bbox, in0)
    @alias(mask, out)
''',
    cpu_src=r'''
double dcos = std::cos(@bbox(4));
double dsin = std::sin(@bbox(4));
double cx = @bbox(0), cy = @bbox(1);
double w = @bbox(2), h = @bbox(3);
int bw = mask_shape0, bh = mask_shape1;
for(int i=0;i<bw;i++)
    for(int j=0;j<bh;j++){
        double x=(j-cx+0.5)*dcos+(i-cy+0.5)*dsin;
        double y=(i-cy+0.5)*dcos-(j-cx+0.5)*dsin;
        @mask(i,j)=(-w/2<x&&x<=w/2&&-h/2<y&&y<=h/2);
    }
''');
def main():
    jt.sort = fake_sort2
    jt.argsort = fake_argsort2
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    args = parser.parse_args()

    jt.flags.use_cuda=1
    jt.set_global_seed(223)
    np.random.seed(0)
    random.seed(0)
    init_cfg("configs/faster_rcnn_RoITrans_r50_fpn_1x_dota_test.py")
    cfg = get_cfg()
    return
    sum = 0
    cnt_file_name = {}
    flag = False
    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi)
    for images, anns in tqdm(train_dataset):
        for i in range(len(anns['gt_bboxes'])):
            cnt = anns['gt_bboxes'][i].shape[0]
            sum = sum + cnt
            name = anns['img_meta'][i]['img_file']
            if name == "P1584__1__3268___2472.png":
                print(anns['gt_bboxes'][i])
                flag = True
                break
            if name not in cnt_file_name.keys():
                cnt_file_name[name] = cnt
            else:
                cnt_file_name[name] = cnt_file_name[name] + cnt
        if flag:
            break
    print(sum)#33894

    coco = COCO("/mnt/disk/zwy/dota1_1024/trainval1024/DOTA_trainval1024.json")
    coco_img_ids = coco.getImgIds()
    coco_dict = {}
    sum = 0
    for i in coco_img_ids:
        info = coco.loadImgs([i])[0]
        info['filename'] = info['file_name']
        if info["filename"] in coco_dict.keys():
            print("error")
            print(info["filename"])
            break
        else:
            coco_dict[info['file_name']] = info
        coco_ann = coco.loadAnns(coco.getAnnIds(info["id"]))
        if info['filename'] == "P1584__1__3268___2472.png":
            print(coco_ann)
            break
        if info['file_name'] in cnt_file_name.keys() and len(coco_ann) != cnt_file_name[info['file_name']]:
            print(info['file_name'], len(coco_ann), cnt_file_name[info['file_name']])

    return


    annotation_file = "/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/trainval1024.pkl"
    dota_img_info = jt.load(annotation_file)
    save_dict = {}
    for idx, info in enumerate(dota_img_info):
        file_name = info["filename"].replace("_1.0_", "_1_")
        if file_name not in coco_dict.keys():
            print("error")
            print(file_name)
            break
        coco_info = coco_dict.pop(file_name)
        coco_ann = coco.loadAnns(coco.getAnnIds(coco_info["id"]))
        if file_name in save_dict:
            print("error!!")
            break
        save_dict[file_name] = True
        if info["ann"]["bboxes"].shape[0] != len(coco_ann):
            labels = []
            for single_coco_ann in coco_ann:
                labels.append(single_coco_ann["category_id"])
                print(single_coco_ann)
            print(np.array(labels))
            print("==========")
            #print(rotated_box_to_bbox_np(info["ann"]["bboxes"]))
            print(info["ann"]["labels"])
            print(file_name)
            break
    for file_name in coco_dict.keys():
        coco_info = coco_dict[file_name]
        coco_ann = coco.loadAnns(coco.getAnnIds(coco_info["id"]))
        if len(coco_ann) > 0:
            print("error")
            break
    print(len(dota_img_info), len(coco_dict.keys()))
    return
    sum = 0
    print(train_dataset[0][1])
    for images, anns in tqdm(train_dataset):
        for ann in anns:
            sum = sum + ann['hboxes'].shape[0]
    print(sum)#261783
    sum = 0
    for i,anns in tqdm(enumerate(train_dataset)):
        for ann in anns:
            sum = sum + len(ann['gt_bboxes'])
    print(sum)#33894
    return
if __name__ == "__main__":
    main()