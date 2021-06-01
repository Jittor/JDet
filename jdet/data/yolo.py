import jittor as jt 
from jittor.dataset import Dataset

import numpy as np 
from PIL import Image
import cv2
import glob
import os 

from jdet.utils.registry import DATASETS
from .transforms import Compose

def xywhn2xyxy(x,w,h,padw=0,padh=0):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y 

@DATASETS.register_module()
class YOLODataset(Dataset):

    CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    
    def __init__(self,path,transforms=[
                          dict(
                              type="Resize",
                              min_size=[800],
                              max_size=1333
                          ),
                          dict(
                              type="Pad",
                              size_divisor=32
                          ),
                          dict(
                              type="Normalize",
                              mean=[123.675, 116.28, 103.53],
                              std = [58.395, 57.12, 57.375],
                          )
                      ],batch_size=1,num_workers=0,shuffle=False,filter_empty_gt=True):
        super(YOLODataset,self).__init__(batch_size=batch_size,num_workers=num_workers,shuffle=shuffle)
        self.path = path 
        
        if isinstance(transforms,list):
            transforms = Compose(transforms)
        if transforms is not None and not callable(transforms):
            raise TypeError("transforms must be list or callable")

        self.transforms = transforms
        
        self.img_files = sorted(glob.glob(os.path.join(path,"*.jpg")))

        if filter_empty_gt:
            self.img_files = self._filter_imgs()

        self.total_len = len(self.img_files)

    def _filter_imgs(self):
        img_files = []
        for img_f in self.img_files:
            label_f = img_f.replace("images","labels",1).replace(".jpg",".txt")
            with open(label_f) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if len(l)>0:
                    img_files.append(img_f)
        return img_files

    def _read_ann_info(self, img_f):
        label_f = img_f.replace("images","labels",1).replace(".jpg",".txt")
        img = Image.open(img_f).convert("RGB")
        w,h = img.size 
        with open(label_f) as f:
            l = [x.split() for x in f.read().strip().splitlines() if len(x)]
            l = np.array(l,dtype=np.float32).reshape(-1,5)
            gt_labels = 1+l[:,0].astype(np.int32)
            gt_bboxes = xywhn2xyxy(l[:,1:],w=w,h=h)

        ann = dict(
            img_file=img_f,
            bboxes=gt_bboxes,
            labels=gt_labels,
            classes=self.CLASSES,
            ori_img_size=(w,h),
            img_size=(w,h))

        return img,ann
    

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        image,anno = self._read_ann_info(img_file)

        if self.transforms is not None:
            image, anno = self.transforms(image, anno)

        return image, anno 

    def collate_batch(self,batch):
        imgs = []
        anns = []
        max_width = 0
        max_height = 0
        for image,ann in batch:
            height,width = image.shape[-2],image.shape[-1]
            max_width = max(max_width,width)
            max_height = max(max_height,height)
            imgs.append(image)
            anns.append(ann)
        N = len(imgs)
        batch_imgs = np.zeros((N,3,max_height,max_width),dtype=np.float32)
        for i,image in enumerate(imgs):
            batch_imgs[i,:,:image.shape[-2],:image.shape[-1]] = image
        
        return batch_imgs,anns