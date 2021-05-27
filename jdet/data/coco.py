# reference a little of mmdetection cocodataset
import jittor as jt
from jittor.dataset import Dataset 


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os 
from PIL import Image
import numpy as np 

from jdet.utils.registry import DATASETS
from .transforms import Compose

@DATASETS.register_module()
class COCODataset(Dataset):

    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    def __init__(self,root,anno_file,transforms=None,batch_size=1,num_workers=0,shuffle=False,filter_empty_gt=True,use_anno_cats=False):
        super(COCODataset,self).__init__(batch_size=batch_size,num_workers=num_workers,shuffle=shuffle)
        self.root = root 
        self.coco = COCO(anno_file)
        
        if isinstance(transforms,list):
            transforms = Compose(transforms)
        if not callable(transforms):
            raise TypeError("transforms must be list or callable")

        self.transforms = transforms
        
        if use_anno_cats:
            self.CLASSES = [cat['name'] for cat in self.coco.cats.values()]

        self.cat_ids = self.coco.getCatIds(self.CLASSES)
        self.cat2label = {cat_id: i+1 for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = list(sorted(self.coco.imgs.keys()))

        if filter_empty_gt:
            self.img_ids = self._filter_imgs()

        self.total_len = len(self.img_ids)

    def _filter_imgs(self):
        """Filter images without ground truths."""
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.catToImgs[class_id])
        # merge the image id sets of the two conditions and use the merged set
        ids_in_cat &= ids_with_ann

        img_ids = [img_id for img_id in self.img_ids if img_id in ids_in_cat]
        return img_ids

    def _read_ann_info(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        width,height = image.size 
        assert width == img_info['width'] and height == img_info["height"],"image size is different from annotations"

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            ori_img_size=(width,height),
            img_size=(width,height))

        return image,ann
    

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image,anno = self._read_ann_info(img_id)

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



def test():
    dataset = COCODataset(root="/mnt/disk/lxl/dataset/coco/images/val2017",anno_file="/mnt/disk/lxl/dataset/coco/annotations/instances_val2017.json")

    for i,(image,target) in enumerate(dataset):
        print(image)


if __name__ == "__main__":
    test()