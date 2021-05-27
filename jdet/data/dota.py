import jittor as jt
from jittor.dataset import Dataset 


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os 
from PIL import Image
import numpy as np 

from jdet.utils.registry import DATASETS
from .coco import COCODataset

@DATASETS.register_module()
class DOTADataset(COCODataset):
    def __init__(self,root,anno_file,transforms=None,batch_size=1,num_workers=0,shuffle=False,filter_empty_gt=True):
        super(DOTADataset,self).__init__(root,anno_file,transforms=transforms,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        shuffle=shuffle,
                                                        filter_empty_gt=filter_empty_gt,
                                                        use_anno_cats=True)
    def _filter_imgs(self):
        tmp_img_ids = super(DOTADataset,self)._filter_imgs()
        
        img_ids = []
        for img_id in tmp_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)

            # remove ignored or crowd box
            anno = [obj for obj in anno if obj["iscrowd"] == 0 and obj["ignore"] == 0 ]
            # if it's empty, there is no annotation
            if len(anno) == 0:
                continue
            # if all boxes have close to zero area, there is no annotation
            if all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno):
                continue
            img_ids.append(img_id)

        # sort indices for reproducible results
        img_ids = sorted(img_ids)
        return img_ids

    def _convert_seg2box(self,seg):
        seg_np = np.array(seg).reshape(4,2)
        arearect = cv2.minAreaRect(seg_np)
        rotation_rect = cv2.boxPoints(arearect)
        min_x = rotation_rect[:,0].min()
        max_x = rotation_rect[:,0].max()
        min_y = rotation_rect[:,1].min()
        max_y = rotation_rect[:,1].max()
        xyxy = np.array([min_x,min_y,max_x,max_y])
        return rotation_rect,xyxy

        
    def _read_ann_info(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        # if "angle" in loaded_img and loaded_img["angle"] is not 0:
        if 'angle' in img_info.keys() and img_info["angle"] is not 0:
            if img_info["angle"] == 90:
                image = image.rotate( 270, expand=True )
            elif img_info["angle"] == 180:
                image = image.rotate( 180, expand=True )
            elif img_info["angle"] == 270:
                image = image.rotate( 90, expand=True )
            else:
                raise ValueError()


        width,height = image.size 
        assert width == img_info['width'] and height == img_info["height"],"image size is different from annotations"

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            seg = ann["segmentation"]
            seg,bbox = self._convert_seg2box(seg)
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(seg)

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
            img_size=(width,height))

        return image,ann


    
