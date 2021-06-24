import jittor as jt
from jittor.dataset import Dataset 


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os 
from PIL import Image
import numpy as np 

from jdet.utils.registry import DATASETS
from jdet.config.constant import DOTA1_ClASSES
from .coco import COCODataset

@DATASETS.register_module()
class DOTADataset(COCODataset):
    CLASSES = DOTA1_CLASSES

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


    
