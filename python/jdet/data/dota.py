from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.utils.registry import DATASETS
from jdet.config.constant import DOTA1_CLASSES
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.data.custom import CustomDataset
import os
import numpy as np

@DATASETS.register_module()
class DOTADataset(CustomDataset):
    CLASSES = DOTA1_CLASSES

    def __init__(self,*arg,balance_category=False,**kwargs):
        super().__init__(*arg,**kwargs)
        if balance_category:
            self.img_infos = self._balance_categories()
            self.total_len = len(self.img_infos)

    def _balance_categories(self):
        img_infos = self.img_infos
        cate_dict = {}
        for idx,img_info in enumerate(img_infos):
            unique_labels = np.unique(img_info["ann"]["labels"])
            for label in unique_labels:
                if label not in cate_dict:
                    cate_dict[label]=[]
                cate_dict[label].append(idx)
        new_idx = []
        balance_dict={
            "storage-tank":(1,526),
            "baseball-diamond":(2,202),
            "ground-track-field":(1,575),
            "swimming-pool":(2,104),
            "soccer-ball-field":(1,962),
            "roundabout":(1,711),
            "tennis-court":(1,655),
            "basketball-court":(4,0),
            "helicopter":(8,0)
        }

        for k,d in cate_dict.items():
            classname = self.CLASSES[k-1]
            l1,l2 = balance_dict.get(classname,(1,0))
            new_d = d*l1+d[:l2]
            new_idx.extend(new_d)
        img_infos = [self.img_infos[idx] for idx in new_idx]
        return img_infos

    def parse_result(self,results,save_path):
        check_dir(save_path)
        data = {}
        for (dets,labels),img_name in results:
            img_name = os.path.splitext(img_name)[0]
            for det,label in zip(dets,labels):
                bbox = det[:5]
                score = det[5]
                classname = self.CLASSES[label-1]
                bbox = rotated_box_to_poly_single(bbox)
                temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                            img_name, score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                            bbox[5], bbox[6], bbox[7])
                if classname not in data:
                    data[classname] = []
                data[classname].append(temp_txt)
        for classname,lines in data.items():
            f_out = open(os.path.join(save_path, classname + '.txt'), 'w')
            f_out.writelines(lines)
            f_out.close()

    def evaluate(self,results,work_dir,epoch,logger=None):
        save_path = os.path.join(work_dir,f"detections/val_{epoch}")
        self.parse_result(results,save_path)