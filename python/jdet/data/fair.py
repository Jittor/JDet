from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.utils.registry import DATASETS
from jdet.config.constant import FAIR_CLASSES_
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.data.dota import DOTADataset
import os
import numpy as np

@DATASETS.register_module()
class FAIRDataset(DOTADataset):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.CLASSES = FAIR_CLASSES_

    def _balance_categories(self):
        assert(False) #TODO: FAIR balance
        # img_infos = self.img_infos
        # cate_dict = {}
        # for idx,img_info in enumerate(img_infos):
        #     unique_labels = np.unique(img_info["ann"]["labels"])
        #     for label in unique_labels:
        #         if label not in cate_dict:
        #             cate_dict[label]=[]
        #         cate_dict[label].append(idx)
        # new_idx = []
        # balance_dict={
        #     "storage-tank":(1,526),
        #     "baseball-diamond":(2,202),
        #     "ground-track-field":(1,575),
        #     "swimming-pool":(2,104),
        #     "soccer-ball-field":(1,962),
        #     "roundabout":(1,711),
        #     "tennis-court":(1,655),
        #     "basketball-court":(4,0),
        #     "helicopter":(8,0)
        # }

        # for k,d in cate_dict.items():
        #     classname = self.CLASSES[k-1]
        #     l1,l2 = balance_dict.get(classname,(1,0))
        #     new_d = d*l1+d[:l2]
        #     new_idx.extend(new_d)
        # img_infos = [self.img_infos[idx] for idx in new_idx]
        # return img_infos