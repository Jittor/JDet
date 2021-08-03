from jdet.data.devkits.voc_eval import voc_eval_dota
from jdet.models.boxes.box_ops import rotated_box_to_poly_np
from jdet.utils.registry import DATASETS
from jdet.config.constant import DOTA1_CLASSES
from jdet.data.custom import CustomDataset
from jdet.ops.nms_poly import iou_poly
import os
import jittor as jt
import numpy as np

def s2anet_post(result):
    dets,labels = result 
    labels = labels+1 
    scores = dets[:,-1]
    dets = dets[:,:-1]
    polys = rotated_box_to_poly_np(dets)
    return polys,scores,labels

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

    def evaluate(self,results,work_dir,epoch,logger=None):
        save_path = os.path.join(work_dir,f"detections/val_{epoch}/val.pkl")
        jt.save(results,save_path)
        dets = []
        gts = []
        for img_idx,(result,target) in enumerate(results):
            if len(result)==2:
                det_polys,det_scores,det_labels = s2anet_post(result)
            else:
                det_polys,det_scores,det_labels =  result
            idx1 = np.ones(det_labels.shape[0],1)*img_idx
            dets.append(np.concatenate([idx1,det_polys,det_scores.reshape(-1,1),det_labels.reshape(-1,1)]))
            gt_polys = target["polys"]
            gt_labels = target["labels"].reshape(-1,1)
            idx2 = np.ones(gt_labels.shape[0],1)*img_idx
            gts.append(np.concatenate([idx2,gt_polys,gt_labels]))
        dets = np.concatenate(dets)
        gts = np.concatenate(gts)
        aps = {}
        for i,classname in enumerate(self.CLASSES):
            c_dets = dets[dets[:,-1]==(i+1),:-1]
            c_gts = gts[gts[:,-1]==(i+1),:-1]
            img_idx = c_gts[:,0]
            classname_gts = {}
            for idx in np.unique(img_idx):
                g = c_gts[img_idx==idx,1:]
                # TODO add diffculty into eval
                classname_gts[idx] = {"box":g,"det":[False]*len(g)}
            rec, prec, ap = voc_eval_dota(c_dets,classname_gts,iou_func=iou_poly)
            aps[classname]=ap 
        map = sum(list(aps.values()))/len(aps)
        aps["map"]=map
        logger.print_log(aps)
        return aps


        
            
            
        
