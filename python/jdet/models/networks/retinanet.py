import jittor as jt 
from jittor import nn 
from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS
import numpy as np
import jdet
import copy
from jdet.models.boxes.box_ops import rotated_box_to_bbox

@MODELS.register_module()
class RetinaNet(nn.Module):

    def __init__(self,backbone,neck=None,rpn_net=None):
        super(RetinaNet,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.rpn_net = build_from_cfg(rpn_net,HEADS)

    def train(self):
        super().train()
        self.backbone.train()

    def execute(self,images,targets):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            results: detections, #((cx,cy,w,h,a(pi),score), type)
            losses (dict): losses
        '''
        if ("rboxes" in targets[0]):
            # limit gt [-pi/4,pi*3/4] to [-pi/2,0) TODO: move to dataloader
            for i in range(len(targets)):
                gt_bbox = targets[i]["rboxes"].data#xywha
                out_bbox = []
                for j in range(gt_bbox.shape[0]):
                    box = gt_bbox[j]
                    x, y, w, h, a = box[0], box[1], box[2], box[3], box[4]
                    if (a >= 0):
                        a -= np.pi
                    if (a < -np.pi / 2):
                        a += np.pi / 2
                        w, h = h, w
                    out_bbox.append(np.array([x, y, w, h, a])[np.newaxis, :])
                out_bbox = np.concatenate(out_bbox, 0)
                temp = jt.array(out_bbox)
                temp = self.rpn_net.cvt2_w_greater_than_h(temp, False)#TODO: yxe?
                targets[i]["rboxes"] = temp
                temp_ = temp.copy()
                temp_[:, 4] += np.pi / 2
                targets[i]["rboxes_h"] = rotated_box_to_bbox(temp_)
        features = self.backbone(images)
        if self.neck:
            features = self.neck(features)
        results,losses = self.rpn_net(features, targets)

        if self.is_training():
            return losses 
        else:
            return results
