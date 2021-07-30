import jittor as jt 
from jittor import nn 
from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS
import numpy as np
import jdet
import copy

@MODELS.register_module()
class RetinaNet(nn.Module):
    """
    """

    def __init__(self,backbone,neck=None,rpn_net=None):
        super(RetinaNet,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.rpn_net = build_from_cfg(rpn_net,HEADS)

    def draw(self, images, results, targets, out_path):
        for i in range(images.shape[0]):
            img = images[i].data
            img = np.transpose(((img + 2.117904) / 5 * 255), [1,2,0]).astype(np.uint8)
            result = copy.deepcopy(results[i])
            target = targets[i]
            result[0][:, [0,2]] = result[0][:, [0,2]] / target["ori_img_size"][0] * target["img_size"][0]
            result[0][:, [1,3]] = result[0][:, [1,3]] / target["ori_img_size"][1] * target["img_size"][1]
            jdet.utils.visualization.visualize_r_result(img.copy(), result, out_path)
            break

    def execute(self,images,targets):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            results: detections, #((x0,y0,x1,y1,a(pi),score), type)
            losses (dict): losses
        '''
        if ("bboxes" in targets[0]):
            # limit gt [-pi/4,pi*3/4] to [-pi/2,0) TODO: move to dataloader
            for i in range(len(targets)):
                gt_bbox = targets[i]["bboxes"].data#xywha
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
                targets[i]["bboxes"] = jt.array(out_bbox)
        features = self.backbone(images)
        if self.neck:
            features = self.neck(features)
        results,losses = self.rpn_net(features, targets)

        if self.is_training():
            return losses 
        else:
            return results
