import jittor as jt 
from jittor import nn 

from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS


@MODELS.register_module()
class RetinaNet(nn.Module):
    """
    """

    def __init__(self,backbone,neck=None,rpn_net=None):
        super(RetinaNet,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.rpn_net = build_from_cfg(rpn_net,HEADS)

    def execute(self,images,targets):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            results: detections
            losses (dict): losses
        '''
        features = self.backbone(images)
        
        if self.neck:
            features = self.neck(features)
        
        results,losses = self.rpn_net(features, targets)
        if self.is_training():
            return losses 
        else:
            return results
