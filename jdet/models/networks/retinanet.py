import jittor as jt 
from jittor import nn 

from jdet.utils.registry import META_ARCHS,build_from_cfg,BACKBONES,ROI_HEADS,NECKS


@META_ARCHS.register_module()
class RetinaNet(nn.Module):
    """
    """

    def __init__(self,backbone,neck=None,roi_head=None):
        super(RetinaNet,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.roi_heads = build_from_cfg(roi_head,ROI_HEADS)

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
        
        results,losses = self.roi_heads(features, targets)
        
        if self.is_training():
            return losses 
        else:
            return results
