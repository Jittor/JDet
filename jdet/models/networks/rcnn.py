import jittor as jt 
from jittor import nn 

from jdet.config.config import get_cfg
from jdet.utils.registry import META_ARCHS,build_from_cfg,BACKBONES,ROI_HEADS,NECKS


@META_ARCHS.register_module()
class RCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self,backbone,neck=None,rpn=None,roi_heads=None):
        super(RCNN,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.rpn = build_from_cfg(rpn,ROI_HEADS)
        self.roi_heads = build_from_cfg(roi_heads,ROI_HEADS)

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

        proposals,rpn_losses = self.rpn(features,targets)
        result, detector_losses = self.roi_heads(features, proposals, targets)
        
        losses = 0.
        for loss1,loss2 in zip(rpn_losses,detector_losses):
            loss1.update(loss2)
            losses+=sum(loss1.values())
            
        return result,losses



