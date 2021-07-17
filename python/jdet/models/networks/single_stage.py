import jittor as jt
from jittor import nn
from jdet.utils.registry import MODELS, build_from_cfg, BACKBONES, HEADS, NECKS


@MODELS.register_module()
class SingleStageDetector(nn.Module):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 roi_heads=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        if neck is not None:
            self.neck = build_from_cfg(neck, NECKS)
        else:
            self.neck = None
        self.bbox_head = build_from_cfg(roi_heads, HEADS)

    def execute(self, images, targets):

        feat = self.backbone(images)
        if self.neck:
            feat = self.neck(feat)

        return self.bbox_head(feat, targets)
