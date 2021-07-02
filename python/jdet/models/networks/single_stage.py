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
        self.ssd_heads = build_from_cfg(roi_heads, HEADS)

    def execute(self, images, targets):
        features = self.backbone(images)

        if self.neck:
            features = self.neck(features)

        results, losses = self.ssd_heads(features, targets)

        if self.is_training():
            return losses
        else:
            return results
