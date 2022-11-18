import numpy as np
import jittor as jt
from jittor import nn

from jdet.utils.registry import HEADS, BOXES, build_from_cfg
from .rotated_retina_head import RotatedRetinaHead

@HEADS.register_module()
class KFIoURRetinaHead(RotatedRetinaHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    The difference from `RetinaHead` is that its loss_bbox requires bbox_pred,
    bbox_targets, pred_decode and targets_decode as inputs.
    """

    #TODO: check 'H' mode
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_ratios=[1.0, 0.5, 2.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 test_cfg=dict(
                    nms_pre=2000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms_rotated', iou_thr=0.1),
                    max_per_img=2000),
                train_cfg=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.4,
                        min_pos_iou=0,
                        ignore_iof_thr=-1,
                        iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                    bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                    target_means=(0., 0., 0., 0., 0.),
                                    target_stds=(1., 1., 1., 1., 1.),
                                    clip_border=True),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False)):
        super(KFIoURRetinaHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            octave_base_scale=octave_base_scale,
            scales_per_octave=scales_per_octave,
            anchor_ratios=anchor_ratios,
            anchor_strides=anchor_strides,
            anchor_base_sizes=anchor_base_sizes,
            target_means=target_means,
            target_stds=target_stds,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            test_cfg=test_cfg,
            train_cfg=train_cfg,
        )

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
        # is applied directly on the decoded bounding boxes, it
        # decodes the already encoded coordinates to absolute format.
        bbox_coder_cfg = cfg.get('bbox_coder', '')
        if bbox_coder_cfg == '':
            bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
        bbox_coder = build_from_cfg(bbox_coder_cfg,BOXES)
        anchors = anchors.reshape(-1, 5)
        bbox_pred_decode = bbox_coder.decode(anchors, bbox_pred)
        bbox_targets_decode = bbox_coder.decode(anchors, bbox_targets)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_pred_decode,
            bbox_targets_decode,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox