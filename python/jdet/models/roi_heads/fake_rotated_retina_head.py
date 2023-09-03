from .anchor_head import NewBaseAnchorHead

from jdet.utils.registry import HEADS
from jdet.models.utils.modules import ConvModule
from jdet.models.utils.weight_init import normal_init,bias_init_with_prob
from jdet.models.boxes.box_ops import rotated_box_to_poly
from jdet.ops.nms_rotated import multiclass_nms_rotated

import numpy as np
import jittor as jt
from jittor import nn

@HEADS.register_module()
class NewRotatedRetinaHead(NewBaseAnchorHead):
    def __init__(self, *args, stacked_convs=4, **kwargs):
        self.stacked_convs = stacked_convs
        super(NewRotatedRetinaHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU()
        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1))
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1))

        self.retina_reg = nn.Conv2d(self.feat_channels, self.anchor_generator.num_base_anchors[0] * 5, 3, padding=1)
        self.retina_cls = nn.Conv2d(self.feat_channels,
                                    self.anchor_generator.num_base_anchors[0] * self.cls_out_channels, 3, padding=1)
        self.init_weights()

    def init_weights(self):
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_reg, std=0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)

    def forward_single(self, x):
        reg_feat = x
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = self.retina_reg(reg_feat)

        cls_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        cls_score = self.retina_cls(cls_feat)
        return cls_score, bbox_pred
    
    def get_bboxes_single(self, cls_score_list, bbox_pred_list, multi_level_anchors, target):
        assert len(cls_score_list) == len(bbox_pred_list) == len(multi_level_anchors)
        cfg = self.cfg
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list, bbox_pred_list, multi_level_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            # anchors = rect2rbox(anchors)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores = scores.max(dim=1)
                else:
                    max_scores = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, target['img_size'][::-1])
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = jt.contrib.concat(mlvl_bboxes)
        rescale = cfg.get('rescale', False)
        if rescale:
            mlvl_bboxes[..., :4] /= target['scale_factor']
        mlvl_scores = jt.contrib.concat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = jt.zeros((mlvl_scores.shape[0], 1),dtype=mlvl_scores.dtype)
            mlvl_scores = jt.contrib.concat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes,
                                                        mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels

@HEADS.register_module()
class NewRotatedRetinaHeadKFIoU(NewRotatedRetinaHead):
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, self.reg_dim)
        bbox_weights = bbox_weights.reshape(-1, self.reg_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.reg_dim)

        assert not self.reg_decoded_bbox

        anchors = anchors.reshape(-1, anchors.shape[-1])
        bbox_pred_decode = self.bbox_coder.decode(anchors, bbox_pred)
        bbox_targets_decode = self.bbox_coder.decode(anchors, bbox_targets)

        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, 
                                   bbox_pred_decode, bbox_targets_decode,
                                   bbox_weights, avg_factor=num_total_samples)

        return loss_cls, loss_bbox
