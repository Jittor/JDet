from .gliding_roi_head import GlidingRoIHead

from jdet.utils.registry import HEADS
from jdet.models.boxes.box_ops import rotated_box_to_poly
from jdet.ops.nms_rotated import multiclass_nms_rotated

import jittor as jt
from jittor import nn

@HEADS.register_module()
class SmoothGlidingRoIHead(GlidingRoIHead):
    def __init__(self, *args, fix_type='sigmoid', ratio_type='sigmoid', **kwargs):
        self.fix_type = fix_type
        self.ratio_type = ratio_type
        super(SmoothGlidingRoIHead, self).__init__(*args, **kwargs)
        
    def get_targets_gt_parse(self, target):
        hboxes = target['hboxes']
        rboxes = target['rboxes']
        return hboxes, rboxes

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_fix = x
        x_ratio = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.ndim > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.ndim > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        for conv in self.fix_convs:
            x_fix = conv(x_fix)
        if x_fix.ndim > 2:
            if self.with_avg_pool:
                x_fix = self.avg_pool(x_fix)
            x_fix = x_fix.view(x_fix.size(0), -1)
        for fc in self.fix_fcs:
            x_fix = self.relu(fc(x_fix))

        for conv in self.ratio_convs:
            x_ratio = conv(x_ratio)
        if x_ratio.ndim > 2:
            if self.with_avg_pool:
                x_ratio = self.avg_pool(x_ratio)
            x_ratio = x_ratio.view(x_ratio.size(0), -1)
        for fc in self.ratio_fcs:
            x_ratio = self.relu(fc(x_ratio))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.fix_type == 'cos':
            fix_pred = jt.cos(self.fc_fix(x)) * 0.5 + 0.5
        elif self.fix_type == 'sigmoid':
            fix_pred = jt.sigmoid(self.fc_fix(x))
        elif self.fix_type == 'add0.5':
            fix_pred = self.fc_fix(x) + 0.5
        elif self.fix_type == 'softmax':
            fix_pred = jt.softmax(self.fc_fix(x), dim=-1)
        else:
            fix_pred = self.fc_fix(x)
        if self.ratio_type == 'cos':
            ratio_pred = jt.cos(self.fc_ratio(x)) * 0.5 + 0.5
        elif self.ratio_type ==' sigmoid':
            ratio_pred = jt.sigmoid(self.fc_ratio(x))
        elif self.ratio_type == 'add0.5':
            ratio_pred = self.fc_ratio(x) + 0.5
        else:
            ratio_pred = self.fc_ratio(x)
        return cls_score, (bbox_pred, fix_pred, ratio_pred)

    def get_targets_single(self, proposal, target, sampling_result):
        if sampling_result is None:
            gt_bbox, gt_bbox_ignore, gt_label, img_shape = self.get_targets_assign_parse(target)
            assign_result = self.assigner.assign(
                proposal, gt_bbox, gt_bbox_ignore, gt_label
            )
            sampling_result = self.sampler.sample(
                assign_result, proposal, gt_bbox, gt_label
            )

        if self.target_type is not None:
            hboxes, rboxes = self.get_targets_gt_parse(target)
            hboxes = hboxes[sampling_result.pos_assigned_gt_inds]
            rboxes = rboxes[sampling_result.pos_assigned_gt_inds]
        else:
            raise NotImplementedError

        num_pos = sampling_result.pos_bboxes.shape[0]
        num_neg = sampling_result.neg_bboxes.shape[0]
        num_samples = num_pos + num_neg
        labels = jt.zeros(num_samples, dtype=jt.int32)
        label_weights = jt.zeros(num_samples)
        bbox_targets = jt.zeros((num_samples, 4))
        bbox_weights = jt.zeros((num_samples, 4))
        fix_targets = jt.zeros((num_samples, 4))
        fix_weights = jt.zeros((num_samples, 4))
        ratio_targets = jt.zeros((num_samples, 1))
        ratio_weights = jt.zeros((num_samples, 1))

        if num_pos > 0:
            labels[:num_pos] = sampling_result.pos_gt_labels
            pos_weight = 1.0 if self.pos_weight < 0 else self.pos_weight
            label_weights[:num_pos] = pos_weight

            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, hboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1.0

            pos_ratio_targets, pos_fix_targets = self.fix_coder.encode(rboxes)
            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1.0
            fix_targets[:num_pos, :] = pos_fix_targets
            fix_weights[:num_pos, :] = 1.0

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, fix_targets, ratio_targets, \
                    bbox_weights, fix_weights, ratio_weights, sampling_result

    def get_det_bboxes_single(self, proposals, cls_score, bbox_pred, target):
        bbox_pred, fix_pred, ratio_pred = bbox_pred
        cfg = self.cfg

        if not self.reg_class_agnostic:
            bbox_pred = jt.reshape(bbox_pred, (bbox_pred.shape[0], -1, 4))
            proposals = jt.expand(proposals[:, None, :], bbox_pred.shape)
            bbox_pred = jt.reshape(bbox_pred, (-1, 4))
            proposals = jt.reshape(proposals, (-1, 4))
            fix_pred = jt.reshape(fix_pred, (-1, 4))
        bboxes = self.bbox_coder.decode(proposals, bbox_pred, target['img_size'][::-1])
        rbboxes = self.fix_coder.decode(bboxes, ratio_pred, fix_pred)

        if not self.reg_class_agnostic:
            rbboxes = jt.reshape(rbboxes, (-1, self.num_classes * 5))

        scores = nn.softmax(cls_score, dim=-1) if cls_score is not None else None
        if cfg.rescale:
            rbboxes[:, 0::5] /= target['scale_factor']
            rbboxes[:, 1::5] /= target['scale_factor']
            rbboxes[:, 2::5] /= target['scale_factor']
            rbboxes[:, 3::5] /= target['scale_factor']
        det_bboxes, det_labels = multiclass_nms_rotated(rbboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels


