# from .base_roi_head import BaseRoIHead
from .convfc_roi_head import ConvFCRoIHead

from jdet.utils.registry import build_from_cfg, LOSSES, HEADS, BOXES
from jdet.ops.bbox_transforms import hbb2obb
from jdet.ops.nms_rotated import multiclass_nms_rotated
from jdet.models.boxes.box_ops import rotated_box_to_poly, rotated_box_to_bbox, poly_to_rotated_box
from jdet.utils.general import multi_apply

from jittor import nn
import jittor as jt

@HEADS.register_module()
class GlidingRoIHead(ConvFCRoIHead):
    def __init__(self,
                 num_fix_convs=0,
                 num_fix_fcs=0,
                 num_ratio_convs=0,
                 num_ratio_fcs=0,
                 with_fix=True,
                 with_ratio=True,
                 ratio_thr=0.8,
                 loss_fix=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=1.0,
                     ),
                 loss_ratio=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=16.0
                     ),
                 fix_coder=dict(type='GVFixCoder'),
                 ratio_coder=dict(type='GVRatioCoder'),
                 **kwargs
                 ):
        self.num_fix_convs = num_fix_convs
        self.num_fix_fcs = num_fix_fcs
        self.num_ratio_convs = num_ratio_convs
        self.num_ratio_fcs = num_ratio_fcs
        self.with_fix = with_fix
        self.with_ratio = with_ratio
        self.ratio_thr = ratio_thr

        self.loss_fix = build_from_cfg(loss_fix, LOSSES)
        self.loss_ratio = build_from_cfg(loss_ratio, LOSSES)
        self.fix_coder = build_from_cfg(fix_coder, BOXES)
        self.ratio_coder = build_from_cfg(ratio_coder, BOXES)

        super(GlidingRoIHead, self).__init__(**kwargs)

    def _init_layers(self):
        super(GlidingRoIHead, self)._init_layers()
        # add fix specific branch
        self.fix_convs, self.fix_fcs, self.fix_last_dim = \
            self._add_conv_fc_branch(
                self.num_fix_convs, self.num_fix_fcs, self.shared_out_channels)
        # add ratio specific branch
        self.ratio_convs, self.ratio_fcs, self.ratio_last_dim = \
            self._add_conv_fc_branch(
                self.num_ratio_convs, self.num_ratio_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_fix_fcs == 0:
                self.fix_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_ratio_fcs == 0:
                self.ratio_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        if self.with_fix:
            out_dim_fix = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_fix = nn.Linear(self.fix_last_dim, out_dim_fix)
        if self.with_ratio:
            out_dim_ratio = 1 if self.reg_class_agnostic else self.num_classes
            self.fc_ratio = nn.Linear(self.fix_last_dim, out_dim_ratio)

    def init_weights(self):
        super(GlidingRoIHead, self).init_weights()
        for module_list in [self.fix_fcs, self.ratio_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        if self.with_fix:
            nn.init.gauss_(self.fc_fix.weight, 0, 0.001)
            nn.init.constant_(self.fc_fix.bias, 0)
        if self.with_ratio:
            nn.init.gauss_(self.fc_ratio.weight, 0, 0.001)
            nn.init.constant_(self.fc_ratio.bias, 0)

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
        fix_pred = jt.sigmoid(self.fc_fix(x_fix)) if self.with_fix else None
        ratio_pred = jt.sigmoid(self.fc_ratio(x_ratio)) if self.with_ratio else None
        return cls_score, (bbox_pred, fix_pred, ratio_pred)

    def get_targets_gt_parse(self, target):
        hboxes = target['hboxes']
        polys = target['polys']
        return hboxes, polys

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
            hboxes, polys = self.get_targets_gt_parse(target)
            hboxes = hboxes[sampling_result.pos_assigned_gt_inds]
            polys = polys[sampling_result.pos_assigned_gt_inds]
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

            pos_fix_targets = self.fix_coder.encode(polys)
            fix_targets[:num_pos, :] = pos_fix_targets
            fix_weights[:num_pos, :] = 1.0

            pos_ratio_targets = self.ratio_coder.encode(polys)
            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1.0

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, fix_targets, ratio_targets, \
                    bbox_weights, fix_weights, ratio_weights, sampling_result
    
    def get_targets(self, proposals, targets, sampling_results=None, concat=True):
        if sampling_results is None:
            sampling_results = [None] * len(targets)
        (labels, label_weights, bbox_targets, fix_targets, ratio_targets, bbox_weights, fix_weights, ratio_weights,\
            sampling_result) = multi_apply(self.get_targets_single, proposals, targets, sampling_results)

        if concat:
            labels = jt.concat(labels, 0)
            label_weights = jt.concat(label_weights, 0)
            bbox_targets = jt.concat(bbox_targets, 0)
            fix_targets = jt.concat(fix_targets, 0)
            ratio_targets = jt.concat(ratio_targets, 0)
            bbox_weights = jt.concat(bbox_weights, 0)
            fix_weights = jt.concat(fix_weights, 0)
            ratio_weights = jt.concat(ratio_weights, 0)

        return labels, label_weights, (bbox_targets, fix_targets, ratio_targets),\
            (bbox_weights, fix_weights, ratio_weights), sampling_result

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, 
             bbox_targets, bbox_weights, sampling_results):
        bbox_pred, fix_pred, ratio_pred = bbox_pred
        bbox_targets, fix_targets, ratio_targets = bbox_targets
        bbox_weights, fix_weights, ratio_weights = bbox_weights
        if self.with_cls:
            loss_cls = self.loss_cls(cls_score, labels, label_weights)
        if self.with_reg:
            assert self.with_fix
            assert self.with_ratio
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.reshape(-1, self.reg_dim)[pos_inds]
                pos_fix_pred = fix_pred.reshape(-1, 4)[pos_inds]
                pos_ratio_pred = ratio_pred.reshape(-1, 1)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.reshape(-1, self.num_classes, self.reg_dim)[pos_inds, labels[pos_inds]]
                pos_fix_pred = fix_pred.reshape(-1, self.num_classes, 4)[pos_inds, labels[pos_inds]]
                pos_ratio_pred = ratio_pred.reshape(-1, self.num_classes, 1)[pos_inds, labels[pos_inds]]
            loss_bbox = self.loss_bbox(pos_bbox_pred,
                                       bbox_targets[pos_inds],
                                       bbox_weights[pos_inds],
                                       avg_factor=bbox_targets.shape[0])
            loss_fix = self.loss_fix(pos_fix_pred,
                                     fix_targets[pos_inds],
                                     fix_weights[pos_inds],
                                     avg_factor=fix_targets.shape[0])
            loss_ratio = self.loss_ratio(pos_ratio_pred,
                                         ratio_targets[pos_inds],
                                         ratio_weights[pos_inds],
                                         avg_factor=ratio_targets.shape[0])
        return dict(gliding_loss_cls=loss_cls, gliding_loss_bbox=loss_bbox,
                    gliding_loss_fix=loss_fix, gliding_loss_ratio=loss_ratio)

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
        polys = self.fix_coder.decode(bboxes, fix_pred)

        bboxes = bboxes.view(ratio_pred.numel(), 4)
        rbboxes = poly_to_rotated_box(polys.view(ratio_pred.numel(), 8))
        ratio_pred = jt.flatten(ratio_pred)
        rbboxes[ratio_pred > self.ratio_thr] = hbb2obb(bboxes[ratio_pred > self.ratio_thr])

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

    def get_det_bboxes(self, rois, cls_scores, bbox_preds, targets):
        bbox_preds, fix_preds, ratio_preds = bbox_preds
        img_idx = rois[:, 0]
        results = []
        for i, target in enumerate(targets):
            cls_score = cls_scores[img_idx == i]
            bbox_pred = bbox_preds[img_idx == i]
            fix_pred = fix_preds[img_idx == i]
            ratio_pred = ratio_preds[img_idx == i]
            proposals = rois[img_idx == i, 1:]
            results.append(self.get_det_bboxes_single(proposals, cls_score, (bbox_pred, fix_pred, ratio_pred), target))
        return results

