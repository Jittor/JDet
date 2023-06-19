from .anchor_target import anchor_target, images_to_levels, anchor_inside_flags
from ..boxes.sampler import PseudoSampler

from jdet.utils.registry import LOSSES, BOXES, build_from_cfg
from jdet.utils.general import multi_apply
from jdet.ops.bbox_transforms import get_bbox_type, bbox2type, get_bbox_dim, hbb2obb

import numpy as np
import jittor as jt
from jittor import nn, init

class NewBaseAnchorHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes=1,
                 feat_channels=256,
                 bbox_type='hbb',
                 reg_dim=4,
                 allowed_border=0,
                 background_label=0,
                 pos_weight=-1,
                 detach_proposals=False,
                 anchor_generator=None,
                 reg_decoded_bbox=False,
                 bbox_coder=None,
                 assigner=None,
                 sampler=None,
                 loss_cls=None,
                 loss_bbox=None,
                 cfg=dict(
                     min_bbox_size=0,
                     nms_thresh=0.8,
                     nms_pre=2000,
                     nms_post=2000,
                 ), 
                ):
        super(NewBaseAnchorHead, self).__init__()
        assert loss_cls is not None
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))
        
        assert bbox_type in ['hbb', 'obb', 'hbb_as_obb'], "bbox_type must be 'hbb' or 'obb' or 'hbb_as_obb'"
        self.bbox_type = bbox_type
        self.reg_dim = reg_dim
        self.reg_decoded_bbox = reg_decoded_bbox
        self.allowed_border = allowed_border
        self.background_label = background_label
        self.pos_weight = pos_weight
        self.unmap_outputs = True
        self.detach_proposals = detach_proposals

        self.loss_cls = build_from_cfg(loss_cls, LOSSES) if loss_cls is not None else None
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES) if loss_bbox is not None else None
        self.anchor_generator = build_from_cfg(anchor_generator,BOXES) if anchor_generator is not None else None
        self.bbox_coder = build_from_cfg(bbox_coder, BOXES) if bbox_coder is not None else None
        self.assigner = build_from_cfg(assigner, BOXES) if assigner is not None else None
        self.sampler = build_from_cfg(sampler, BOXES) if sampler is not None else PseudoSampler()
        
        self.cfg = cfg
        
        self._init_layers()

    def _init_layers(self):
        pass

    def init_weights(self):
        raise NotImplementedError

    def forward_single(self, x):
        raise NotImplementedError
    
    def get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           multi_level_anchors,
                           target,
                           ):
        raise NotImplementedError
    
    def unmap(self, data, count, inds, fill=0):
        """ Unmap a subset of item (data) back to the original set of items (of
        size count) """
        if data.ndim == 1:
            ret = jt.full((count, ), fill, dtype=data.dtype)
            ret[inds.astype(jt.bool)] = data
        else:
            new_size = (count, ) + data.size()[1:]
            ret = jt.full(new_size, fill, dtype=data.dtype)
            ret[inds.astype(jt.bool), :] = data
        return ret

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   targets):
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        num_levels = len(cls_scores)
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes)
        result_list = []
        for img_id, target in enumerate(targets):
            if self.detach_proposals:
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
            else:
                cls_score_list = [
                    cls_scores[i][img_id] for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id] for i in range(num_levels)
                ]                
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, multi_level_anchors, target)
            result_list.append(proposals)
        return result_list
    
    def get_targets_parse(self, target):
        if self.bbox_type == 'hbb':
            if target["hboxes"] is None:
                gt_bboxes = None
            else:
                gt_bboxes = target["hboxes"].clone()

            if target["hboxes_ignore"] is None or target["hboxes_ignore"].numel() == 0: 
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = target["hboxes_ignore"].clone()
        elif self.bbox_type == 'obb':
            if target["rboxes"] is None:
                gt_bboxes = None
            else:
                gt_bboxes = target["rboxes"].clone()
                # gt_bboxes[:, -1] *= -1

            if target["rboxes_ignore"] is None or target["rboxes_ignore"].numel() == 0: 
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = target["rboxes_ignore"].clone()
        elif self.bbox_type == 'hbb_as_obb':
            if target["hboxes"] is None:
                gt_bboxes = None
            else:
                gt_bboxes = hbb2obb(target["hboxes"].clone())

            if target["hboxes_ignore"] is None or target["hboxes_ignore"].numel() == 0: 
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = hbb2obb(target["hboxes_ignore"].clone())
        gt_labels = target["labels"].clone()
        # img_size:(w, h) img_shape:(h, w)
        img_shape = target["img_size"][::-1]
        return gt_bboxes, gt_bboxes_ignore, gt_labels, img_shape
    
    def get_targets_single(self, anchor_list, valid_flag_list, target):
        gt_bboxes, gt_bboxes_ignore, gt_labels, img_shape = self.get_targets_parse(target)
        flat_anchors = jt.concat(anchor_list)
        valid_flags = jt.concat(valid_flag_list)
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_shape, self.allowed_border)
        if not inside_flags.any_():
            return (None, ) * 7
        anchors = flat_anchors[inside_flags, :]

        # assign gt and sample anchors
        anchor_bbox_type = get_bbox_type(anchors)
        gt_bbox_type = get_bbox_type(gt_bboxes)
        target_bboxes = bbox2type(gt_bboxes, anchor_bbox_type)
        target_bboxes_ignore = None if gt_bboxes_ignore is None or gt_bboxes_ignore.numel() == 0 else \
                bbox2type(gt_bboxes_ignore, anchor_bbox_type)

        assign_result = self.assigner.assign(anchors, target_bboxes, target_bboxes_ignore, None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, target_bboxes)

        if anchor_bbox_type != gt_bbox_type:
            if gt_bboxes.numel() == 0:
                sampling_result.pos_gt_bboxes = jt.empty(gt_bboxes.shape).view(-1, get_bbox_dim(gt_bbox_type))
            else:
                sampling_result.pos_gt_bboxes = gt_bboxes[sampling_result.pos_assigned_gt_inds, :]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = jt.zeros((anchors.size(0), self.reg_dim))
        bbox_weights = jt.zeros((anchors.size(0), self.reg_dim))
        labels = jt.full((num_valid_anchors, ), self.background_label).long()
        label_weights = jt.zeros((num_valid_anchors,)).float()

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if  self.unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = self.unmap(
                labels,
                num_total_anchors,
                inside_flags,
                fill=self.background_label)  # fill bg label
            label_weights = self.unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = self.unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = self.unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result)


    def get_targets(self, anchor_lists, valid_flag_lists, targets):
        assert len(anchor_lists) == len(valid_flag_lists) == len(targets)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_lists[0]]

        # compute targets for each image
        (all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, 
         sampling_results_list) = multi_apply(self.get_targets_single, anchor_lists, valid_flag_lists, targets)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return labels_list, label_weights_list, bbox_targets_list,bbox_weights_list, num_total_pos, num_total_neg

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.
        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, self.reg_dim)
        bbox_weights = bbox_weights.reshape(-1, self.reg_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.reg_dim)
        
        if self.reg_decoded_bbox:
            anchor_dim = anchors.size(-1)
            anchors = anchors.reshape(-1, anchor_dim)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)

        return loss_cls, loss_bbox

    def loss(self, cls_scores, bbox_preds, targets):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(len(targets))]

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(jt.contrib.concat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,num_level_anchors)

        valid_flag_list = []
        for img_id, target in enumerate(targets):
            multi_level_flags = self.anchor_generator.valid_flags(featmap_sizes, target['pad_shape'][::-1])
            valid_flag_list.append(multi_level_flags)

        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, targets)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
    
    def execute_train(self, cls_scores, bbox_preds, targets):
        return self.loss(cls_scores, bbox_preds, targets)
    
    def execute_test(self, cls_scores, bbox_preds, targets):
        return self.get_bboxes(cls_scores, bbox_preds, targets)

    def execute(self, features, targets):
        outs = multi_apply(self.forward_single, features)
        if self.is_training():
            return self.execute_train(*outs, targets)
        return self.execute_test(*outs, targets)
