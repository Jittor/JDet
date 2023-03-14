from .rotated_retina_head import RotatedRetinaHead

import numpy as np
import jittor as jt
from jittor import nn

from jdet.models.utils.weight_init import normal_init,bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.utils.general import multi_apply, unmap
from jdet.utils.registry import HEADS,LOSSES,BOXES,build_from_cfg


from jdet.ops.nms_rotated import multiclass_nms_rotated
from jdet.models.boxes.box_ops import delta2bbox_rotated, rotated_box_to_poly
from jdet.models.boxes.anchor_target import images_to_levels,anchor_target, anchor_inside_flags, assign_and_sample
from jdet.models.boxes.sampler import PseudoSampler
from jdet.models.boxes.anchor_generator import AnchorGeneratorRotatedRetinaNet

@HEADS.register_module()
class RotatedATSSHead(RotatedRetinaHead):
    def anchor_target_single(
                            self,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            cfg=None,
                            label_channels=1,
                            sampling=True,
                            unmap_outputs=True):
        bbox_coder_cfg = cfg.get('bbox_coder', '')
        if bbox_coder_cfg == '':
            bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
        bbox_coder = build_from_cfg(bbox_coder_cfg, BOXES)
        # Set True to use IoULoss
        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                        img_meta['img_shape'][:2],
                                        cfg.get('allowed_border', -1))
        if not inside_flags.any(0):
            return (None,) * 6
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        if sampling:
            bbox_assigner = build_from_cfg(cfg.get('assigner', ''), BOXES)
            bbox_sampler = build_from_cfg(cfg.get('sampler', ''), BOXES)
            assign_result = bbox_assigner.assign(anchors, num_level_anchors_inside, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
            sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes,
                                                gt_labels)
        else:
            bbox_assigner = build_from_cfg(cfg.get('assigner', ''), BOXES)
            assign_result = bbox_assigner.assign(anchors, num_level_anchors_inside, gt_bboxes,
                                                gt_bboxes_ignore, gt_labels)
            bbox_sampler = PseudoSampler()
            sampling_result = bbox_sampler.sample(assign_result, anchors,
                                                gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = jt.zeros_like(anchors)
        bbox_weights = jt.zeros_like(anchors)
        labels = jt.zeros(num_valid_anchors).int()
        label_weights = jt.zeros(num_valid_anchors).float()
        # num_classes = 80
        # labels = jt.full((num_valid_anchors,), num_classes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not reg_decoded_bbox:
                pos_bbox_targets = bbox_coder.encode(sampling_result.pos_bboxes,
                                                    sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets.cast(bbox_targets.dtype)
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = cfg.get('pos_weight', -1)
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def anchor_target(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        target_means,
        target_stds,
        cfg,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        sampling=True,
        unmap_outputs=True
    ):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            valid_flag_list (list[list]): Multi level valid flags of each image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            target_means (Iterable): Mean value of regression targets.
            target_stds (Iterable): Std value of regression targets.
            cfg (dict): RPN train configs.

        Returns:
            tuple
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = jt.contrib.concat(anchor_list[i])
            valid_flag_list[i] = jt.contrib.concat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
        pos_inds_list, neg_inds_list) = multi_apply(
            self.anchor_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            cfg=cfg,
            label_channels=label_channels,
            sampling=sampling,
            unmap_outputs=unmap_outputs)
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
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        
        cfg = self.train_cfg.copy()
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_init_anchors(featmap_sizes, img_metas)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(jt.contrib.concat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,num_level_anchors)

        # Feature Alignment Module
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,num_total_pos, num_total_neg = cls_reg_targets
        
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)

        return dict(loss_cls=losses_cls,
                    loss_bbox=losses_bbox)

def get_num_level_anchors_inside(num_level_anchors, inside_flags):
    """Get number of every level anchors inside.

    Args:
        num_level_anchors (List[int]): List of number of every level's anchors.
        inside_flags (torch.Tensor): Flags of all anchors.

    Returns:
        List[int]: List of number of inside anchors.
    """
    split_inside_flags = jt.split(inside_flags, num_level_anchors)
    num_level_anchors_inside = [
        int(flags.sum()) for flags in split_inside_flags
    ]
    return num_level_anchors_inside
