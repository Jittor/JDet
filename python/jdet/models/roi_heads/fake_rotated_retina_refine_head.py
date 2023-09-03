from .fake_rotated_retina_head import NewRotatedRetinaHead
from .anchor_target import images_to_levels

from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS

import jittor as jt

@HEADS.register_module()
class RotatedRetinaRefineHead(NewRotatedRetinaHead):
    def filter_bboxes(self, cls_scores, bbox_preds):
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes)

        bboxes_list = [[] for _ in range(num_imgs)]
        num_base_anchors = self.anchor_generator.num_base_anchors[0]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = multi_level_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(-1, num_base_anchors, self.cls_out_channels)
            cls_score = jt.max(cls_score, dim=-1)
            best_ind, _ = jt.argmax(cls_score, dim=-1)
            best_ind = best_ind + jt.arange(best_ind.shape[0], dtype=jt.int32) * num_base_anchors

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(-1, 5)
            best_pred = bbox_pred[best_ind].reshape(num_imgs, -1, 5)
            best_ind = best_ind.reshape(num_imgs, -1)

            anchors = anchors.reshape(-1, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_ind_i = best_ind_i - img_id * best_ind.shape[1] * num_base_anchors
                best_pred_i = best_pred[img_id]
                anchors_i = anchors[best_ind_i]

                best_bbox_i = self.bbox_coder.decode(anchors_i, best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    def refine_bboxes(self, anchor_list, cls_scores, bbox_preds):
        raise NotImplementedError

    def loss_refine(self, anchor_list, cls_scores, bbox_preds, targets):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

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

    def get_bboxes_refine(self, anchor_list, cls_scores, bbox_preds, targets):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        result_list = []
        for img_id, target in enumerate(targets):
            cls_score_list = [
                cls_scores[i][img_id] for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id] for i in range(num_levels)
            ]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, anchor_list[img_id], target)
            result_list.append(proposals)
        return result_list

    # execute: base on generated anchors
    def execute_train(self, cls_scores, bbox_preds, targets):
        loss = super().loss(cls_scores, bbox_preds, targets)
        rois = self.filter_bboxes(cls_scores, bbox_preds)
        return rois, loss

    def execute_test(self, cls_scores, bbox_preds, targets):
        return self.filter_bboxes(cls_scores, bbox_preds)

    # refine: base on given anchors
    def refine_train(self, anchor_list, cls_scores, bbox_preds, targets, return_anchors=False):
        anchors = None
        if return_anchors:
            anchors = self.refine_bboxes(anchor_list, cls_scores, bbox_preds)
        return anchors, self.loss_refine(anchor_list, cls_scores, bbox_preds, targets)

    def refine_test(self, anchor_list, cls_scores, bbox_preds, targets, return_anchors=False):
        if return_anchors:
            return self.refine_bboxes(anchor_list, cls_scores, bbox_preds)
        return self.get_bboxes_refine(anchor_list, cls_scores, bbox_preds, targets)

    def refine(self, anchor_list, features, targets, return_anchors=False):
        outs = multi_apply(self.forward_single, features)
        if self.is_training():
            return self.refine_train(anchor_list, *outs, targets)
        return self.refine_test(anchor_list, *outs, targets, return_anchors=return_anchors)
