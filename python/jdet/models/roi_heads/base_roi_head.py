from jdet.utils.registry import build_from_cfg, MODELS, BOXES, HEADS, LOSSES, ROI_EXTRACTORS
from jdet.utils.general import multi_apply
from jdet.ops.bbox_transforms import bbox2roi, dbbox2roi
from jdet.ops.nms_rotated import ml_nms_rotated
from jdet.models.boxes.box_ops import rotated_box_to_poly

import jittor as jt
from jittor import nn

class BaseRoIHead(nn.Module):
    def __init__(self,
                 proposal_type='hbb',
                 target_type='obb',
                 reg_dim=5,
                 pos_weight=-1,
                 with_cls=True,
                 with_reg=True,
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 assigner=None,
                 sampler=None,
                 roi_extractor=None,
                 bbox_coder=None,
                 loss_cls=None,
                 loss_bbox=None,
                 cfg=None,
                 ) -> None:
        assert proposal_type in ['obb', 'hbb']
        self.proposal_type = proposal_type

        # None: use default pos gt bboxes.
        assert target_type in ['obb', 'hbb', None]
        self.target_type = target_type

        self.reg_dim = reg_dim
        self.pos_weight = pos_weight
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.assigner = build_from_cfg(assigner, BOXES) if assigner is not None else None
        self.sampler = build_from_cfg(sampler, BOXES) if assigner is not None else None
        self.roi_extractor = build_from_cfg(roi_extractor, ROI_EXTRACTORS) if roi_extractor is not None else None
        self.bbox_coder = build_from_cfg(bbox_coder, BOXES) if bbox_coder is not None else None
        self.loss_cls = build_from_cfg(loss_cls, LOSSES) if loss_cls is not None else None
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES) if loss_bbox is not None else None
        self.cfg = cfg

    def get_targets_assign_parse(self, target):
        if self.proposal_type == 'hbb':
            if target["hboxes"] is None:
                gt_bboxes = None
            else:
                gt_bboxes = target["hboxes"].clone()

            if target["hboxes_ignore"] is None or target["hboxes_ignore"].numel() == 0: 
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = target["hboxes_ignore"].clone()
        elif self.proposal_type == 'obb':
            if target["rboxes"] is None:
                gt_bboxes = None
            else:
                gt_bboxes = target["rboxes"].clone()

            if target["rboxes_ignore"] is None or target["rboxes_ignore"].numel() == 0: 
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = target["rboxes_ignore"].clone()
        gt_labels = target["labels"].clone()
        # img_size:(w, h) img_shape:(h, w)
        img_shape = target["img_size"][::-1]
        return gt_bboxes, gt_bboxes_ignore, gt_labels, img_shape
    
    def get_targets_gt_parse(self, target):
        if self.target_type == 'hbb':
            gt_targets = target['hboxes'].clone()
        elif self.target_type == 'obb':
            gt_targets = target['rboxes'].clone()
        else:
            raise NotImplementedError
        return gt_targets

    def get_targets_single(self, proposal, target, sampling_result):
        gt_bbox, gt_bbox_ignore, gt_label, img_shape = self.get_targets_assign_parse(target)
        if sampling_result is None:
            if self.proposal_type == 'hbb':
                tproposal = proposal[:, :4]
            elif self.proposal_type == 'obb':
                tproposal = proposal[:, :5]
            assign_result = self.assigner.assign(
                tproposal, gt_bbox, gt_bbox_ignore, gt_label
            )
            sampling_result = self.sampler.sample(
                assign_result, tproposal, gt_bbox, gt_label
            )

        if self.target_type is not None:
            gt_target = self.get_targets_gt_parse(target)
            gt_target = gt_target[sampling_result.pos_assigned_gt_inds]
        else:
            gt_target = sampling_result.pos_gt_bboxes

        num_pos = sampling_result.pos_bboxes.shape[0]
        num_neg = sampling_result.neg_bboxes.shape[0]
        num_samples = num_pos + num_neg
        labels = jt.zeros(num_samples, dtype=jt.int32)
        label_weights = jt.zeros(num_samples)
        bbox_targets = jt.zeros((num_samples, gt_target.shape[1]))
        bbox_weights = jt.zeros((num_samples, gt_target.shape[1])) 

        if num_pos > 0:
            labels[:num_pos] = sampling_result.pos_gt_labels
            pos_weight = 1.0 if self.pos_weight < 0 else self.pos_weight
            label_weights[:num_pos] = pos_weight
            if self.reg_decoded_bbox:
                pos_bbox_targets = gt_target
            else:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, gt_target)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1.0
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, sampling_result

    def get_targets(self, proposals, targets, sampling_results=None, concat=True):
        if sampling_results is None:
            sampling_results = [None] * len(targets)
        (labels, label_weights, bbox_targets, bbox_weights,
            sampling_result) = multi_apply(self.get_targets_single, proposals, targets, sampling_results)

        if concat:
            labels = jt.concat(labels, 0)
            label_weights = jt.concat(label_weights, 0)
            bbox_targets = jt.concat(bbox_targets, 0)
            bbox_weights = jt.concat(bbox_weights, 0)

        return labels, label_weights, bbox_targets, bbox_weights, sampling_result
    
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, 
             bbox_targets, bbox_weights, sampling_results):
        if self.with_cls:
            loss_cls = self.loss_cls(cls_score, labels, label_weights)
        if self.with_reg:
            pos_inds = labels > 0

            if self.reg_decoded_bbox:
                proposals = rois[:, 1:]
                bbox_pred = self.bbox_coder.decode(proposals, bbox_pred)

            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.reshape(-1, self.reg_dim)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.reshape(-1, self.num_classes, self.reg_dim)[pos_inds, labels[pos_inds]]
            loss_bbox = self.loss_bbox(pos_bbox_pred,
                                       bbox_targets[pos_inds],
                                       bbox_weights[pos_inds],
                                       avg_factor=bbox_targets.shape[0])
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)
    
    def get_results(self, multi_bboxes, multi_scores, score_factors=None):
        cfg = self.cfg

        num_classes = multi_scores.size(1) - 1
        # exclude background category
        if multi_bboxes.shape[1] > 5:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)[:, 1:]
        else:
            bboxes = multi_bboxes[:, None].expand((multi_bboxes.shape[0], num_classes, 5))
        scores = multi_scores[:, 1:]

        # filter out boxes with low scores
        score_thr = cfg.get('score_thr', 0.05)
        valid_mask = scores > score_thr
        bboxes = bboxes[valid_mask]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]
        labels = valid_mask.nonzero()[:, 1]

        if bboxes.numel() == 0:
            return jt.zeros((0,6), dtype=multi_bboxes.dtype), jt.zeros((0,)).int()
        nms_cfg = cfg.get('nms', None)
        if nms_cfg is not None:
            nms_cfg_ = nms_cfg.copy()
            nms_type = nms_cfg_.pop('type', 'nms')
            iou_thr = nms_cfg_.pop('iou_thr', 0.1)
            keep = ml_nms_rotated(bboxes, scores, labels, iou_thr)
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        max_num = cfg.get('max_per_img', None)
        if max_num is not None:
            inds,_ = scores.argsort(descending=True)

            if keep.size(0) > max_num:
                inds = inds[:max_num]
            bboxes = bboxes[inds]
            scores = scores[inds]
            labels = labels[inds]

        return jt.contrib.concat([bboxes, scores[:, None]], 1), labels

    def get_det_bboxes_single(self, proposals, cls_score, bbox_pred, target):
        cfg = self.cfg
        bbox_pred = self.bbox_coder.decode(proposals, bbox_pred, target['img_size'][::-1])
        scores = nn.softmax(cls_score, dim=-1) if cls_score is not None else None
        if cfg.rescale:
            bbox_pred[:, 0::5] /= target['scale_factor']
            bbox_pred[:, 1::5] /= target['scale_factor']
            bbox_pred[:, 2::5] /= target['scale_factor']
            bbox_pred[:, 3::5] /= target['scale_factor']
        det_bboxes, det_labels = self.get_results(bbox_pred, scores)
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels

    def get_det_bboxes(self, rois, cls_scores, bbox_preds, targets):
        img_idx = rois[:, 0]
        results = []
        for i, target in enumerate(targets):
            cls_score = cls_scores[img_idx == i]
            bbox_pred = bbox_preds[img_idx == i]
            proposals = rois[img_idx == i, 1:]
            results.append(self.get_det_bboxes_single(proposals, cls_score, bbox_pred, target))
        return results
    
    def get_refine_proposals_single(self, proposals, bbox_pred, target, label=None, sampling_result=None, filter_gt=False):
        assert self.reg_dim == 5
        if not self.reg_class_agnostic:
            assert label is not None
            idx_label = label * 5
            inds = jt.stack((idx_label, idx_label + 1, idx_label + 2, idx_label + 3, idx_label + 4), 1)
            bbox_pred = jt.gather(bbox_pred, 1, inds)
        assert bbox_pred.shape[1] == 5

        bbox_pred = self.bbox_coder.decode(proposals, bbox_pred, target['img_size'][::-1])

        # filter gt_bboxes
        if filter_gt:
            assert sampling_result is not None
            # num_bbox = bbox_pred.shape[0]
            # keep = jt.full((num_bbox), True, dtype=jt.bool)
            # keep[sampling_result.pos_inds] = jt.logical_not(sampling_result.pos_is_gt)
            num_rois = bbox_pred.shape[0]
            pos_keep = 1 - sampling_result.pos_is_gt
            keep = jt.ones((num_rois), dtype=jt.bool)
            keep[:len(sampling_result.pos_is_gt)] = pos_keep
            return bbox_pred[keep]
        else:
            return bbox_pred
    
    def get_refine_proposals(self, rois, bbox_preds, targets, labels=None, sampling_results=None, filter_gt=False):

        img_idx = rois[:, 0]
        results = []
        for i, target in enumerate(targets):
            keep_inds = img_idx == i
            label = labels[keep_inds] if labels is not None else None
            bbox_pred = bbox_preds[keep_inds]
            proposals = rois[keep_inds, 1:]
            sampling_result = sampling_results[i] if sampling_results is not None else None
            results.append(self.get_refine_proposals_single(proposals, bbox_pred, target, label, sampling_result, filter_gt))

        return results

    def forward(self, features):
        raise NotImplementedError

    def execute_train(self, features, proposals, targets, as_proposals=False):
        encoded_target = self.get_targets(proposals, targets)
        (labels, _, _, _, sampling_results) = encoded_target

        if self.proposal_type == 'hbb':
            rois = bbox2roi([res.bboxes for res in sampling_results])
        elif self.proposal_type == 'obb':
            rois = dbbox2roi([res.bboxes for res in sampling_results])
        else:
            raise NotImplementedError
        bbox_feats = self.roi_extractor(
            features[:self.roi_extractor.num_inputs], rois)
        cls_scores, bbox_pred = self.forward(bbox_feats)

        losses = self.loss(cls_scores, bbox_pred, rois, *encoded_target)
        if as_proposals:
            with jt.no_grad():
                proposals = self.get_refine_proposals(rois, bbox_pred, targets, labels, sampling_results, filter_gt=True)
            return losses, proposals
        return losses
    
    def execute_test(self, features, proposals, targets, as_proposals=False):
        if self.proposal_type == 'hbb':
            rois = bbox2roi(proposals)
        elif self.proposal_type == 'obb':
            rois = dbbox2roi(proposals)
        else:
            raise NotImplementedError
        bbox_feats = self.roi_extractor(
            features[:self.roi_extractor.num_inputs], rois)
        cls_scores, bbox_pred = self.forward(bbox_feats)
        
        if as_proposals:
            labels = jt.argmax(cls_scores, dim=1)[0]
            proposals = self.get_refine_proposals(rois, bbox_pred, targets, labels, None, filter_gt=False)
            return proposals
        return self.get_det_bboxes(rois, cls_scores, bbox_pred, targets)
    
    def execute(self, features, proposals, targets, as_proposals=False) -> None:
        if self.is_training():
            return self.execute_train(features, proposals, targets, as_proposals=as_proposals)
        return self.execute_test(features, proposals, targets, as_proposals=as_proposals)
