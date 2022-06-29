import jittor as jt 
from jittor import nn
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS,BOXES,LOSSES, ROI_EXTRACTORS,build_from_cfg
from jdet.ops.nms_poly import multiclass_poly_nms

from jdet.ops.bbox_transforms import *

@HEADS.register_module()
class GlidingHead(nn.Module):

    def __init__(self,
                 num_classes=15,
                 in_channels=256,
                 representation_dim = 1024,
                 pooler_resolution =  7, 
                 pooler_scales = [1/4.,1/8., 1/16., 1/32., 1/64.],
                 pooler_sampling_ratio = 0,
                 score_thresh=0.05,
                 nms_thresh=0.1,
                 detections_per_img=2000,
                 box_weights = (10., 10., 5., 5.),
                 assigner=dict(
                     type='MaxIoUAssigner',
                     pos_iou_thr=0.5,
                     neg_iou_thr=0.5,
                     min_pos_iou=0.5,
                     ignore_iof_thr=-1,
                     match_low_quality=False,
                     iou_calculator=dict(type='BboxOverlaps2D')),
                 sampler=dict(
                     type='RandomSampler',
                     num=512,
                     pos_fraction=0.25,
                     neg_pos_ub=-1,
                     add_gt_as_proposals=True),
                 bbox_coder=dict(
                     type='GVDeltaXYWHBBoxCoder',
                     target_means=(.0, .0, .0, .0),
                     target_stds=(0.1, 0.1, 0.2, 0.2)),
                 fix_coder=dict(type='GVFixCoder'),
                 ratio_coder=dict(type='GVRatioCoder'),
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=2, version=1),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 cls_loss=dict(
                     type='CrossEntropyLoss',
                    ),
                 bbox_loss=dict(
                     type='SmoothL1Loss', 
                     beta=1.0, 
                     loss_weight=1.0
                     ),
                 fix_loss=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=1.0,
                     ),
                 ratio_loss=dict(
                     type='SmoothL1Loss', 
                     beta=1.0 / 3.0, 
                     loss_weight=16.0
                     ),
                 with_bbox=True,
                 with_shared_head=False,
                 start_bbox_type='hbb',
                 end_bbox_type='poly',
                 with_avg_pool=False,
                 pos_weight=-1,
                 reg_class_agnostic=False,
                 ratio_thr=0.8,
                 max_per_img=2000,
     ):
        super().__init__()
        self.representation_dim = representation_dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooler_resolution = pooler_resolution
        self.pooler_scales = pooler_scales
        self.pooler_sampling_ratio = pooler_sampling_ratio
        self.box_weights = box_weights
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        
        self.with_bbox = with_bbox
        self.with_shared_head = with_shared_head
        self.start_bbox_type = start_bbox_type
        self.end_bbox_type = end_bbox_type
        self.with_avg_pool = with_avg_pool
        self.pos_weight = pos_weight
        self.reg_class_agnostic = reg_class_agnostic
        self.ratio_thr = ratio_thr
        self.max_per_img = max_per_img

        self.assigner = build_from_cfg(assigner, BOXES)
        self.sampler = build_from_cfg(sampler, BOXES)
        self.bbox_coder = build_from_cfg(bbox_coder, BOXES)
        self.fix_coder = build_from_cfg(fix_coder, BOXES)
        self.ratio_coder = build_from_cfg(ratio_coder, BOXES)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        self.cls_loss = build_from_cfg(cls_loss, LOSSES)
        self.bbox_loss = build_from_cfg(bbox_loss, LOSSES)
        self.fix_loss = build_from_cfg(fix_loss, LOSSES)
        self.ratio_loss = build_from_cfg(ratio_loss, LOSSES)
        
        self._init_layers()
        self.init_weights()

    
    def _init_layers(self):

        in_dim = self.pooler_resolution * self.pooler_resolution * self.in_channels
        self.fc1 = nn.Linear(in_dim, self.representation_dim)
        self.fc2 = nn.Linear(self.representation_dim, self.representation_dim)

        self.cls_score = nn.Linear(self.representation_dim, self.num_classes + 1)
        self.bbox_pred = nn.Linear(self.representation_dim, self.num_classes * 4)
        self.fix_pred = nn.Linear(self.representation_dim, self.num_classes * 4)
        self.ratio_pred = nn.Linear(self.representation_dim, self.num_classes * 1)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        nn.init.gauss_(self.cls_score.weight,std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        for l in [self.bbox_pred, self.fix_pred, self.ratio_pred]:
            nn.init.gauss_(l.weight,std=0.001)
            nn.init.constant_(l.bias, 0)

    def arb2roi(self, bbox_list, bbox_type='hbb'):

        assert bbox_type in ['hbb', 'obb', 'poly']
        bbox_dim = get_bbox_dim(bbox_type)

        rois_list = []
        for img_id, bboxes in enumerate(bbox_list):
            if bboxes.size(0) > 0:
                img_inds = jt.full((bboxes.size(0), 1), img_id, dtype=bboxes.dtype)
                rois = jt.concat([img_inds, bboxes[:, :bbox_dim]], dim=-1)
            else:
                rois = jt.zeros((0, bbox_dim + 1), dtype=bboxes.dtype)
            rois_list.append(rois)
        rois = jt.concat(rois_list, 0)
        return rois
    
    def get_results(self, multi_bboxes, multi_scores, score_factors=None, bbox_type='hbb'):
        
        bbox_dim = get_bbox_dim(bbox_type)
        num_classes = multi_scores.size(1) - 1

        # exclude background category
        if multi_bboxes.shape[1] > bbox_dim:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, bbox_dim)
        else:
            bboxes = multi_bboxes[:, None].expand(-1, num_classes, bbox_dim)
        scores = multi_scores[:, :-1]

        # filter out boxes with low scores
        valid_mask = scores > self.score_thresh
        bboxes = bboxes[valid_mask]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]
        labels = valid_mask.nonzero()[:, 1]

        if bboxes.numel() == 0:
            bboxes =jt.zeros((0, bbox_dim+1), dtype=multi_bboxes.dtype)
            labels = jt.zeros((0, ), dtype="int64")
            return bboxes, labels
        
        if self.nms_thresh is None:
            dets = jt.concat([bboxes, scores.unsqueeze(1)], dim=1)
        else:
            dets,labels = multiclass_poly_nms(bboxes,scores,labels,self.nms_thresh)

        return dets, labels
        
    def forward_single(self, x, sampling_results, test=False):

        if test:
            rois = self.arb2roi(sampling_results, bbox_type=self.start_bbox_type)
        else:
            rois = self.arb2roi([res.bboxes for res in sampling_results])
        
        x = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            x = self.shared_head(x)
        
        if self.with_avg_pool:
            x = self.avg_pool2d(x)

        x = x.view(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        fixes = self.fix_pred(x)
        fixes = fixes.sigmoid()
        ratios = self.ratio_pred(x)
        ratios = ratios.sigmoid()

        return scores, bbox_deltas, fixes, ratios, rois
    
    def loss(self, cls_score, bbox_pred, fix_pred, ratio_pred, rois, labels, label_weights, bbox_targets, bbox_weights,
             fix_targets, fix_weights, ratio_targets, ratio_weights, reduction_override=None):

        losses = dict()
        avg_factor = max(jt.sum(label_weights > 0).float().item(), 1.)
        if cls_score.numel() > 0:
            losses['gliding_cls_loss'] = self.cls_loss(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)

        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any_():
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.astype(jt.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.astype(jt.bool),
                       labels[pos_inds.astype(jt.bool)]]

            losses['gliding_bbox_loss'] = self.bbox_loss(
                pos_bbox_pred,
                bbox_targets[pos_inds.astype(jt.bool)],
                bbox_weights[pos_inds.astype(jt.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['gliding_bbox_loss'] = bbox_pred.sum() * 0

        if pos_inds.any_():
            if self.reg_class_agnostic:
                pos_fix_pred = fix_pred.view(
                    fix_pred.size(0), 4)[pos_inds.astype(jt.bool)]
            else:
                pos_fix_pred = fix_pred.view(
                    fix_pred.size(0), -1,
                    4)[pos_inds.astype(jt.bool),
                       labels[pos_inds.astype(jt.bool)]]
            losses['gliding_fix_loss'] = self.fix_loss(
                pos_fix_pred,
                fix_targets[pos_inds.astype(jt.bool)],
                fix_weights[pos_inds.astype(jt.bool)],
                avg_factor=fix_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['gliding_fix_loss'] = fix_pred.sum() * 0

        if pos_inds.any_():
            if self.reg_class_agnostic:
                pos_ratio_pred = ratio_pred.view(
                    ratio_pred.size(0), 1)[pos_inds.astype(jt.bool)]
            else:
                pos_ratio_pred = ratio_pred.view(
                    ratio_pred.size(0), -1,
                    1)[pos_inds.astype(jt.bool),
                       labels[pos_inds.astype(jt.bool)]]
            losses['gliding_ratio_loss'] = self.ratio_loss(
                pos_ratio_pred,
                ratio_targets[pos_inds.astype(jt.bool)],
                ratio_weights[pos_inds.astype(jt.bool)],
                avg_factor=ratio_targets.size(0),
                reduction_override=reduction_override)
        else:
            losses['gliding_ratio_loss'] = ratio_pred.sum() * 0

        return losses

    def get_bboxes_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels):

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes - 1]
        labels = jt.full((num_samples,), self.num_classes, dtype="int64")
        label_weights = jt.zeros((num_samples,), dtype=pos_bboxes.dtype)
        bbox_targets = jt.zeros((num_samples, 4), dtype=pos_bboxes.dtype)
        bbox_weights = jt.zeros((num_samples, 4), dtype=pos_bboxes.dtype)
        fix_targets = jt.zeros((num_samples, 4), dtype=pos_bboxes.dtype)
        fix_weights = jt.zeros((num_samples, 4), dtype=pos_bboxes.dtype)
        ratio_targets = jt.zeros((num_samples, 1), dtype=pos_bboxes.dtype)
        ratio_weights = jt.zeros((num_samples, 1), dtype=pos_bboxes.dtype)

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if self.pos_weight <= 0 else self.pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, bbox2type(pos_gt_bboxes, 'hbb'))
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1

            pos_fix_targets = self.fix_coder.encode(bbox2type(pos_gt_bboxes, 'poly'))

            fix_targets[:num_pos, :] = pos_fix_targets
            fix_weights[:num_pos, :] = 1

            pos_ratio_targets = self.ratio_coder.encode(bbox2type(pos_gt_bboxes, 'poly'))
            
            ratio_targets[:num_pos, :] = pos_ratio_targets
            ratio_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets, fix_weights, ratio_targets, ratio_weights)
        
    def get_bboxes_targets(self, sampling_results, concat=True):

        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        
        outputs = multi_apply(
            self.get_bboxes_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list)

        (labels, label_weights, bbox_targets, bbox_weights, fix_targets, fix_weights, ratio_targets, ratio_weights) = outputs

        if concat:
            labels = jt.concat(labels, 0)
            label_weights = jt.concat(label_weights, 0)
            bbox_targets = jt.concat(bbox_targets, 0)
            bbox_weights = jt.concat(bbox_weights, 0)
            fix_targets = jt.concat(fix_targets, 0)
            fix_weights = jt.concat(fix_weights, 0)
            ratio_targets = jt.concat(ratio_targets, 0)
            ratio_weights = jt.concat(ratio_weights, 0)

        return (labels, label_weights, bbox_targets, bbox_weights, fix_targets, fix_weights, ratio_targets, ratio_weights)

    def get_bboxes(self, rois, cls_score, bbox_pred, fix_pred, ratio_pred, img_shape, scale_factor, rescale=False):
        
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        
        scores = nn.softmax(cls_score, dim=1)

        bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_shape)
        polys = self.fix_coder.decode(bboxes, fix_pred)

        bboxes = bboxes.view(*ratio_pred.size(), 4)
        polys = polys.view(*ratio_pred.size(), 8)
        polys[ratio_pred > self.ratio_thr] = hbb2poly(bboxes[ratio_pred > self.ratio_thr])

        if rescale:
            if isinstance(scale_factor, float):
                scale_factor = [scale_factor for _ in range(4)]
            scale_factor = jt.array(scale_factor, dtype=bboxes.dtype)
            polys /= scale_factor.repeat(2)
        polys = polys.view(polys.size(0), -1)

        det_bboxes, det_labels = self.get_results(polys, scores, bbox_type='poly')
        # det_labels = det_labels + 1 # output label range should be adjusted back to [1, self.class_NUm]

        return det_bboxes, det_labels

    def execute(self, x, proposal_list, targets):

        if self.is_training():

            gt_obboxes = []
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_obboxes_ignore = []

            for target in targets:
                gt_obboxes.append(target['polys'])
                gt_bboxes.append(target['hboxes'])
                gt_labels.append(target['labels'] - 1)
                gt_bboxes_ignore.append(target['hboxes_ignore'])
                gt_obboxes_ignore.append(target['polys_ignore'])

            # assign gts and sample proposals
            if self.with_bbox:
                start_bbox_type = self.start_bbox_type
                end_bbox_type = self.end_bbox_type
                target_bboxes = gt_bboxes if start_bbox_type == 'hbb' else gt_obboxes
                target_bboxes_ignore = gt_bboxes_ignore if start_bbox_type == 'hbb' else gt_obboxes_ignore

                num_imgs = len(targets)
                if target_bboxes_ignore is None:
                    target_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []

                for i in range(num_imgs):

                    assign_result = self.assigner.assign(proposal_list[i], target_bboxes[i], target_bboxes_ignore[i], gt_labels[i])

                    sampling_result = self.sampler.sample(
                        assign_result,
                        proposal_list[i],
                        target_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])

                    if start_bbox_type != end_bbox_type:
                        if gt_obboxes[i].numel() == 0:
                            sampling_result.pos_gt_bboxes = jt.zeros((0, gt_obboxes[0].size(-1)), dtype=gt_obboxes[i].dtype)
                        else:
                            sampling_result.pos_gt_bboxes = gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

                    sampling_results.append(sampling_result)
                    

            scores, bbox_deltas, fixes, ratios, rois = self.forward_single(x, sampling_results, test=False)

            bbox_targets = self.get_bboxes_targets(sampling_results)

            return self.loss(scores, bbox_deltas, fixes, ratios, rois, *bbox_targets)
            
        else:
            
            result = []
            for i in range(len(targets)):

                x_ = []
                for j in range(len(x)):
                    x_.append(x[j][i:i+1])
                scores, bbox_deltas, fixes, ratios, rois = self.forward_single(x_, [proposal_list[i]], test=True)
                img_shape = targets[i]['img_size']
                scale_factor = targets[i]['scale_factor']
                
                det_bboxes, det_labels = self.get_bboxes(rois, scores, bbox_deltas, fixes, ratios, img_shape, scale_factor)

                poly = det_bboxes[:, :8]
                scores = det_bboxes[:, 8]
                labels = det_labels

                result.append((poly, scores, labels))
            
            return result
