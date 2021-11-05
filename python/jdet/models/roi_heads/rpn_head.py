import jittor as jt 
from jittor import nn 
from jdet.utils.registry import BOXES, LOSSES, build_from_cfg,HEADS
from jdet.utils.general import multi_apply
from jdet.models.boxes.anchor_target import images_to_levels

@HEADS.register_module()
class RPNHead(nn.Module):
    """RPN head.
    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 num_classes=2,
                 min_bbox_size = -1,
                 nms_thresh = 0.3,
                 nms_pre = 1200,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[4, 8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64,128]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='L1Loss', loss_weight=1.0),
                 assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False)
                ):
        super(RPNHead, self).__init__()
        
        self.min_bbox_size = min_bbox_size
        self.nms_thresh  = nms_thresh
        self.nms_pre = nms_pre
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes

        self.bbox_coder = build_from_cfg(bbox_coder,BOXES)
        self.loss_cls = build_from_cfg(loss_cls,LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox,LOSSES)
        self.assigner = build_from_cfg(assigner,BOXES)
        self.sampler = build_from_cfg(sampler,BOXES)
        self.anchor_generator = build_from_cfg(anchor_generator,BOXES)

        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.num_classes, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = nn.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    
    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape):
        """Transform outputs for a single batch item into bbox predictions.
          Args:
            cls_scores (list[Tensor]): Box scores of all scale level
                each item has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas of all
                scale level, each item has shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        # bboxes from different level should be independent during NMS,
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            rpn_cls_score = rpn_cls_score.reshape(-1, 2)
            # We set FG labels to [0, num_class-1] and BG label to
            # num_class in RPN head since mmdet v2.5, which is unified to
            # be consistent with other head since mmdet v2.0. In mmdet v2.0
            # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
            scores = nn.softmax(rpn_cls_score,dim=1)[:, 0]

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]

            if self.nms_pre > 0 and scores.shape[0] > self.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                rank_inds,ranked_scores = scores.argsort(descending=True)
                topk_inds = rank_inds[:self.nms_pre]
                scores = ranked_scores[:self.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            proposals = self.bbox_coder.decode(anchors,rpn_bbox_pred,max_shape=img_shape)

            if self.min_bbox_size >= 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_mask = (w > self.min_bbox_size) & (h > self.min_bbox_size)
                if not valid_mask.all():
                    proposals = proposals[valid_mask]
                    scores = scores[valid_mask]

            dets = jt.concat([proposals,scores.unsqueeze(1)],dim=1)
            keep = jt.nms(dets,self.nms_thresh)
            proposals = proposals[keep,:]
            # scores = scores[keep]
            # mlvl_proposals.append((proposals,scores))
            mlvl_proposals.append(proposals)

        return mlvl_proposals

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   targets):
        """Transform network output for a batch into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes)

        result_list = []
        for img_id,target in enumerate(targets):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            # W,H
            img_shape = target['img_size']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,mlvl_anchors, img_shape)
            result_list.append(proposals)
        return result_list


    def _get_targets_single(self,mlvl_anchors,target):
        """Compute regression and classification targets for anchors in a
        single image.
        """
        # w,h
        gt_bboxes = target["hboxes"]
        gt_bboxes_ignore = target["hboxes_ignore"]
        anchors = jt.concat(mlvl_anchors)
        # print(gt_bboxes)

        # # filter TODO
        # w,h = target["img_size"]
        # inside_flags = (anchors[:,0]>=0) & (anchors[:,1]>=0 )& (anchors[:,2]<w) & (anchors[:,3]<h)
        # anchors = anchors[inside_flags,:]

        # print(anchors[:10],anchors.shape)

        # assign gt and sample anchors
        assign_result = self.assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, anchors,gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = jt.zeros_like(anchors)
        bbox_weights = jt.zeros_like(anchors)
        # 1 is background label
        labels = jt.ones((num_valid_anchors, )).int()
        label_weights = jt.zeros((num_valid_anchors,)).float()

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # which is box delta
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            labels[pos_inds] = 0
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    targets):
        """Compute regression and classification targets for anchors in
        multiple images.
        """

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]


        # compute targets for each image
        (all_labels, all_label_weights, all_bbox_targets, 
           all_bbox_weights, pos_inds_list, neg_inds_list, sampling_results_list) = multi_apply(self._get_targets_single,anchor_list,targets)

        
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)

        return labels_list, label_weights_list, bbox_targets_list,bbox_weights_list, num_total_pos, num_total_neg

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
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
        cls_score = cls_score.permute(0, 2, 3,1).reshape(-1, 2)
        loss_cls = self.loss_cls(cls_score, labels, weight=label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(bbox_pred,bbox_targets,bbox_weights,avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self,
             cls_scores,
             bbox_preds,
             targets):
        """Compute losses of the head.
        Args:
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(len(targets))]

        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,num_total_pos, num_total_neg = self.get_targets(anchor_list,targets)

        num_total_samples =  num_total_pos + num_total_neg 

        # anchor number of multi levels
        # num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # # concat all level anchors and flags to a single tensor
        # concat_anchor_list = []
        # for i in range(len(anchor_list)):
        #     concat_anchor_list.append(jt.concat(anchor_list[i]))
        # all_anchor_list = images_to_levels(concat_anchor_list,num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            # all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def execute(self,features,targets):
        outs = multi_apply(self.forward_single,features)
        if self.is_training():
            losses = self.loss(*outs,targets)
        else:
            losses = dict()
        proposals = self.get_bboxes(*outs,targets)
        return proposals,losses