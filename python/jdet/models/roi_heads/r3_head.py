import numpy as np
import jittor as jt
from jittor import nn

from jdet.ops.fr import FeatureRefineModule
from jdet.models.utils.weight_init import normal_init,bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS,LOSSES,BOXES,build_from_cfg


from jdet.ops.nms_rotated import multiclass_nms_rotated
from jdet.models.boxes.box_ops import delta2bbox_rotated, rotated_box_to_poly
from jdet.models.boxes.anchor_target import images_to_levels,anchor_target
from jdet.models.boxes.anchor_generator import PseudoAnchorGenerator
from jdet.models.boxes.anchor_generator import AnchorGeneratorRotatedRetinaNet


@HEADS.register_module()
class R3Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_ratios=[1.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 loss_init_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_init_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_refine_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_refine_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 test_cfg=dict(
                    nms_pre=2000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms_rotated', iou_thr=0.1),
                    max_per_img=2000),
                train_cfg=dict(
                    init_cfg=dict(
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
                        debug=False),
                    refine_cfg=dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.6,
                            neg_iou_thr=0.5,
                            min_pos_iou=0,
                            ignore_iof_thr=-1,
                            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                        target_means=(0., 0., 0., 0., 0.),
                                        target_stds=(1., 1., 1., 1., 1.),
                                        clip_border=True),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False))):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_refine_cls.get('use_sigmoid', False)
        self.sampling = loss_refine_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))
        self.loss_init_cls = build_from_cfg(loss_init_cls,LOSSES)
        self.loss_init_bbox = build_from_cfg(loss_init_bbox,LOSSES)
        self.loss_refine_cls = build_from_cfg(loss_refine_cls,LOSSES)
        self.loss_refine_bbox = build_from_cfg(loss_refine_bbox,LOSSES)
        self.feat_refine_module = FeatureRefineModule(in_channels=in_channels, featmap_strides=anchor_strides)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.anchor_generators = []
        self.refine_anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGeneratorRotatedRetinaNet(anchor_base, None, anchor_ratios, 
                octave_base_scale=octave_base_scale, scales_per_octave=scales_per_octave))
            self.refine_anchor_generators.append(PseudoAnchorGenerator(anchor_base))
        self.num_anchors = self.anchor_generators[0].num_base_anchors
        # anchor cache
        self.base_anchors = dict()
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU()
        self.init_reg_convs = nn.ModuleList()
        self.init_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.init_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.init_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))

        self.init_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 5, 3, padding=1)
        self.init_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        # self.init_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 5, 1)
        # self.init_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)

        self.refine_reg_convs = nn.ModuleList()
        self.refine_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.refine_reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.refine_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))

        self.refine_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.refine_reg = nn.Conv2d(self.feat_channels, 5, 3, padding=1)

        self.init_weights()

    def init_weights(self):
        for m in self.init_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.init_cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.init_reg, std=0.01)
        normal_init(self.init_cls, std=0.01, bias=bias_cls)

        for m in self.refine_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.refine_cls_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.refine_cls, std=0.01, bias=bias_cls)
        normal_init(self.refine_reg, std=0.01)

    def init_forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.init_cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.init_reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.init_cls(cls_feat)
        bbox_pred = self.init_reg(reg_feat)
        return cls_score, bbox_pred

    def refine_forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.refine_cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.refine_reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.refine_cls(cls_feat)
        bbox_pred = self.refine_reg(reg_feat)
        return cls_score, bbox_pred

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
            list[list[Tensor]]: best or refined rbboxes of each level \
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]
            anchors = self.anchor_generators[lvl].grid_anchors(featmap_sizes[lvl], self.anchor_strides[lvl])

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score = cls_score.max(dim=3, keepdims=True)
            best_ind, _ = cls_score.argmax(dim=2, keepdims=True)
            best_ind = best_ind.expand(-1, -1, -1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)
            best_pred = bbox_pred.gather(
                dim=-2, index=best_ind).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)

                best_bbox_i = delta2bbox_rotated(best_anchor_i, best_pred_i, self.target_means,
                                            self.target_stds, wh_ratio_clip=1e-6)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    def get_init_anchors(self,
                         featmap_sizes,
                         img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                w,h = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def get_refine_anchors(self,
                           featmap_sizes,
                           refine_anchors,
                           img_metas,
                           is_train=True):
        num_levels = len(featmap_sizes)

        # refine_anchors_list = []
        # for img_id, img_meta in enumerate(img_metas):
        #     mlvl_refine_anchors = []
        #     for i in range(num_levels):
        #         refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
        #         mlvl_refine_anchors.append(refine_anchor)
        #     refine_anchors_list.append(mlvl_refine_anchors)
        refine_anchors_list = [[
            bboxes_img_lvl.clone().detach() for bboxes_img_lvl in bboxes_img
        ] for bboxes_img in refine_anchors]

        valid_flag_list = []
        if is_train:
            for img_id, img_meta in enumerate(img_metas):
                multi_level_flags = []
                for i in range(num_levels):
                    anchor_stride = self.anchor_strides[i]
                    feat_h, feat_w = featmap_sizes[i]
                    w,h = img_meta['pad_shape'][:2]
                    valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                    valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                    flags = self.refine_anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                    multi_level_flags.append(flags)
                valid_flag_list.append(multi_level_flags)
        return refine_anchors_list, valid_flag_list

    def loss(self,
             init_cls_scores,
             init_bbox_preds,
             refine_anchors,
             refine_cls_scores,
             refine_bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        
        cfg = self.train_cfg.copy()
        featmap_sizes = [featmap.size()[-2:] for featmap in refine_cls_scores]
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
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list.copy(),
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.init_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,num_total_pos, num_total_neg = cls_reg_targets
        
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos

        losses_init_cls, losses_init_bbox = multi_apply(
            self.loss_init_single,
            init_cls_scores,
            init_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.init_cfg)


        # refine stage
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas)
        # refine_anchors_list = [[
        #     bboxes_img_lvl.clone().detach() for bboxes_img_lvl in bboxes_img
        # ] for bboxes_img in refine_anchors]

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0)
                             for anchors in refine_anchors_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(refine_anchors_list)):
            concat_anchor_list.append(jt.contrib.concat(refine_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            refine_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.refine_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_refine_cls, losses_refine_bbox = multi_apply(
            self.loss_refine_single,
            refine_cls_scores,
            refine_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.refine_cfg)

        return dict(loss_init_cls=losses_init_cls,
                    loss_init_bbox=losses_init_bbox,
                    loss_refine_cls=losses_refine_cls,
                    loss_refine_bbox=losses_refine_bbox)

    def loss_init_single(self,
                        init_cls_score,
                        init_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        init_cls_score = init_cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_init_cls = self.loss_init_cls(
            init_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        init_bbox_pred = init_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_from_cfg(bbox_coder_cfg,BOXES)
            anchors = anchors.reshape(-1, 5)
            init_bbox_pred = bbox_coder.decode(anchors, init_bbox_pred)
        loss_init_bbox = self.loss_init_bbox(
            init_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_init_cls, loss_init_bbox

    def loss_refine_single(self,
                        refine_cls_score,
                        refine_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        refine_cls_score = refine_cls_score.permute(0, 2, 3,
                                              1).reshape(-1, self.cls_out_channels)
        loss_refine_cls = self.loss_refine_cls(
            refine_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        refine_bbox_pred = refine_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_from_cfg(bbox_coder_cfg,BOXES)
            anchors = anchors.reshape(-1, 5)
            refine_bbox_pred = bbox_coder.decode(anchors, refine_bbox_pred)
        loss_refine_bbox = self.loss_refine_bbox(
            refine_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_refine_cls, loss_refine_bbox

    def get_bboxes(self,
                   refine_anchors,
                   refine_cls_scores,
                   refine_bbox_preds,
                   img_metas,
                   rescale=True):
        assert len(refine_cls_scores) == len(refine_bbox_preds)
        cfg = self.test_cfg.copy()

        featmap_sizes = [featmap.size()[-2:] for featmap in refine_cls_scores]
        num_levels = len(refine_cls_scores)

        refine_anchors = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, is_train=False)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                refine_cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                refine_bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               refine_anchors[0][img_id], img_shape,
                                               scale_factor, cfg, rescale)

            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)

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
            bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
                                        self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = jt.contrib.concat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= scale_factor
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

    
    def parse_targets(self,targets,is_train=True):
        img_metas = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_labels = []

        for target in targets:
            if is_train:
                gt_bboxes.append(target["rboxes"])
                gt_labels.append(target["labels"])
                gt_bboxes_ignore.append(target["rboxes_ignore"])
            img_metas.append(dict(
                img_shape=target["img_size"][::-1],
                scale_factor=target["scale_factor"],
                pad_shape = target["pad_shape"]
            ))
        if not is_train:
            return img_metas
        return gt_bboxes,gt_labels,img_metas,gt_bboxes_ignore

    def execute(self, feats, targets):

        init_outs = multi_apply(self.init_forward_single, feats)
        rois = self.filter_bboxes(*init_outs)
        x_refine = self.feat_refine_module(feats, rois)
        refine_outs =  multi_apply(self.refine_forward_single, x_refine)

        if self.is_training():
            return self.loss(*init_outs, rois, *refine_outs, *self.parse_targets(targets))
        else:
            return self.get_bboxes(rois, *refine_outs, self.parse_targets(targets,is_train=False))
