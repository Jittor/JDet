import numpy as np
import jittor as jt
from jittor import nn

from jdet.models.utils.weight_init import normal_init,bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.utils.general import multi_apply,unmap
from jdet.utils.registry import HEADS,LOSSES,BOXES,build_from_cfg


from jdet.ops.nms_rotated import multiclass_nms_rotated
from jdet.models.boxes.box_ops import delta2bbox_rotated, rotated_box_to_poly
from jdet.models.boxes.anchor_target import images_to_levels, anchor_inside_flags
from jdet.models.boxes.anchor_generator import AnchorGeneratorRotatedRetinaNet
from jdet.models.boxes.sampler import PseudoSampler

@HEADS.register_module()
class CSLRRetinaHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_ratios=[1.0, 0.5, 2.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                angle_coder=dict(
                    type='CSLCoder',
                    omega=4,
                    window='gaussian',
                    radius=3),
                loss_angle=dict(
                    type='SmoothFocalLoss', gamma=2.0, alpha=0.25, loss_weight=0.8),
                 test_cfg=dict(
                    nms_pre=2000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms_rotated', iou_thr=0.1),
                    max_per_img=2000),
                train_cfg=dict(
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
                    debug=False)):
        super(CSLRRetinaHead, self).__init__()
        self.angle_coder = build_from_cfg(angle_coder,BOXES)
        self.coding_len = self.angle_coder.coding_len

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

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))
        self.loss_cls = build_from_cfg(loss_cls,LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox,LOSSES)
        self.loss_angle = build_from_cfg(loss_angle,LOSSES)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            
            self.anchor_generators.append(AnchorGeneratorRotatedRetinaNet(anchor_base, None, anchor_ratios, 
                octave_base_scale=octave_base_scale, scales_per_octave=scales_per_octave))
        self.num_anchors = self.anchor_generators[0].num_base_anchors
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU()
        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))

        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 5, 1)
        self.retina_cls = nn.Conv2d(self.feat_channels, 
                                    self.num_anchors * self.cls_out_channels, 1)
        self.retina_angle_cls = nn.Conv2d(self.feat_channels, 
                                          self.num_anchors * self.coding_len,
                                          3,
                                          padding=1)
        self.init_weights()

    def init_weights(self):
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_reg, std=0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_angle_cls, std=0.01, bias=bias_cls)

    def forward_single(self, x, stride):
        reg_feat = x
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        bbox_pred = self.retina_reg(reg_feat)
        angle_cls = self.retina_angle_cls(reg_feat)

        cls_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        cls_score = self.retina_cls(cls_feat)
        
        return cls_score, bbox_pred, angle_cls

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

    def loss(self,
             cls_scores,
             bbox_preds,
             angle_clses,
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
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,num_total_pos, num_total_neg, angle_targets_list, angle_weights_list = cls_reg_targets
        
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos

        losses_cls, losses_bbox, losses_angle = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            angle_clses,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            angle_targets_list,
            angle_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)

        return dict(loss_cls=losses_cls,
                    loss_bbox=losses_bbox,
                    loss_angle=losses_angle)

    def loss_single(self,
                        cls_score,
                        bbox_pred,
                        angle_cls,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        angle_targets,
                        angle_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

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
            bbox_pred = bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        angle_cls = angle_cls.permute(0, 2, 3, 1).reshape(-1, self.coding_len)
        angle_targets = angle_targets.reshape(-1, self.coding_len)
        angle_weights = angle_weights.reshape(-1, 1)

        loss_angle = self.loss_angle(
            angle_cls,
            angle_targets,
            weight=angle_weights,
            avg_factor=num_total_samples)
        
        return loss_cls, loss_bbox, loss_angle

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_clses,
                   img_metas,
                   rescale=True):
        assert len(cls_scores) == len(bbox_preds)
        cfg = self.test_cfg.copy()

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        num_levels = len(cls_scores)
        anchor_list, _ = self.get_init_anchors(featmap_sizes, img_metas)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_cls_list = [
                angle_clses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, angle_cls_list,
                                               anchor_list[img_id], img_shape,
                                               scale_factor, cfg, rescale)

            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          angle_cls_list,
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
        for cls_score, bbox_pred, angle_cls, anchors in zip(cls_score_list,
                                                 bbox_pred_list, angle_cls_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)

            angle_cls = angle_cls.permute(1, 2, 0).reshape(
                -1, self.coding_len).sigmoid()

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
                angle_cls = angle_cls[topk_inds, :]

            # Angle decoder
            angle_pred = self.angle_coder.decode(angle_cls)
            bbox_pred[..., -1] = angle_pred
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

    def execute(self, feats,targets):
        outs = multi_apply(self.forward_single, feats, self.anchor_strides)
        if self.is_training():
            return self.loss(*outs,*self.parse_targets(targets))
        else:
            return self.get_bboxes(*outs,self.parse_targets(targets,is_train=False))

    def anchor_target(self, 
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
                    unmap_outputs=True):
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
        pos_inds_list, neg_inds_list, all_angle_targets, all_angle_weights) = multi_apply(
            self.anchor_target_single,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            target_means=target_means,
            target_stds=target_stds,
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
        angle_targets_list = images_to_levels(all_angle_targets, num_level_anchors)
        angle_weights_list = images_to_levels(all_angle_weights, num_level_anchors)

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, angle_targets_list, angle_weights_list)

    def anchor_target_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            target_means,
                            target_stds,
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

        if sampling:
            assign_result, sampling_result = assign_and_sample(
                anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
        else:
            bbox_assigner = build_from_cfg(cfg.get('assigner', ''), BOXES)
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                gt_bboxes_ignore, gt_labels)
            bbox_sampler = PseudoSampler()
            sampling_result = bbox_sampler.sample(assign_result, anchors,
                                                gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = jt.zeros_like(anchors)
        bbox_weights = jt.zeros_like(anchors)
        angle_targets = jt.zeros_like(bbox_targets[:, 4:5])
        angle_weights = jt.zeros_like(bbox_targets[:, 4:5])
        labels = jt.zeros(num_valid_anchors).int()
        label_weights = jt.zeros(num_valid_anchors).float()
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

            angle_targets[pos_inds, :] = pos_bbox_targets[:, 4:5]
            # Angle encoder
            angle_targets = self.angle_coder.encode(angle_targets)
            angle_weights[pos_inds, :] = 1.0

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
            angle_targets = unmap(angle_targets, num_total_anchors, inside_flags)
            angle_weights = unmap(angle_weights, num_total_anchors, inside_flags)
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, angle_targets, angle_weights)
