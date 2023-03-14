from jdet.utils.registry import HEADS, LOSSES, BOXES, build_from_cfg
from jdet.models.utils.modules import ConvModule
from jdet.ops.dcn_v1 import DeformConv
from jdet.ops.bbox_transforms import obb2poly, poly2obb
from jdet.utils.general import multi_apply, unmap
from jdet.ops.nms_rotated import multiclass_nms_rotated
from jdet.models.boxes.anchor_target import images_to_levels
from jdet.ops.reppoints_convex_iou import reppoints_convex_iou
from jdet.ops.reppoints_min_area_bbox import reppoints_min_area_bbox
from jdet.models.boxes.box_ops import rotated_box_to_poly

import jittor as jt
from jittor import nn
import numpy as np
from jittor.nn import _pair


import numpy as np
def deleteme(a, b, size = 10):
    if a is None and b is None:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        print('-' * size)
        for a1, b1 in zip(a.values(), b.values()):
            deleteme(a1, b1, size + 10)
        print('-' * size)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        print('-' * size)
        for a1, b1 in zip(a, b):
            deleteme(a1, b1, size + 10)
        print('-' * size)
    elif isinstance(a, jt.Var) and isinstance(b, np.ndarray):
        print((a - b).abs().max().item())
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        print(np.max(np.abs(a - b)))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        print("number diff:", a - b)
    else:
        print(type(a))
        print(type(b))
        raise NotImplementedError
def transpose_to(a, b):
    if a is None:
        return None
    if isinstance(a, list) and isinstance(b, list):
        rlist = []
        for a1, b1 in zip(a, b):
            rlist.append(transpose_to(a1, b1))
        return rlist
    elif isinstance(a, dict) and isinstance(b, dict):
        rdict = []
        for k in b.keys():
            rdict[k] = transpose_to(a[k], b[k])
        return rdict
    elif isinstance(a, np.ndarray) and isinstance(b, jt.Var):
        return jt.array(a)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a
    elif isinstance(a, tuple) and isinstance(b, tuple):
        rlist = [transpose_to(a1, b1) for a1, b1 in zip(a, b)]
        return tuple(rlist)
    elif isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)):
        assert(type(a) == type(b))
        return a
    else:
        print(type(a))
        print(type(b))
        raise NotImplementedError



@HEADS.register_module()
class RotatedRepPointsHead(nn.Module):
    """Rotated RepPoints head.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        feat_channels (int): Number of feature channels.
        point_feat_channels (int, optional): Number of channels of points
            features.
        stacked_convs (int, optional): Number of stacked convolutions.
        num_points (int, optional): Number of points in points set.
        gradient_mul (float, optional): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable, optional): points strides.
        point_base_scale (int, optional): Bbox scale for assigning labels.
        conv_bias (str, optional): The bias of convolution.
        loss_cls (dict, optional): Config of classification loss.
        loss_bbox_init (dict, optional): Config of initial points loss.
        loss_bbox_refine (dict, optional): Config of points loss in refinement.
        conv_cfg (dict, optional): The config of convolution.
        norm_cfg (dict, optional): The config of normlization.
        train_cfg (dict, optional): The config of train.
        test_cfg (dict, optional): The config of test.
        center_init (bool, optional): Whether to use center point assignment.
        transform_method (str, optional): The methods to transform RepPoints
            to bbox.
        use_reassign (bool, optional): Whether to reassign samples.
        topk (int, optional): Number of the highest topk points. Defaults to 9.
        anti_factor (float, optional): Feature anti-aliasing coefficient.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 background_label=0,
                 conv_bias='auto',
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 center_init=True,
                 transform_method='rotrect',
                 use_reassign=False,
                 topk=6,
                 anti_factor=0.75,
                 **kwargs):
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.center_init = center_init

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = jt.array(dcn_base_offset).view(1, -1, 1, 1)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(self.point_strides, offset=0.)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.init_assigner = build_from_cfg(self.train_cfg.init.assigner, BOXES)
            self.refine_assigner = build_from_cfg(self.train_cfg.refine.assigner, BOXES)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_from_cfg(sampler_cfg, BOXES)
        self.transform_method = transform_method
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.loss_bbox_init = build_from_cfg(loss_bbox_init, LOSSES)
        self.loss_bbox_refine = build_from_cfg(loss_bbox_refine, LOSSES)
        self.use_reassign = use_reassign
        self.topk = topk
        self.anti_factor = anti_factor
        self.background_label = background_label
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1,
                                             self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def points2rotrect(self, pts, y_first=True):
        """Convert points to oriented bboxes."""
        if y_first:
            pts = pts.reshape(-1, self.num_points, 2)
            pts_dy = pts[:, :, 0::2]
            pts_dx = pts[:, :, 1::2]
            pts = jt.concat([pts_dx, pts_dy],
                            dim=2).reshape(-1, 2 * self.num_points)
        if self.transform_method == 'rotrect':
            rotrect_pred = reppoints_min_area_bbox(pts)
            return rotrect_pred
        else:
            raise NotImplementedError

    def forward_single(self, x):
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        pts_out_refine = pts_out_refine + pts_out_init.detach()

        return cls_out, pts_out_init, pts_out_refine

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl, _ in enumerate(self.point_strides):
            pts_lvl = []
            for i_img, _ in enumerate(center_list):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = jt.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = jt.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             overlaps,
                             stage='init',
                             unmap_outputs=True):
        """Single point target function."""
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight

        # if stage != 'init':
        #     import pickle
        #     input_dict = pickle.load(open("/mnt/disk/flowey/remote/JDet-debug/weights/input_dict.pkl", "rb"))
        #     proposals = transpose_to(input_dict['proposals'], proposals)
        #     gt_bboxes = transpose_to(input_dict['gt_bboxes'], gt_bboxes)
        #     gt_labels = transpose_to(input_dict['gt_labels'], gt_labels)
        #     gt_bboxes_ignore = transpose_to(input_dict['gt_bboxes_ignore'], gt_bboxes_ignore)

        # convert gt from obb to poly
        gt_bboxes = obb2poly(gt_bboxes)

        assign_result = assigner.assign(proposals, gt_bboxes,
                                        gt_bboxes_ignore,
                                        None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)

        # if stage != 'init':
        #     out_list = [sampling_result.pos_inds, sampling_result.neg_inds,
        #                 sampling_result.pos_gt_bboxes, sampling_result.pos_assigned_gt_inds,
        #                 assign_result.gt_inds]
        #     result_list = pickle.load(open("/mnt/disk/flowey/remote/JDet-debug/weights/result_dict.pkl", "rb"))
        #     deleteme(out_list, result_list)
        #     exit(0)

        num_valid_proposals = proposals.shape[0]
        bbox_gt = jt.zeros([num_valid_proposals, 8], dtype=proposals.dtype)
        pos_proposals = jt.zeros_like(proposals)
        proposals_weights = jt.zeros(num_valid_proposals, dtype=proposals.dtype)
        labels = jt.full((num_valid_proposals, ),
                                    self.background_label,
                                    dtype=jt.int32)
        label_weights = jt.zeros((num_valid_proposals,), dtype=jt.float32)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                # TODO(514flowey): first class is 1
                # labels[pos_inds] = 0
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, sampling_result)

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of \
                    each level.
                - proposal_weights_list (list[Tensor]): Proposal weights of \
                    each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = jt.concat(proposals_list[i])
            valid_flag_list[i] = jt.concat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * len(proposals_list)
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_result) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             all_overlaps_rotate_list,
             stage=stage,
             unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)

        return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg, None)

    def get_cfa_targets(self,
                        proposals_list,
                        valid_flag_list,
                        gt_bboxes_list,
                        img_metas,
                        gt_bboxes_ignore_list=None,
                        gt_labels_list=None,
                        stage='init',
                        label_channels=1,
                        unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                - all_labels (list[Tensor]): Labels of each level.
                - all_label_weights (list[Tensor]): Label weights of each \
                    level.
                - all_bbox_gt (list[Tensor]): Ground truth bbox of each level.
                - all_proposals (list[Tensor]): Proposals(points/bboxes) of \
                    each level.
                - all_proposal_weights (list[Tensor]): Proposal weights of \
                    each level.
                - pos_inds (list[Tensor]): Index of positive samples in all \
                    images.
                - gt_inds (list[Tensor]): Index of ground truth bbox in all \
                    images.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = jt.concat(proposals_list[i])
            valid_flag_list[i] = jt.concat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * len(proposals_list)
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_result) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             all_overlaps_rotate_list,
             stage=stage,
             unmap_outputs=unmap_outputs)
        pos_inds = []
        # pos_gt_index = []
        for i, single_labels in enumerate(all_labels):
            pos_mask = (0 < single_labels) & (
                single_labels <= self.num_classes) #TODO(514flowey): num_class not include background
            pos_inds.append(pos_mask.nonzero().view(-1))

        gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]

        return (all_labels, all_label_weights, all_bbox_gt, all_proposals,
                all_proposal_weights, pos_inds, gt_inds)

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels,
                    label_weights, rbbox_gt_init, convex_weights_init,
                    rbbox_gt_refine, convex_weights_refine, stride,
                    num_total_samples_refine):
        """Single loss function."""
        normalize_term = self.point_base_scale * stride
        if self.use_reassign:
            rbbox_gt_init = rbbox_gt_init.reshape(-1, 8)
            convex_weights_init = convex_weights_init.reshape(-1)
            pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
            pos_ind_init = (convex_weights_init > 0).nonzero().reshape(-1)
            pts_pred_init_norm = pts_pred_init[pos_ind_init]
            rbbox_gt_init_norm = rbbox_gt_init[pos_ind_init]
            convex_weights_pos_init = convex_weights_init[pos_ind_init]
            loss_pts_init = self.loss_bbox_init(
                pts_pred_init_norm / normalize_term,
                rbbox_gt_init_norm / normalize_term, convex_weights_pos_init)
            return 0, loss_pts_init, 0
        else:
            rbbox_gt_init = rbbox_gt_init.reshape(-1, 8)
            convex_weights_init = convex_weights_init.reshape(-1)
            # init points loss
            pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
            pos_ind_init = (convex_weights_init > 0).nonzero().reshape(-1)
            pts_pred_init_norm = pts_pred_init[pos_ind_init]
            rbbox_gt_init_norm = rbbox_gt_init[pos_ind_init]
            convex_weights_pos_init = convex_weights_init[pos_ind_init]
            loss_pts_init = self.loss_bbox_init(
                pts_pred_init_norm / normalize_term,
                rbbox_gt_init_norm / normalize_term, convex_weights_pos_init)
            # refine points loss
            rbbox_gt_refine = rbbox_gt_refine.reshape(-1, 8)
            pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
            convex_weights_refine = convex_weights_refine.reshape(-1)
            pos_ind_refine = (convex_weights_refine > 0).nonzero().reshape(-1)
            pts_pred_refine_norm = pts_pred_refine[pos_ind_refine]
            rbbox_gt_refine_norm = rbbox_gt_refine[pos_ind_refine]
            convex_weights_pos_refine = convex_weights_refine[pos_ind_refine]
            loss_pts_refine = self.loss_bbox_refine(
                pts_pred_refine_norm / normalize_term,
                rbbox_gt_refine_norm / normalize_term,
                convex_weights_pos_refine)
            # classification loss
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=num_total_samples_refine)
            return loss_cls, loss_pts_init, loss_pts_refine

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Loss function of CFA head."""

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1



        # import pickle
        # input_dict = pickle.load(open("/mnt/disk/flowey/remote/JDet-debug/weights/input_dict.pkl", "rb"))
        # featmap_sizes = input_dict['featmap_sizes']
        # img_metas = input_dict['img_metas']
        # gt_bboxes = transpose_to(input_dict['gt_bboxes'], gt_bboxes)
        # gt_labels = transpose_to(input_dict['gt_labels'], gt_labels)
        # gt_bboxes_ignore = transpose_to(input_dict['gt_bboxes_ignore'], gt_bboxes_ignore)
        # cls_scores = transpose_to(input_dict['cls_scores'], cls_scores)
        # pts_preds_init = transpose_to(input_dict['pts_preds_init'], pts_preds_init)
        # pts_preds_refine = transpose_to(input_dict['pts_preds_refine'], pts_preds_refine)


        # target for initial stage
        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if self.use_reassign:  # get num_proposal_each_lvl and lvl_num
            num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                        for featmap in cls_scores]
            num_level = len(featmap_sizes)
            assert num_level == len(pts_coordinate_preds_init)
        if self.train_cfg.init.assigner['type'] == 'ConvexAssigner':
            candidate_list = center_list
        else:
            raise NotImplementedError
        cls_reg_targets_init = self.get_targets(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, rbbox_gt_list_init, candidate_list_init, convex_weights_list_init,
         num_total_pos_init, num_total_neg_init, _) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        points_list = []
        for i_img, center in enumerate(center_list):
            points = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(
                    points_preds_init_.shape[0], -1,
                    *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(
                    0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                points.append(
                    points_center +
                    points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)

        if self.use_reassign:
            cls_reg_targets_refine = self.get_cfa_targets(
                points_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                stage='refine',
                label_channels=label_channels)
            (labels_list, label_weights_list, rbbox_gt_list_refine, _,
             convex_weights_list_refine, pos_inds_list_refine,
             pos_gt_index_list_refine) = cls_reg_targets_refine
            cls_scores = levels_to_images(cls_scores)
            cls_scores = [
                item.reshape(-1, self.cls_out_channels) for item in cls_scores
            ]
            pts_coordinate_preds_init_cfa = levels_to_images(
                pts_coordinate_preds_init, flatten=True)
            pts_coordinate_preds_init_cfa = [
                item.reshape(-1, 2 * self.num_points)
                for item in pts_coordinate_preds_init_cfa
            ]
            pts_coordinate_preds_refine = levels_to_images(
                pts_coordinate_preds_refine, flatten=True)
            pts_coordinate_preds_refine = [
                item.reshape(-1, 2 * self.num_points)
                for item in pts_coordinate_preds_refine
            ]
            with jt.no_grad():
                pos_losses_list, = multi_apply(
                    self.get_pos_loss, cls_scores,
                    pts_coordinate_preds_init_cfa, labels_list,
                    rbbox_gt_list_refine, label_weights_list,
                    convex_weights_list_refine, pos_inds_list_refine)
                labels_list, label_weights_list, convex_weights_list_refine, \
                    num_pos, pos_normalize_term = multi_apply(
                        self.reassign,
                        pos_losses_list,
                        labels_list,
                        label_weights_list,
                        pts_coordinate_preds_init_cfa,
                        convex_weights_list_refine,
                        gt_bboxes,
                        pos_inds_list_refine,
                        pos_gt_index_list_refine,
                        num_proposals_each_level=num_proposals_each_level,
                        num_level=num_level
                    )
                num_pos = sum(num_pos)
            # convert all tensor list to a flatten tensor
            cls_scores = jt.concat(cls_scores, 0).view(-1,
                                                       cls_scores[0].size(-1))
            pts_preds_refine = jt.concat(pts_coordinate_preds_refine, 0).view(
                -1, pts_coordinate_preds_refine[0].size(-1))
            labels = jt.concat(labels_list, 0).view(-1)
            labels_weight = jt.concat(label_weights_list, 0).view(-1)
            rbbox_gt_refine = jt.concat(rbbox_gt_list_refine, 0).view(
                -1, rbbox_gt_list_refine[0].size(-1))
            convex_weights_refine = jt.concat(convex_weights_list_refine,
                                              0).view(-1)
            pos_normalize_term = jt.concat(pos_normalize_term, 0).reshape(-1)
            pos_inds_flatten = ((0 <= labels) &
                                (labels < self.num_classes)).nonzero(
                                    as_tuple=False).reshape(-1)
            assert len(pos_normalize_term) == len(pos_inds_flatten)
            if num_pos:
                losses_cls = self.loss_cls(
                    cls_scores, labels, labels_weight, avg_factor=num_pos)
                pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
                pos_rbbox_gt_refine = rbbox_gt_refine[pos_inds_flatten]
                pos_convex_weights_refine = convex_weights_refine[
                    pos_inds_flatten]
                losses_pts_refine = self.loss_bbox_refine(
                    pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                    pos_rbbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                    pos_convex_weights_refine)
            else:
                losses_cls = cls_scores.sum() * 0
                losses_pts_refine = pts_preds_refine.sum() * 0
            None_list = [None] * num_level
            _, losses_pts_init, _ = multi_apply(
                self.loss_single,
                None_list,
                pts_coordinate_preds_init,
                None_list,
                None_list,
                None_list,
                rbbox_gt_list_init,
                convex_weights_list_init,
                None_list,
                None_list,
                self.point_strides,
                num_total_samples_refine=None,
            )
            loss_dict_all = {
                'loss_cls': losses_cls,
                'loss_pts_init': losses_pts_init,
                'loss_pts_refine': losses_pts_refine
            }
            return loss_dict_all
        else:
            cls_reg_targets_refine = self.get_targets(
                points_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                stage='refine',
                label_channels=label_channels)
            (labels_list, label_weights_list, rbbox_gt_list_refine,
             candidate_list_refine, convex_weights_list_refine,
             num_total_pos_refine, num_total_neg_refine,
             _) = cls_reg_targets_refine

            num_total_samples_refine = (
                num_total_pos_refine + num_total_neg_refine
                if self.sampling else num_total_pos_refine)

            losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
                self.loss_single,
                cls_scores,
                pts_coordinate_preds_init,
                pts_coordinate_preds_refine,
                labels_list,
                label_weights_list,
                rbbox_gt_list_init,
                convex_weights_list_init,
                rbbox_gt_list_refine,
                convex_weights_list_refine,
                self.point_strides,
                num_total_samples_refine=num_total_samples_refine)
            loss_dict_all = {
                'loss_cls': losses_cls,
                'loss_pts_init': losses_pts_init,
                'loss_pts_refine': losses_pts_refine
            }
            return loss_dict_all

    def get_pos_loss(self, cls_score, pts_pred, label, bbox_gt, label_weight,
                     convex_weight, pos_inds):
        """Calculate loss of all potential positive samples obtained from first
        match process.

        Args:
            cls_score (Tensor): Box scores of single image with shape
                (num_anchors, num_classes)
            pts_pred (Tensor): Box energies / deltas of single image
                with shape (num_anchors, 4)
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            bbox_gt (Tensor): Ground truth box.
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            convex_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.

        Returns:
            Tensor: Losses of all positive samples in single image.
        """
        if pos_inds.size(0) == 0:
            pos_loss = jt.zeros((0))
            return pos_loss,
        pos_scores = cls_score[pos_inds]
        pos_pts_pred = pts_pred[pos_inds]
        pos_bbox_gt = bbox_gt[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_convex_weight = convex_weight[pos_inds]
        loss_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        loss_bbox = self.loss_bbox_refine(
            pos_pts_pred,
            pos_bbox_gt,
            pos_convex_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        loss_cls = loss_cls.sum(-1)
        pos_loss = loss_bbox + loss_cls
        return pos_loss,

    def reassign(self,
                 pos_losses,
                 label,
                 label_weight,
                 pts_pred_init,
                 convex_weight,
                 gt_bbox,
                 pos_inds,
                 pos_gt_inds,
                 num_proposals_each_level=None,
                 num_level=None):
        """CFA reassign process.

        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            pts_pred_init (Tensor):
            convex_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            gt_bbox (Tensor): Ground truth box.
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            num_proposals_each_level (list, optional): Number of proposals
                of each level.
            num_level (int, optional): Number of level.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - label (Tensor): classification target of each anchor after \
                  paa assign, with shape (num_anchors,)
                - label_weight (Tensor): Classification loss weight of each \
                  anchor after paa assign, with shape (num_anchors).
                - convex_weight (Tensor): Bbox weight of each anchor with \
                    shape (num_anchors, 4).
                - pos_normalize_term (list): pos normalize term for refine \
                  points losses.
        """
        if len(pos_inds) == 0:
            return label, label_weight, convex_weight, 0, jt.array([]).type_as(convex_weight)

        num_gt = pos_gt_inds.max() + 1
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)

        # convert gt from obb to poly
        gt_bbox = obb2poly(gt_bbox)
        overlaps_matrix = reppoints_convex_iou(pts_pred_init, gt_bbox)
        overlaps_matrix = jt.transpose(overlaps_matrix)
        pos_inds_after_cfa = []
        ignore_inds_after_cfa = []
        re_assign_weights_after_cfa = []
        for gt_ind in range(num_gt):
            pos_inds_cfa = []
            pos_loss_cfa = []
            pos_overlaps_init_cfa = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                value, topk_inds = pos_losses[level_gt_mask].topk(
                    min(level_gt_mask.sum(), self.topk), largest=False)
                pos_inds_cfa.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_cfa.append(value)
                pos_overlaps_init_cfa.append(
                    overlaps_matrix[:, pos_inds[level_gt_mask][topk_inds]])
            pos_inds_cfa = jt.concat(pos_inds_cfa)
            pos_loss_cfa = jt.concat(pos_loss_cfa)
            pos_overlaps_init_cfa = jt.concat(pos_overlaps_init_cfa, 1)
            if len(pos_inds_cfa) < 2:
                pos_inds_after_cfa.append(pos_inds_cfa)
                ignore_inds_after_cfa.append(jt.empty((0)))
                re_assign_weights_after_cfa.append(jt.ones([len(pos_inds_cfa)]))
            else:
                pos_loss_cfa, sort_inds = pos_loss_cfa.sort()
                pos_inds_cfa = pos_inds_cfa[sort_inds]
                pos_overlaps_init_cfa = pos_overlaps_init_cfa[:, sort_inds] \
                    .reshape(-1, len(pos_inds_cfa))
                pos_loss_cfa = pos_loss_cfa.reshape(-1)
                loss_mean = pos_loss_cfa.mean()
                loss_var = pos_loss_cfa.var()

                gauss_prob_density = \
                    (-(pos_loss_cfa - loss_mean) ** 2 / loss_var) \
                    .exp() / loss_var.sqrt()
                index_inverted, _ = jt.arange(
                    len(gauss_prob_density)).sort(descending=True)
                gauss_prob_inverted = jt.cumsum(
                    gauss_prob_density[index_inverted], 0)
                gauss_prob = gauss_prob_inverted[index_inverted]
                gauss_prob_norm = (gauss_prob - gauss_prob.min()) / \
                                  (gauss_prob.max() - gauss_prob.min())

                # splitting by gradient consistency
                loss_curve = gauss_prob_norm * pos_loss_cfa
                _, max_thr = loss_curve.topk(1)
                reweights = gauss_prob_norm[:max_thr + 1]
                # feature anti-aliasing coefficient
                pos_overlaps_init_cfa = pos_overlaps_init_cfa[:, :max_thr + 1]
                overlaps_level = pos_overlaps_init_cfa[gt_ind] / (
                    pos_overlaps_init_cfa.sum(0) + 1e-6)
                reweights = \
                    self.anti_factor * overlaps_level * reweights + \
                    1e-6
                re_assign_weights = \
                    reweights.reshape(-1) / reweights.sum() * \
                    jt.ones(len(reweights)).type_as(
                        gauss_prob_norm).sum()
                pos_inds_temp = pos_inds_cfa[:max_thr + 1]
                ignore_inds_temp = pos_inds_cfa.new_tensor([])

                pos_inds_after_cfa.append(pos_inds_temp)
                ignore_inds_after_cfa.append(ignore_inds_temp)
                re_assign_weights_after_cfa.append(re_assign_weights)

        pos_inds_after_cfa = jt.concat(pos_inds_after_cfa)
        ignore_inds_after_cfa = jt.concat(ignore_inds_after_cfa)
        re_assign_weights_after_cfa = jt.concat(re_assign_weights_after_cfa)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_cfa).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_cfa] = 0
        convex_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_cfa)

        re_assign_weights_mask = (
            pos_inds.unsqueeze(1) == pos_inds_after_cfa).any(1)
        reweight_ids = pos_inds[re_assign_weights_mask]
        label_weight[reweight_ids] = re_assign_weights_after_cfa
        convex_weight[reweight_ids] = re_assign_weights_after_cfa

        pos_level_mask_after_cfa = []
        for i in range(num_level):
            mask = (pos_inds_after_cfa >= inds_level_interval[i]) & (
                pos_inds_after_cfa < inds_level_interval[i + 1])
            pos_level_mask_after_cfa.append(mask)
        pos_level_mask_after_cfa = jt.stack(pos_level_mask_after_cfa,
                                               0).type_as(label)
        pos_normalize_term = pos_level_mask_after_cfa * (
            self.point_base_scale *
            jt.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[
            pos_normalize_term > 0].type_as(convex_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_cfa)

        return label, label_weight, convex_weight, num_pos, pos_normalize_term

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            pts_preds_init (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 18D-tensor, has shape
                (batch_size, num_points * 2, H, W).
            pts_preds_refine (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 18D-tensor, has shape
                (batch_size, num_points * 2, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(pts_preds_refine)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype)

        result_list = []

        for img_id, _ in enumerate(img_metas):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            point_pred_list = select_single_mlvl(pts_preds_refine, img_id)

            results = self._get_bboxes_single(cls_score_list, point_pred_list,
                                              mlvl_priors, img_meta, cfg,
                                              rescale, with_nms, **kwargs)
            result_list.append(results)

        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           point_pred_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RepPoints head does not need
                this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (cx, cy, w, h, a) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(point_pred_list)
        scale_factor = img_meta['scale_factor']

        mlvl_bboxes = []
        mlvl_scores = []
        for level_idx, (cls_score, points_pred, points) in enumerate(
                zip(cls_score_list, point_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            points_pred = points_pred.permute(1, 2, 0).reshape(
                -1, 2 * self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                if self.use_sigmoid_cls:
                    max_scores = scores.max(dim=1)
                else:
                    max_scores = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            poly_pred = self.points2rotrect(points_pred, y_first=True)
            bbox_pos_center = points[:, :2].repeat(1, 4)
            polys = poly_pred * self.point_strides[level_idx] + bbox_pos_center
            bboxes = poly2obb(polys)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = jt.concat(mlvl_bboxes)

        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes[..., :4].new_tensor(
                scale_factor)
        mlvl_scores = jt.concat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = jt.zeros((mlvl_scores.shape[0], 1), dtype=mlvl_scores.dtype)
            mlvl_scores = jt.concat([padding, mlvl_scores], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img)
            boxes = det_bboxes[:, :5]
            scores = det_bboxes[:, 5]
            polys = rotated_box_to_poly(boxes)
            return polys, scores, det_labels
        else:
            raise NotImplementedError

    def parse_targets(self, targets):
        img_metas = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_labels = []

        for target in targets:
            if self.is_training():
                gt_bboxes.append(target["rboxes"])
                gt_labels.append(target["labels"])
                gt_bboxes_ignore.append(target["rboxes_ignore"])
            img_metas.append(dict(
                img_shape=target["img_size"][::-1],
                scale_factor=target["scale_factor"],
                pad_shape = target["pad_shape"]
            ))
        if not self.is_training():
            return dict(img_metas = img_metas)
        return dict(
            gt_bboxes = gt_bboxes,
            gt_labels = gt_labels,
            img_metas = img_metas,
            gt_bboxes_ignore = gt_bboxes_ignore,
        )

    def execute(self, feats, targets):
        outs = multi_apply(self.forward_single, feats)
        if self.is_training():
            return self.loss(*outs, **self.parse_targets(targets))
        return self.get_bboxes(*outs, **self.parse_targets(targets))


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list

def levels_to_images(mlvl_tensor, flatten=False):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)
        flatten (bool, optional): if shape of mlvl_tensor is (N, C, H, W)
            set False, if shape of mlvl_tensor is  (N, H, W, C) set True.

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels)
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [jt.concat(item, 0) for item in batch_list]

class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self, strides, offset=0.5):
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = jt.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
                    featmap_sizes,
                    dtype=jt.float32,
                    with_stride=False):
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str): The device where the anchors will be put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=jt.float32,
                                 with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (jt.arange(0, feat_w) +
                   self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (jt.arange(0, feat_h) +
                   self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = jt.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = jt.full((shift_xx.shape[0], ), stride_w, dtype=dtype)
            stride_h = jt.full((shift_yy.shape[0], ), stride_h, dtype=dtype)
            shifts = jt.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape):
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                 arrange as (h, w).
            device (str): The device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w))
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size):
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str, optional): The device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = jt.zeros(feat_w, dtype=jt.bool)
        valid_y = jt.zeros(feat_h, dtype=jt.bool)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=jt.float32):
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = jt.stack([x, y], 1).to(dtype)
        return prioris



