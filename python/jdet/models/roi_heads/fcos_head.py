import jittor as jt
from jittor import nn 

from jdet.models.boxes.box_ops import mintheta_obb,distance2obb,rotated_box_to_poly
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS,LOSSES,build_from_cfg
from jdet.models.utils.weight_init import normal_init,bias_init_with_prob
from jdet.models.utils.modules import ConvModule

from jdet.ops.nms_rotated import multiclass_nms_rotated

INF = 1e8

class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = jt.array(scale).float()

    def execute(self, x):
        return x * self.scale

@HEADS.register_module()
class FCOSHead(nn.Module):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
    """ 

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels = 256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 conv_bias='auto',
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 scale_theta=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='PolyIoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_bce=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, is_train=True),
                 test_cfg = None,
                 conv_cfg = None
                 ):
        super(FCOSHead, self).__init__()

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.scale_theta = scale_theta

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.bbox_type = 'obb'
        self.reg_dim = 4
        self.stacked_convs = stacked_convs
        self.strides = strides
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_from_cfg(loss_cls,LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox,LOSSES)
        self.loss_centerness = build_from_cfg(loss_centerness,LOSSES)
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)

        normal_init(self.conv_centerness, std=0.01)
        normal_init(self.conv_theta, std=0.01)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.reg_dim, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_theta = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.scale_theta:
            self.scale_t = Scale(1.0)

    def execute(self, feats,targets):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                centernesses (list[Tensor]): Centerss for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        feats = multi_apply(self.forward_single, feats, self.scales,self.strides)
        if self.is_training():
            return self.loss(*feats,targets) 
        else:
            return self.get_bboxes(*feats,targets) 

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness
                predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level

        bbox_pred = scale(bbox_pred)
        if self.norm_on_bbox:
            bbox_pred = nn.relu(bbox_pred)
            if not self.is_training(): 
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()

        theta_pred = self.conv_theta(reg_feat)
        if self.scale_theta:
            theta_pred = self.scale_t(theta_pred)
        return cls_score, bbox_pred, theta_pred, centerness

    def loss(self,
             cls_scores,
             bbox_preds,
             theta_preds,
             centernesses,
             targets):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            targets (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype)
        labels, bbox_targets = self.get_targets(all_level_points, targets)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_theta_preds = [
            theta_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for theta_pred in theta_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = jt.concat(flatten_cls_scores)
        flatten_bbox_preds = jt.concat(flatten_bbox_preds)
        flatten_theta_preds = jt.concat(flatten_theta_preds)
        flatten_centerness = jt.concat(flatten_centerness)
        flatten_labels = jt.concat(labels)
        flatten_bbox_targets = jt.concat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = jt.concat([points.repeat(num_imgs, 1) for points in all_level_points])

        # cat bbox_preds and theta_preds to obb bbox_preds
        flatten_bbox_preds = jt.concat([flatten_bbox_preds, flatten_theta_preds], dim=1)

        # FG cat_id: [1, num_classes], BG cat_id: 0
        # pos_inds = jt.where(flatten_labels >0)[0]
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)

        num_pos = len(pos_inds)

        flatten_labels+=1
        flatten_labels[flatten_labels==bg_class_ind+1]=0
        loss_cls = self.loss_cls( flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2obb(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2obb(pos_points, pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)


    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   theta_preds,
                   centernesses,
                   targets,
                   rescale=True):
        """ Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            targets (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

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

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype)
        result_list = []
        for img_id in range(len(targets)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            theta_pred_list = [
                theta_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = targets[img_id]['img_size']
            scale_factor = targets[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 theta_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           theta_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, theta_pred, centerness, points in zip(
                cls_scores, bbox_preds, theta_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            theta_pred = theta_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred = jt.concat([bbox_pred, theta_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)

            centerness = centerness + cfg.get("centerness_factor",0.)

            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                points = points[topk_inds,:]
                centerness = centerness[topk_inds]
            bboxes = distance2obb(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = jt.concat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = jt.concat(mlvl_scores)
        padding = jt.zeros((mlvl_scores.shape[0], 1),dtype=mlvl_scores.dtype)

        mlvl_centerness = jt.concat(mlvl_centerness)
        mlvl_scores = jt.concat([padding,mlvl_scores], dim=1)

        det_bboxes, det_labels  = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)

        return polys,scores, det_labels


    def get_points(self, featmap_sizes, dtype, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (jt.dtype): Type of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],dtype))
        return mlvl_points

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype):
        """Get points according to feature map sizes."""
        h, w = featmap_size
        x_range = jt.arange(w, dtype=dtype)
        y_range = jt.arange(h, dtype=dtype)
        y, x = jt.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        points = jt.stack((x * stride, y * stride),dim=-1) + stride // 2
        return points

    def get_targets(self, points, targets):
        """Compute regression, classification and centerss targets for points
            in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            jt.array(self.regress_ranges[i],dtype=points[i].dtype)[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = jt.concat(expanded_regress_ranges, dim=0)
        concat_points = jt.concat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        gt_bboxes_list = [t["rboxes"] for t in targets]
        gt_labels_list = [t["labels"] for t in targets]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                jt.concat([labels[i] for labels in labels_list]))
            bbox_targets = jt.concat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets[:, :4] = bbox_targets[:, :4] / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return jt.zeros((num_points,),dtype=gt_labels.dtype), \
                   jt.zeros((num_points, 5),dtype=gt_bboxes.dtype)

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = mintheta_obb(gt_bboxes)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_thetas = jt.split(gt_bboxes, [2, 2, 1], dim=2)

        Cos, Sin = jt.cos(gt_thetas), jt.sin(gt_thetas)
        Matrix = jt.concat([Cos, -Sin, Sin, Cos], dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = jt.matmul(Matrix, offset[..., None])
        offset = offset.squeeze(-1)

        W, H = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = W / 2 + offset_x
        right = W / 2 - offset_x
        top = H / 2 + offset_y
        bottom = H / 2 - offset_y
        bbox_targets = jt.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        # if center_sampling is true, also in center bbox.
        inside_gt_bbox_mask = bbox_targets.min(-1) > 0
        if self.center_sampling:
            # inside a `center bbox`
            radius = self.center_sample_radius
            stride = jt.zeros_like(offset)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = jt.logical_and(
                inside_center_bbox_mask, inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area_inds,min_area  = areas.argmin(dim=1)

        labels = gt_labels[min_area_inds]-1
        labels[min_area == INF] = self.num_classes  # set as BG

        bbox_targets = bbox_targets[jt.index((num_points,),dim=0), min_area_inds]

        theta_targets = gt_thetas[jt.index((num_points,),dim=0), min_area_inds]
        bbox_targets = jt.concat([bbox_targets, theta_targets], dim=1)
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1) / left_right.max(dim=-1)) * (
                top_bottom.min(dim=-1) / top_bottom.max(dim=-1))
        return jt.sqrt(centerness_targets)
