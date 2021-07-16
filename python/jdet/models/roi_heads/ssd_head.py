from re import M
import jittor as jt
from jittor import nn, init
from jdet.utils.general import multi_apply
from jdet.utils.registry import build_from_cfg, HEADS, BOXES, MODELS
# from jdet.models.losses.ssd_loss import ssd_loss
from jdet.models.boxes.anchor_target import anchor_target
from jdet.ops.nms import multiclass_nms


@HEADS.register_module()
class SSDHead(nn.Module):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     basesize_ratio_range=(0.15, 0.9),
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
                 bbox_coder_cfg=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=(0., 0., 0., 0.),
                     target_stds=(1, 1, 1, 1)),
                 train_cfg=None,
                 test_cfg=None,
                 ):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes + 1  # add background class
        self.anchor_generator = build_from_cfg(anchor_generator, MODELS)
        n_anchor = self.anchor_generator.num_base_anchors

        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    n_anchor[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    n_anchor[i] * (num_classes + 1),
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        self.bbox_coder = build_from_cfg(bbox_coder_cfg, BOXES)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        for i in self.reg_convs:
            init.gauss_(i.weight, 0, 0.001)
            init.constant_(i.bias, 0)
        for i in self.cls_convs:
            init.gauss_(i.weight, 0, 0.001)
            init.constant_(i.bias, 0)

    def get_bboxes(
            self,
            cls_scores,
            bbox_preds,
            img_metas,
            cfg=None,
            rescale=True,
            with_nms=True):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes)

        mlvl_cls_scores = [cls_scores[i] for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i] for i in range(num_levels)]
        img_shapes = [
            img_metas[i]['img_shape']
            for i in range(cls_scores[0].shape[0])
        ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_var = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        use_sigmoid_cls = cfg.get('use_sigmoid_cls', False)
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)

            if use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = nn.softmax(cls_score, -1)
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)
            nms_pre = get_k_for_topk(nms_pre_var, bbox_pred.shape[1])
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if use_sigmoid_cls:
                    max_scores = scores.max(-1)
                else:
                    max_scores = scores[..., :-1].max(-1)
                _, topk_inds = jt.topk(max_scores, int(nms_pre))
                batch_inds = jt.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shapes)
            # print(bboxes[0, :10, 0])
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = jt.concat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= jt.array([[0.46875,
                                          0.625, 0.46875, 0.625]]).unsqueeze(1)
        batch_mlvl_scores = jt.concat(mlvl_scores, dim=1)

        if use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = jt.zeros((batch_size, batch_mlvl_scores.shape[1], 1))
            batch_mlvl_scores = jt.concat([batch_mlvl_scores, padding], dim=-1)
        if with_nms:
            result_list = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                                  batch_mlvl_scores):
                det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.get(
                    'score_thr'), cfg.get('nms'), cfg.get('max_per_img'))

                result_list.append(tuple([det_bbox, det_label]))
        else:
            result_list = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]

        return result_list

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            targets (list[dict]): Image info.

        Returns:
            tuple:
                anchor_list (list[jt.Var]): Anchors of each image.
                valid_flag_list (list[jt.Var]]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss(self, cls_scores, bbox_preds, targets):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore = self.parse_targets(
            targets)

        assert len(featmap_sizes) == self.anchor_generator.num_levels

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)

        if cls_reg_targets is None:
            return None

        (labels, label_weights, bbox_targets, bbox_weights,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(targets)

        all_cls_scores = jt.concat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)

        all_labels = jt.concat(labels, -1).view(num_images, -1)
        all_label_weights = jt.concat(label_weights,
                                      -1).view(num_images, -1)
        all_bbox_preds = jt.concat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = jt.concat(bbox_targets,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = jt.concat(bbox_weights,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(jt.concat(anchor_list[i]))

        losses_cls, losses_bbox = multi_apply(
            ssd_loss,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_pos,
            num_classes=self.n_class,
            train_cfg=self.train_cfg)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def execute(self, feats, targets):

        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))

        if self.is_training():
            return self.loss(cls_scores, bbox_preds, targets)
        else:
            return self.get_bboxes(cls_scores, bbox_preds, self.parse_targets(targets, is_train=False))

    def parse_targets(self, targets, is_train=True):
        img_metas = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_labels = []

        for target in targets:
            if is_train:
                gt_bboxes.append(target["bboxes"])
                gt_labels.append(target["labels"])
                gt_bboxes_ignore.append(target["bboxes_ignore"])
            img_metas.append(dict(
                img_shape=target["img_size"][::-1],
                scale_factor=target["scale_factor"],
                pad_shape=target["pad_shape"]
            ))
        if not is_train:
            return img_metas
        return gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore


def get_k_for_topk(k, size):
    ret_k = -1
    if k <= 0 or size <= 0:
        return ret_k
    if k < size:
        ret_k = k
    else:
        # ret_k is -1
        pass
    return ret_k
