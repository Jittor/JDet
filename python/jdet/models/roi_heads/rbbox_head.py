import jittor as jt
import jittor.nn as nn

from jdet.utils.registry import HEADS, LOSSES, build_from_cfg
from jdet.utils.general import multi_apply
from jdet.ops.bbox_transforms import bbox2delta, mask2poly, obb2poly_v0, get_best_begin_point, polygonToRotRectangle_batch, hbb2obb_v2, dbbox2delta_v3, best_match_dbbox2delta, delta2dbbox_v3, delta2dbbox_v2, choose_best_Rroi_batch, choose_best_obb_batch
from jdet.ops.nms_rotated import multiclass_nms_rotated
from jdet.models.boxes.box_ops import rotated_box_to_poly

def bbox_target_rbbox(pos_bboxes_list,
                neg_bboxes_list,
                pos_assigned_gt_inds_list,
                gt_obbs_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                concat=True,
                with_module=True,
                hbb_trans='hbb2obb_v2'):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_rbbox_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_assigned_gt_inds_list,
        gt_obbs_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds,
        with_module=with_module,
        hbb_trans=hbb_trans)

    if concat:
        labels = jt.contrib.concat(labels, 0)
        label_weights = jt.contrib.concat(label_weights, 0)
        bbox_targets = jt.contrib.concat(bbox_targets, 0)
        bbox_weights = jt.contrib.concat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_rbbox_single(pos_bboxes,
                       neg_bboxes,
                       pos_assigned_gt_inds,
                       gt_obbs,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                       with_module=True,
                       hbb_trans='hbb2obb_v2'):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = jt.zeros(num_samples, dtype=jt.int)        #origin: torch.long
    label_weights = jt.zeros(num_samples)
    bbox_targets = jt.zeros((num_samples, 5))
    bbox_weights = jt.zeros((num_samples, 5))
    pos_gt_obbs = choose_best_obb_batch(gt_obbs[pos_assigned_gt_inds])

    if pos_bboxes.size(1) == 4:
        pos_ext_bboxes = hbb2obb_v2(pos_bboxes)
    else:
        pos_ext_bboxes = pos_bboxes

    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg['pos_weight'] <= 0 else cfg['pos_weight']
        label_weights[:num_pos] = pos_weight
        if with_module:
            pos_bbox_targets = dbbox2delta(pos_ext_bboxes, pos_gt_obbs, target_means,
                                          target_stds)
        else:
            pos_bbox_targets = dbbox2delta_v3(pos_ext_bboxes, pos_gt_obbs, target_means,
                                              target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights

def rbbox_target_rbbox(pos_rbboxes_list,
                         neg_rbboxes_list,
                         pos_gt_rbboxes_list,
                         pos_gt_labels_list,
                         cfg,
                         reg_classes=1,
                         target_means=[.0, .0, .0, .0, 0],
                         target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                         concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        rbbox_target_rbbox_single,
        pos_rbboxes_list,
        neg_rbboxes_list,
        pos_gt_rbboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = jt.contrib.concat(labels, 0)
        label_weights = jt.contrib.concat(label_weights, 0)
        bbox_targets = jt.contrib.concat(bbox_targets, 0)
        bbox_weights = jt.contrib.concat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights

def rbbox_target_rbbox_single(pos_rbboxes,
                       neg_rbboxes,
                       pos_gt_rbboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]):
    assert pos_rbboxes.size(1) == 5
    num_pos = pos_rbboxes.size(0)
    num_neg = neg_rbboxes.size(0)
    num_samples = num_pos + num_neg
    labels = jt.zeros(num_samples, dtype=jt.int)        #origin: torch.long
    label_weights = jt.zeros(num_samples)
    bbox_targets = jt.zeros((num_samples, 5))
    bbox_weights = jt.zeros((num_samples, 5))

    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = best_match_dbbox2delta(pos_rbboxes, pos_gt_rbboxes, target_means, target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights

def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.equal(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdims=True)
#        res.append(correct_k.mul_(100.0 / pred.size(0)))
        correct_k *= 100.0 / pred.shape[0]
        res.append(correct_k)
    return res[0] if return_single else res

@HEADS.register_module()
class BBoxHeadRbbox(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=19,
                 target_means=[0., 0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
                 reg_class_agnostic=False,
                 with_module=True,
                 hbb_trans='hbb2obb_v2',
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHeadRbbox, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic

        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            if isinstance(self.roi_feat_size, int):
                in_channels *= (self.roi_feat_size * self.roi_feat_size)
            elif isinstance(self.roi_feat_size, tuple):
                assert len(self.roi_feat_size) == 2
                assert isinstance(self.roi_feat_size[0], int)
                assert isinstance(self.roi_feat_size[1], int)
                in_channels *= (self.roi_feat_size[0] * self.roi_feat_size[1])
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 5 if reg_class_agnostic else 5 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None
        self.with_module = with_module
        self.hbb_trans = hbb_trans

    def init_weights(self):
        if self.with_cls:
#            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.gauss_(self.fc_cls,0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
#            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.gauss_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def excute(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_obbs, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds  for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_assigned_gt_inds,
            gt_obbs,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            with_module=self.with_module,
            hbb_trans=self.hbb_trans)
        return cls_reg_targets

    def get_target_rbbox(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = rbbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = nn.softmax(cls_score, dim=1) if cls_score is not None else None

        if rois.size(1) == 5:
            obbs = hbb2obb_v2(rois[:, 1:])
        elif rois.size(1) == 6:
            obbs = rois[:, 1:]
        else:
            print('strange size')
            import pdb
            pdb.set_trace()
        if bbox_pred is not None:
            dbboxes = delta2dbbox_v3(obbs, bbox_pred, self.target_means,
                                    self.target_stds, img_shape)
        else:
            dbboxes = obbs

        if rescale:
            dbboxes[:, 0::5] /= scale_factor
            dbboxes[:, 1::5] /= scale_factor
            dbboxes[:, 2::5] /= scale_factor
            dbboxes[:, 3::5] /= scale_factor
        det_bboxes, det_labels = multiclass_nms_rotated(dbboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        det_bboxes = jt.contrib.concat([rotated_box_to_poly(det_bboxes), det_bboxes[:,-1:]], -1)
        return det_bboxes, det_labels

    def get_det_rbboxes(self,
                       rrois,
                       cls_score,
                       rbbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = nn.softmax(cls_score, dim=1) if cls_score is not None else None

        if rbbox_pred is not None:
            # bboxes = delta2dbbox(rois[:, 1:], bbox_pred, self.target_means,
            #                     self.target_stds, img_shape)
            dbboxes = delta2dbbox_v2(rrois[:, 1:], rbbox_pred, self.target_means,
                                     self.target_stds, img_shape)
        else:
            # bboxes = rois[:, 1:]
            dbboxes = rrois[:, 1:]
            # TODO: add clip here

        if rescale:
            # bboxes /= scale_factor
            # dbboxes[:, :4] /= scale_factor
            dbboxes[:, 0::5] /= scale_factor
            dbboxes[:, 1::5] /= scale_factor
            dbboxes[:, 2::5] /= scale_factor
            dbboxes[:, 3::5] /= scale_factor
        if cfg is None:
            return dbboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms_rotated(dbboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            det_bboxes = jt.contrib.concat([rotated_box_to_poly(det_bboxes), det_bboxes[:,-1:]], -1)
            return det_bboxes, det_labels

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduce=True):
        losses = dict()
        if cls_score is not None:
            losses['rbbox_loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, reduce=reduce)
            losses['rbbox_acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               5)[pos_inds, labels[pos_inds]]
            losses['rbbox_loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0))
        return losses
    def refine_rbboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 5) or (n*bs, 5*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique()
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = jt.nonzero(rois[:, 0] == i).squeeze(-1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class_rbbox(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = jt.ones((num_rois), dtype=pos_is_gts_.dtype)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list
    def regress_by_class_rbbox(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 5) or (n, 6)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 5*(#class+1)) or (n, 5)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        # import pdb
        # pdb.set_trace()
        assert rois.size(1) == 5 or rois.size(1) == 6

        if not self.reg_class_agnostic:
            label = label * 5
            inds = jt.stack((label, label + 1, label + 2, label + 3, label + 4), 1)
            bbox_pred = jt.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 5

        if rois.size(1) == 5:
            new_rois = delta2dbbox_v3(rois, bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            # choose best Rroi
            new_rois = choose_best_Rroi_batch(new_rois)
        else:
            bboxes = delta2dbbox_v3(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            bboxes = choose_best_Rroi_batch(bboxes)
            new_rois = jt.contrib.concat((rois[:, [0]], bboxes), dim=1)

        return new_rois