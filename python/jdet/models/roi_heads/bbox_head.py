import jittor as jt
import jittor.nn as nn

from jdet.utils.registry import HEADS, LOSSES, build_from_cfg
from jdet.utils.general import multi_apply
from jdet.ops.bbox_transfomrs import bbox2delta

def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
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


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = jt.zeros(num_samples, dtype=jt.long)
    label_weights = jt.zeros(num_samples)
    bbox_targets = jt.zeros(num_samples, 4)
    bbox_weights = jt.zeros(num_samples, 4)
    # import pdb
    # pdb.set_trace()
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
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
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#        res.append(correct_k.mul_(100.0 / pred.size(0)))
        correct_k *= 100.0 / pred.shape[0]
        res.append(correct_k)
    return res[0] if return_single else res



@HEADS.register_module()
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
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
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

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

    def get_target(self, sampling_results, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

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
            losses['loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, reduce=reduce)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0))
        return losses