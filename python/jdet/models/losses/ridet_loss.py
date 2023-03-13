# https://github.com/ming71/RIDet/blob/0ca6882cc4f28c92260caff3d84baaece8fd7c5e/mmdet/models/anchor_heads/anchor_head.py

import jittor as jt 
from jdet.utils.registry import LOSSES
from jittor import nn
import math

def __smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = jt.abs(pred - target)
    flag = (diff<beta).float()
    loss = flag*0.5*diff.sqr()/beta + (1-flag)*(diff - 0.5 * beta)
    return loss

def rotation_mapping(input, target):
    temp_ratios_w = jt.abs(input[:, 0] / (target[:, 0] + 1e-6))
    temp_thetas = input[:, -1] - target[:, -1]

    flag_ratios_w = (temp_ratios_w > 1).float()
    ratios_w = flag_ratios_w*(1 / (temp_ratios_w + 1e-6)) + (1-flag_ratios_w)*temp_ratios_w
    ### original code:
    # dtheta = torch.where(temp_thetas > torch.zeros_like(temp_thetas), temp_thetas, -temp_thetas) % math.pi
    # delta_theta = torch.where((dtheta > torch.zeros_like(dtheta)) & (dtheta < (math.pi * 0.5 * torch.ones_like(dtheta))), \
    #     dtheta, math.pi - dtheta)
    # rotation_metric = 1 / (1 + 1e-6 + ratios_w * torch.cos(delta_theta)) - 0.5
    ### The code is confusing. Cosine value of the acute angle can be represented simply.
    rotation_metric = 1 / (1 + 1e-6 + ratios_w * jt.abs(jt.cos(temp_thetas))) - 0.5
    return rotation_metric

def wh_iou(input, target):
    inter = jt.minimum(input[:, 0], target[:, 0]) * jt.minimum(input[:, 1] ,target[:, 1])
    union = input[:, 0] * input[:, 1]  + target[:, 0] * target[:, 1] - inter
    areac = jt.maximum(input[:, 0] ,target[:, 0]) * jt.maximum(input[:, 1] ,target[:, 1])
    hiou_loss = - jt.log( inter / (union + 1e-6) + 1e-6) + (areac - union) / (areac + 1e-6)
    return hiou_loss

def shape_mapping(input, target):
    return jt.minimum(wh_iou(input[:, [1,0]], target[:, :2]), wh_iou(input[:, [0,1]], target[:, :2]))

def hungarian_shape(input, target):
    target_plus = jt.concat([target[:, [1, 0]], (target[:, -1] + math.pi * 0.5).unsqueeze(1)], -1)
    loss = jt.minimum(10*rotation_mapping(input, target_plus) + 0.1 * shape_mapping(input, target_plus),
                      10*rotation_mapping(input, target) + 0.1 * shape_mapping(input, target))
    return loss

def hungarian_loss_obb(inputs, targets, weight=None, beta=1.0, reduction="mean", avg_factor=None):
    # center-metric
    temp_box_ratio = targets[:, 2] / (targets[:, 3] + 1e-6)
    flag = (temp_box_ratio > jt.ones_like(temp_box_ratio)).float()
    box_ratios = temp_box_ratio * flag + 1 / (temp_box_ratio + 1e-6) * (1-flag)
    smoothl1 = True
    if smoothl1:
        center_dist = __smooth_l1_loss(inputs[:, :2], targets[:, :2], beta).sum(1) 
    else:
        center_dist = (inputs[:, 0] - targets[:, 0])**2 + (inputs[:, 1] - targets[:, 1])**2
    diagonal = (targets[:, 2]**2 + targets[:, 3]**2) 
    center_metric = box_ratios * 0.25 * center_dist / (diagonal + 1e-6)
    # geometry-metric
    geometry_metric = hungarian_shape(inputs[:, 2:], targets[:, 2:])
    loss = center_metric + geometry_metric

    if weight is not None:
        if weight.ndim!=1:
            weight = weight.reshape((weight.shape[0], -1))
            weight = jt.mean(weight, dim=1)
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == "mean":
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()

    return loss 

@LOSSES.register_module()
class RIDetLoss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(RIDetLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = hungarian_loss_obb(
            pred,
            target,
            weight,
            self.beta,
            reduction=reduction,
            avg_factor=avg_factor
        ) * self.loss_weight
        return loss_bbox
