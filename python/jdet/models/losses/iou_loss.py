import jittor as jt 
from jittor import nn 
from jdet.utils.registry import LOSSES
import warnings
from jdet.models.boxes.iou_calculator import bbox_overlaps


def iou_loss(pred, target, weight=None, avg_factor=None,
             reduction="mean", linear=False, mode='log', eps=1e-6):

    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in '
                      'iou_loss is deprecated, please use "mode=`linear`" '
                      'instead.')
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min_v=eps)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious ** 2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError

    if weight is not None:
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0], 1)

    if reduction == "mean":
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()

    return loss 


@LOSSES.register_module()
class IoULoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_bbox