import jittor as jt
from jittor import nn
from .focal_loss import binary_cross_entropy_with_logits

from jdet.utils.registry import LOSSES

def smooth_focal_loss(pred,
                      target,
                      gamma=2.0,
                      alpha=0.25,
                      reduction='mean',
                      avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = binary_cross_entropy_with_logits(pred, 
        target, weight=None, reduction="none") * focal_weight

    if reduction == "mean":
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss

@LOSSES.register_module()
class SmoothFocalLoss(nn.Module):
    """Smooth Focal Loss. Implementation of `Circular Smooth Label (CSL).`__
    __ https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SmoothFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
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
        if (weight is not None) and (not jt.any(weight > 0)) and (reduction != 'none'):
            mask = (weight > 0).detach()
            return (pred[mask] * weight[mask].reshape(-1, 1)).sum()
        if weight is not None and weight.ndim > 1:
            weight = weight.mean(-1)
    
        mask = (weight > 0)
        pred = pred[mask]
        target = target[mask]

        loss_cls = self.loss_weight * smooth_focal_loss(
            pred,
            target,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls
