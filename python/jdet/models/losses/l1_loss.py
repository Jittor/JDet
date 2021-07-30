import jittor as jt 
from jittor import nn 
from jdet.utils.registry import LOSSES

def l1_loss(pred,target,weight=None,avg_factor=None,reduction="mean"):
    loss = jt.abs(pred-target)

    if weight is not None:
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == "mean":
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()

    return loss 

@LOSSES.register_module()
class L1Loss(nn.Module):

    def __init__(self,reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
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
        loss_bbox = self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_bbox