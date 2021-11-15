import jittor as jt 
from jdet.utils.registry import LOSSES
from jittor import nn

def smooth_l1_loss(pred,target,weight=None,beta=1.,avg_factor=None,reduction="mean"):
    diff = jt.abs(pred-target)
    if beta!=0.:
        flag = (diff<beta).float()
        loss = flag*0.5* diff.sqr()/beta + (1-flag)*(diff - 0.5 * beta)
    else:
        loss = diff 

    if weight is not None:
        if weight.ndim==1:
            weight = weight[:,None]
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == "mean":
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()

    return loss 


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
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
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_bbox