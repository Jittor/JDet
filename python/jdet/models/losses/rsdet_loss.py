import jittor as jt 
from jdet.utils.registry import LOSSES
from jittor import nn

@LOSSES.register_module()
class RSDetLoss(nn.Module):
    # https://github.com/yangxue0827/RotationDetection/blob/507b03f85bff592666a114bfaec526be8d37eac2/alpharotate/libs/models/losses/losses_rsdet.py
    def modulated_rotation_5p_loss(self, preds, targets, anchors, weight=None, sigma=3.0, reduction="mean", avg_factor=None):
        assert targets.shape[0] == preds.shape[0] == anchors.shape[0]
        sigma_squared = sigma ** 2
        reg_diff = jt.abs(preds - targets)
        flags = jt.float32(reg_diff < 1.0 / sigma_squared)
        loss1 = flags * 0.5 * sigma_squared * jt.sqr(reg_diff) + (1-flags) * (reg_diff - 0.5/sigma_squared)
        loss1 = jt.sum(loss1, dim=1)

        # log(w/w_a) - log(h/h_a) + log(w_a) - log(h_a) = log(w/h)
        logr = jt.safe_log(anchors[:, 2]) - jt.safe_log(anchors[:, 3])
        loss2_1 = preds[:, 0] - targets[:, 0]
        loss2_2 = preds[:, 1] - targets[:, 1]
        ### according to original article
        loss2_3 = preds[:, 2] - targets[:, 3] - logr
        loss2_4 = preds[:, 3] - targets[:, 2] + logr
        loss2_5 = jt.abs(preds[:, 4] - targets[:, 4]) - 0.5
        ### according to codes
        # loss2_3 = preds[:, 2] - targets[:, 3] + logr
        # loss2_4 = preds[:, 3] - targets[:, 2] - logr
        # loss2_5 = jt.minimum((jt.abs(preds[:, 4] - targets[:, 4]) - 0.5),
        #                      (jt.abs(targets[:, 4] - preds[:, 4]) - 0.5))
        loss2 = jt.stack([loss2_1, loss2_2, loss2_3, loss2_4, loss2_5], dim=1)
        loss2 = jt.abs(loss2)
        loss2 = jt.sum(loss2, dim=1)
        
        loss = jt.minimum(loss1, loss2)
        
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

    # beta = 1/sigma**2
    def __init__(self, sigma=3.0, loss_weight=1.0, reg_type='5p', reduction='mean') -> None:
        assert reg_type in ['5p', '8p']
        self.sigma = sigma
        self.loss_weight = loss_weight
        self.reg_type = reg_type
        self.reduction = reduction
    
    def execute(self,
                pred,
                target,
                anchors,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.reg_type == '5p':
            loss_bbox = self.loss_weight * self.modulated_rotation_5p_loss(
                pred,
                target,
                anchors,
                weight,
                sigma=self.sigma,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_bbox
