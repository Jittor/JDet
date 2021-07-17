import jittor.nn as nn
import jittor as jt

from jdet.utils.registry import LOSSES

def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(jt.sum(weight > 0).float().item(), 1.)
    raw = nn.cross_entropy(pred, label, reduction='none')
    if reduce:
        return jt.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.loss_weight = loss_weight

        if self.use_sigmoid:
#            self.cls_criterion = weighted_binary_cross_entropy
            raise NotImplementedError
        elif self.use_mask:
#            self.cls_criterion = mask_cross_entropy
            raise NotImplementedError
        else:
            self.cls_criterion = weighted_cross_entropy

    def forward(self, cls_score, label, label_weight, *args, **kwargs):
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, label, label_weight, *args, **kwargs)
        return loss_cls
