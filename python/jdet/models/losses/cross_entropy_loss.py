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

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = jt.full((labels.size(0), label_channels), 0)
#    inds = jt.nonzero(labels >= 1).squeeze()
    inds = jt.nonzero(labels >= 1)              #TODO: add default squeeze
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        (label_weights.size(0), label_channels))
    return bin_labels, bin_label_weights

def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if pred.ndim != label.ndim:
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if avg_factor is None:
        avg_factor = max(jt.sum(weight > 0).float().item(), 1.)
    return (nn.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        size_average=False)[None] / avg_factor).squeeze(0)          #squeeze(not sure)

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = weighted_binary_cross_entropy
        elif self.use_mask:
#            self.cls_criterion = mask_cross_entropy
            raise NotImplementedError
        else:
            self.cls_criterion = weighted_cross_entropy

    def execute(self, cls_score, label, label_weight, *args, **kwargs):
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, label, label_weight, *args, **kwargs)
        return loss_cls
