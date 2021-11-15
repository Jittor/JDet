import jittor.nn as nn
import jittor as jt

from jdet.utils.registry import LOSSES

def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(jt.sum(weight > 0).float().item(), 1.)
    raw = nn.cross_entropy_loss(pred, label, reduction='none')
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
class CrossEntropyLossForRcnn(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, loss_weight=1.0):
        super(CrossEntropyLossForRcnn, self).__init__()
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

def cross_entropy_loss(pred,target,weight=None,avg_factor=None,reduction="mean"):
    target = target.reshape((-1, ))
    
    target = target.broadcast(pred, [1])
    target = target.index(1) == target
    
    output = pred - pred.max([1], keepdims=True)
    logsum = output.exp().sum(1).safe_log()
    loss = (logsum - (output*target).sum(1))

    if weight is not None:
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == "mean":
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()

    return loss 

def binary_cross_entropy_with_logits(output, target, pos_weight=None):
    max_val = jt.clamp(-output,min_v=0)
    if pos_weight is not None:
        log_weight = (pos_weight-1)*target + 1
        loss = (1-target)*output+(log_weight*(((-max_val).exp()+(-output - max_val).exp()).log()+max_val))
    else:
        loss = (1-target)*output+max_val+((-max_val).exp()+(-output -max_val).exp()).log()
    
    return loss 

def binary_cross_entropy_loss(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.ndim == label.ndim
    assert class_weight is None 

    loss = binary_cross_entropy_with_logits(pred, label.float())

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
class CrossEntropyLoss(nn.Module):

    def __init__(self,reduction='mean',use_bce=False, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_bce = use_bce

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_func = cross_entropy_loss
        if self.use_bce:
            loss_func = binary_cross_entropy_loss
        loss_bbox = self.loss_weight * loss_func(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_bbox